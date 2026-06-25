"""
Phase 4f — Power Transmission Line Tower Detection
VFR Chart Extraction Pipeline — Cognida.ai

UNIVERSAL MODE — works on any FAA VFR Sectional Chart.

APPROACH — ROTATION-NORMALIZED SHAPE MATCHING
---------------------------------------------
Templates are built ONCE from the Washington chart using known tower
coordinates, then saved to disk. On any subsequent map (Chicago, Seattle,
etc.), the saved templates are loaded directly — no known coordinates needed.

Two modes:
  MODE 1 — BUILD:  run on Washington chart with known coords → saves templates.pkl
  MODE 2 — DETECT: run on any chart → loads templates.pkl, runs detection

Pipeline (detect mode):
  1. Load saved templates (rotation-normalized 48x48 binary shapes)
  2. Per tile: extract black ink (HSV S<60 V<80), label blobs, geometry pre-filter
  3. Rotation-normalize each candidate, compute max IoU vs all templates
  4. Accept if IoU >= IOU_THRESHOLD. NMS. Save outputs.

OUTPUTS
-------
outputs/phase4_symbol_detection/phase4f_towers/
    templates.pkl       — saved template bank (build mode only)
    detections.json     — confirmed detections
    confirmed_map.png   — detections overlaid on input RGB
    confirmed_crops.png — contact sheet of RGB crops

Usage (PowerShell):
  Build templates (run once on Washington):
    python phase4f_tower_detection.py --build

  Detect on any map (default):
    python phase4f_tower_detection.py
    python phase4f_tower_detection.py --map outputs\\phase1_preprocessing\\Chicago_rgb.png
"""

import sys, json, math, time, pickle
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
from skimage.measure import label, regionprops

# =============================================================================
# CONFIG
# =============================================================================

# Source for template building — Washington chart
WASHINGTON_RGB  = r"outputs\phase1_preprocessing\Washington_rgb_300dpi.png"

# Template bank — saved once, reused on all maps
TEMPLATES_PKL   = r"outputs\phase4_symbol_detection\phase4f_towers\templates.pkl"

# Detection input — override with --map argument
DEFAULT_RGB     = r"outputs\phase1_preprocessing\Washington_rgb_300dpi.png"

OUTPUT_DIR      = r"outputs\phase4_symbol_detection\phase4f_towers"

TILE_SIZE       = 2048
TILE_OVERLAP    = 128

# HSV black ink mask (Phase 2d calibration)
HSV_S_MAX       = 60
HSV_V_MAX       = 80

# Loose geometry pre-filter (generous margins — just cuts candidate count)
# All values derived from 26 confirmed Washington tower instances
PF_MIN_AREA     = 35
PF_MAX_AREA     = 220
PF_MIN_SOL      = 0.18
PF_MAX_SOL      = 0.52
PF_MIN_ECCEN    = 0.65
PF_MAX_ECCEN    = 0.995
PF_MIN_MAJ      = 15.0
PF_MAX_MAJ      = 50.0

# Shape matching
TEMPLATE_SIZE   = 48
IOU_THRESHOLD   = 0.45   # all 25 found Washington towers scored 1.00 at this threshold

# Exclusion zones (fraction of image)
LEGEND_FRAC     = 0.098
BOTTOM_FRAC     = 0.088

NMS_RADIUS      = 20
CROP_HALF       = 48

# Isolation check — reject if any OTHER black blob centroid is within this
# radius of the candidate. Text letters always have adjacent letter blobs.
# Tower symbols sit in isolation. Measured gap between adjacent letters ~8-15px.
ISOLATION_R     = 18   # px — reject if another blob centroid is within this

# Known tower coordinates on Washington chart — ONLY used in build mode
WASHINGTON_KNOWN = [
    (1967,1656),(3065,1694),(3592,1671),(4405,1632),(5404,1773),
    (5486,1834),(6853,1812),(7057,1785),(7220,1625),(7602,1608),
    (9253,1762),(7715,1515),(7642,1486),(7617,65),(11068,570),
    (11460,561),(8228,865),(7918,268),(7615,67),(7289,263),
    (7000,161),(5791,554),(5637,556),(4749,434),(4064,144),(3326,354),
]

# =============================================================================
# HELPERS
# =============================================================================

def extract_black_hsv(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return ((hsv[:,:,1] < HSV_S_MAX) & (hsv[:,:,2] < HSV_V_MAX)).astype(np.uint8)


def normalize_shape(blob_mask, orientation_rad, size=TEMPLATE_SIZE):
    """Rotation-normalize a blob to canonical orientation, resize to size×size."""
    h, w = blob_mask.shape
    s    = int(max(h, w) * 1.6) + 4
    padded = np.zeros((s, s), dtype=np.uint8)
    y0 = (s - h) // 2;  x0 = (s - w) // 2
    padded[y0:y0+h, x0:x0+w] = blob_mask
    angle   = -math.degrees(orientation_rad)
    M       = cv2.getRotationMatrix2D((s/2.0, s/2.0), angle, 1.0)
    rotated = cv2.warpAffine(padded, M, (s, s), flags=cv2.INTER_NEAREST)
    ys, xs  = np.where(rotated > 0)
    if len(ys) == 0:
        return np.zeros((size, size), dtype=np.uint8)
    crop = rotated[ys.min():ys.max()+1, xs.min():xs.max()+1]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_NEAREST)


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def match_score(norm_shape, templates):
    """Max IoU vs all templates, testing both 180° polarities."""
    flipped = np.rot90(norm_shape, 2)
    best    = 0.0
    for t in templates:
        s = max(iou(norm_shape, t), iou(flipped, t))
        if s > best:
            best = s
    return best


def nms(detections, radius):
    if not detections: return []
    scored = sorted(detections, key=lambda d: -d["score"])
    kept   = []
    for det in scored:
        r, c = det["map_r"], det["map_c"]
        if not any(math.sqrt((r-k["map_r"])**2+(c-k["map_c"])**2) < radius
                   for k in kept):
            kept.append(det)
    return kept

# =============================================================================
# MODE 1 — BUILD TEMPLATE BANK
# =============================================================================

def build_and_save_templates():
    """
    Extract rotation-normalized templates from Washington chart using known
    tower coordinates. Save to TEMPLATES_PKL for reuse on any chart.
    """
    print(f"BUILD MODE — loading Washington chart: {WASHINGTON_RGB}")
    rgb  = np.array(Image.open(WASHINGTON_RGB).convert("RGB"))
    H, W = rgb.shape[:2]
    print(f"  {W}x{H}px")

    templates = []
    win       = 60   # half-window around each known coord

    for tx, ty in WASHINGTON_KNOWN:
        r0 = max(0, ty-win); r1 = min(H, ty+win)
        c0 = max(0, tx-win); c1 = min(W, tx+win)
        crop   = rgb[r0:r1, c0:c1]
        binary = extract_black_hsv(crop)
        lab    = label(binary, connectivity=2)
        props  = regionprops(lab)

        best_p, best_d = None, 1e9
        for p in props:
            if p.area < PF_MIN_AREA: continue
            cr, cc = p.centroid
            d = math.sqrt((cr-(ty-r0))**2 + (cc-(tx-c0))**2)
            if d < best_d:
                best_d, best_p = d, p

        if best_p is None or best_d > 15:
            print(f"  ({tx},{ty})  SKIPPED (nearest={best_d:.0f}px)")
            continue

        blob = (lab == best_p.label).astype(np.uint8)
        rr0, cc0, rr1, cc1 = best_p.bbox
        tmpl = normalize_shape(blob[rr0:rr1, cc0:cc1], best_p.orientation)
        templates.append(tmpl)
        print(f"  ({tx},{ty})  OK  area={best_p.area:.0f}  dist={best_d:.1f}")

    print(f"\n  Templates extracted: {len(templates)}/{len(WASHINGTON_KNOWN)}")

    # Save
    pkl_path = Path(TEMPLATES_PKL)
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pkl_path, "wb") as f:
        pickle.dump(templates, f)
    print(f"  Saved → {pkl_path}")

    # Also save a visual sheet
    sheet_path = str(pkl_path.parent / "templates.png")
    cell  = TEMPLATE_SIZE * 3
    sheet = Image.new("RGB", (cell*len(templates), cell), (20,20,20))
    for i, t in enumerate(templates):
        img = Image.fromarray(t*255).resize((cell,cell), Image.NEAREST).convert("RGB")
        sheet.paste(img, (i*cell, 0))
    sheet.save(sheet_path)
    print(f"  Visual → {sheet_path}")
    return templates

# =============================================================================
# MODE 2 — DETECT ON ANY CHART
# =============================================================================

def load_templates():
    pkl_path = Path(TEMPLATES_PKL)
    if not pkl_path.exists():
        print(f"ERROR: templates.pkl not found at {pkl_path}")
        print("  Run with --build first to generate templates from Washington chart.")
        sys.exit(1)
    with open(pkl_path, "rb") as f:
        templates = pickle.load(f)
    print(f"  Loaded {len(templates)} templates from {pkl_path}")
    return templates


def isolation_check(cr_t, cc_t, props, own_label, map_r_off, map_c_off,
                    pf_min_area=8):
    """
    Return True (isolated = keep) if no OTHER blob of area >= pf_min_area
    has its centroid within ISOLATION_R of this candidate.
    Checks blobs in the same tile (props from same tile label array).
    """
    for p in props:
        if p.label == own_label:
            continue
        if p.area < pf_min_area:
            continue
        pr, pc = p.centroid
        dist = math.sqrt((pr - cr_t)**2 + (pc - cc_t)**2)
        if dist < ISOLATION_R:
            return False   # neighbour found — likely text
    return True


def detect(rgb_path, templates):
    print(f"Loading map: {rgb_path}")
    rgb_full = np.array(Image.open(rgb_path).convert("RGB"))
    H, W     = rgb_full.shape[:2]
    print(f"  {W}x{H}px")

    legend_x = int(W * LEGEND_FRAC)
    bottom_y = int(H * (1.0 - BOTTOM_FRAC))
    print(f"  Exclusion: x<{legend_x}  y>{bottom_y}")
    print(f"  IOU_THRESHOLD = {IOU_THRESHOLD}")

    tile_rows = list(range(0, H, TILE_SIZE))
    tile_cols = list(range(0, W, TILE_SIZE))
    n_tiles   = len(tile_rows) * len(tile_cols)
    idx = 0; detections = []; n_pf = 0

    print(f"\nTiled detection ({n_tiles} tiles) ...")

    for tr in tile_rows:
        for tc in tile_cols:
            idx += 1
            inner_r0 = tr;                    inner_c0 = tc
            inner_r1 = min(tr+TILE_SIZE, H);  inner_c1 = min(tc+TILE_SIZE, W)
            pad_r0   = max(0, tr-TILE_OVERLAP); pad_c0 = max(0, tc-TILE_OVERLAP)
            pad_r1   = min(H, tr+TILE_SIZE+TILE_OVERLAP)
            pad_c1   = min(W, tc+TILE_SIZE+TILE_OVERLAP)

            tile   = rgb_full[pad_r0:pad_r1, pad_c0:pad_c1]
            binary = extract_black_hsv(tile)
            lab    = label(binary, connectivity=2)
            props  = regionprops(lab)

            hits = 0
            for p in props:
                cr_t, cc_t = p.centroid
                map_r = cr_t + pad_r0;  map_c = cc_t + pad_c0
                if not (inner_r0 <= map_r < inner_r1 and
                        inner_c0 <= map_c < inner_c1):
                    continue
                if map_c < legend_x or map_r > bottom_y:
                    continue

                area = p.area; sol = float(p.solidity)
                ecc  = float(p.eccentricity)
                maj  = float(p.axis_major_length)

                if not (PF_MIN_AREA  <= area <= PF_MAX_AREA  and
                        PF_MIN_SOL   <= sol  <= PF_MAX_SOL   and
                        PF_MIN_ECCEN <= ecc  <= PF_MAX_ECCEN and
                        PF_MIN_MAJ   <= maj  <= PF_MAX_MAJ):
                    continue
                n_pf += 1

                rr0, cc0, rr1, cc1 = p.bbox
                blob  = (lab[rr0:rr1, cc0:cc1] == p.label).astype(np.uint8)
                shape = normalize_shape(blob, p.orientation)
                score = match_score(shape, templates)
                if score < IOU_THRESHOLD:
                    continue

                # Isolation check — reject if adjacent blob exists (text context)
                if not isolation_check(cr_t, cc_t, props, p.label,
                                       pad_r0, pad_c0):
                    continue

                detections.append({
                    "map_r":    round(float(map_r), 1),
                    "map_c":    round(float(map_c), 1),
                    "area":     int(area),
                    "solidity": round(sol, 3),
                    "eccen":    round(ecc, 3),
                    "maj_ax":   round(maj, 1),
                    "score":    round(score, 3),
                })
                hits += 1

            print(f"  [{idx:3d}/{n_tiles}] "
                  f"rows {inner_r0:5d}-{inner_r1:5d}  "
                  f"cols {inner_c0:5d}-{inner_c1:5d}  → {hits:3d} hits")

    print(f"\n  Pre-filter passed : {n_pf}")
    print(f"  Shape match passed: {len(detections)}")
    confirmed = nms(detections, NMS_RADIUS)
    print(f"  After NMS         : {len(confirmed)}")
    return confirmed

# =============================================================================
# OUTPUTS
# =============================================================================

def save_outputs(rgb_path, confirmed, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir/"detections.json", "w") as f:
        json.dump({"source": str(rgb_path),
                   "total_confirmed": len(confirmed),
                   "iou_threshold": IOU_THRESHOLD,
                   "confirmed": confirmed}, f, indent=2)
    print(f"  detections.json → {out_dir/'detections.json'}")

    # confirmed_map
    print("  Building confirmed map ...")
    img  = Image.open(rgb_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for d in confirmed:
        r,c = int(round(d["map_r"])), int(round(d["map_c"]))
        draw.line([(c-10,r),(c+10,r)], fill=(255,0,0), width=2)
        draw.line([(c,r-10),(c,r+10)], fill=(255,0,0), width=2)
        draw.ellipse([c-5,r-5,c+5,r+5], outline=(255,0,0), width=2)
    img.save(str(out_dir/"confirmed_map.png"))
    print(f"  confirmed_map → {out_dir/'confirmed_map.png'}")

    # confirmed_crops
    print("  Building crops ...")
    rgb  = np.array(Image.open(rgb_path).convert("RGB"))
    H,W  = rgb.shape[:2]
    cell = CROP_HALF*2; cols = 16
    n    = min(len(confirmed), 400)
    rows_n = math.ceil(n/cols) if n > 0 else 1
    sheet  = Image.new("RGB", (cell*cols, cell*rows_n), (20,20,20))
    sdraw  = ImageDraw.Draw(sheet)
    for i, d in enumerate(confirmed[:400]):
        r,c = int(round(d["map_r"])), int(round(d["map_c"]))
        r0=max(0,r-CROP_HALF); r1=min(H,r+CROP_HALF)
        c0=max(0,c-CROP_HALF); c1=min(W,c+CROP_HALF)
        crop = rgb[r0:r1,c0:c1]
        if crop.shape[0]==0 or crop.shape[1]==0: continue
        patch = Image.fromarray(crop).resize((cell,cell), Image.LANCZOS)
        pd = ImageDraw.Draw(patch)
        pd.line([(cell//2-8,cell//2),(cell//2+8,cell//2)], fill=(255,0,0), width=1)
        pd.line([(cell//2,cell//2-8),(cell//2,cell//2+8)], fill=(255,0,0), width=1)
        pd.text((2,2), f"{d['score']:.2f}", fill=(255,255,0))
        col_i=i%cols; row_i=i//cols
        sheet.paste(patch, (col_i*cell, row_i*cell))
        sdraw.rectangle([col_i*cell, row_i*cell,
                         (col_i+1)*cell-1, (row_i+1)*cell-1],
                        outline=(60,60,60), width=1)
    sheet.save(str(out_dir/"confirmed_crops.png"))
    print(f"  confirmed_crops → {out_dir/'confirmed_crops.png'}  ({n} shown)")

# =============================================================================
# MAIN
# =============================================================================

def main():
    t0       = time.time()
    build    = "--build" in sys.argv
    map_arg  = next((sys.argv[sys.argv.index("--map")+1]
                     for i,a in enumerate(sys.argv) if a=="--map"), None) \
               if "--map" in sys.argv else None
    rgb_path = map_arg if map_arg else DEFAULT_RGB

    out_dir  = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    if build:
        templates = build_and_save_templates()
    else:
        print("DETECT MODE")
        templates = load_templates()

    confirmed = detect(rgb_path, templates)

    # Recall check (only meaningful on Washington chart with known coords)
    if str(rgb_path) == str(WASHINGTON_RGB) or "Washington" in str(rgb_path):
        known = WASHINGTON_KNOWN
        print(f"\nRecall check ({len(known)} known Washington towers):")
        found = 0
        for tx, ty in known:
            if not confirmed:
                print(f"  ({tx:5d},{ty:4d})  MISSED"); continue
            nearest = min(confirmed,
                          key=lambda d: math.sqrt((d["map_r"]-ty)**2+(d["map_c"]-tx)**2))
            dist = math.sqrt((nearest["map_r"]-ty)**2+(nearest["map_c"]-tx)**2)
            hit  = dist < 30
            found += int(hit)
            print(f"  ({tx:5d},{ty:4d})  {'FOUND' if hit else 'MISSED'}  "
                  f"dist={dist:.0f}  score={nearest['score']:.2f}")
        print(f"\n  Recall: {found}/{len(known)}  ({100*found/len(known):.0f}%)")

    save_outputs(rgb_path, confirmed, OUTPUT_DIR)

    elapsed = round(time.time()-t0, 1)
    print(f"\n=== DONE in {elapsed}s ===")
    print(f"  Templates : {len(templates)}")
    print(f"  Threshold : {IOU_THRESHOLD}")
    print(f"  Confirmed : {len(confirmed)}")
    print(f"  Outputs   : {out_dir.resolve()}")

if __name__ == "__main__":
    main()