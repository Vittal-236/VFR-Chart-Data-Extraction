"""
Circle Detector v4 — Step 1 of Letter-in-Circle Pipeline
VFR Chart Extraction Pipeline (FAA Base Model)

What changed from v3
--------------------
Problem 1 (from rejected_crops.png):
    Hundreds of genuine R-in-circle symbols were rejected.
    Root cause: eccentricity threshold 0.90 was too tight.
    The R letter at 12-18px interior scale has a dominant vertical stroke
    → largest blob eccentricity often 0.88-0.95.
    Fix: raise INTERIOR_MAX_ECCENTRICITY back to 0.96.
    Pure diagonal slashes are 0.97-0.99, so 0.96 still catches them.

Problem 2 (from crops.png):
    Large bold text characters (G, Q, 8, B, etc.) were confirmed.
    These are NOT circle symbols — they are large map text that accidentally
    has a closed ink ring around it from surrounding map features.
    Root cause: the Hough correctly finds the ring, but the interior
    contains a very large blob (the letter G fills most of the interior).
    Fix: cap largest_area relative to the inner disk area.
    For inner_rad r, disk area = π×r². A real letter fits in ≤ 40% of that.
    Large text blobs fill 55-90% of the interior → rejected.
    New parameter: INTERIOR_LARGEST_BLOB_MAX_FRAC = 0.45

Filter B2 (diagonal bbox) — REMOVED
    With eccentricity raised to 0.96, the diagonal bbox check is redundant
    and was adding noise. Removed for simplicity.

Final filter set in v4
----------------------
  Step 1  Hough Circle Transform (tiled)
  Step 2  Circumference coverage >= 0.88        (open arcs → reject)
  Step 3  NMS                                   (duplicates → remove)
  Step 4  Interior ink fraction 0.10 – 0.60    (empty/filled → reject)
  Step 5A Interior CC count <= 4               (compass noise → reject)
  Step 5B Eccentricity of largest blob < 0.96  (pure slashes → reject)
  Step 5C Largest blob area <= 45% of disk     (large text → reject)

Expected output: ~150-350 confirmed circles.
The remaining false positives will be handled by the letter reader
(Tesseract/EasyOCR) which will return low confidence for garbage crops.

Outputs
-------
  circles.json        — confirmed circles → feed to letter_reader.py
  annotated_map.png   — confirmed circles on map
  crops.png           — confirmed contact sheet (green border)
  rejected_crops.png  — rejected contact sheet (red border, reason labelled)

HOW TO USE
----------
1. Set BINARY_PATH and OUTPUT_DIR below.
2. python circle_detector_v4.py
3. Inspect crops.png and rejected_crops.png.
4. If crops look clean → run letter_reader.py with circles.json.

Requirements: pip install numpy pillow scikit-image
"""

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from skimage.measure import label, regionprops
from skimage.transform import hough_circle, hough_circle_peaks

Image.MAX_IMAGE_PIXELS = None


# =============================================================================
# CONFIG
# =============================================================================

BINARY_PATH = r"outputs/phase1_preprocessing/Washington_binary.png"
OUTPUT_DIR  = r"outputs/phase4_symbol_detection/circle_detector_v4"
DPI         = 150


# =============================================================================
# TUNING PARAMETERS
# =============================================================================

RADIUS_MIN                 = 17
RADIUS_MAX                 = 26
HOUGH_THRESHOLD            = 0.60
MIN_CIRCUMFERENCE_COVERAGE = 0.88

# Interior ink — letter contributes 0.10-0.55 of interior area
INTERIOR_INK_MIN = 0.10
INTERIOR_INK_MAX = 0.60

# Filter A: max CC blobs inside circle interior
# Single letter = 1-4 blobs. Compass/noise = 5+.
INTERIOR_MAX_CC      = 4
INTERIOR_CC_MIN_AREA = 8    # px² — ignore dust

# Filter B: eccentricity of largest interior blob
# Raised from 0.90 (v3) back to 0.96.
# R letter at small scale: eccentricity 0.88-0.95 (vertical stroke dominates)
# Pure diagonal slash: 0.97-0.99
# Setting 0.96 catches slashes while keeping real letters.
INTERIOR_MAX_ECCENTRICITY = 0.96

# Filter C: largest blob area as fraction of inner disk area
# Inner disk area = π × inner_rad²  where inner_rad = 0.68 × circle_radius
# A real letter fills at most 45% of the inner disk.
# Large text characters (G, Q, B, 8) fill 55-90% of the disk.
INTERIOR_LARGEST_BLOB_MAX_FRAC = 0.45

# NMS
NMS_DIST_FACTOR    = 1.0

# Legend strip
LEGEND_STRIP_WIDTH = 140

# Tiling
TILE_SIZE          = 2048


# =============================================================================
# UTILITIES
# =============================================================================

def load_binary(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("L")) > 128


def _inner_disk(binary: np.ndarray, circ: dict):
    """
    Extract the circular interior of the detected ring (68% of radius).
    Returns (masked_interior bool, mask bool, inner_rad int, disk_area float)
    or (None, None, 0, 0) if out of bounds.
    """
    H, W      = binary.shape
    cr        = circ["row"]
    cc        = circ["col"]
    inner_rad = max(3, int(circ["radius"] * 0.68))

    r0 = max(0, cr - inner_rad)
    r1 = min(H, cr + inner_rad + 1)
    c0 = max(0, cc - inner_rad)
    c1 = min(W, cc + inner_rad + 1)

    patch = binary[r0:r1, c0:c1]
    ph, pw = patch.shape
    if ph == 0 or pw == 0:
        return None, None, 0, 0.0

    pr_c = cr - r0
    pc_c = cc - c0
    rr, cg = np.ogrid[:ph, :pw]
    mask = (rr - pr_c) ** 2 + (cg - pc_c) ** 2 <= inner_rad ** 2

    disk_area = float(mask.sum())
    return patch.copy() & mask, mask, inner_rad, disk_area


def convert_json_safe(obj):
    """
    Recursively convert NumPy types into standard Python types
    so json.dump() can serialize safely.
    """
    import numpy as np

    if isinstance(obj, dict):
        return {k: convert_json_safe(v) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [convert_json_safe(v) for v in obj]

    elif isinstance(obj, np.integer):
        return int(obj)

    elif isinstance(obj, np.floating):
        return float(obj)

    elif isinstance(obj, np.bool_):
        return bool(obj)

    elif isinstance(obj, np.ndarray):
        return obj.tolist()

    return obj

# =============================================================================
# STEP 1 — Hough Circle Transform (tiled)
# =============================================================================

def detect_circles_hough(binary: np.ndarray) -> list:
    H, W    = binary.shape
    radii   = np.arange(RADIUS_MIN, RADIUS_MAX + 1)
    overlap = RADIUS_MAX + 10
    all_c   = []

    tile_rows = list(range(0, H, TILE_SIZE))
    tile_cols = list(range(0, W, TILE_SIZE))
    n_tiles   = len(tile_rows) * len(tile_cols)
    idx       = 0

    for tr in tile_rows:
        for tc in tile_cols:
            idx += 1
            inner_r0, inner_c0 = tr, tc
            inner_r1 = min(tr + TILE_SIZE, H)
            inner_c1 = min(tc + TILE_SIZE, W)

            pad_r0 = max(0, tr - overlap)
            pad_c0 = max(0, tc - overlap)
            pad_r1 = min(H, tr + TILE_SIZE + overlap)
            pad_c1 = min(W, tc + TILE_SIZE + overlap)

            tile = binary[pad_r0:pad_r1, pad_c0:pad_c1].astype(np.uint8)
            print(f"  Tile {idx:3d}/{n_tiles}  "
                  f"[{inner_r0}:{inner_r1}, {inner_c0}:{inner_c1}]", end="")

            hspaces = hough_circle(tile, radii)
            accums, cx_arr, cy_arr, rad_arr = hough_circle_peaks(
                hspaces, radii,
                min_xdistance=RADIUS_MIN,
                min_ydistance=RADIUS_MIN,
                threshold=HOUGH_THRESHOLD,
                num_peaks=500,
            )

            found = 0
            for acc, cx, cy, rad in zip(accums, cx_arr, cy_arr, rad_arr):
                map_r = int(cy) + pad_r0
                map_c = int(cx) + pad_c0
                if not (inner_r0 <= map_r < inner_r1 and
                        inner_c0 <= map_c < inner_c1):
                    continue
                if map_c < LEGEND_STRIP_WIDTH:
                    continue
                all_c.append({
                    "row": map_r, "col": map_c,
                    "radius": int(rad), "hough_score": float(acc),
                })
                found += 1
            print(f"  → {found}")

    return all_c


# =============================================================================
# STEP 2 — Circumference coverage
# =============================================================================

def circumference_coverage(binary: np.ndarray, circ: dict) -> float:
    H, W   = binary.shape
    cr, cc = circ["row"], circ["col"]
    rad    = circ["radius"]
    N      = max(36, int(2 * math.pi * rad))
    hits   = 0
    for i in range(N):
        a = 2 * math.pi * i / N
        r = int(round(cr + rad * math.sin(a)))
        c = int(round(cc + rad * math.cos(a)))
        if 0 <= r < H and 0 <= c < W and binary[r, c]:
            hits += 1
    return hits / N


# =============================================================================
# STEP 3 — NMS
# =============================================================================

def nms_circles(circles: list) -> list:
    if not circles:
        return []
    circles    = sorted(circles, key=lambda x: -x["hough_score"])
    kept       = []
    suppressed = set()
    for i, c in enumerate(circles):
        if i in suppressed:
            continue
        kept.append(c)
        for j, d in enumerate(circles[i + 1:], start=i + 1):
            if j in suppressed:
                continue
            dist = math.hypot(c["row"] - d["row"], c["col"] - d["col"])
            if dist < NMS_DIST_FACTOR * (c["radius"] + d["radius"]) / 2:
                suppressed.add(j)
    return kept


# =============================================================================
# STEP 4 — Interior ink fraction
# =============================================================================

def interior_ink(binary: np.ndarray, circ: dict) -> float:
    interior, mask, _, disk_area = _inner_disk(binary, circ)
    if interior is None or disk_area == 0:
        return 0.0
    return float(interior.sum()) / disk_area


# =============================================================================
# STEP 5 — Interior blob analysis (A, B, C)
# =============================================================================

def analyse_interior(binary: np.ndarray, circ: dict) -> dict:
    """
    Filter A — CC count <= INTERIOR_MAX_CC
        Rejects compass fragments and noisy interiors.

    Filter B — Eccentricity of largest blob < INTERIOR_MAX_ECCENTRICITY (0.96)
        Pure diagonal slashes: 0.97-0.99.
        Letter R vertical stroke at small scale: 0.88-0.95 → passes.

    Filter C — Largest blob area <= INTERIOR_LARGEST_BLOB_MAX_FRAC × disk_area
        Real letters fill at most 45% of the inner disk.
        Large text characters (G, Q, 8) fill 55-90% → rejected.
        disk_area = π × inner_rad²
    """

    result = {
        "cc_count": 0,
        "max_eccentricity": 0.0,
        "largest_area": 0,
        "largest_blob_frac": 0.0,
        "disk_area": 0.0,
        "passes_A": False,
        "passes_B": False,
        "passes_C": False,
    }

    interior, mask, inner_rad, disk_area = _inner_disk(binary, circ)

    if interior is None or disk_area == 0:
        return result

    result["disk_area"] = float(round(disk_area, 1))

    lbl = label(interior, connectivity=2)

    props = [
        p for p in regionprops(lbl)
        if int(p.area) >= INTERIOR_CC_MIN_AREA
    ]

    if not props:
        return result

    result["cc_count"] = int(len(props))
    result["passes_A"] = bool(len(props) <= INTERIOR_MAX_CC)

    largest = max(props, key=lambda p: p.area)

    largest_area = int(largest.area)
    largest_ecc = float(round(float(largest.eccentricity), 4))

    blob_frac = float(largest.area) / float(disk_area)

    result["largest_area"] = largest_area
    result["max_eccentricity"] = largest_ecc
    result["largest_blob_frac"] = float(round(blob_frac, 4))

    result["passes_B"] = bool(
        float(largest.eccentricity) < INTERIOR_MAX_ECCENTRICITY
    )

    result["passes_C"] = bool(
        blob_frac <= INTERIOR_LARGEST_BLOB_MAX_FRAC
    )

    return result


# =============================================================================
# STEP 6 — Save outputs
# =============================================================================

def save_annotated_map(binary: np.ndarray, circles: list, out_path: str):
    rgb  = np.stack([binary * 255] * 3, axis=-1).astype(np.uint8)
    img  = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    for c in circles:
        cr, cc, rad = c["row"], c["col"], c["radius"]
        draw.ellipse([cc-rad-3, cr-rad-3, cc+rad+3, cr+rad+3],
                     outline=(0, 255, 0), width=2)
        draw.ellipse([cc-2, cr-2, cc+2, cr+2], fill=(0, 255, 0))
    img.save(out_path)
    print(f"  Annotated map → {out_path}")


def _crop_sheet(binary: np.ndarray, circles: list, out_path: str,
                max_show: int = 300,
                border_colour: tuple = (0, 220, 80)):
    if not circles:
        print(f"  No circles — skipping {out_path}")
        return

    H, W     = binary.shape
    sorted_c = sorted(circles, key=lambda x: -x["hough_score"])
    cell = 80
    cols = 15
    n    = min(len(sorted_c), max_show)
    rows = math.ceil(n / cols)

    sheet = Image.new("RGB", (cell * cols, cell * rows), (20, 20, 20))
    draw  = ImageDraw.Draw(sheet)

    for i, c in enumerate(sorted_c[:n]):
        cr, cc, rad = c["row"], c["col"], c["radius"]
        pad = rad + 8
        r0 = max(0, cr-pad); r1 = min(H, cr+pad)
        c0 = max(0, cc-pad); c1 = min(W, cc+pad)
        crop = binary[r0:r1, c0:c1]
        ch, cw = crop.shape
        if ch == 0 or cw == 0:
            continue

        scale = min((cell-6)/ch, (cell-6)/cw, 4.0)
        nw = max(1, int(cw*scale))
        nh = max(1, int(ch*scale))
        patch = (Image.fromarray((crop*255).astype(np.uint8))
                 .resize((nw, nh), Image.NEAREST).convert("RGB"))

        col_i = i % cols
        row_i = i // cols
        ox = col_i*cell + (cell-nw)//2
        oy = row_i*cell + (cell-nh)//2
        sheet.paste(patch, (ox, oy))

        draw.rectangle(
            [col_i*cell, row_i*cell,
             (col_i+1)*cell-1, (row_i+1)*cell-1],
            outline=border_colour, width=2)

        # Labels
        ink  = c.get("interior_ink_frac", 0)
        ecc  = c.get("max_eccentricity", 0)
        bfrac = c.get("largest_blob_frac", 0)
        rej  = c.get("reject_reason", "")
        draw.text((col_i*cell+2, row_i*cell+2),
                  f"i={ink:.2f} e={ecc:.2f}",
                  fill=(255, 255, 200))
        draw.text((col_i*cell+2, row_i*cell+13),
                  f"bf={bfrac:.2f}",
                  fill=(200, 220, 200))
        if rej:
            draw.text((col_i*cell+2, row_i*cell+24),
                      rej[:14], fill=(255, 120, 80))

    sheet.save(out_path)
    print(f"  Crop sheet → {out_path}  ({n}/{len(circles)} shown)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {BINARY_PATH}")
    binary = load_binary(BINARY_PATH)
    H, W   = binary.shape
    print(f"  {W} × {H} px  |  ink {binary.mean()*100:.2f}%\n")

    print("Step 1 — Hough Circle Transform ...")
    raw = detect_circles_hough(binary)
    print(f"  Raw: {len(raw)}\n")

    print(f"Step 2 — Circumference coverage (≥{MIN_CIRCUMFERENCE_COVERAGE}) ...")
    after_circ = []
    for c in raw:
        cov = circumference_coverage(binary, c)
        c["circumference_coverage"] = round(cov, 4)
        if cov >= MIN_CIRCUMFERENCE_COVERAGE:
            after_circ.append(c)
    print(f"  {len(after_circ)} pass  ({len(raw)-len(after_circ)} rejected)\n")

    print("Step 3 — NMS ...")
    after_nms = nms_circles(after_circ)
    print(f"  {len(after_nms)} pass  ({len(after_circ)-len(after_nms)} removed)\n")

    print(f"Step 4 — Interior ink ({INTERIOR_INK_MIN}–{INTERIOR_INK_MAX}) ...")
    after_ink    = []
    rejected_ink = []
    for c in after_nms:
        ink = interior_ink(binary, c)
        c["interior_ink_frac"] = round(ink, 4)
        if INTERIOR_INK_MIN <= ink <= INTERIOR_INK_MAX:
            after_ink.append(c)
        else:
            c["reject_reason"] = f"ink={ink:.3f}"
            rejected_ink.append(c)
    print(f"  {len(after_ink)} pass  ({len(rejected_ink)} rejected)\n")

    print("Step 5 — Blob analysis (A=CC, B=eccentricity, C=blob fraction) ...")
    confirmed     = []
    rejected_blob = []

    for c in after_ink:
        info = analyse_interior(binary, c)
        c.update(info)

        if info["passes_A"] and info["passes_B"] and info["passes_C"]:
            confirmed.append(c)
        else:
            reasons = []
            if not info["passes_A"]:
                reasons.append(f"cc={info['cc_count']}")
            if not info["passes_B"]:
                reasons.append(f"ecc={info['max_eccentricity']:.2f}")
            if not info["passes_C"]:
                reasons.append(f"bf={info['largest_blob_frac']:.2f}")
            c["reject_reason"] = " | ".join(reasons)
            rejected_blob.append(c)

    n_A = sum(1 for c in rejected_blob if not c.get("passes_A", True))
    n_B = sum(1 for c in rejected_blob if not c.get("passes_B", True))
    n_C = sum(1 for c in rejected_blob if not c.get("passes_C", True))

    print(f"  Confirmed : {len(confirmed)}")
    print(f"  Rejected  : {len(rejected_blob)}")
    print(f"    A (CC count)      : {n_A}  compass/noise")
    print(f"    B (eccentricity)  : {n_B}  diagonal slashes")
    print(f"    C (blob fraction) : {n_C}  large text chars\n")

    # Save
    print("Saving ...")
    out_json = {
        "total_confirmed": len(confirmed),
        "params": {
            "binary_path":                    BINARY_PATH,
            "dpi":                            DPI,
            "radius_min":                     RADIUS_MIN,
            "radius_max":                     RADIUS_MAX,
            "hough_threshold":                HOUGH_THRESHOLD,
            "min_circumference_coverage":     MIN_CIRCUMFERENCE_COVERAGE,
            "interior_ink_min":               INTERIOR_INK_MIN,
            "interior_ink_max":               INTERIOR_INK_MAX,
            "interior_max_cc":                INTERIOR_MAX_CC,
            "interior_max_eccentricity":      INTERIOR_MAX_ECCENTRICITY,
            "interior_largest_blob_max_frac": INTERIOR_LARGEST_BLOB_MAX_FRAC,
        },
        "circles": confirmed,
    }
    json_path = out_dir / "circles.json"
    
    safe_json = convert_json_safe(out_json)

    with open(json_path, "w") as f:
        json.dump(safe_json, f, indent=2)
    print(f"  circles.json → {json_path}")

    save_annotated_map(binary, confirmed, str(out_dir / "annotated_map.png"))
    _crop_sheet(binary, confirmed, str(out_dir / "crops.png"),
                border_colour=(0, 220, 80))
    _crop_sheet(binary, rejected_blob, str(out_dir / "rejected_crops.png"),
                max_show=300, border_colour=(200, 60, 0))

    print()
    print("=" * 55)
    print("CIRCLE DETECTION v4 COMPLETE")
    print("=" * 55)
    print(f"  Raw Hough           : {len(raw)}")
    print(f"  After circumference : {len(after_circ)}")
    print(f"  After NMS           : {len(after_nms)}")
    print(f"  After ink           : {len(after_ink)}")
    print(f"  Confirmed           : {len(confirmed)}  ← feed to letter_reader.py")
    print()
    print("  Each cell shows:")
    print("    i=  interior ink fraction")
    print("    e=  eccentricity (< 0.96 = pass)")
    print("    bf= blob fraction of disk (< 0.45 = pass)")
    print()
    print(f"  Output: {out_dir.resolve()}")
    print()
    print("Tuning:")
    print(f"  Slashes still passing → lower INTERIOR_MAX_ECCENTRICITY "
          f"(now {INTERIOR_MAX_ECCENTRICITY})")
    print(f"  Large text still in  → lower INTERIOR_LARGEST_BLOB_MAX_FRAC "
          f"(now {INTERIOR_LARGEST_BLOB_MAX_FRAC})")
    print(f"  Real symbols lost    → raise INTERIOR_LARGEST_BLOB_MAX_FRAC "
          f"or INTERIOR_MAX_ECCENTRICITY")
    print()
    print("Next: run letter_reader.py with circles.json as input.")


if __name__ == "__main__":
    main()