"""
PB3 Full-Map Circle Detector
VFR Chart Extraction Pipeline

Scales the 3-Point Perpendicular Bisector (PB3) method to the full chart.
Processes the entire image in 2048px tiles with RADIUS_MAX+10 px overlap
so circles at tile boundaries are never missed.

Core geometry (same as pb3_circle_detector.py)
-----------------------------------------------
  3 points on a circle → two perpendicular bisectors → exact centre
  No voting accumulator. Cluster count = confidence score.
  Data-driven thresholds:
    MIN_CLUSTER_COUNT = 80  (true circles: n=228-933, curves: n=13-54)
    MIN_COVERAGE      = 0.25 (true circles: c=0.27-0.78)

Tiling strategy
---------------
  - Tile size: 2048 × 2048 px
  - Overlap:   RADIUS_MAX + 10 = 38 px on every side
  - Each tile is padded with the overlap before processing
  - A detection is kept only if its centre falls inside the INNER tile
    (not the padded region) → prevents duplicate detections at boundaries
  - Legend strip (left 140 px) is suppressed

Changes vs test script
-----------------------
  - N_TRIPLETS raised to 1000 for better recall on borderline circles
  - Tile overlap added (38 px) to fix clipped-ring false negatives
  - Per-tile visualisation removed (too many tiles) — single annotated map saved
  - Full circles.json output added with centre, radius, cluster count, coverage
  - Progress bar printed per tile (tile N/total)

Outputs
-------
  circles.json        — all confirmed circles with pixel coords + metadata
  annotated_map.png   — full chart with green rings on detections
  summary.txt         — total count and parameter log

HOW TO USE
----------
1. Set IMAGE_PATH and OUTPUT_DIR below.
2. python pb3_fullmap.py
3. Inspect annotated_map.png — green rings should land on symbol circles.
4. Pass circles.json to letter_reader.py.

Requirements: numpy, pillow, scikit-image, scipy
"""

import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage import feature, measure

Image.MAX_IMAGE_PIXELS = None


# =============================================================================
# CONFIG
# =============================================================================

# Input image — use the binary or the annotated_map from circle_detector_v4
# Clean binary gives stronger edges and higher accuracy.
# annotated_map.png works but is noisier.
IMAGE_PATH = r"outputs/phase4_symbol_detection/circle_detector_v4/annotated_map.png"
IS_BINARY  = False   # True = clean binary (0/255), False = greyscale chart

OUTPUT_DIR = r"outputs/phase4_symbol_detection/pb3_fullmap"

RANDOM_SEED = 42


# =============================================================================
# TUNING PARAMETERS
# (data-driven from tile_A/B/C test — do not change unless results are bad)
# =============================================================================

RADIUS_MIN = 15      # px — minimum expected symbol circle radius
RADIUS_MAX = 28      # px — maximum expected symbol circle radius

CANNY_SIGMA = 1.0    # edge detection smoothing

MIN_CC_LEN  = 12     # skip edge components shorter than this

N_TRIPLETS  = 120   # random triplets sampled per edge component
                     # (raised from 500 to catch borderline circles)

MIN_CHORD   = 8      # px — minimum distance between any two points in a triplet

CLUSTER_CELL = 4     # px — grid cell size for clustering centre estimates

MIN_CLUSTER_COUNT = 80   # minimum triplet estimates in a cluster to confirm
                          # gap between true circles (228-933) and curves (13-54)

MIN_COVERAGE = 0.25  # minimum circumference edge coverage to confirm

MIN_CIRCLE_DIST = 15 # px — NMS distance (remove duplicate detections)

LEGEND_STRIP_WIDTH = 140  # px from left edge — suppress legend symbols

# Tiling
TILE_SIZE = 2048     # px — tile width and height
TILE_OVERLAP = RADIUS_MAX + 10  # px — overlap between adjacent tiles (38 px)


# =============================================================================
# GEOMETRY
# =============================================================================

def perp_bisector_intersection(ax, ay, bx, by, cx, cy):
    """
    Exact circle centre from 3 points via perpendicular bisector intersection.
    Returns (ox, oy) or None if points are collinear.
    """
    mx1, my1 = (ax + bx) / 2.0, (ay + by) / 2.0
    mx2, my2 = (bx + cx) / 2.0, (by + cy) / 2.0

    dx1, dy1 = -(by - ay), (bx - ax)
    dx2, dy2 = -(cy - by), (cx - bx)

    denom = dx1 * (-dy2) - (-dx2) * dy1
    if abs(denom) < 1e-8:
        return None

    t = ((mx2 - mx1) * (-dy2) - (-dx2) * (my2 - my1)) / denom
    return mx1 + t * dx1, my1 + t * dy1


def circumference_coverage(edges, cx, cy, r, H, W):
    """Fraction of circle circumference that lies on Canny edge pixels."""
    N = max(24, int(2 * math.pi * r))
    hits = 0
    for k in range(N):
        angle = 2 * math.pi * k / N
        row = int(round(cy + r * math.sin(angle)))
        col = int(round(cx + r * math.cos(angle)))
        if 0 <= row < H and 0 <= col < W and edges[row, col]:
            hits += 1
    return hits / N


# =============================================================================
# PB3 DETECTOR (on a single tile)
# =============================================================================

def detect_pb3_tile(gray_tile: np.ndarray) -> list:
    """
    Run PB3 on one greyscale tile.
    Returns list of dicts: {cx, cy, r, cov, count}
    Coordinates are in TILE space (not map space — caller adds offset).
    """
    H, W = gray_tile.shape

    # Canny edges
    edges = feature.canny(gray_tile, sigma=CANNY_SIGMA)

    # Connected edge components
    lbl_map = measure.label(edges, connectivity=2)
    comps   = [p.coords for p in measure.regionprops(lbl_map)
               if len(p.coords) >= MIN_CC_LEN]

    if not comps:
        return []

    # Sample triplets → collect centre estimates
    estimates = []
    for comp in comps:
        n = len(comp)
        if n < 6:
            continue
        n_sample = min(N_TRIPLETS, n * (n - 1) * (n - 2) // 6)
        for _ in range(n_sample):
            i, j, k = random.sample(range(n), 3)
            ay, ax   = int(comp[i][0]), int(comp[i][1])
            by_, bx  = int(comp[j][0]), int(comp[j][1])
            cy_, cx_ = int(comp[k][0]), int(comp[k][1])

            d_ab = math.hypot(bx  - ax,  by_ - ay)
            d_bc = math.hypot(cx_ - bx,  cy_ - by_)
            d_ac = math.hypot(cx_ - ax,  cy_ - ay)
            if min(d_ab, d_bc, d_ac) < MIN_CHORD:
                continue
            if max(d_ab, d_bc, d_ac) > RADIUS_MAX * 2:
                continue

            result = perp_bisector_intersection(ax, ay, bx, by_, cx_, cy_)
            if result is None:
                continue
            ox, oy = result
            r_est = math.hypot(ox - ax, oy - ay)

            if not (RADIUS_MIN <= r_est <= RADIUS_MAX):
                continue
            if not (0 <= ox < W and 0 <= oy < H):
                continue

            estimates.append((ox, oy, r_est))

    if not estimates:
        return []

    # Cluster estimates
    bins = defaultdict(list)
    for ox, oy, r_est in estimates:
        gx = int(ox) // CLUSTER_CELL
        gy = int(oy) // CLUSTER_CELL
        bins[(gx, gy)].append((ox, oy, r_est))

    # Extract dense clusters
    candidates = []
    visited = set()
    sorted_bins = sorted(bins.items(), key=lambda x: -len(x[1]))

    for (gx, gy), pts in sorted_bins:
        if (gx, gy) in visited:
            continue
        if len(pts) < MIN_CLUSTER_COUNT:
            break

        merged = list(pts)
        for dgx in [-1, 0, 1]:
            for dgy in [-1, 0, 1]:
                if dgx == 0 and dgy == 0:
                    continue
                nb = (gx + dgx, gy + dgy)
                if nb in bins:
                    merged.extend(bins[nb])
                    visited.add(nb)
        visited.add((gx, gy))

        if len(merged) < MIN_CLUSTER_COUNT:
            continue

        cx_r = float(np.median([p[0] for p in merged]))
        cy_r = float(np.median([p[1] for p in merged]))
        r_r  = float(np.median([p[2] for p in merged]))
        candidates.append({"cx": cx_r, "cy": cy_r, "r": r_r, "count": len(merged)})

    # Coverage verification
    confirmed_raw = []
    for cand in candidates:
        cx_i = int(round(cand["cx"]))
        cy_i = int(round(cand["cy"]))
        r_i  = int(round(cand["r"]))
        cov  = circumference_coverage(edges, cx_i, cy_i, r_i, H, W)
        if cov >= MIN_COVERAGE:
            confirmed_raw.append({
                "cx": cx_i, "cy": cy_i, "r": r_i,
                "cov": round(cov, 3), "count": cand["count"],
            })

    # NMS
    confirmed_raw = sorted(confirmed_raw, key=lambda x: -x["count"])
    confirmed = []
    suppressed = set()
    for i, c in enumerate(confirmed_raw):
        if i in suppressed:
            continue
        confirmed.append(c)
        for j, d in enumerate(confirmed_raw[i + 1:], start=i + 1):
            if j in suppressed:
                continue
            if math.hypot(c["cx"] - d["cx"], c["cy"] - d["cy"]) < MIN_CIRCLE_DIST:
                suppressed.add(j)

    return confirmed


# =============================================================================
# FULL MAP — TILED PROCESSING
# =============================================================================

def run_fullmap(gray_full: np.ndarray) -> list:
    """
    Tile the full image and run PB3 on each tile.
    Returns list of all confirmed circles in MAP coordinates.
    """
    H, W = gray_full.shape
    all_circles = []

    tile_rows = list(range(0, H, TILE_SIZE))
    tile_cols = list(range(0, W, TILE_SIZE))
    n_tiles   = len(tile_rows) * len(tile_cols)
    idx       = 0

    print(f"  Image: {W} × {H} px")
    print(f"  Tile size: {TILE_SIZE} px, overlap: {TILE_OVERLAP} px")
    print(f"  Total tiles: {n_tiles}\n")

    seen_centres = set()  # dedup grid (4px cells) across tile boundaries

    for tr in tile_rows:
        for tc in tile_cols:
            idx += 1

            # Inner tile bounds — only keep detections whose centre is here
            inner_r0 = tr
            inner_c0 = tc
            inner_r1 = min(tr + TILE_SIZE, H)
            inner_c1 = min(tc + TILE_SIZE, W)

            # Padded tile bounds — fed to detector
            pad_r0 = max(0, tr - TILE_OVERLAP)
            pad_c0 = max(0, tc - TILE_OVERLAP)
            pad_r1 = min(H, tr + TILE_SIZE + TILE_OVERLAP)
            pad_c1 = min(W, tc + TILE_SIZE + TILE_OVERLAP)

            tile = gray_full[pad_r0:pad_r1, pad_c0:pad_c1]

            t0 = time.time()
            tile_results = detect_pb3_tile(tile)
            elapsed = time.time() - t0

            kept = 0
            for d in tile_results:
                # Convert tile coords → map coords
                map_cx = d["cx"] + pad_c0
                map_cy = d["cy"] + pad_r0

                # Keep only if centre is in the inner tile
                if not (inner_c0 <= map_cx < inner_c1 and
                        inner_r0 <= map_cy < inner_r1):
                    continue

                # Suppress legend strip
                if map_cx < LEGEND_STRIP_WIDTH:
                    continue

                # Dedup across tile boundaries (4px grid)
                key = (int(map_cx) // 4, int(map_cy) // 4)
                if key in seen_centres:
                    continue
                seen_centres.add(key)

                all_circles.append({
                    "centre_col":  int(round(map_cx)),
                    "centre_row":  int(round(map_cy)),
                    "radius":      d["r"],
                    "cov":         d["cov"],
                    "cluster_n":   d["count"],
                })
                kept += 1

            print(f"  Tile {idx:4d}/{n_tiles}  "
                  f"map[{inner_r0}:{inner_r1},{inner_c0}:{inner_c1}]  "
                  f"found={len(tile_results)}  kept={kept}  "
                  f"({elapsed:.1f}s)")

    return all_circles


# =============================================================================
# SAVE OUTPUTS
# =============================================================================

def save_annotated_map(gray_full: np.ndarray, circles: list, out_path: str):
    """
    Save the full chart with a green ring drawn at each confirmed circle.
    Downscales for manageable file size if image is very large.
    """
    H, W = gray_full.shape

    # Work at half resolution for annotated map (saves memory/disk)
    scale = 0.5 if W > 8000 else 1.0
    new_w = int(W * scale)
    new_h = int(H * scale)

    img = Image.fromarray((gray_full * 255).astype(np.uint8)).convert("RGB")
    if scale < 1.0:
        img = img.resize((new_w, new_h), Image.LANCZOS)

    draw = ImageDraw.Draw(img)
    for c in circles:
        cx = int(round(c["centre_col"] * scale))
        cy = int(round(c["centre_row"] * scale))
        r  = max(1, int(round(c["radius"] * scale)))
        draw.ellipse([cx-r-2, cy-r-2, cx+r+2, cy+r+2],
                     outline=(0, 255, 0), width=2)
        draw.ellipse([cx-2, cy-2, cx+2, cy+2], fill=(255, 80, 80))

    img.save(out_path)
    print(f"  Annotated map → {out_path}  ({new_w}×{new_h}px)")


def save_json(circles: list, out_path: str, params: dict):
    out = {
        "total": len(circles),
        "params": params,
        "circles": circles,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  circles.json  → {out_path}  ({len(circles)} circles)")


def save_summary(circles: list, out_path: str, params: dict, elapsed: float):
    lines = [
        "PB3 Full-Map Circle Detector — Summary",
        "=" * 60,
        f"Image          : {params['image_path']}",
        f"Total circles  : {len(circles)}",
        f"Runtime        : {elapsed:.1f}s",
        "",
        "Parameters:",
    ]
    for k, v in params.items():
        if k != "image_path":
            lines.append(f"  {k:30s}: {v}")

    if circles:
        counts = [c["cluster_n"] for c in circles]
        covs   = [c["cov"]       for c in circles]
        lines += [
            "",
            "Detection statistics:",
            f"  cluster_n : min={min(counts)}  max={max(counts)}  "
            f"mean={sum(counts)/len(counts):.0f}",
            f"  coverage  : min={min(covs):.3f}  max={max(covs):.3f}  "
            f"mean={sum(covs)/len(covs):.3f}",
        ]

    lines += [
        "",
        "Next step:",
        "  Run letter_reader.py with circles.json as input.",
        "  The letter reader reads the letter inside each confirmed circle",
        "  (R=private airspace, H=heliport, U=unverified, F=ultralight, etc.)",
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  summary.txt   → {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    random.seed(RANDOM_SEED)

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "image_path":        IMAGE_PATH,
        "is_binary":         IS_BINARY,
        "radius_min":        RADIUS_MIN,
        "radius_max":        RADIUS_MAX,
        "canny_sigma":       CANNY_SIGMA,
        "min_cc_len":        MIN_CC_LEN,
        "n_triplets":        N_TRIPLETS,
        "min_chord":         MIN_CHORD,
        "cluster_cell":      CLUSTER_CELL,
        "min_cluster_count": MIN_CLUSTER_COUNT,
        "min_coverage":      MIN_COVERAGE,
        "min_circle_dist":   MIN_CIRCLE_DIST,
        "legend_strip_width":LEGEND_STRIP_WIDTH,
        "tile_size":         TILE_SIZE,
        "tile_overlap":      TILE_OVERLAP,
    }

    print("PB3 FULL-MAP CIRCLE DETECTOR")
    print("=" * 55)
    print(f"Image    : {IMAGE_PATH}")
    print(f"Output   : {OUTPUT_DIR}")
    print(f"Params   : MIN_CLUSTER={MIN_CLUSTER_COUNT}  "
          f"MIN_COV={MIN_COVERAGE}  "
          f"RADIUS={RADIUS_MIN}-{RADIUS_MAX}px  "
          f"N_TRIPLETS={N_TRIPLETS}")
    print()

    # Load image
    print("Loading image ...")
    img = Image.open(IMAGE_PATH).convert("L")
    gray_full = np.array(img).astype(float) / 255.0
    print(f"  {gray_full.shape[1]} × {gray_full.shape[0]} px\n")

    # Run full-map detection
    print("Running PB3 across all tiles ...")
    t_start = time.time()
    circles = run_fullmap(gray_full)
    elapsed = time.time() - t_start

    print(f"\nTotal confirmed circles: {len(circles)}")
    print(f"Total runtime: {elapsed:.1f}s  ({elapsed/60:.1f} min)\n")

    # Save outputs
    print("Saving outputs ...")
    save_json(circles,
              str(out_dir / "circles.json"),
              params)

    save_annotated_map(gray_full, circles,
                       str(out_dir / "annotated_map.png"))

    save_summary(circles,
                 str(out_dir / "summary.txt"),
                 params, elapsed)

    print()
    print("=" * 55)
    print("DONE")
    print("=" * 55)
    print(f"  Confirmed circles : {len(circles)}")
    print(f"  Output folder     : {out_dir.resolve()}")
    print()
    print("  Files:")
    print("    circles.json    — feed to letter_reader.py")
    print("    annotated_map.png — inspect detections visually")
    print("    summary.txt     — run log")
    print()
    print("  Tuning (if needed):")
    print(f"    Too many FP → raise MIN_CLUSTER_COUNT (now {MIN_CLUSTER_COUNT})")
    print(f"    Too many FP → raise MIN_COVERAGE      (now {MIN_COVERAGE})")
    print(f"    Missing circles → lower MIN_CLUSTER_COUNT")
    print(f"    Missing circles → raise N_TRIPLETS    (now {N_TRIPLETS})")


if __name__ == "__main__":
    main()