"""
Phase 4b — Restricted Area Symbol Detection (R-in-circle)
VFR Chart Extraction Pipeline (FAA Base Model)

Detects the FAA Restricted Area symbol: a hollow circle with the letter R
inside it. This is geometrically distinct from all other map symbols —
it is the only symbol that combines a closed circular ring with an enclosed
letter containing a hole (the R's counter/bowl).

DETECTION STRATEGY
------------------
Two-stage pipeline:

  Stage 1 — Circle detection via Hough Transform
    Uses skimage.transform.hough_circle on the binary map.
    Searches for circles in the expected radius range.
    The circle ring on the map is the primary anchor — it is distinctive
    and rare (most other circles on a VFR chart are airspace arcs, which
    are partial arcs, not closed rings).

  Stage 2 — R verification inside each circle candidate
    For each circle found:
      a) Check the circle is closed (high fraction of circumference is ink).
      b) Crop the interior of the circle.
      c) Verify the interior contains an R-like blob:
           - Has an enclosed hole (Euler number < 0 → bowl of the R)
           - Is roughly vertically centred in the circle interior
           - Has moderate solidity (R is not a filled blob)

WHY NOT NCC FOR THIS SYMBOL
----------------------------
NCC template matching works well for the obstacle ∧ because the template
has a unique spatial ink pattern. For the R-in-circle:
  - The circle radius varies slightly between instances
  - The R inside is small and variable (font weight, scale)
  - Hough circle detection is mathematically optimal for ring-shaped ink
  - The enclosed hole of the R is a topological feature that NCC cannot
    directly exploit — Euler number checks it directly

HOW TO USE
----------
1. Set paths in CONFIG below.
2. Hit Run — no CLI needed.
3. Outputs saved to OUTPUT_DIR:
     detections.json          — confirmed detections with pixel coords
     confirmed_map.png        — map with green circles drawn on detections
     confirmed_crops.png      — contact sheet of cropped instances

SYMBOL GEOMETRY (from uploaded reference image, ~110x110 px)
-------------------------------------------------------------
  Outer circle: hollow ring, stroke ~6-8px
  Inner clear zone between ring and R: ~8-12px
  R glyph: ~40px tall, ~30px wide inside the circle
  R has an enclosed counter (bowl) — Euler number < 0
  Overall symbol diameter: ~90px at reference scale
  At map scale (300 DPI): estimated 30-50px diameter
"""

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None  # allow large images (crop sheets can be huge)
from skimage.draw import disk
from skimage.measure import label, regionprops
from skimage.transform import hough_circle, hough_circle_peaks

# =============================================================================
# CONFIG — edit these paths, then hit Run
# =============================================================================

SYMBOLS_PATH = r"outputs/phase2_layer_segmentation/phase2c_only_symbols_binary/washington_symbols_only.png"  # Phase 2c binary
OUTPUT_DIR   = r"outputs/phase4_symbol_detection/phase4b_binary_private"

# =============================================================================
# TUNING PARAMETERS
# =============================================================================

# Expected radius range of the circle at map scale (pixels).
CIRCLE_RADIUS_MIN = 17    # px
CIRCLE_RADIUS_MAX = 26    # px

# Hough accumulator threshold. Raised to 0.60 — text clusters produce spurious
# circle centres at 0.50 but genuine closed rings score 0.65+.
HOUGH_THRESHOLD = 0.60

# Minimum circumference coverage. Raised to 0.88 — genuine closed rings cover
# 88-100% of their circumference. Text digits (0, 3, 6, 8) and annotation
# boxes were passing at 0.75. Raising to 0.88 cuts those while keeping real rings.
MIN_CIRCUMFERENCE_COVERAGE = 0.88

# Interior ink range (fraction of interior circular area).
INTERIOR_INK_MIN = 0.12
INTERIOR_INK_MAX = 0.55

# Quadrant check — br_bl tightened from 0.85 to 0.80 to cut F and H.
# Data shows confirmed R: br_bl max = 0.827. H/F: br_bl 0.88-1.25.
BR_BL_MAX_RATIO = 0.83

# Top half ink minimum.
TOPINK_MIN_FRAC = 0.15

LEGEND_STRIP_WIDTH = 140


# =============================================================================
# STEP 1 — Circle detection via Hough Transform
# =============================================================================

def detect_circles(binary: np.ndarray) -> list:
    """
    Find closed circular rings in the binary map using the Hough circle transform.

    The Hough circle transform votes for circle centres given edge pixels.
    We run it tiled (same memory strategy as the obstacle NCC) to avoid OOM.

    Returns list of dicts: {row, col, radius, accumulator_score}
      row, col = centre of the circle in MAP coordinates
    """
    H, W      = binary.shape
    radii     = np.arange(CIRCLE_RADIUS_MIN, CIRCLE_RADIUS_MAX + 1)
    all_circs = []

    TILE   = 2048
    OVERLAP = CIRCLE_RADIUS_MAX + 10   # enough to capture circles at tile edges

    tile_rows = list(range(0, H, TILE))
    tile_cols = list(range(0, W, TILE))
    n_tiles   = len(tile_rows) * len(tile_cols)
    idx       = 0

    for tr in tile_rows:
        for tc in tile_cols:
            idx += 1
            inner_r0, inner_c0 = tr, tc
            inner_r1 = min(tr + TILE, H)
            inner_c1 = min(tc + TILE, W)

            pad_r0 = max(0, tr - OVERLAP)
            pad_c0 = max(0, tc - OVERLAP)
            pad_r1 = min(H, tr + TILE + OVERLAP)
            pad_c1 = min(W, tc + TILE + OVERLAP)

            tile = binary[pad_r0:pad_r1, pad_c0:pad_c1].astype(np.uint8)

            print(f"  Tile {idx}/{n_tiles}  [{inner_r0}:{inner_r1}, {inner_c0}:{inner_c1}]",
                  end="")

            hspaces = hough_circle(tile, radii)
            accums, cx_arr, cy_arr, rad_arr = hough_circle_peaks(
                hspaces, radii,
                min_xdistance=CIRCLE_RADIUS_MIN,
                min_ydistance=CIRCLE_RADIUS_MIN,
                threshold=HOUGH_THRESHOLD,
                num_peaks=500,
            )

            found = 0
            for acc, cx, cy, rad in zip(accums, cx_arr, cy_arr, rad_arr):
                # cx, cy are in padded-tile coordinates
                map_r = int(cy) + pad_r0
                map_c = int(cx) + pad_c0

                # Keep only if centre is in inner tile (dedup)
                if not (inner_r0 <= map_r < inner_r1 and inner_c0 <= map_c < inner_c1):
                    continue
                if map_c < LEGEND_STRIP_WIDTH:
                    continue

                all_circs.append({
                    "row": map_r,
                    "col": map_c,
                    "radius": int(rad),
                    "hough_score": float(acc),
                })
                found += 1

            print(f"  → {found} circles")

    return all_circs


# =============================================================================
# STEP 2 — Circumference coverage check (reject partial arcs)
# =============================================================================

def check_circumference(binary: np.ndarray, circ: dict) -> float:
    """
    Measure what fraction of the expected circle circumference has ink.

    Airspace boundary arcs are partial — they cover 20-60% of a circle.
    The R-in-circle symbol is a closed ring — it covers 70-100%.

    Method: sample N points around the circumference, check how many are ink.
    Returns coverage fraction 0.0-1.0.
    """
    H, W    = binary.shape
    cr, cc  = circ["row"], circ["col"]
    rad     = circ["radius"]
    N       = max(36, int(2 * math.pi * rad))   # sample ~1px spacing

    hits = 0
    for i in range(N):
        angle = 2 * math.pi * i / N
        r     = int(round(cr + rad * math.sin(angle)))
        c     = int(round(cc + rad * math.cos(angle)))
        if 0 <= r < H and 0 <= c < W and binary[r, c]:
            hits += 1

    return hits / N


# =============================================================================
# STEP 3 — R verification inside the circle
# =============================================================================

def verify_r_inside(binary: np.ndarray, circ: dict) -> dict:
    """
    Verify the circle interior contains an R glyph using quadrant ink analysis.

    WHY QUADRANT ANALYSIS INSTEAD OF NCC
    -------------------------------------
    The synthesised R template produces NCC scores of 0.00-0.26 for all
    interior types (R, H, X, empty) — useless for discrimination. This is
    because the interior crop at map scale (radius 17-26px → interior ~24px
    diameter) is too small for meaningful template correlation after resize.

    QUADRANT APPROACH
    -----------------
    Split the circular interior into 4 quadrants (top-left, top-right,
    bottom-left, bottom-right) and measure ink in each.

    R ink pattern:
      Top-left:     HIGH  (top of vertical stroke)
      Top-right:    HIGH  (bowl of the R)
      Bottom-left:  HIGH  (bottom of vertical stroke)
      Bottom-right: LOW   (open space below the diagonal leg)

    Key discriminators:
      - H: top-right ≈ top-left (symmetric), bottom symmetric too
      - X: all 4 quadrants roughly equal (diagonals)
      - Empty O: all quadrants near zero
      - Filled: all quadrants high
      - R: bottom-right is noticeably lower than bottom-left

    Checks applied:
      1. Ink coverage 12-55%  (kills empty O and filled blobs)
      2. Bottom-right ink < bottom-left ink * BR_BL_MAX_RATIO
         (R has open space bottom-right; H/X do not)
      3. Top half total ink > TOPINK_MIN_FRAC of interior
         (R always has ink in both top quadrants from the bowl and stroke)
    """
    H, W   = binary.shape
    cr, cc = circ["row"], circ["col"]
    rad    = circ["radius"]

    # Extract interior: disk of radius * 0.68
    inner_rad = max(3, int(rad * 0.68))
    bbox_r0   = max(0, cr - inner_rad)
    bbox_r1   = min(H, cr + inner_rad + 1)
    bbox_c0   = max(0, cc - inner_rad)
    bbox_c1   = min(W, cc + inner_rad + 1)

    patch = binary[bbox_r0:bbox_r1, bbox_c0:bbox_c1].copy().astype(np.float32)
    ph, pw = patch.shape
    if ph < 4 or pw < 4:
        circ.update({"r_verified": False, "interior_ink_frac": 0.0,
                     "br_bl_ratio": 0.0, "top_ink_frac": 0.0})
        return circ

    # Circular mask
    mask = np.zeros((ph, pw), dtype=bool)
    pc, qc = ph // 2, pw // 2
    for r in range(ph):
        for c in range(pw):
            if (r - pc) ** 2 + (c - qc) ** 2 <= inner_rad ** 2:
                mask[r, c] = True

    interior = patch * mask
    total_pix = float(mask.sum())
    ink_frac  = float(interior.sum() / max(total_pix, 1))
    circ["interior_ink_frac"] = ink_frac

    # CHECK 1 — ink coverage
    if not (INTERIOR_INK_MIN <= ink_frac <= INTERIOR_INK_MAX):
        circ.update({"r_verified": False, "br_bl_ratio": 0.0, "top_ink_frac": 0.0})
        return circ

    # Quadrant split
    mid_r = ph // 2
    mid_c = pw // 2

    tl = (interior[:mid_r, :mid_c] * mask[:mid_r, :mid_c]).sum()
    tr = (interior[:mid_r, mid_c:] * mask[:mid_r, mid_c:]).sum()
    bl = (interior[mid_r:, :mid_c] * mask[mid_r:, :mid_c]).sum()
    br = (interior[mid_r:, mid_c:] * mask[mid_r:, mid_c:]).sum()

    total_top = tl + tr
    top_frac  = float(total_top / max(total_pix / 2, 1))
    circ["top_ink_frac"] = top_frac

    # CHECK 2 — bottom-right must be lower than bottom-left (R's open leg space)
    # R:   bl >> br  (vertical stroke on left continues down, right is open)
    # H:   bl ≈ br   (symmetric)
    # X:   bl ≈ br   (symmetric diagonals)
    br_bl_ratio = float(br / max(bl, 1))
    circ["br_bl_ratio"] = br_bl_ratio

    if br_bl_ratio > BR_BL_MAX_RATIO:
        circ["r_verified"] = False
        return circ

    # CHECK 3 — top half must have substantial ink (bowl + top of stroke)
    if top_frac < TOPINK_MIN_FRAC:
        circ["r_verified"] = False
        return circ

    circ["r_verified"] = True
    return circ


# =============================================================================
# STEP 4 — Save outputs
# =============================================================================

def save_confirmed_map(binary: np.ndarray, confirmed: list, out_path: str):
    """Draw green circles on confirmed detections."""
    rgb  = np.stack([binary * 255] * 3, axis=-1).astype(np.uint8)
    img  = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)

    for c in confirmed:
        cr, cc, rad = c["row"], c["col"], c["radius"]
        bbox = [cc - rad - 2, cr - rad - 2, cc + rad + 2, cr + rad + 2]
        draw.ellipse(bbox, outline=(0, 255, 0), width=2)

    img.save(out_path)
    print(f"  Confirmed map → {out_path}")


def save_confirmed_crops(binary: np.ndarray, confirmed: list, out_path: str,
                         max_show: int = 200):
    """Contact sheet of cropped circle instances, sorted by Hough score desc."""
    H, W     = binary.shape
    sorted_c = sorted(confirmed, key=lambda x: -x["hough_score"])
    cell     = 72
    cols     = 15
    rows     = math.ceil(min(len(sorted_c), max_show) / cols)
    sheet    = Image.new("RGB", (cell * cols, cell * rows), (15, 15, 15))
    draw     = ImageDraw.Draw(sheet)

    for i, c in enumerate(sorted_c[:max_show]):
        cr, cc  = c["row"], c["col"]
        rad     = c["radius"]
        pad     = 6
        crop_r0 = max(0, cr - rad - pad)
        crop_r1 = min(H, cr + rad + pad)
        crop_c0 = max(0, cc - rad - pad)
        crop_c1 = min(W, cc + rad + pad)

        crop = binary[crop_r0:crop_r1, crop_c0:crop_c1]
        ch, cw = crop.shape
        if ch == 0 or cw == 0:
            continue

        scale = min((cell - 4) / ch, (cell - 4) / cw, 3.0)
        nw    = max(1, int(cw * scale))
        nh    = max(1, int(ch * scale))
        patch = Image.fromarray((crop * 255).astype(np.uint8)).resize(
            (nw, nh), Image.NEAREST).convert("RGB")

        col_i = i % cols
        row_i = i // cols
        ox    = col_i * cell + (cell - nw) // 2
        oy    = row_i * cell + (cell - nh) // 2
        sheet.paste(patch, (ox, oy))

        draw.rectangle(
            [col_i * cell, row_i * cell,
             (col_i + 1) * cell - 1, (row_i + 1) * cell - 1],
            outline=(0, 200, 80), width=1)

    sheet.save(out_path)
    print(f"  Confirmed crops → {out_path}  ({min(len(sorted_c), max_show)} shown)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load binary
    print(f"Loading symbols binary: {SYMBOLS_PATH}")
    binary = np.array(Image.open(SYMBOLS_PATH).convert("L")) > 128
    H, W   = binary.shape
    print(f"  Size: {W} x {H} px    Ink: {binary.mean()*100:.2f}%")

    # Stage 1: Hough circle detection
    print(f"\nHough circle detection (radius {CIRCLE_RADIUS_MIN}-{CIRCLE_RADIUS_MAX}px) ...")
    raw_circles = detect_circles(binary)
    print(f"  Raw circles found: {len(raw_circles)}")

    # Stage 2: Circumference coverage filter (reject partial arcs)
    print("\nCircumference coverage check ...")
    after_circ = []
    for c in raw_circles:
        cov = check_circumference(binary, c)
        c["circumference_coverage"] = float(cov)
        if cov >= MIN_CIRCUMFERENCE_COVERAGE:
            after_circ.append(c)
    print(f"  After circumference filter: {len(after_circ)}  "
          f"(rejected {len(raw_circles) - len(after_circ)})")

    # Stage 3: R verification inside circle
    print("R interior verification ...")
    confirmed = []
    for c in after_circ:
        verify_r_inside(binary, c)
        if c["r_verified"]:
            confirmed.append(c)
    print(f"  Confirmed (circle + R inside): {len(confirmed)}  "
          f"(rejected {len(after_circ) - len(confirmed)})")

    # Centre coordinates
    for c in confirmed:
        c["centre_row"] = c["row"]
        c["centre_col"] = c["col"]

    # Save
    print("\nSaving outputs ...")

    # Diagnostic: print verification results for first 20 after circumference
    print("\nDiagnostic — verification results (first 20):")
    for i, c in enumerate(after_circ[:20]):
        ink   = c.get("interior_ink_frac", 0)
        br_bl = c.get("br_bl_ratio", 0)
        top   = c.get("top_ink_frac", 0)
        vrfy  = c.get("r_verified", False)
        print(f"  {i:2d}  r={c['radius']:2d}  ink={ink:.3f}  br/bl={br_bl:.3f}  top={top:.3f}  ok={vrfy}")

    with open(out_dir / "detections.json", "w") as f:
        json.dump({
            "total_confirmed": len(confirmed),
            "params": {
                "radius_range": [CIRCLE_RADIUS_MIN, CIRCLE_RADIUS_MAX],
                "hough_threshold": HOUGH_THRESHOLD,
                "min_circumference_coverage": MIN_CIRCUMFERENCE_COVERAGE,
                "interior_ink_range": [INTERIOR_INK_MIN, INTERIOR_INK_MAX],
                "br_bl_max_ratio": BR_BL_MAX_RATIO,
                "topink_min_frac": TOPINK_MIN_FRAC,
            },
            "confirmed": confirmed,
            "after_circ_sample": after_circ[:50],
        }, f, indent=2)
    print(f"  detections.json saved")

    save_confirmed_map(binary, confirmed, str(out_dir / "confirmed_map.png"))
    save_confirmed_crops(binary, confirmed, str(out_dir / "confirmed_crops.png"))

    # Save ALL circumference-passing candidates regardless of R check (diagnosis)
    print("  Saving all-candidates sheet for diagnosis ...")
    save_confirmed_crops(binary, after_circ,
                         str(out_dir / "all_candidates_crops.png"), max_show=200)

    print("\n=== RESULT ===")
    print(f"  Raw Hough circles     : {len(raw_circles)}")
    print(f"  After circumference   : {len(after_circ)}")
    print(f"  Confirmed (R inside)  : {len(confirmed)}")
    print(f"\nOutputs in: {out_dir.resolve()}")
    print("\nTuning guide:")
    print(f"  Too few R:   lower MIN_CIRCUMFERENCE_COVERAGE (now {MIN_CIRCUMFERENCE_COVERAGE})")
    print(f"               or lower TOPINK_MIN_FRAC (now {TOPINK_MIN_FRAC})")
    print(f"  H/X still:   lower BR_BL_MAX_RATIO (now {BR_BL_MAX_RATIO})")
    print(f"  Wrong size:  adjust CIRCLE_RADIUS_MIN/MAX (now {CIRCLE_RADIUS_MIN}-{CIRCLE_RADIUS_MAX})")


if __name__ == "__main__":
    main()