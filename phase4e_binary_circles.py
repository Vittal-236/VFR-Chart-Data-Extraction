"""
Phase 4b — Restricted Area Symbol Detection (Circle Ring Only)
VFR Chart Extraction Pipeline

Modified Version: Outputs results directly after circle ring detection and 
circumference filtering, skipping the internal 'R' glyph validation step.
Saves a pristine binary mask containing ONLY the detected features.
"""

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None  # allow large images

from skimage.transform import hough_circle, hough_circle_peaks

# =============================================================================
# CONFIG — edit these paths, then hit Run
# =============================================================================

SYMBOLS_PATH = r"outputs/phase2_layer_segmentation/phase2d_clean_binary/Washington_magenta_only.png"
OUTPUT_DIR   = r"outputs/phase4_symbol_detection/phase4d_binary_circles"

# =============================================================================
# TUNING PARAMETERS
# =============================================================================

# Expected radius range of the circle at map scale (pixels).
CIRCLE_RADIUS_MIN = 17    # px
CIRCLE_RADIUS_MAX = 26    # px

# Hough accumulator threshold.
HOUGH_THRESHOLD = 0.60

# Minimum circumference coverage. Closed rings cover 88-100% of their circumference.
MIN_CIRCUMFERENCE_COVERAGE = 0.88

LEGEND_STRIP_WIDTH = 140


# =============================================================================
# STEP 1 — Circle detection via Hough Transform
# =============================================================================

def detect_circles(binary: np.ndarray) -> list:
    """
    Find closed circular rings in the binary map using the Hough circle transform.
    Runs tiled to avoid out-of-memory errors on large chart layers.
    """
    H, W      = binary.shape
    radii     = np.arange(CIRCLE_RADIUS_MIN, CIRCLE_RADIUS_MAX + 1)
    all_circs = []

    TILE   = 2048
    OVERLAP = CIRCLE_RADIUS_MAX + 10

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

            print(f"  Tile {idx}/{n_tiles}  [{inner_r0}:{inner_r1}, {inner_c0}:{inner_c1}]", end="")

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
                map_r = int(cy) + pad_r0
                map_c = int(cx) + pad_c0

                # Deduplication check
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
    Filters out partial airspace arcs while keeping true closed symbols.
    """
    H, W    = binary.shape
    cr, cc  = circ["row"], circ["col"]
    rad     = circ["radius"]
    N       = max(36, int(2 * math.pi * rad))

    hits = 0
    for i in range(N):
        angle = 2 * math.pi * i / N
        r     = int(round(cr + rad * math.sin(angle)))
        c     = int(round(cc + rad * math.cos(angle)))
        if 0 <= r < H and 0 <= c < W and binary[r, c]:
            hits += 1

    return hits / N


# =============================================================================
# STEP 3 — Save outputs
# =============================================================================

def save_isolated_mask(shape: tuple, confirmed: list, out_path: str):
    """
    Creates a pristine black canvas and draws ONLY the detected features 
    as solid white filled masks without original map background.
    """
    H, W = shape
    # Create a completely blank black binary image (L mode, 0 = black)
    mask_img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask_img)

    for c in confirmed:
        cr, cc, rad = c["row"], c["col"], c["radius"]
        # Define the exact bounding box for the circle ring
        bbox = [cc - rad, cr - rad, cc + rad, cr + rad]
        
        # CHANGED: Use fill=255 to make the circle a solid white disc
        draw.ellipse(bbox, fill=255)

    mask_img.save(out_path)
    print(f"  Pristine isolated feature mask → {out_path}")   
    """
    Creates a pristine black canvas and draws ONLY the detected features 
    as white binary shapes without original map background or green markers.
    """
    H, W = shape
    # Create a completely blank black binary image (L mode, 0 = black)
    mask_img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask_img)

    for c in confirmed:
        cr, cc, rad = c["row"], c["col"], c["radius"]
        # Define the exact bounding box for the circle ring
        bbox = [cc - rad, cr - rad, cc + rad, cr + rad]
        
        # Draw the ring matching the visual style (hollow white line, thickness matches reference)
        draw.ellipse(bbox, outline=255, width=4)

    mask_img.save(out_path)
    print(f"  Pristine isolated feature mask → {out_path}")


def save_confirmed_crops(binary: np.ndarray, confirmed: list, out_path: str, max_show: int = 200):
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
        patch = Image.fromarray((crop * 255).astype(np.uint8)).resize((nw, nh), Image.NEAREST).convert("RGB")

        col_i = i % cols
        row_i = i // cols
        ox    = col_i * cell + (cell - nw) // 2
        oy    = row_i * cell + (cell - nh) // 2
        sheet.paste(patch, (ox, oy))

        draw.rectangle(
            [col_i * cell, row_i * cell, (col_i + 1) * cell - 1, (row_i + 1) * cell - 1],
            outline=(0, 200, 80), width=1
        )

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
    confirmed = []
    for c in raw_circles:
        cov = check_circumference(binary, c)
        c["circumference_coverage"] = float(cov)
        if cov >= MIN_CIRCUMFERENCE_COVERAGE:
            c["centre_row"] = c["row"]
            c["centre_col"] = c["col"]
            confirmed.append(c)
            
    print(f"  After circumference filter (Confirmed Detections): {len(confirmed)}  "
          f"(rejected {len(raw_circles) - len(confirmed)})")

    # Save outputs
    print("\nSaving outputs ...")
    with open(out_dir / "detections.json", "w") as f:
        json.dump({
            "total_confirmed": len(confirmed),
            "params": {
                "radius_range": [CIRCLE_RADIUS_MIN, CIRCLE_RADIUS_MAX],
                "hough_threshold": HOUGH_THRESHOLD,
                "min_circumference_coverage": MIN_CIRCUMFERENCE_COVERAGE,
            },
            "confirmed": confirmed,
        }, f, indent=2)
    print(f"  detections.json saved")

    # Generates a pristine black map containing only your isolated white features
    save_isolated_mask((H, W), confirmed, str(out_dir / "confirmed_isolated_mask.png"))
    save_confirmed_crops(binary, confirmed, str(out_dir / "confirmed_crops.png"))

    print("\n=== RESULT ===")
    print(f"  Raw Hough circles     : {len(raw_circles)}")
    print(f"  Confirmed closed rings: {len(confirmed)}")
    print(f"\nOutputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()