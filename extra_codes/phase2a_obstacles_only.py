"""
Phase 2a — Obstacle Symbol Extractor  
VFR Chart Extraction Pipeline (FAA Base Model)

Restored to the exact parameters that produced the Washington mask with
coverage 0.3155% and visible obstacle symbols throughout the map.

Parameters (verified from Washington_obstacle_mask_log.json):
    HSV:       H 0.53-0.72, S >= 0.25, V >= 0.25
    Aspect:    0.78 - 1.25
    Solidity:  0.50 - 0.62
    Coverage:  ~0.31%
    Time:      ~65s at 150 DPI

Usage (PowerShell):
    python phase2a_extract_obstacles.py `
        --rgb  outputs/phase1_150/Washington_rgb.png `
        --output-dir outputs/phase2a `
        --dpi 150
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import rgb2hsv
from skimage.measure import label, regionprops

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase2a")


# ---------------------------------------------------------------------------
# FAA Obstacle Blue — HSV ranges (measured from Washington sectional)
# ---------------------------------------------------------------------------
OBSTACLE_HSV = dict(h_min=0.53, h_max=0.72, s_min=0.25, v_min=0.25)

# ---------------------------------------------------------------------------
# Size and shape profile at 150 DPI (scales automatically with DPI)
#
# These values produced the confirmed working mask on Washington sectional.
# The blobs kept here are sub-symbol fragments (one leg of the triangle,
# the dot alone) — NOT complete merged symbols. Phase 4 scores overlap
# against each fragment individually.
#
#   Area:        65-240 px²
#   W / H:       12-26 px each
#   Aspect H/W:  0.78-1.25
#   Solidity:    0.50-0.62
#   Eccentricity < 0.85
# ---------------------------------------------------------------------------
BASE_DPI         = 150
BASE_MIN_AREA    = 65
BASE_MAX_AREA    = 240
BASE_MIN_W       = 12
BASE_MAX_W       = 26
BASE_MIN_H       = 12
BASE_MAX_H       = 26
MAX_ECCENTRICITY = 0.85


def _scale(value, dpi):
    return max(1, round(value * dpi / BASE_DPI))


def extract_blue_mask(rgb: np.ndarray, tile_size: int = 1024) -> np.ndarray:
    """
    Extract pixels matching FAA obstacle blue HSV range.
    Tiled to keep RAM usage constant regardless of image size.
    """
    H, W = rgb.shape[:2]
    mask = np.zeros((H, W), dtype=bool)
    n_ty = int(np.ceil(H / tile_size))
    n_tx = int(np.ceil(W / tile_size))
    total = n_ty * n_tx
    done  = 0

    for ty in range(n_ty):
        for tx in range(n_tx):
            y0, y1 = ty * tile_size, min((ty + 1) * tile_size, H)
            x0, x1 = tx * tile_size, min((tx + 1) * tile_size, W)
            tile = rgb[y0:y1, x0:x1].astype(np.float64) / 255.0
            hsv  = rgb2hsv(tile)
            mask[y0:y1, x0:x1] = (
                (hsv[..., 0] >= OBSTACLE_HSV['h_min']) &
                (hsv[..., 0] <= OBSTACLE_HSV['h_max']) &
                (hsv[..., 1] >= OBSTACLE_HSV['s_min']) &
                (hsv[..., 2] >= OBSTACLE_HSV['v_min'])
            )
            done += 1
            log.info(f"  Blue tile {done}/{total}")

    log.info(f"  Raw blue coverage: {mask.mean() * 100:.2f}%")
    return mask


def filter_obstacle_blobs(
    blue_mask:    np.ndarray,
    dpi:          int,
    min_aspect:   float,
    max_aspect:   float,
    min_solidity: float,
    max_solidity: float,
) -> np.ndarray:
    """
    Label connected blue blobs and keep only those matching
    the obstacle symbol size and shape profile.

    Filters:
        1. Area:         BASE_MIN_AREA – BASE_MAX_AREA px² (DPI-scaled)
        2. Bbox size:    width and height within [min_w, max_w]
        3. Aspect H/W:   min_aspect – max_aspect
        4. Solidity:     min_solidity – max_solidity
        5. Eccentricity: < MAX_ECCENTRICITY
    """
    min_area = int(BASE_MIN_AREA * (dpi / BASE_DPI) ** 2)
    max_area = int(BASE_MAX_AREA * (dpi / BASE_DPI) ** 2)
    min_w    = _scale(BASE_MIN_W, dpi)
    max_w    = _scale(BASE_MAX_W, dpi)
    min_h    = _scale(BASE_MIN_H, dpi)
    max_h    = _scale(BASE_MAX_H, dpi)

    log.info(f"  Area:      {min_area} – {max_area} px²")
    log.info(f"  W range:   {min_w} – {max_w} px")
    log.info(f"  H range:   {min_h} – {max_h} px")
    log.info(f"  Aspect:    {min_aspect} – {max_aspect}")
    log.info(f"  Solidity:  {min_solidity} – {max_solidity}")
    log.info(f"  Eccentric < {MAX_ECCENTRICITY}")

    H, W     = blue_mask.shape
    labelled = label(blue_mask, connectivity=2)
    props    = regionprops(labelled)
    log.info(f"  Total blue blobs: {len(props)}")

    output = np.zeros((H, W), dtype=np.uint8)
    kept   = 0
    rej    = dict(area=0, size=0, aspect=0, solidity=0, eccentric=0)

    for p in props:
        r0, c0, r1, c1 = p.bbox
        bh = r1 - r0
        bw = c1 - c0

        # Filter 1: Area
        if not (min_area <= int(p.area) <= max_area):
            rej['area'] += 1
            continue

        # Filter 2: Bounding box dimensions
        if not (min_w <= bw <= max_w and min_h <= bh <= max_h):
            rej['size'] += 1
            continue

        # Filter 3: Aspect ratio
        if bw == 0:
            continue
        aspect = bh / bw
        if not (min_aspect <= aspect <= max_aspect):
            rej['aspect'] += 1
            continue

        # Filter 4: Solidity
        if not (min_solidity <= p.solidity <= max_solidity):
            rej['solidity'] += 1
            continue

        # Filter 5: Eccentricity
        if p.eccentricity >= MAX_ECCENTRICITY:
            rej['eccentric'] += 1
            continue

        output[labelled == p.label] = 255
        kept += 1

    log.info(f"  Rejected — area:{rej['area']} size:{rej['size']} "
             f"aspect:{rej['aspect']} solidity:{rej['solidity']} "
             f"eccentric:{rej['eccentric']}")
    log.info(f"  Kept blobs: {kept}")
    return output


def extract_obstacle_mask(
    rgb:          np.ndarray,
    dpi:          int   = 150,
    min_aspect:   float = 0.78,
    max_aspect:   float = 1.25,
    min_solidity: float = 0.50,
    max_solidity: float = 0.62,
    output_dir:   str   = None,
    stem:         str   = "Washington",
) -> np.ndarray:
    """
    Full pipeline:
        1. Extract blue pixels from RGB
        2. Filter blobs to obstacle candidates
        3. Save mask PNG + log JSON
        4. Return binary mask array
    """
    t0 = time.time()
    log.info("=" * 55)
    log.info("Phase 2a — Obstacle Mask Extraction  (v7 baseline)")
    log.info("=" * 55)
    log.info(f"  Image: {rgb.shape[1]}x{rgb.shape[0]}px at {dpi} DPI")

    log.info("Step 1: Extracting blue pixels ...")
    blue_mask = extract_blue_mask(rgb)

    log.info("Step 2: Filtering blobs ...")
    obstacle_mask = filter_obstacle_blobs(
        blue_mask, dpi, min_aspect, max_aspect, min_solidity, max_solidity
    )

    coverage = (obstacle_mask > 0).mean() * 100
    log.info(f"  Final coverage: {coverage:.4f}%")

    elapsed = round(time.time() - t0, 2)
    log.info(f"Done in {elapsed}s")

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        mask_path = out / f"{stem}_obstacle_mask.png"
        Image.fromarray(obstacle_mask).save(str(mask_path))
        log.info(f"  Mask → {mask_path}")

        meta = {
            "stem":         stem,
            "dpi":          dpi,
            "version":      7,
            "hsv_range":    OBSTACLE_HSV,
            "min_aspect":   min_aspect,
            "max_aspect":   max_aspect,
            "min_solidity": min_solidity,
            "max_solidity": max_solidity,
            "coverage_pct": round(coverage, 4),
            "elapsed_sec":  elapsed,
        }
        with open(out / f"{stem}_obstacle_mask_log.json", "w") as f:
            json.dump(meta, f, indent=2)

    return obstacle_mask


if __name__ == "__main__":
    Image.MAX_IMAGE_PIXELS = None

    p = argparse.ArgumentParser(
        description="Phase 2a — Extract obstacle symbol mask (v7 baseline)"
    )
    p.add_argument("--rgb",          required=True)
    p.add_argument("--output-dir",   default="outputs/phase2a")
    p.add_argument("--stem",         default=None)
    p.add_argument("--dpi",          type=int,   default=150)
    p.add_argument("--min-aspect",   type=float, default=0.78)
    p.add_argument("--max-aspect",   type=float, default=1.25)
    p.add_argument("--min-solidity", type=float, default=0.50)
    p.add_argument("--max-solidity", type=float, default=0.62)
    args = p.parse_args()

    log.info(f"Loading: {args.rgb}")
    rgb  = np.array(Image.open(args.rgb).convert("RGB"), dtype=np.uint8)
    stem = args.stem or Path(args.rgb).stem.replace("_rgb", "")

    extract_obstacle_mask(
        rgb=rgb,
        dpi=args.dpi,
        min_aspect=args.min_aspect,
        max_aspect=args.max_aspect,
        min_solidity=args.min_solidity,
        max_solidity=args.max_solidity,
        output_dir=args.output_dir,
        stem=stem,
    )