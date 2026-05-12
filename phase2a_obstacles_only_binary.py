"""
Phase 2a — Obstacle Symbol Extractor (Binary Map Version)
VFR Chart Extraction Pipeline (FAA Base Model)

Uses the binary (black/white) map from Phase 1 instead of the RGB map.
Advantages over RGB approach:
    - No HSV colour thresholding needed — no colour ambiguity
    - Obstacle symbol (open ∧ + dot) is ONE connected white blob in binary
    - No dilation/merge step needed
    - Much faster — no tiled HSV computation
    - Parameters derived directly from real blob measurements on the map

Algorithm:
    1. Load binary PNG, threshold to bool
    2. Label all connected white blobs
    3. Filter by shape profile measured from real obstacle symbols:
           Area:      100 – 400 px²   (at 150 DPI)
           W:          12 –  36 px
           H:          14 –  40 px
           Aspect H/W: 0.70 – 1.40
           Solidity:   0.45 – 0.65    (open-V + dot = partially hollow)
           Eccen:      < 0.70         (not lines or elongated letters)
    4. Save mask PNG + log JSON

Parameters measured from Washington_binary.png at 150 DPI:
    344 candidate blobs in sample region
    Area:     median 108 px², range 100-349
    Width:    median 17 px,   range 12-34
    Height:   median 18 px,   range 14-37
    Solidity: median 0.574,   range 0.456-0.644
    Eccen:    median 0.453,   range 0.123-0.700

Usage (PowerShell):
    python phase2a_binary.py `
        --binary outputs/phase1_preprocessing/Washington_binary.png `
        --output-dir outputs/phase2_layer_segmentation/phase2a_obstacles_only_binary `
        --dpi 150
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.measure import label, regionprops

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase2a_bin")


# ---------------------------------------------------------------------------
# Shape profile at 150 DPI — measured from Washington_binary.png
# Scales automatically with DPI via _scale() and _scale_area()
# ---------------------------------------------------------------------------
BASE_DPI      = 150

BASE_MIN_AREA = 100
BASE_MAX_AREA = 400
BASE_MIN_W    = 12
BASE_MAX_W    = 36
BASE_MIN_H    = 14
BASE_MAX_H    = 40

MIN_SOLIDITY  = 0.45
MAX_SOLIDITY  = 0.65
MAX_ECCEN     = 0.70
MIN_ASPECT    = 0.70
MAX_ASPECT    = 1.40

# Legend strip: left 18% of image width is the chart legend, not map content
LEGEND_FRAC   = 0.18

# Bottom strip: bottom ~8% is the chart data table, not map content
BOTTOM_FRAC   = 0.08


def _scale(value: float, dpi: int) -> int:
    return max(1, round(value * dpi / BASE_DPI))


def _scale_area(value: float, dpi: int) -> int:
    return max(1, round(value * (dpi / BASE_DPI) ** 2))


def load_binary(path: str) -> np.ndarray:
    """
    Load the Phase 1 binary PNG and return a bool array.
    Handles both true binary (0/255) and greyscale outputs.
    """
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.uint8)
    binary = arr > 128
    log.info(f"  Loaded: {arr.shape[1]}x{arr.shape[0]}px  "
             f"white coverage={binary.mean()*100:.2f}%")
    return binary


def filter_obstacle_blobs(
    binary:   np.ndarray,
    dpi:      int,
) -> np.ndarray:
    """
    Label all white blobs and keep only those matching the obstacle
    symbol shape profile.

    The FAA obstacle symbol in the binary map is ONE connected blob:
        - Open ∧ (two edges, no base) + filled dot at base-centre
        - Connected because the dot nearly touches the triangle base
        - Distinctive solidity: 0.45-0.65 (partially hollow due to
          the interior of the ∧ being empty)
        - Nearly square bounding box: aspect 0.70-1.40
        - Not elongated: eccentricity < 0.70

    Filters applied in order (cheapest first):
        1. Area
        2. Bounding box W and H
        3. Aspect ratio H/W
        4. Solidity
        5. Eccentricity
    """
    min_area = _scale_area(BASE_MIN_AREA, dpi)
    max_area = _scale_area(BASE_MAX_AREA, dpi)
    min_w    = _scale(BASE_MIN_W, dpi)
    max_w    = _scale(BASE_MAX_W, dpi)
    min_h    = _scale(BASE_MIN_H, dpi)
    max_h    = _scale(BASE_MAX_H, dpi)

    H, W = binary.shape
    legend_x = int(W * LEGEND_FRAC)
    bottom_y = int(H * (1.0 - BOTTOM_FRAC))

    log.info(f"  Area:      {min_area} – {max_area} px²")
    log.info(f"  W range:   {min_w} – {max_w} px")
    log.info(f"  H range:   {min_h} – {max_h} px")
    log.info(f"  Aspect:    {MIN_ASPECT} – {MAX_ASPECT}")
    log.info(f"  Solidity:  {MIN_SOLIDITY} – {MAX_SOLIDITY}")
    log.info(f"  Eccen <    {MAX_ECCEN}")
    log.info(f"  Legend strip: x < {legend_x}px  Bottom strip: y > {bottom_y}px")

    log.info("  Labelling blobs ...")
    labelled = label(binary, connectivity=2)
    props    = regionprops(labelled)
    log.info(f"  Total white blobs: {len(props)}")

    output = np.zeros((H, W), dtype=np.uint8)
    kept   = 0
    rej    = dict(position=0, area=0, size=0, aspect=0, solidity=0, eccen=0)

    for p in props:
        r0, c0, r1, c1 = p.bbox
        bh  = r1 - r0
        bw  = c1 - c0
        cx  = (c0 + c1) // 2
        cy  = (r0 + r1) // 2

        # Filter 0: Position — skip legend and bottom data strip
        if cx < legend_x or cy > bottom_y:
            rej['position'] += 1
            continue

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
        if not (MIN_ASPECT <= aspect <= MAX_ASPECT):
            rej['aspect'] += 1
            continue

        # Filter 4: Solidity — open-V + dot scores 0.45-0.65
        if not (MIN_SOLIDITY <= p.solidity <= MAX_SOLIDITY):
            rej['solidity'] += 1
            continue

        # Filter 5: Eccentricity — rejects lines and elongated letters
        if p.eccentricity >= MAX_ECCEN:
            rej['eccen'] += 1
            continue

        output[labelled == p.label] = 255
        kept += 1

    log.info(f"  Rejected — position:{rej['position']} area:{rej['area']} "
             f"size:{rej['size']} aspect:{rej['aspect']} "
             f"solidity:{rej['solidity']} eccen:{rej['eccen']}")
    log.info(f"  Kept blobs: {kept}")
    return output


def extract_obstacle_mask(
    binary_path: str,
    dpi:         int = 150,
    output_dir:  str = None,
    stem:        str = None,
) -> np.ndarray:
    """
    Full pipeline:
        1. Load binary PNG
        2. Filter blobs to obstacle candidates
        3. Save mask PNG + log JSON
        4. Return mask array
    """
    t0 = time.time()
    log.info("=" * 55)
    log.info("Phase 2a (Binary) — Obstacle Mask Extraction")
    log.info("=" * 55)

    log.info("Step 1: Loading binary map ...")
    binary = load_binary(binary_path)

    log.info("Step 2: Filtering blobs ...")
    obstacle_mask = filter_obstacle_blobs(binary, dpi)

    coverage = (obstacle_mask > 0).mean() * 100
    log.info(f"  Final mask coverage: {coverage:.4f}%")

    elapsed = round(time.time() - t0, 2)
    log.info(f"Done in {elapsed}s")

    if output_dir:
        out  = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        _stem = stem or Path(binary_path).stem.replace("_binary", "")

        mask_path = out / f"{_stem}_obstacle_mask.png"
        Image.fromarray(obstacle_mask).save(str(mask_path))
        log.info(f"  Mask → {mask_path}")

        meta = {
            "stem":         _stem,
            "dpi":          dpi,
            "version":      "2a_binary",
            "source":       "binary_map",
            "min_area":     _scale_area(BASE_MIN_AREA, dpi),
            "max_area":     _scale_area(BASE_MAX_AREA, dpi),
            "min_solidity": MIN_SOLIDITY,
            "max_solidity": MAX_SOLIDITY,
            "max_eccen":    MAX_ECCEN,
            "min_aspect":   MIN_ASPECT,
            "max_aspect":   MAX_ASPECT,
            "coverage_pct": round(coverage, 4),
            "elapsed_sec":  elapsed,
        }
        with open(out / f"{_stem}_obstacle_mask_log.json", "w") as f:
            json.dump(meta, f, indent=2)

    return obstacle_mask


if __name__ == "__main__":
    Image.MAX_IMAGE_PIXELS = None

    # ── CONFIG ──────────────────────────────────────────────────────────────────
    BINARY_PATH = "outputs/phase1_preprocessing/chart_binary.png"  # Phase 1 binary PNG
    OUTPUT_DIR  = "outputs/phase2_layer_segmentation/phase2a_obstacles_only_binary"
    STEM        = None  # Custom output stem; None = derived from input filename
    DPI         = 150
    # ────────────────────────────────────────────────────────────────────────────

    extract_obstacle_mask(
        binary_path=BINARY_PATH,
        dpi=DPI,
        output_dir=OUTPUT_DIR,
        stem=STEM,
    )