"""
Phase 2d — Keep Only Black, Remove Everything Else
VFR Chart Extraction Pipeline

Keeps only black-coloured pixels in the binary mask.
Everything else is removed — magenta obstacle symbols,
blue airspace, brown terrain, cyan water, all of it.

Black on FAA VFR charts = text labels, road outlines,
airport symbols, and general cartographic linework
printed in black ink.

A pixel is kept if it is BOTH:
  - Foreground in the binary mask
  - Black in the RGB source image

Black in HSV:
  Saturation low : S < S_MAX  (desaturated — no real hue)
  Brightness low : V < V_MAX  (dark, not grey or white)
  No hue constraint needed — black has no meaningful hue.

Two outputs are saved:
  1. Binary mask  — white foreground on black background
  2. RGB overlay  — original black pixels on white background
"""

import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── CONFIG ─────────────────────────────────────────────────────────────────────
BINARY_PATH     = r"outputs\phase1_preprocessing\Washington_binary.png"
RGB_PATH        = r"outputs\phase1_preprocessing\Washington_rgb_300dpi.png"
OUTPUT_PATH     = r"outputs\phase2_layer_segmentation\phase2d_colour_based_binary\Washington_black_only.png"
OUTPUT_RGB_PATH = r"outputs\phase2_layer_segmentation\phase2d_colour_based_binary\Washington_black_only_rgb.png"

# HSV range for black
# OpenCV HSV: H in [0,179], S in [0,255], V in [0,255]
# Black has no real hue — only saturation and brightness matter:
#   S must be low  (desaturated)
#   V must be low  (dark)
S_MAX = 60    # maximum saturation — must be near-grey/desaturated
V_MAX = 80    # maximum brightness — must be dark

TILE_SIZE = 2048
TILE_OVL  = 128
# ───────────────────────────────────────────────────────────────────────────────


def keep_black(binary_path: str, rgb_path: str, output_path: str, output_rgb_path: str) -> None:
    t0  = time.time()
    dst     = Path(output_path)
    dst_rgb = Path(output_rgb_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────
    log.info(f"Loading binary : {Path(binary_path).name}")
    arr    = np.array(Image.open(binary_path).convert("L"))
    binary = (arr > 128).astype(np.uint8)
    H, W   = binary.shape
    log.info(f"  {W} x {H} px  |  fg: {binary.mean()*100:.2f}%")

    log.info(f"Loading RGB    : {Path(rgb_path).name}")
    rgb = np.array(Image.open(rgb_path).convert("RGB"))
    log.info(f"  {rgb.shape[1]} x {rgb.shape[0]} px")

    assert rgb.shape[0] == H and rgb.shape[1] == W, (
        f"Size mismatch: binary {binary.shape} vs RGB {rgb.shape}."
    )

    # ── Tiled black-keep processing ────────────────────────────────────────
    log.info(f"Keeping black pixels only ...")
    log.info(f"  S < {S_MAX},  V < {V_MAX}")

    out        = np.zeros_like(binary)               # binary output
    out_rgb    = np.full((H, W, 3), 255, dtype=np.uint8)  # RGB output — white background
    ys         = list(range(0, H, TILE_SIZE))
    xs         = list(range(0, W, TILE_SIZE))
    px_kept    = 0
    log.info(f"  Grid: {len(ys)} rows x {len(xs)} cols = {len(ys)*len(xs)} tiles")

    for ri, y0 in enumerate(ys):
        for ci, x0 in enumerate(xs):
            ty0 = max(0, y0 - TILE_OVL);  ty1 = min(H, y0 + TILE_SIZE + TILE_OVL)
            tx0 = max(0, x0 - TILE_OVL);  tx1 = min(W, x0 + TILE_SIZE + TILE_OVL)

            bin_tile = binary[ty0:ty1, tx0:tx1]
            rgb_tile = rgb[ty0:ty1, tx0:tx1]

            # Convert to HSV
            bgr_tile = cv2.cvtColor(rgb_tile, cv2.COLOR_RGB2BGR)
            hsv_tile = cv2.cvtColor(bgr_tile, cv2.COLOR_BGR2HSV)

            Sc = hsv_tile[:, :, 1]
            Vc = hsv_tile[:, :, 2]

            # Black mask: low saturation AND low brightness
            hue_match = (
                (Sc < S_MAX) &
                (Vc < V_MAX)
            ).astype(np.uint8)

            # Keep only binary foreground pixels that are black in RGB
            kept = bin_tile & hue_match

            # Write centre region only
            cy0 = y0 - ty0;  cy1 = cy0 + min(TILE_SIZE, H - y0)
            cx0 = x0 - tx0;  cx1 = cx0 + min(TILE_SIZE, W - x0)
            kept_centre = kept[cy0:cy1, cx0:cx1]

            # Binary output
            out[y0:y0+(cy1-cy0), x0:x0+(cx1-cx0)] = kept_centre
            px_kept += int(kept_centre.sum())

            # RGB output — paste original pixels where kept, leave white elsewhere
            mask_bool = kept_centre.astype(bool)
            rgb_centre = rgb_tile[cy0:cy1, cx0:cx1]
            out_rgb_region = out_rgb[y0:y0+(cy1-cy0), x0:x0+(cx1-cx0)]
            out_rgb_region[mask_bool] = rgb_centre[mask_bool]

        log.info(f"  Row {ri+1}/{len(ys)} done")

    # ── Save ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    out_fg  = float(out.mean() * 100)

    log.info("=== Done ===")
    log.info(f"  Input  fg      : {binary.mean()*100:.2f}%")
    log.info(f"  Output fg      : {out_fg:.2f}%")
    log.info(f"  Pixels kept    : {px_kept:,}")
    log.info(f"  Elapsed        : {elapsed:.1f}s")

    # Save binary mask
    Image.fromarray((out * 255).astype(np.uint8)).save(
        str(dst), format="PNG", compress_level=3
    )
    log.info(f"  Binary saved → {dst}")

    # Save RGB overlay
    Image.fromarray(out_rgb).save(
        str(dst_rgb), format="PNG", compress_level=3
    )
    log.info(f"  RGB    saved → {dst_rgb}")

    with open(dst.with_suffix(".json"), "w") as f:
        json.dump({
            "binary_path":     binary_path,
            "rgb_path":        rgb_path,
            "output_path":     output_path,
            "rgb_output_path": output_rgb_path,
            "img_width":       W,
            "img_height":      H,
            "hsv_thresholds": {
                "S_MAX": S_MAX,
                "V_MAX": V_MAX,
            },
            "input_fg_pct":  round(float(binary.mean() * 100), 4),
            "output_fg_pct": round(out_fg, 4),
            "px_kept":       px_kept,
            "elapsed_sec":   round(elapsed, 2),
        }, f, indent=2)


if __name__ == "__main__":
    keep_black(BINARY_PATH, RGB_PATH, OUTPUT_PATH, OUTPUT_RGB_PATH)