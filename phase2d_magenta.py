"""
Phase 2d — Keep Only Magenta, Remove Everything Else
VFR Chart Extraction Pipeline

Keeps only magenta-coloured pixels in the binary mask.
Everything else is removed — black text, blue airspace,
brown terrain, cyan water, all of it.

Magenta on FAA VFR charts = obstacle symbols, airways,
and related aeronautical features printed in magenta ink.

A pixel is kept if it is BOTH:
  - Foreground in the binary mask
  - Magenta in the RGB source image

Magenta in HSV:
  Hue wraps around red: H in [0, 10] OR [160, 179]
  Saturation high: S > S_MIN  (vivid colour, not grey)
  Brightness high: V > V_MIN  (not dark/black)
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
BINARY_PATH = r"outputs\phase1_preprocessing\Washington_binary.png"
RGB_PATH    = r"outputs\phase1_preprocessing\Washington_rgb_300dpi.png"
OUTPUT_PATH = r"outputs\phase2_layer_segmentation\phase2d_clean_binary\Washington_magenta_only_v2.png"

# HSV range for magenta
# OpenCV HSV: H in [0,179], S in [0,255], V in [0,255]
# Magenta hue wraps around 0/179 boundary:
#   lower band: H in [0,  H_LOW_MAX]
#   upper band: H in [H_HIGH_MIN, 179]
H_LOW_MAX   = 15    # upper end of low hue band
H_HIGH_MIN  = 148   # lower end of high hue band
S_MIN       = 20    # minimum saturation — must be vivid
V_MIN       = 50    # minimum brightness — must not be black

TILE_SIZE = 2048
TILE_OVL  = 128
# ───────────────────────────────────────────────────────────────────────────────


def keep_magenta(binary_path: str, rgb_path: str, output_path: str) -> None:
    t0  = time.time()
    dst = Path(output_path)
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

    # ── Tiled magenta-keep processing ─────────────────────────────────────
    log.info(f"Keeping magenta pixels only ...")
    log.info(f"  H in [0,{H_LOW_MAX}] or [{H_HIGH_MIN},179]")
    log.info(f"  S > {S_MIN},  V > {V_MIN}")

    out        = np.zeros_like(binary)
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

            Hc = hsv_tile[:, :, 0]
            Sc = hsv_tile[:, :, 1]
            Vc = hsv_tile[:, :, 2]

            # Magenta mask: hue wraps around red boundary
            hue_match = (
                ((Hc <= H_LOW_MAX) | (Hc >= H_HIGH_MIN)) &
                (Sc > S_MIN) &
                (Vc > V_MIN)
            ).astype(np.uint8)

            # Keep only binary foreground pixels that are magenta in RGB
            kept = bin_tile & hue_match

            # Write centre region only
            cy0 = y0 - ty0;  cy1 = cy0 + min(TILE_SIZE, H - y0)
            cx0 = x0 - tx0;  cx1 = cx0 + min(TILE_SIZE, W - x0)
            out[y0:y0+(cy1-cy0), x0:x0+(cx1-cx0)] = kept[cy0:cy1, cx0:cx1]
            px_kept += int(kept[cy0:cy1, cx0:cx1].sum())

        log.info(f"  Row {ri+1}/{len(ys)} done")

    # ── Save ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    out_fg  = float(out.mean() * 100)

    log.info("=== Done ===")
    log.info(f"  Input  fg      : {binary.mean()*100:.2f}%")
    log.info(f"  Output fg      : {out_fg:.2f}%")
    log.info(f"  Pixels kept    : {px_kept:,}")
    log.info(f"  Elapsed        : {elapsed:.1f}s")

    Image.fromarray((out * 255).astype(np.uint8)).save(
        str(dst), format="PNG", compress_level=3
    )
    with open(dst.with_suffix(".json"), "w") as f:
        json.dump({
            "binary_path":   binary_path,
            "rgb_path":      rgb_path,
            "output_path":   output_path,
            "img_width":     W,
            "img_height":    H,
            "hsv_thresholds": {
                "H_LOW_MAX":  H_LOW_MAX,
                "H_HIGH_MIN": H_HIGH_MIN,
                "S_MIN":      S_MIN,
                "V_MIN":      V_MIN,
            },
            "input_fg_pct":  round(float(binary.mean() * 100), 4),
            "output_fg_pct": round(out_fg, 4),
            "px_kept":       px_kept,
            "elapsed_sec":   round(elapsed, 2),
        }, f, indent=2)
    log.info(f"  Saved → {dst}")


if __name__ == "__main__":
    keep_magenta(BINARY_PATH, RGB_PATH, OUTPUT_PATH)