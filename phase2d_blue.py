"""
Phase 2d — Keep Only Blue Airport/Airspace Features
VFR Chart Extraction Pipeline

Input  : Phase 1 binary PNG  (white fg, black bg)
         Phase 1 RGB PNG     (denoised colour image, 300dpi)
Output : Binary keeping only blue-ink features:
           - Towered airport rings
           - Class B / C / D airspace boundary circles
           - VOR compass roses
           - Blue airspace boundary arcs

COLOUR PROFILE (confirmed by pixel sampling):
  Ocean background : H~104, S~31,  V~246  → excluded by S_MIN
  Airport/airspace : H~93-103, S~200-240, V~100-175

HSV ranges used:
  H : 88 – 105   (blue-teal hue band)
  S : 180 – 255  (highly saturated — excludes washed-out ocean background)
  V : 80  – 190  (medium brightness — excludes white and very dark pixels)
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
OUTPUT_PATH = r"outputs\phase2_layer_segmentation\phase2d_clean_binary\Washington_blue_only.png"

# HSV range for airport/airspace blue ink
# OpenCV HSV: H in [0,179], S in [0,255], V in [0,255]
H_MIN = 88    # blue-teal hue lower bound
H_MAX = 105   # blue-teal hue upper bound
S_MIN = 180   # high saturation — ocean background (S~31) is excluded here
V_MIN = 80    # not too dark
V_MAX = 190   # not too bright (white excluded)

TILE_SIZE = 2048
TILE_OVL  = 128
# ───────────────────────────────────────────────────────────────────────────────


def keep_blue(binary_path: str, rgb_path: str, output_path: str) -> None:
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

    # ── Tiled blue-keep processing ─────────────────────────────────────────
    log.info(f"Keeping blue pixels: H=[{H_MIN},{H_MAX}], S>={S_MIN}, V=[{V_MIN},{V_MAX}]")
    out     = np.zeros_like(binary)
    ys      = list(range(0, H, TILE_SIZE))
    xs      = list(range(0, W, TILE_SIZE))
    px_kept = 0
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

            # Blue mask
            blue_mask = (
                (Hc >= H_MIN) & (Hc <= H_MAX) &
                (Sc >= S_MIN) &
                (Vc >= V_MIN) & (Vc <= V_MAX)
            ).astype(np.uint8)

            # Keep only binary foreground pixels that are blue in RGB
            kept = bin_tile & blue_mask

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
            "binary_path":    binary_path,
            "rgb_path":       rgb_path,
            "output_path":    output_path,
            "img_width":      W,
            "img_height":     H,
            "hsv_thresholds": {
                "H_MIN": H_MIN, "H_MAX": H_MAX,
                "S_MIN": S_MIN,
                "V_MIN": V_MIN, "V_MAX": V_MAX,
            },
            "input_fg_pct":  round(float(binary.mean() * 100), 4),
            "output_fg_pct": round(out_fg, 4),
            "px_kept":       px_kept,
            "elapsed_sec":   round(elapsed, 2),
        }, f, indent=2)
    log.info(f"  Saved → {dst}")


if __name__ == "__main__":
    keep_blue(BINARY_PATH, RGB_PATH, OUTPUT_PATH)