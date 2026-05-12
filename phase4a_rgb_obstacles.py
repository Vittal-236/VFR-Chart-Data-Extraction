"""
Phase 4 — Symbol Detection (Sliding Window Template Matching)
VFR Chart Extraction Pipeline (FAA Base Model)

Algorithm (as specified by supervisor):
    1. Take the obstacle symbol template (hollow triangle with dot)
    2. Slide it across the entire map left to right, top to bottom
    3. At each position, count pixel overlap between template and map
    4. If overlap exceeds threshold → candidate detection
    5. Zoom into candidate region and verify shape
    6. Mark confirmed detections on output image

Usage (PowerShell):
    python phase4_obstacle_detection.py `
        --rgb     outputs/phase1_preprocessing/Washington_rgb.png `
        --output-dir outputs/phase4_symbol_detection/phase4a_rgb_obstacles `
        --dpi 150
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw
from skimage.color import rgb2hsv
from skimage.feature import match_template, peak_local_max

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase4")


def build_template(dpi: int = 150) -> np.ndarray:
    """
    Step 1: Build the obstacle triangle template.

    FAA obstacle symbol (verified from chart sample):
      - Hollow upward-pointing triangle
      - Interior dot centred inside
      - ~14px wide x 14px tall at 150 DPI

    Returns binary float64 array: 1.0=ink, 0.0=empty.
    """
    size = max(8, round(14 * dpi / 150))
    img  = np.zeros((size, size), dtype=np.float64)

    apex_r  = 1
    apex_c  = size // 2
    base_r  = size - 2
    base_lc = 1
    base_rc = size - 2

    def draw_edge(r0, c0, r1, c1):
        steps = max(abs(r1-r0), abs(c1-c0), 1)
        for i in range(steps + 1):
            t = i / steps
            r = int(round(r0 + t*(r1-r0)))
            c = int(round(c0 + t*(c1-c0)))
            for dr, dc in [(0,0),(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < size and 0 <= nc < size:
                    img[nr, nc] = 1.0

    draw_edge(apex_r, apex_c, base_r, base_lc)
    draw_edge(apex_r, apex_c, base_r, base_rc)
    draw_edge(base_r, base_lc, base_r, base_rc)

    dot_r = apex_r + int((base_r - apex_r) * 0.50)
    dot_c = apex_c
    dot_r2 = max(1, round(2 * dpi / 150))
    for dr in range(-dot_r2, dot_r2+1):
        for dc in range(-dot_r2, dot_r2+1):
            if dr*dr + dc*dc <= dot_r2*dot_r2:
                rr, cc = dot_r+dr, dot_c+dc
                if 0 <= rr < size and 0 <= cc < size:
                    img[rr, cc] = 1.0

    return img


def prepare_search_image(rgb: np.ndarray, tile_size: int = 2048) -> np.ndarray:
    """
    Step 2: Extract blue channel from RGB chart as search image.

    Obstacle triangles are blue (H:0.53-0.72, S>=0.20, V>=0.30).
    Processed in tiles to avoid large RAM allocation.
    """
    H, W = rgb.shape[:2]
    search = np.zeros((H, W), dtype=np.float64)
    n_ty = int(np.ceil(H / tile_size))
    n_tx = int(np.ceil(W / tile_size))
    total = n_ty * n_tx
    done = 0

    for ty in range(n_ty):
        for tx in range(n_tx):
            y0, y1 = ty*tile_size, min((ty+1)*tile_size, H)
            x0, x1 = tx*tile_size, min((tx+1)*tile_size, W)
            tile = rgb[y0:y1, x0:x1].astype(np.float64) / 255.0
            hsv  = rgb2hsv(tile)
            blue = (
                (hsv[..., 0] >= 0.53) & (hsv[..., 0] <= 0.72) &
                (hsv[..., 1] >= 0.20) &
                (hsv[..., 2] >= 0.30)
            ).astype(np.float64)
            search[y0:y1, x0:x1] = blue
            done += 1
            log.info(f"  Blue extraction tile {done}/{total}")

    log.info(f"  Blue coverage: {search.mean()*100:.1f}%")
    return search


def slide_template(search: np.ndarray, template: np.ndarray,
                   ncc_threshold: float) -> list:
    """
    Step 3: Slide the template across the entire map left to right, top to bottom.

    At every pixel position the NCC score measures how much the template
    overlaps with the map. Positions above ncc_threshold are candidates.

    Uses FFT-accelerated NCC (match_template) — mathematically equivalent
    to manually sliding a window across every position but ~1000x faster.

    Results are returned sorted left-to-right, top-to-bottom.
    """
    tH, tW = template.shape
    H, W   = search.shape
    log.info(f"  Template: {tW}x{tH}px  |  Map: {W}x{H}px")
    log.info(f"  Total positions to evaluate: {W*H:,}")

    response = match_template(search, template, pad_input=True)

    peaks = peak_local_max(
        response,
        min_distance=max(tH, tW),   # one detection per template-sized region
        threshold_abs=ncc_threshold,
    )

    candidates = [(int(r), int(c), float(response[r, c]))
                  for (r, c) in peaks
                  if response[r, c] >= ncc_threshold]

    # Sort left to right, top to bottom as specified
    candidates.sort(key=lambda x: (x[1], x[0]))

    log.info(f"  Candidates above NCC {ncc_threshold}: {len(candidates)}")
    return candidates


def verify_candidates(candidates: list, search: np.ndarray,
                      rgb: np.ndarray, template: np.ndarray,
                      ncc_threshold: float,
                      overlap_threshold: float,
                      legend_frac: float = 0.18,
                      blue_frac_min: float = 0.08) -> list:
    """
    Step 4: Three-gate verification for each sliding window candidate.

    Gate 1 — NCC score >= ncc_threshold
        Re-applies the threshold explicitly so the caller can pass a tighter
        value than what was used in the sliding window scan without re-running
        the full NCC pass.

    Gate 2 — Legend strip exclusion
        Rejects anything in the left 18% of the chart (legend, margin text).
        The Washington sectional legend occupies ~15–17% of chart width.

    Gate 3 — Blue colour verification
        Extracts a 20×20px patch from the original RGB image at the detection
        centre. Computes the fraction of pixels that fall in the FAA obstacle
        blue HSV range (H:0.53–0.72, S≥0.20, V≥0.30). Rejects if < 8%.
        This eliminates false positives on magenta airport symbols, black text
        characters (like 'A' in WARNING), and red/brown terrain features.

    Gate 4 — Pixel overlap check
        Counts how many of the template's ink pixels are matched by blue ink
        in the map patch. Overlap ratio must meet overlap_threshold.
    """
    from skimage.color import rgb2hsv as _rgb2hsv

    H, W   = search.shape
    tH, tW = template.shape
    tmpl_binary = template > 0.5
    legend_col  = int(W * legend_frac)

    n_g1 = n_g2 = n_g3 = n_g4 = 0
    confirmed = []

    for (r, c, ncc_score) in candidates:

        # Gate 1: NCC threshold
        if ncc_score < ncc_threshold:
            n_g1 += 1
            continue

        # Gate 2: Legend strip
        if c < legend_col:
            n_g2 += 1
            continue

        # Gate 3: Blue colour verification on RGB patch
        pad = 10
        pr0 = max(0, r-pad); pr1 = min(H, r+pad)
        pc0 = max(0, c-pad); pc1 = min(W, c+pad)
        rgb_patch = rgb[pr0:pr1, pc0:pc1].astype(np.float64) / 255.0
        if rgb_patch.size == 0:
            n_g3 += 1
            continue
        hsv = _rgb2hsv(rgb_patch)
        blue_mask = (
            (hsv[..., 0] >= 0.53) & (hsv[..., 0] <= 0.72) &
            (hsv[..., 1] >= 0.20) &
            (hsv[..., 2] >= 0.30)
        )
        if blue_mask.mean() < blue_frac_min:
            n_g3 += 1
            continue

        # Gate 4: Pixel overlap on search (blue binary) patch
        r0 = max(0, r - tH//2);  r1 = min(H, r0 + tH)
        c0 = max(0, c - tW//2);  c1 = min(W, c0 + tW)
        patch     = search[r0:r1, c0:c1] > 0.5
        tmpl_crop = tmpl_binary[:patch.shape[0], :patch.shape[1]]
        n_tmpl    = tmpl_crop.sum()
        if n_tmpl == 0:
            continue
        overlap       = np.logical_and(tmpl_crop, patch).sum()
        overlap_ratio = float(overlap) / float(n_tmpl)

        if overlap_ratio < overlap_threshold:
            n_g4 += 1
            continue

        confirmed.append({
            "pixel_x":       int(c),
            "pixel_y":       int(r),
            "ncc_score":     round(ncc_score, 4),
            "overlap_ratio": round(overlap_ratio, 4),
            "bbox":          [int(r0), int(c0), int(r1), int(c1)],
        })

    log.info(f"  Gate 1 (NCC)    rejected: {n_g1}")
    log.info(f"  Gate 2 (legend) rejected: {n_g2}")
    log.info(f"  Gate 3 (colour) rejected: {n_g3}")
    log.info(f"  Gate 4 (overlap)rejected: {n_g4}")
    log.info(f"  Confirmed detections    : {len(confirmed)}")
    return confirmed


def draw_detections(rgb: np.ndarray, detections: list,
                    template_shape: tuple) -> Image.Image:
    """Step 5: Draw red circle around every confirmed detection."""
    vis  = Image.fromarray(rgb)
    draw = ImageDraw.Draw(vis)
    tH, tW = template_shape
    r = max(tH, tW) // 2 + 5

    for d in detections:
        cx, cy = d["pixel_x"], d["pixel_y"]
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=(255,0,0), width=2)
        draw.text((cx+r+2, cy-6), f"{d['ncc_score']:.2f}", fill=(255,0,0))

    return vis


def detect_obstacles(
    rgb: np.ndarray,
    output_dir: Optional[str] = None,
    stem: str = "Washington",
    dpi: int = 150,
    ncc_threshold: float = 0.65,
    overlap_threshold: float = 0.25,
) -> dict:
    """
    Full 5-step pipeline:
        1. Build template
        2. Prepare blue search image
        3. Slide template across map (left→right, top→bottom)
        4. Verify by pixel overlap, reject legend FPs
        5. Draw and save
    """
    t0 = time.time()
    log.info("=" * 55)
    log.info("Phase 4 — Sliding Window Obstacle Detection")
    log.info("=" * 55)

    log.info("Step 1: Building template …")
    template = build_template(dpi)
    log.info(f"  Template: {template.shape[1]}x{template.shape[0]}px")

    log.info("Step 2: Extracting blue search image …")
    search = prepare_search_image(rgb)

    log.info("Step 3: Sliding template across map …")
    candidates = slide_template(search, template, ncc_threshold)

    log.info("Step 4: Verifying candidates by pixel overlap …")
    detections = verify_candidates(
        candidates, search, rgb, template,
        ncc_threshold=ncc_threshold,
        overlap_threshold=overlap_threshold,
    )

    log.info("Step 5: Drawing detections …")
    vis = draw_detections(rgb, detections, template.shape)

    elapsed = round(time.time() - t0, 2)
    result  = {
        "stem":               stem,
        "dpi":                dpi,
        "template_size_px":   list(template.shape),
        "ncc_threshold":      ncc_threshold,
        "overlap_threshold":  overlap_threshold,
        "n_candidates":       len(candidates),
        "n_detections":       len(detections),
        "detections":         detections,
        "elapsed_sec":        elapsed,
    }

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        img_path = out / f"{stem}_phase4_obstacles.png"
        vis.save(str(img_path))
        log.info(f"  Saved → {img_path}")
        log_path = out / f"{stem}_phase4_log.json"
        with open(log_path, "w") as f:
            json.dump(result, f, indent=2)
        log.info(f"  Saved → {log_path}")

    log.info(f"Done in {elapsed}s  |  {len(detections)} detections")
    return result


if __name__ == "__main__":
    Image.MAX_IMAGE_PIXELS = None

    # ── CONFIG ──────────────────────────────────────────────────────────────────
    RGB_PATH   = "outputs/phase1_preprocessing/chart_rgb.png"  # RGB chart image
    OUTPUT_DIR = "outputs/phase4_symbol_detection/phase4a_rgb_obstacles"
    STEM       = None   # None = derived from RGB filename
    DPI        = 150
    NCC        = 0.65   # NCC threshold for template matching
    OVERLAP    = 0.25   # Overlap threshold for NMS
    # ────────────────────────────────────────────────────────────────────────────

    log.info(f"Loading: {RGB_PATH}")
    rgb  = np.array(Image.open(RGB_PATH).convert("RGB"), dtype=np.uint8)
    stem = STEM or Path(RGB_PATH).stem.replace("_rgb", "")

    detect_obstacles(
        rgb=rgb,
        output_dir=OUTPUT_DIR,
        stem=stem,
        dpi=DPI,
        ncc_threshold=NCC,
        overlap_threshold=OVERLAP,
    )