"""
Phase 4 (Combined) — Single and Double Obstacle Detection on Binary Map
VFR Chart Extraction Pipeline (FAA Base Model)

Runs both the single-obstacle (Phase 4a) and double-obstacle (Phase 4c)
detection pipelines on the same loaded images, then draws all detections
onto one shared RGB overlay.

  Single obstacle  → GREEN  bounding boxes on the RGB overlay
  Double obstacle  → MAGENTA bounding boxes on the RGB overlay

All images are loaded once and shared between both pipelines.
Per-type diagnostic outputs (crops, dot samples, raw peak crops) are
saved under sub-folders: OUTPUT_DIR/single/ and OUTPUT_DIR/double/.
The combined JSON and the combined RGB overlay are saved directly to
OUTPUT_DIR.

TILING RULE
-----------
The full map is NEVER processed at once. NCC runs on 2048x2048 tiles
with overlap so symbols at tile boundaries are not missed.

HOW TO USE
----------
1. Set the paths in CONFIG below.
2. Run:  python phase4_combined_obstacles.py
3. Main outputs in OUTPUT_DIR:
     detections_combined.json     all confirmed detections (both types)
     rgb_overlay_combined.png     colour map with both types annotated
   Sub-folder outputs:
     single/confirmed_map.png     binary map with single-obstacle boxes
     single/confirmed_crops.png   contact sheet of single-obstacle crops
     single/dot_search_samples.png
     single/raw_peak_crops.png
     double/confirmed_map.png
     double/confirmed_crops.png
     double/dot_samples.png
     double/raw_peak_crops.png
"""

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None

from skimage.feature import match_template, peak_local_max
from skimage.measure import label, regionprops


# =============================================================================
# CONFIG — edit these paths, then run
# =============================================================================

SYMBOLS_PATH = r"outputs/phase2_layer_segmentation/phase2c_symbols_only_binary/Washington_symbols_only.png"
BINARY_PATH  = r"outputs/phase1_preprocessing/Washington_binary.png"
RGB_PATH     = r"outputs/phase1_preprocessing/Washington_rgb_150dpi.png"
OUTPUT_DIR   = r"outputs/phase4_symbol_detection/phase4_combined"

# =============================================================================
# SHARED CONSTANTS
# =============================================================================

TILE_SIZE    = 2048
TILE_OVERLAP = 100
LEGEND_STRIP_WIDTH = 140

# =============================================================================
# SINGLE-OBSTACLE PARAMETERS  (Phase 4a)
# =============================================================================

S_NCC_THRESHOLD = 0.62
S_NCC_THRESHOLD_PER_SCALE = {
    24: 0.65,
    28: 0.62,
    32: 0.62,
}
S_PEAK_MIN_DISTANCE = 10
S_TEMPLATE_WIDTHS   = [24, 28, 32]
S_MIN_SCALE_W       = 24

S_DOT_BELOW_FRAC    = 0.0
S_DOT_SEARCH_H_FRAC = 0.30
S_DOT_SEARCH_W_FRAC = 0.40
S_DOT_MIN_AREA      = 3
S_DOT_MAX_AREA      = 25
S_DOT_MAX_ECC       = 0.65

S_BASE_CHECK_FRAC = 0.15
S_BASE_INK_LIMIT  = 0.10

S_APEX_CHECK_FRAC = 0.15
S_APEX_MAX_SPREAD = 0.35

# --- Arm continuity check ---
# Sample N points along each expected arm diagonal and measure what fraction
# of sample positions actually contain ink. A real ∧ scores ~0.70+.
# Arc fragments and letter halves often have gaps and score below 0.50.
S_ARM_SAMPLES      = 12    # number of points sampled along each arm
S_ARM_RADIUS       = 1     # pixel search radius around each sample point
S_ARM_MIN_COVERAGE = 0.50  # each arm must hit ink at >= 50% of sample points

# =============================================================================
# DOUBLE-OBSTACLE PARAMETERS  (Phase 4c)
# =============================================================================

D_NCC_THRESHOLD = 0.50
D_NCC_THRESHOLD_PER_SCALE = {
    40: 0.52,
    48: 0.50,
    56: 0.50,
    64: 0.50,
}
D_PEAK_MIN_DISTANCE = 18
D_TEMPLATE_WIDTHS   = [40, 48, 56, 64]
D_MIN_SCALE_W       = 40

D_BASE_CHECK_FRAC     = 0.12
D_BASE_ARM_ZONE       = 0.15
D_BASE_ARM_MIN_INK    = 0.05
D_BASE_CENTRE_ZONE    = 0.25
D_BASE_CENTRE_MAX_INK = 0.08

D_APEX_CHECK_FRAC  = 0.15
D_APEX_LEFT_ZONE   = 0.40
D_APEX_RIGHT_ZONE  = 0.40
D_APEX_MIN_INK     = 0.05
D_APEX_GAP_ZONE_LO = 0.40
D_APEX_GAP_ZONE_HI = 0.60
D_APEX_MAX_GAP_INK = 0.10

D_VALLEY_ROW_LO   = 0.37
D_VALLEY_ROW_HI   = 0.74
D_VALLEY_GAP_PX   = 4
D_VALLEY_N_GROUPS = 3

D_SYMMETRY_MIN         = 0.40
D_ISOLATION_MAX_W      = 1.8

# --- Valley-band ink density check ---
# In the crossing region, the double obstacle must have ink spread across
# a reasonable fraction of the width on average. Sparse/partial matches
# (single arm fragment, letter M) tend to have very low mean ink density
# across the valley band.
D_VALLEY_DENSITY_MIN = 0.04   # mean ink fraction across valley-band rows

D_DOT_BELOW_FRAC        = 0.86
D_DOT_SEARCH_W_FRAC     = 0.12
D_DOT_LEFT_CENTRE_FRAC  = 0.344
D_DOT_RIGHT_CENTRE_FRAC = 0.653
D_DOT_MIN_AREA          = 2
D_DOT_MAX_AREA          = 20
D_DOT_MAX_ECC           = 0.85


# =============================================================================
# TEMPLATE BUILDERS
# =============================================================================

def build_single_template(width_px: int) -> np.ndarray:
    """Open inverted-V for single obstacle. H/W ratio = 126/118."""
    height_px = int(round(width_px * 126 / 118))
    tmpl  = np.zeros((height_px, width_px), dtype=np.float32)
    stroke = 1 if width_px < 24 else 2

    apex_r  = 0
    apex_c  = (width_px - 1) / 2.0
    base_l  = (height_px - 1, 0.0)
    base_r  = (height_px - 1, float(width_px - 1))

    def draw_arm(r0, c0, r1, c1):
        steps = max(abs(r1 - r0), abs(c1 - c0), 1)
        for i in range(int(steps) + 1):
            t  = i / steps
            r  = r0 + t * (r1 - r0)
            c  = c0 + t * (c1 - c0)
            for dr in range(-stroke + 1, stroke):
                for dc in range(-stroke + 1, stroke):
                    rr = int(round(r)) + dr
                    cc = int(round(c)) + dc
                    if 0 <= rr < height_px and 0 <= cc < width_px:
                        tmpl[rr, cc] = 1.0

    draw_arm(apex_r, apex_c, *base_l)
    draw_arm(apex_r, apex_c, *base_r)
    return tmpl


def build_double_template(width_px: int) -> np.ndarray:
    """Double inverted-V (M-shape) for double obstacle. H/W ratio = 0.772."""
    height_px = int(round(width_px * 0.772))
    tmpl  = np.zeros((height_px, width_px), dtype=np.float32)
    stroke = 2 if width_px >= 48 else 1

    apex_L_r, apex_L_c = 0, width_px * 0.344
    apex_R_r, apex_R_c = 0, width_px * 0.650
    base_L   = (height_px - 1, 0.0)
    base_R   = (height_px - 1, float(width_px - 1))
    base_mid = (height_px - 1, (width_px - 1) * 0.5)

    def draw_arm(r0, c0, r1, c1):
        steps = max(abs(r1 - r0), abs(c1 - c0), 1)
        for i in range(int(steps) + 1):
            t  = i / steps
            r  = r0 + t * (r1 - r0)
            c  = c0 + t * (c1 - c0)
            for dr in range(-stroke + 1, stroke):
                for dc in range(-stroke + 1, stroke):
                    rr = int(round(r)) + dr
                    cc = int(round(c)) + dc
                    if 0 <= rr < height_px and 0 <= cc < width_px:
                        tmpl[rr, cc] = 1.0

    draw_arm(apex_L_r, apex_L_c, *base_L)
    draw_arm(apex_L_r, apex_L_c, *base_mid)
    draw_arm(apex_R_r, apex_R_c, *base_mid)
    draw_arm(apex_R_r, apex_R_c, *base_R)
    return tmpl


# =============================================================================
# SHARED NCC RUNNER
# =============================================================================

def run_ncc(binary: np.ndarray, templates: dict,
            ncc_threshold: float, ncc_per_scale: dict,
            peak_min_dist: int, label: str) -> list:
    """
    Tiled multi-scale NCC. Processes 2048x2048 tiles with TILE_OVERLAP overlap.
    The full map image is never processed at once.

    binary       : bool array (H, W), True = ink
    templates    : dict  {width_px: template_array}
    ncc_threshold: global threshold floor
    ncc_per_scale: per-scale override thresholds
    peak_min_dist: minimum pixel distance between peaks
    label        : short string for console output ('single' or 'double')
    """
    H, W      = binary.shape
    all_peaks = []

    tile_rows = list(range(0, H, TILE_SIZE))
    tile_cols = list(range(0, W, TILE_SIZE))
    n_tiles   = len(tile_rows) * len(tile_cols)
    tile_idx  = 0

    for tr in tile_rows:
        for tc in tile_cols:
            tile_idx += 1
            inner_r0 = tr
            inner_c0 = tc
            inner_r1 = min(tr + TILE_SIZE, H)
            inner_c1 = min(tc + TILE_SIZE, W)

            pad_r0 = max(0, tr - TILE_OVERLAP)
            pad_c0 = max(0, tc - TILE_OVERLAP)
            pad_r1 = min(H, tr + TILE_SIZE + TILE_OVERLAP)
            pad_c1 = min(W, tc + TILE_SIZE + TILE_OVERLAP)

            tile = binary[pad_r0:pad_r1, pad_c0:pad_c1].astype(np.float32)

            print(f"  [{label}] Tile {tile_idx}/{n_tiles}  "
                  f"map[{inner_r0}:{inner_r1}, {inner_c0}:{inner_c1}]  "
                  f"padded {tile.shape[1]}x{tile.shape[0]}px", end="")

            tile_peaks = 0

            for w, tmpl in templates.items():
                th = tmpl.shape[0]
                if tile.shape[0] < th or tile.shape[1] < w:
                    continue

                corr  = match_template(tile, tmpl, pad_input=False)
                peaks = peak_local_max(corr, min_distance=peak_min_dist,
                                       threshold_abs=ncc_threshold)

                for pr, pc in peaks:
                    map_r = pr + pad_r0
                    map_c = pc + pad_c0

                    scale_thresh = ncc_per_scale.get(w, ncc_threshold)
                    if corr[pr, pc] < scale_thresh:
                        continue
                    if not (inner_r0 <= map_r < inner_r1 and
                            inner_c0 <= map_c < inner_c1):
                        continue
                    if map_c < LEGEND_STRIP_WIDTH:
                        continue

                    all_peaks.append({
                        "row":       int(map_r),
                        "col":       int(map_c),
                        "scale_w":   w,
                        "scale_h":   th,
                        "ncc_score": float(corr[pr, pc]),
                    })
                    tile_peaks += 1

            print(f"  -> {tile_peaks} peaks")

    return all_peaks


# =============================================================================
# SHARED NMS
# =============================================================================

def nms(candidates: list) -> list:
    """Remove duplicate detections; keeps highest NCC score."""
    if not candidates:
        return []
    candidates = sorted(candidates, key=lambda x: -x["ncc_score"])
    kept = []
    suppressed = set()
    for i, c in enumerate(candidates):
        if i in suppressed:
            continue
        kept.append(c)
        cy = c["row"] + c["scale_h"] / 2
        cx = c["col"] + c["scale_w"] / 2
        for j, d in enumerate(candidates[i + 1:], start=i + 1):
            if j in suppressed:
                continue
            dy = d["row"] + d["scale_h"] / 2
            dx = d["col"] + d["scale_w"] / 2
            if math.hypot(cy - dy, cx - dx) < (c["scale_w"] + d["scale_w"]) / 2 * 0.6:
                suppressed.add(j)
    return kept


# =============================================================================
# SINGLE-OBSTACLE FILTERS
# =============================================================================

def s_passes_open_base(sym: np.ndarray, c: dict) -> bool:
    r0, c0 = c["row"], c["col"]
    tw, th = c["scale_w"], c["scale_h"]
    H, W   = sym.shape
    patch  = sym[r0:min(H, r0+th), c0:min(W, c0+tw)]
    ph, pw = patch.shape
    if ph == 0 or pw == 0:
        return True
    bot_rows   = max(1, int(round(ph * S_BASE_CHECK_FRAC)))
    bot_zone   = patch[ph - bot_rows:, :]
    c_lo, c_hi = pw // 3, pw - pw // 3
    centre     = bot_zone[:, c_lo:c_hi]
    if centre.size == 0:
        return True
    return (centre.sum() / centre.size) <= S_BASE_INK_LIMIT


def s_passes_apex(sym: np.ndarray, c: dict) -> bool:
    r0, c0 = c["row"], c["col"]
    tw, th = c["scale_w"], c["scale_h"]
    H, W   = sym.shape
    patch  = sym[r0:min(H, r0+th), c0:min(W, c0+tw)]
    ph, pw = patch.shape
    if ph == 0 or pw == 0:
        return True
    apex_rows = max(1, int(round(ph * S_APEX_CHECK_FRAC)))
    top       = patch[:apex_rows, :]
    ink_cols  = np.where(top.any(axis=0))[0]
    if len(ink_cols) == 0:
        return True
    spread = (ink_cols[-1] - ink_cols[0] + 1) / pw
    return spread <= S_APEX_MAX_SPREAD


def s_passes_symmetry(sym: np.ndarray, c: dict) -> bool:
    r0, c0 = c["row"], c["col"]
    tw, th = c["scale_w"], c["scale_h"]
    H, W   = sym.shape
    patch  = sym[r0:min(H, r0+th), c0:min(W, c0+tw)]
    ph, pw = patch.shape
    if ph == 0 or pw < 2:
        return True
    left  = patch[:, :pw // 2].sum()
    right = patch[:, pw - pw // 2:].sum()
    total = left + right
    if total == 0:
        return True
    ratio = min(left, right) / max(left, right)
    return ratio >= 0.40


def s_passes_isolation(sym: np.ndarray, c: dict) -> bool:
    r0, c0 = c["row"], c["col"]
    tw, th = c["scale_w"], c["scale_h"]
    H, W   = sym.shape
    patch  = sym[r0:min(H, r0+th), c0:min(W, c0+tw)]
    ph, pw = patch.shape
    if ph == 0 or pw == 0:
        return True
    labelled = label(patch, connectivity=2)
    regions  = regionprops(labelled)
    if not regions:
        return True
    cx      = patch.shape[1] / 2
    nearest = min(regions, key=lambda r: abs(r.centroid[1] - cx))
    comp    = labelled == nearest.label
    comp_cols = np.where(comp.any(axis=0))[0]
    if len(comp_cols) == 0:
        return True
    return (comp_cols[-1] - comp_cols[0] + 1) <= 1.8 * tw


def s_passes_arm_continuity(sym: np.ndarray, c: dict) -> bool:
    """
    Check that ink is present along both expected arm diagonals.

    A real ∧ has ink running continuously from the apex down to each base
    corner. Arc fragments, letter halves, and noise blobs often have large
    gaps along one or both diagonals.

    Method:
      Sample S_ARM_SAMPLES evenly-spaced points along each arm's expected
      diagonal (apex -> bottom-left, apex -> bottom-right). At each sample
      point, check whether any pixel within S_ARM_RADIUS is ink. The fraction
      of sample points that hit ink must be >= S_ARM_MIN_COVERAGE for both arms.
    """
    r0, c0 = c["row"], c["col"]
    tw, th = c["scale_w"], c["scale_h"]
    H, W   = sym.shape
    patch  = sym[r0:min(H, r0+th), c0:min(W, c0+tw)]
    ph, pw = patch.shape
    if ph == 0 or pw == 0:
        return True

    apex_r = 0.0
    apex_c = (pw - 1) / 2.0

    def arm_coverage(end_r, end_c):
        hits = 0
        for i in range(S_ARM_SAMPLES):
            t   = i / max(S_ARM_SAMPLES - 1, 1)
            sr  = int(round(apex_r + t * (end_r - apex_r)))
            sc  = int(round(apex_c + t * (end_c - apex_c)))
            found = False
            for dr in range(-S_ARM_RADIUS, S_ARM_RADIUS + 1):
                for dc in range(-S_ARM_RADIUS, S_ARM_RADIUS + 1):
                    rr, cc = sr + dr, sc + dc
                    if 0 <= rr < ph and 0 <= cc < pw and patch[rr, cc]:
                        found = True
                        break
                if found:
                    break
            if found:
                hits += 1
        return hits / S_ARM_SAMPLES

    left_cov  = arm_coverage(ph - 1, 0.0)
    right_cov = arm_coverage(ph - 1, float(pw - 1))
    return left_cov >= S_ARM_MIN_COVERAGE and right_cov >= S_ARM_MIN_COVERAGE


def s_verify_dot(dot_bin: np.ndarray, c: dict) -> dict:
    r0, c0 = c["row"], c["col"]
    tw, th = c["scale_w"], c["scale_h"]
    H, W   = dot_bin.shape
    base_row    = r0 + th
    centre_col  = c0 + tw // 2
    search_h_px = max(4, int(round(th * S_DOT_SEARCH_H_FRAC)))
    search_w_px = max(6, int(round(tw * S_DOT_SEARCH_W_FRAC)))
    win_r0 = max(0, base_row)
    win_r1 = min(H, win_r0 + search_h_px)
    win_c0 = max(0, centre_col - search_w_px // 2)
    win_c1 = min(W, centre_col + search_w_px // 2)
    c["dot_win"]   = (win_r0, win_c0, win_r1, win_c1)
    c["dot_found"] = False
    c["dot_area"]  = 0
    if win_r1 <= win_r0 or win_c1 <= win_c0:
        return c
    window = dot_bin[win_r0:win_r1, win_c0:win_c1]
    if window.sum() == 0:
        return c
    for p in regionprops(label(window, connectivity=2)):
        if S_DOT_MIN_AREA <= p.area <= S_DOT_MAX_AREA and p.eccentricity <= S_DOT_MAX_ECC:
            c["dot_found"] = True
            c["dot_area"]  = int(p.area)
            break
    return c


# =============================================================================
# DOUBLE-OBSTACLE FILTERS
# =============================================================================

def d_passes_open_base(sym: np.ndarray, c: dict) -> bool:
    r0, c0 = c["row"], c["col"]
    tw, th = c["scale_w"], c["scale_h"]
    H, W   = sym.shape
    patch  = sym[r0:min(H, r0+th), c0:min(W, c0+tw)]
    ph, pw = patch.shape
    if ph == 0 or pw == 0:
        return True
    bot_rows = max(1, int(round(ph * D_BASE_CHECK_FRAC)))
    bot      = patch[ph - bot_rows:, :]
    arm_w    = max(1, int(round(pw * D_BASE_ARM_ZONE)))
    left_z   = bot[:, :arm_w]
    right_z  = bot[:, pw - arm_w:]
    cz_lo    = int(round(pw * D_BASE_CENTRE_ZONE))
    cz_hi    = pw - int(round(pw * D_BASE_CENTRE_ZONE))
    centre_z = bot[:, cz_lo:cz_hi]
    if left_z.size == 0 or right_z.size == 0:
        return True
    left_ink   = left_z.sum()   / left_z.size
    right_ink  = right_z.sum()  / right_z.size
    centre_ink = centre_z.sum() / centre_z.size if centre_z.size > 0 else 0.0
    return (left_ink >= D_BASE_ARM_MIN_INK and
            right_ink >= D_BASE_ARM_MIN_INK and
            centre_ink <= D_BASE_CENTRE_MAX_INK)


def d_passes_dual_apex(sym: np.ndarray, c: dict) -> bool:
    r0, c0 = c["row"], c["col"]
    tw, th = c["scale_w"], c["scale_h"]
    H, W   = sym.shape
    patch  = sym[r0:min(H, r0+th), c0:min(W, c0+tw)]
    ph, pw = patch.shape
    if ph == 0 or pw == 0:
        return True
    apex_rows  = max(1, int(round(ph * D_APEX_CHECK_FRAC)))
    top        = patch[:apex_rows, :]
    left_w     = max(1, int(round(pw * D_APEX_LEFT_ZONE)))
    right_w    = max(1, int(round(pw * D_APEX_RIGHT_ZONE)))
    gap_lo     = int(round(pw * D_APEX_GAP_ZONE_LO))
    gap_hi     = int(round(pw * D_APEX_GAP_ZONE_HI))
    left_zone  = top[:, :left_w]
    right_zone = top[:, pw - right_w:]
    gap_zone   = top[:, gap_lo:gap_hi]
    if left_zone.size == 0 or right_zone.size == 0:
        return True
    left_ink  = left_zone.sum()  / left_zone.size
    right_ink = right_zone.sum() / right_zone.size
    gap_ink   = gap_zone.sum()   / gap_zone.size if gap_zone.size > 0 else 0.0
    return (left_ink >= D_APEX_MIN_INK and
            right_ink >= D_APEX_MIN_INK and
            gap_ink <= D_APEX_MAX_GAP_INK)


def d_passes_valley(sym: np.ndarray, c: dict) -> bool:
    r0, c0 = c["row"], c["col"]
    tw, th = c["scale_w"], c["scale_h"]
    H, W   = sym.shape
    patch  = sym[r0:min(H, r0+th), c0:min(W, c0+tw)]
    ph, pw = patch.shape
    if ph == 0 or pw == 0:
        return True
    row_lo = int(round(ph * D_VALLEY_ROW_LO))
    row_hi = int(round(ph * D_VALLEY_ROW_HI))
    for row_idx in range(row_lo, min(row_hi, ph)):
        ink_cols = np.where(patch[row_idx])[0]
        if len(ink_cols) < 2:
            continue
        groups = 1
        for k in range(1, len(ink_cols)):
            if ink_cols[k] - ink_cols[k - 1] > D_VALLEY_GAP_PX:
                groups += 1
        if groups >= D_VALLEY_N_GROUPS:
            return True
    return False


def d_passes_symmetry(sym: np.ndarray, c: dict) -> bool:
    r0, c0 = c["row"], c["col"]
    tw, th = c["scale_w"], c["scale_h"]
    H, W   = sym.shape
    patch  = sym[r0:min(H, r0+th), c0:min(W, c0+tw)]
    ph, pw = patch.shape
    if ph == 0 or pw < 2:
        return True
    left  = patch[:, :pw // 2].sum()
    right = patch[:, pw - pw // 2:].sum()
    total = left + right
    if total == 0:
        return True
    return min(left, right) / max(left, right) >= D_SYMMETRY_MIN


def d_passes_isolation(sym: np.ndarray, c: dict) -> bool:
    r0, c0 = c["row"], c["col"]
    tw, th = c["scale_w"], c["scale_h"]
    H, W   = sym.shape
    patch  = sym[r0:min(H, r0+th), c0:min(W, c0+tw)]
    ph, pw = patch.shape
    if ph == 0 or pw == 0:
        return True
    labelled = label(patch, connectivity=2)
    regions  = regionprops(labelled)
    if not regions:
        return True
    cx      = pw / 2
    nearest = min(regions, key=lambda r: abs(r.centroid[1] - cx))
    comp    = labelled == nearest.label
    comp_cols = np.where(comp.any(axis=0))[0]
    if len(comp_cols) == 0:
        return True
    return (comp_cols[-1] - comp_cols[0] + 1) <= D_ISOLATION_MAX_W * tw


def d_passes_valley_density(sym: np.ndarray, c: dict) -> bool:
    """
    Check that the valley/crossing region has sufficient ink density overall.

    The double obstacle has four arms converging in the crossing region,
    producing relatively dense ink across the full width. A single-arm
    fragment or letter fragment that somehow passes the valley group check
    typically has very sparse ink in this band.

    Method:
      Take the rows from D_VALLEY_ROW_LO to D_VALLEY_ROW_HI. Compute the
      mean ink fraction across all pixels in that band. If below
      D_VALLEY_DENSITY_MIN, reject.
    """
    r0, c0 = c["row"], c["col"]
    tw, th = c["scale_w"], c["scale_h"]
    H, W   = sym.shape
    patch  = sym[r0:min(H, r0+th), c0:min(W, c0+tw)]
    ph, pw = patch.shape
    if ph == 0 or pw == 0:
        return True
    row_lo  = int(round(ph * D_VALLEY_ROW_LO))
    row_hi  = int(round(ph * D_VALLEY_ROW_HI))
    band    = patch[row_lo:row_hi, :]
    if band.size == 0:
        return True
    return (band.sum() / band.size) >= D_VALLEY_DENSITY_MIN


def d_verify_two_dots(dot_bin: np.ndarray, c: dict) -> dict:
    r0, c0 = c["row"], c["col"]
    tw, th = c["scale_w"], c["scale_h"]
    H, W   = dot_bin.shape
    search_w_px = max(4, int(round(tw * D_DOT_SEARCH_W_FRAC)))
    tick_top    = int(round((r0 + th * D_DOT_BELOW_FRAC)))
    tick_bot    = min(H, r0 + th)

    def check_dot(centre_frac):
        dc     = int(round(c0 + tw * centre_frac))
        wc0    = max(0, dc - search_w_px // 2)
        wc1    = min(W, dc + search_w_px // 2)
        wr0    = max(0, tick_top)
        wr1    = min(H, tick_bot)
        win    = (wr0, wc0, wr1, wc1)
        found, area = False, 0
        if wr1 > wr0 and wc1 > wc0:
            window = dot_bin[wr0:wr1, wc0:wc1]
            if window.sum() > 0:
                for p in regionprops(label(window, connectivity=2)):
                    if D_DOT_MIN_AREA <= p.area <= D_DOT_MAX_AREA and p.eccentricity <= D_DOT_MAX_ECC:
                        found, area = True, int(p.area)
                        break
        return win, found, area

    l_win, l_found, l_area = check_dot(D_DOT_LEFT_CENTRE_FRAC)
    r_win, r_found, r_area = check_dot(D_DOT_RIGHT_CENTRE_FRAC)
    c["dot_left_win"]    = l_win
    c["dot_right_win"]   = r_win
    c["dot_left_found"]  = l_found
    c["dot_right_found"] = r_found
    c["dot_left_area"]   = l_area
    c["dot_right_area"]  = r_area
    c["both_dots_found"] = l_found and r_found
    return c


# =============================================================================
# DIAGNOSTIC SAVERS — SINGLE
# =============================================================================

def save_single_confirmed_map(sym: np.ndarray, confirmed: list, out_dir: Path):
    H, W = sym.shape
    rgb  = np.stack([(sym * 255).astype(np.uint8)] * 3, axis=-1)
    img  = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    for c in confirmed:
        r0, c0_px = c["row"], c["col"]
        draw.rectangle([c0_px, r0, c0_px + c["scale_w"], r0 + c["scale_h"]],
                       outline=(0, 255, 0), width=2)
        wr0, wc0, wr1, wc1 = c["dot_win"]
        draw.rectangle([wc0, wr0, wc1, wr1], outline=(0, 220, 220), width=1)
    path = out_dir / "confirmed_map.png"
    img.save(str(path))
    print(f"  [single] Confirmed map   -> {path}")


def save_single_confirmed_crops(sym: np.ndarray, confirmed: list, out_dir: Path):
    sorted_conf = sorted(confirmed, key=lambda x: x["ncc_score"])
    H, W  = sym.shape
    cell, pad, cols = 56, 8, 20
    n     = min(len(sorted_conf), 400)
    rows  = math.ceil(n / cols) if n > 0 else 1
    sheet = Image.new("RGB", (cell * cols, cell * rows), (15, 15, 15))
    draw  = ImageDraw.Draw(sheet)
    for i, c in enumerate(sorted_conf[:n]):
        r0, c0_px = c["row"], c["col"]
        tw, th    = c["scale_w"], c["scale_h"]
        cr0, cc0  = max(0, r0 - pad), max(0, c0_px - pad)
        cr1, cc1  = min(H, r0 + th + pad), min(W, c0_px + tw + pad)
        crop      = sym[cr0:cr1, cc0:cc1]
        ch, cw    = crop.shape
        if ch == 0 or cw == 0:
            continue
        scale = min((cell - 4) / ch, (cell - 4) / cw, 4.0)
        nw, nh = max(1, int(cw * scale)), max(1, int(ch * scale))
        patch  = Image.fromarray((crop * 255).astype(np.uint8)).resize(
            (nw, nh), Image.NEAREST).convert("RGB")
        col_i, row_i = i % cols, i // cols
        sheet.paste(patch, (col_i * cell + (cell - nw) // 2,
                             row_i * cell + (cell - nh) // 2))
        draw.rectangle([col_i * cell, row_i * cell,
                        (col_i + 1) * cell - 1, (row_i + 1) * cell - 1],
                       outline=(0, 180, 60), width=1)
    path = out_dir / "confirmed_crops.png"
    sheet.save(str(path))
    print(f"  [single] Confirmed crops -> {path}  ({n} shown)")


def save_single_dot_samples(dot_bin: np.ndarray, confirmed: list, out_dir: Path):
    cell, cols = 48, 10
    samples = confirmed[:60]
    n_rows  = math.ceil(len(samples) / cols) if samples else 1
    sheet   = Image.new("RGB", (cell * cols, cell * n_rows + 20), (20, 20, 20))
    draw    = ImageDraw.Draw(sheet)
    draw.text((4, 2), "Dot search windows  GREEN=found  RED=missing", fill=(200, 200, 200))
    for i, c in enumerate(samples):
        wr0, wc0, wr1, wc1 = c["dot_win"]
        win = dot_bin[wr0:wr1, wc0:wc1]
        wh, ww = win.shape
        if wh == 0 or ww == 0:
            continue
        scale = min((cell - 4) / wh, (cell - 4) / ww, 6.0)
        nw, nh = max(1, int(ww * scale)), max(1, int(wh * scale))
        patch  = Image.fromarray((win * 255).astype(np.uint8)).resize(
            (nw, nh), Image.NEAREST).convert("RGB")
        col_i, row_i = i % cols, i // cols
        ox = col_i * cell + (cell - nw) // 2
        oy = row_i * cell + (cell - nh) // 2 + 20
        sheet.paste(patch, (ox, oy))
        colour = (0, 220, 80) if c["dot_found"] else (220, 60, 60)
        draw.rectangle([col_i * cell, row_i * cell + 20,
                        (col_i + 1) * cell - 1, (row_i + 1) * cell + 19],
                       outline=colour, width=2)
    path = out_dir / "dot_search_samples.png"
    sheet.save(str(path))
    print(f"  [single] Dot samples     -> {path}")


def save_single_raw_peaks(sym: np.ndarray, raw_peaks: list, out_dir: Path):
    sorted_peaks = sorted(raw_peaks, key=lambda x: -x["ncc_score"])
    H, W  = sym.shape
    cell, pad, cols = 80, 6, 12
    n     = min(len(sorted_peaks), 200)
    rows  = math.ceil(n / cols) if n > 0 else 1
    sheet = Image.new("RGB", (cell * cols, cell * rows + 20), (15, 15, 15))
    draw  = ImageDraw.Draw(sheet)
    draw.text((4, 2), "RAW NCC peaks (single, no filter) sorted best->worst",
              fill=(200, 200, 200))
    for i, c in enumerate(sorted_peaks[:n]):
        r0, c0_px = c["row"], c["col"]
        tw, th    = c["scale_w"], c["scale_h"]
        cr0, cc0  = max(0, r0 - pad), max(0, c0_px - pad)
        cr1, cc1  = min(H, r0 + th + pad), min(W, c0_px + tw + pad)
        crop = sym[cr0:cr1, cc0:cc1]
        ch, cw = crop.shape
        if ch == 0 or cw == 0:
            continue
        scale = min((cell - 4) / ch, (cell - 4) / cw, 4.0)
        nw, nh = max(1, int(cw * scale)), max(1, int(ch * scale))
        patch  = Image.fromarray((crop * 255).astype(np.uint8)).resize(
            (nw, nh), Image.NEAREST).convert("RGB")
        col_i, row_i = i % cols, i // cols
        sheet.paste(patch, (col_i * cell + (cell - nw) // 2,
                             row_i * cell + (cell - nh) // 2 + 20))
        draw.rectangle([col_i * cell, row_i * cell + 20,
                        (col_i + 1) * cell - 1, (row_i + 1) * cell + 19],
                       outline=(180, 180, 60), width=1)
        draw.text((col_i * cell + 2, row_i * cell + 22),
                  f"{c['ncc_score']:.2f} w{c['scale_w']}", fill=(200, 200, 60))
    path = out_dir / "raw_peak_crops.png"
    sheet.save(str(path))
    print(f"  [single] Raw peak crops  -> {path}  ({n} shown)")


# =============================================================================
# DIAGNOSTIC SAVERS — DOUBLE
# =============================================================================

def save_double_confirmed_map(sym: np.ndarray, confirmed: list, out_dir: Path):
    H, W = sym.shape
    rgb  = np.stack([(sym * 255).astype(np.uint8)] * 3, axis=-1)
    img  = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    for c in confirmed:
        r0, c0_px = c["row"], c["col"]
        draw.rectangle([c0_px, r0, c0_px + c["scale_w"], r0 + c["scale_h"]],
                       outline=(0, 80, 255), width=2)
        for win_key, colour in [("dot_left_win", (0, 220, 220)),
                                 ("dot_right_win", (255, 220, 0))]:
            wr0, wc0, wr1, wc1 = c[win_key]
            draw.rectangle([wc0, wr0, wc1, wr1], outline=colour, width=1)
    path = out_dir / "confirmed_map.png"
    img.save(str(path))
    print(f"  [double] Confirmed map   -> {path}")


def save_double_confirmed_crops(sym: np.ndarray, confirmed: list, out_dir: Path):
    sorted_conf = sorted(confirmed, key=lambda x: x["ncc_score"])
    H, W  = sym.shape
    cell, pad, cols = 80, 8, 12
    n     = min(len(sorted_conf), 300)
    rows  = math.ceil(n / cols) if n > 0 else 1
    sheet = Image.new("RGB", (cell * cols, cell * rows), (15, 15, 15))
    draw  = ImageDraw.Draw(sheet)
    for i, c in enumerate(sorted_conf[:n]):
        r0, c0_px = c["row"], c["col"]
        tw, th    = c["scale_w"], c["scale_h"]
        cr0, cc0  = max(0, r0 - pad), max(0, c0_px - pad)
        cr1, cc1  = min(H, r0 + th + pad), min(W, c0_px + tw + pad)
        crop = sym[cr0:cr1, cc0:cc1]
        ch, cw = crop.shape
        if ch == 0 or cw == 0:
            continue
        scale = min((cell - 4) / ch, (cell - 4) / cw, 4.0)
        nw, nh = max(1, int(cw * scale)), max(1, int(ch * scale))
        patch  = Image.fromarray((crop * 255).astype(np.uint8)).resize(
            (nw, nh), Image.NEAREST).convert("RGB")
        col_i, row_i = i % cols, i // cols
        sheet.paste(patch, (col_i * cell + (cell - nw) // 2,
                             row_i * cell + (cell - nh) // 2))
        if c["both_dots_found"]:
            border = (0, 200, 60)
        elif c["dot_left_found"] or c["dot_right_found"]:
            border = (255, 140, 0)
        else:
            border = (200, 40, 40)
        draw.rectangle([col_i * cell, row_i * cell,
                        (col_i + 1) * cell - 1, (row_i + 1) * cell - 1],
                       outline=border, width=2)
    path = out_dir / "confirmed_crops.png"
    sheet.save(str(path))
    print(f"  [double] Confirmed crops -> {path}  ({n} shown)")


def save_double_dot_samples(dot_bin: np.ndarray, confirmed: list, out_dir: Path):
    cell, cols = 48, 10
    samples = confirmed[:60]
    n_rows  = math.ceil(len(samples) / cols) if samples else 1
    W_sheet = cell * cols
    H_sheet = cell * n_rows + 24
    left_sheet  = Image.new("RGB", (W_sheet, H_sheet), (20, 20, 20))
    right_sheet = Image.new("RGB", (W_sheet, H_sheet), (20, 20, 20))
    dl = ImageDraw.Draw(left_sheet)
    dr = ImageDraw.Draw(right_sheet)
    dl.text((4, 2), "LEFT dot windows   GREEN=found  RED=missing",  fill=(200, 200, 200))
    dr.text((4, 2), "RIGHT dot windows  GREEN=found  RED=missing",  fill=(200, 200, 200))
    for i, c in enumerate(samples):
        for sheet, draw_obj, win_key, found_key in [
            (left_sheet,  dl, "dot_left_win",  "dot_left_found"),
            (right_sheet, dr, "dot_right_win", "dot_right_found"),
        ]:
            wr0, wc0, wr1, wc1 = c[win_key]
            win = dot_bin[wr0:wr1, wc0:wc1]
            wh, ww = win.shape
            if wh == 0 or ww == 0:
                continue
            scale = min((cell - 4) / wh, (cell - 4) / ww, 6.0)
            nw, nh = max(1, int(ww * scale)), max(1, int(wh * scale))
            patch  = Image.fromarray((win * 255).astype(np.uint8)).resize(
                (nw, nh), Image.NEAREST).convert("RGB")
            col_i, row_i = i % cols, i // cols
            sheet.paste(patch, (col_i * cell + (cell - nw) // 2,
                                  row_i * cell + (cell - nh) // 2 + 24))
            colour = (0, 220, 80) if c[found_key] else (220, 60, 60)
            draw_obj.rectangle([col_i * cell, row_i * cell + 24,
                                 (col_i + 1) * cell - 1, (row_i + 1) * cell + 23],
                                outline=colour, width=2)
    combined = Image.new("RGB", (W_sheet * 2 + 8, H_sheet), (10, 10, 10))
    combined.paste(left_sheet,  (0, 0))
    combined.paste(right_sheet, (W_sheet + 8, 0))
    path = out_dir / "dot_samples.png"
    combined.save(str(path))
    print(f"  [double] Dot samples     -> {path}")


def save_double_raw_peaks(sym: np.ndarray, raw_peaks: list, out_dir: Path):
    sorted_peaks = sorted(raw_peaks, key=lambda x: -x["ncc_score"])
    H, W  = sym.shape
    cell, pad, cols = 80, 6, 12
    n     = min(len(sorted_peaks), 200)
    rows  = math.ceil(n / cols) if n > 0 else 1
    sheet = Image.new("RGB", (cell * cols, cell * rows + 20), (15, 15, 15))
    draw  = ImageDraw.Draw(sheet)
    draw.text((4, 2), "RAW NCC peaks (double, no filter) sorted best->worst",
              fill=(200, 200, 200))
    for i, c in enumerate(sorted_peaks[:n]):
        r0, c0_px = c["row"], c["col"]
        tw, th    = c["scale_w"], c["scale_h"]
        cr0, cc0  = max(0, r0 - pad), max(0, c0_px - pad)
        cr1, cc1  = min(H, r0 + th + pad), min(W, c0_px + tw + pad)
        crop = sym[cr0:cr1, cc0:cc1]
        ch, cw = crop.shape
        if ch == 0 or cw == 0:
            continue
        scale = min((cell - 4) / ch, (cell - 4) / cw, 4.0)
        nw, nh = max(1, int(cw * scale)), max(1, int(ch * scale))
        patch  = Image.fromarray((crop * 255).astype(np.uint8)).resize(
            (nw, nh), Image.NEAREST).convert("RGB")
        col_i, row_i = i % cols, i // cols
        sheet.paste(patch, (col_i * cell + (cell - nw) // 2,
                             row_i * cell + (cell - nh) // 2 + 20))
        draw.rectangle([col_i * cell, row_i * cell + 20,
                        (col_i + 1) * cell - 1, (row_i + 1) * cell + 19],
                       outline=(180, 180, 60), width=1)
        draw.text((col_i * cell + 2, row_i * cell + 22),
                  f"{c['ncc_score']:.2f} w{c['scale_w']}", fill=(200, 200, 60))
    path = out_dir / "raw_peak_crops.png"
    sheet.save(str(path))
    print(f"  [double] Raw peak crops  -> {path}  ({n} shown)")


# =============================================================================
# COMBINED RGB OVERLAY
# =============================================================================

def save_combined_rgb_overlay(rgb_path: str,
                               single_confirmed: list,
                               double_confirmed: list,
                               bin_h: int, bin_w: int,
                               out_dir: Path):
    """
    Draw single and double obstacle detections onto the original RGB map.

    Single obstacles -> GREEN  bounding box + green centre dot
    Double obstacles -> MAGENTA bounding box + magenta centre dot
                        + cyan tick (left dot) + yellow tick (right dot)

    Coordinates are scaled from binary-map space to RGB-image space.
    """
    rgb_file = Path(rgb_path)
    if not rgb_file.exists():
        print(f"  WARNING: RGB map not found at '{rgb_path}' — overlay skipped.")
        return

    print(f"  Loading RGB map: {rgb_file.name} ...")
    img   = Image.open(str(rgb_file)).convert("RGB")
    draw  = ImageDraw.Draw(img)
    img_w, img_h = img.size
    scale_x = img_w / bin_w
    scale_y = img_h / bin_h

    GREEN   = (0,   220,  60)
    MAGENTA = (255,   0, 255)
    CYAN    = (0,   220, 220)
    YELLOW  = (255, 220,   0)

    # Draw single obstacles
    for c in single_confirmed:
        r0    = int(round(c["row"]            * scale_y))
        c0_px = int(round(c["col"]            * scale_x))
        tw    = max(2, int(round(c["scale_w"] * scale_x)))
        th    = max(2, int(round(c["scale_h"] * scale_y)))
        cx    = int(round((c["col"] + c["scale_w"] / 2) * scale_x))
        cy    = int(round((c["row"] + c["scale_h"] / 2) * scale_y))
        draw.rectangle([c0_px, r0, c0_px + tw, r0 + th],
                       outline=GREEN, width=2)
        dot_r = max(3, tw // 10)
        draw.ellipse([cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r],
                     fill=GREEN)

    # Draw double obstacles
    for c in double_confirmed:
        r0    = int(round(c["row"]            * scale_y))
        c0_px = int(round(c["col"]            * scale_x))
        tw    = max(2, int(round(c["scale_w"] * scale_x)))
        th    = max(2, int(round(c["scale_h"] * scale_y)))
        cx    = int(round((c["col"] + c["scale_w"] / 2) * scale_x))
        cy    = int(round((c["row"] + c["scale_h"] / 2) * scale_y))
        draw.rectangle([c0_px, r0, c0_px + tw, r0 + th],
                       outline=MAGENTA, width=2)
        dot_r = max(3, tw // 10)
        draw.ellipse([cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r],
                     fill=MAGENTA)
        ldc        = int(round((c["col"] + c["scale_w"] * D_DOT_LEFT_CENTRE_FRAC) * scale_x))
        rdc        = int(round((c["col"] + c["scale_w"] * D_DOT_RIGHT_CENTRE_FRAC) * scale_x))
        tick_top   = int(round((c["row"] + c["scale_h"] * 0.82) * scale_y))
        tick_bot   = int(round((c["row"] + c["scale_h"] * 0.98) * scale_y))
        draw.line([ldc, tick_top, ldc, tick_bot], fill=CYAN,   width=2)
        draw.line([rdc, tick_top, rdc, tick_bot], fill=YELLOW, width=2)

    path = out_dir / "rgb_overlay_combined.png"
    img.save(str(path))
    print(f"  Combined RGB overlay -> {path}  "
          f"({len(single_confirmed)} single, {len(double_confirmed)} double)")


# =============================================================================
# CROSS-TYPE NMS
# =============================================================================

def cross_type_nms(single_confirmed: list, double_confirmed: list) -> list:
    """
    Remove single-obstacle detections that overlap strongly with a confirmed
    double-obstacle detection.

    The single-obstacle NCC template can partially match one half of a double
    obstacle symbol (one of its two inverted-V arms). This produces a spurious
    single detection sitting inside a genuine double detection. The cross-type
    NMS pass removes those cases.

    Method:
      For each single detection, compute the centre point.
      If that centre falls inside the bounding box of any double detection
      (expanded by CROSS_NMS_EXPAND_FRAC on each side), suppress the single.
      The double detection is always preferred — it passed stricter checks.

    Returns the filtered single_confirmed list (double list is unchanged).
    """
    CROSS_NMS_EXPAND_FRAC = 0.15   # expand double bbox by 15% each side

    if not double_confirmed:
        return single_confirmed

    kept = []
    for s in single_confirmed:
        sc_r = s["row"] + s["scale_h"] / 2
        sc_c = s["col"] + s["scale_w"] / 2
        suppressed = False
        for d in double_confirmed:
            exp_h = d["scale_h"] * CROSS_NMS_EXPAND_FRAC
            exp_w = d["scale_w"] * CROSS_NMS_EXPAND_FRAC
            if (d["row"] - exp_h <= sc_r <= d["row"] + d["scale_h"] + exp_h and
                    d["col"] - exp_w <= sc_c <= d["col"] + d["scale_w"] + exp_w):
                suppressed = True
                break
        if not suppressed:
            kept.append(s)
    return kept


# =============================================================================
# MAIN
# =============================================================================

def main():
    out_dir    = Path(OUTPUT_DIR)
    single_dir = out_dir / "single"
    double_dir = out_dir / "double"
    out_dir.mkdir(parents=True, exist_ok=True)
    single_dir.mkdir(parents=True, exist_ok=True)
    double_dir.mkdir(parents=True, exist_ok=True)

    # ── Load images (shared between both pipelines) ───────────────────────────
    print(f"Loading symbols binary (NCC): {SYMBOLS_PATH}")
    symbols_bin = np.array(Image.open(SYMBOLS_PATH).convert("L")) > 128
    H, W = symbols_bin.shape
    print(f"  Size: {W} x {H} px    Ink: {symbols_bin.mean()*100:.2f}%")

    print(f"Loading full binary  (dots):  {BINARY_PATH}")
    dot_bin = np.array(Image.open(BINARY_PATH).convert("L")) > 128
    dH, dW  = dot_bin.shape
    print(f"  Size: {dW} x {dH} px    Ink: {dot_bin.mean()*100:.2f}%")

    if (H, W) != (dH, dW):
        print("  WARNING: size mismatch — resizing full binary to match symbols binary ...")
        dot_bin = np.array(
            Image.fromarray((dot_bin * 255).astype(np.uint8)).resize(
                (W, H), Image.NEAREST)
        ) > 128

    # =========================================================================
    # SINGLE-OBSTACLE PIPELINE
    # =========================================================================
    print("\n" + "=" * 60)
    print("SINGLE-OBSTACLE DETECTION (Phase 4a logic)")
    print("=" * 60)

    s_templates = {w: build_single_template(w) for w in S_TEMPLATE_WIDTHS}

    print(f"\nNCC at scales {S_TEMPLATE_WIDTHS} px (threshold={S_NCC_THRESHOLD}) ...")
    s_raw = run_ncc(symbols_bin, s_templates, S_NCC_THRESHOLD,
                    S_NCC_THRESHOLD_PER_SCALE, S_PEAK_MIN_DISTANCE, "single")
    print(f"  Raw peaks: {len(s_raw)}")

    s_after_nms   = nms(s_raw)
    print(f"  After NMS: {len(s_after_nms)}")

    s_after_scale = [c for c in s_after_nms if c["scale_w"] >= S_MIN_SCALE_W]
    print(f"  After min scale ({S_MIN_SCALE_W}px): {len(s_after_scale)}")

    print("\nOpen-base check ...")
    s_after_base  = [c for c in s_after_scale if s_passes_open_base(symbols_bin, c)]
    print(f"  After open-base: {len(s_after_base)}  (rejected {len(s_after_scale)-len(s_after_base)})")

    print("Apex sharpness check ...")
    s_after_apex  = [c for c in s_after_base if s_passes_apex(symbols_bin, c)]
    print(f"  After apex:      {len(s_after_apex)}  (rejected {len(s_after_base)-len(s_after_apex)})")

    print("Arm symmetry check ...")
    s_after_sym   = [c for c in s_after_apex if s_passes_symmetry(symbols_bin, c)]
    print(f"  After symmetry:  {len(s_after_sym)}  (rejected {len(s_after_apex)-len(s_after_sym)})")

    print("Blob isolation check ...")
    s_after_iso   = [c for c in s_after_sym if s_passes_isolation(symbols_bin, c)]
    print(f"  After isolation: {len(s_after_iso)}  (rejected {len(s_after_sym)-len(s_after_iso)})")

    print("Arm continuity check ...")
    s_after_cont  = [c for c in s_after_iso if s_passes_arm_continuity(symbols_bin, c)]
    print(f"  After arm cont.: {len(s_after_cont)}  (rejected {len(s_after_iso)-len(s_after_cont)})")

    print("Dot check (soft metadata only) ...")
    s_confirmed = [s_verify_dot(dot_bin, c) for c in s_after_cont]
    s_dot_found = sum(1 for c in s_confirmed if c["dot_found"])
    print(f"  Dot found in {s_dot_found}/{len(s_confirmed)} confirmed")

    for c in s_confirmed:
        c["centre_row"]   = c["row"] + c["scale_h"] // 2
        c["centre_col"]   = c["col"] + c["scale_w"] // 2
        c["symbol_class"] = "single_obstacle"

    print("\nSaving single-obstacle diagnostics ...")
    save_single_raw_peaks(symbols_bin, s_raw, single_dir)
    save_single_confirmed_map(symbols_bin, s_confirmed, single_dir)
    save_single_confirmed_crops(symbols_bin, s_confirmed, single_dir)
    save_single_dot_samples(dot_bin, s_confirmed, single_dir)

    print("\n=== SINGLE RESULT ===")
    print(f"  NCC raw peaks   : {len(s_raw)}")
    print(f"  After NMS       : {len(s_after_nms)}")
    print(f"  After min scale : {len(s_after_scale)}")
    print(f"  After open-base : {len(s_after_base)}")
    print(f"  After apex      : {len(s_after_apex)}")
    print(f"  After symmetry  : {len(s_after_sym)}")
    print(f"  After isolation : {len(s_after_iso)}")
    print(f"  After arm cont. : {len(s_after_cont)}")
    print(f"  Confirmed       : {len(s_confirmed)}")
    print(f"  Dot found in    : {s_dot_found}/{len(s_confirmed)}")

    # =========================================================================
    # DOUBLE-OBSTACLE PIPELINE
    # =========================================================================
    print("\n" + "=" * 60)
    print("DOUBLE-OBSTACLE DETECTION (Phase 4c logic)")
    print("=" * 60)

    d_templates = {w: build_double_template(w) for w in D_TEMPLATE_WIDTHS}

    print(f"\nNCC at scales {D_TEMPLATE_WIDTHS} px (threshold={D_NCC_THRESHOLD}) ...")
    d_raw = run_ncc(symbols_bin, d_templates, D_NCC_THRESHOLD,
                    D_NCC_THRESHOLD_PER_SCALE, D_PEAK_MIN_DISTANCE, "double")
    print(f"  Raw peaks: {len(d_raw)}")

    d_after_nms   = nms(d_raw)
    print(f"  After NMS: {len(d_after_nms)}")

    d_after_scale = [c for c in d_after_nms if c["scale_w"] >= D_MIN_SCALE_W]
    print(f"  After min scale ({D_MIN_SCALE_W}px): {len(d_after_scale)}")

    print("\nOpen-base check (arms at outer edges, centre open) ...")
    d_after_base  = [c for c in d_after_scale if d_passes_open_base(symbols_bin, c)]
    print(f"  After open-base:   {len(d_after_base)}  (rejected {len(d_after_scale)-len(d_after_base)})")

    print("Dual-apex check ...")
    d_after_apex  = [c for c in d_after_base if d_passes_dual_apex(symbols_bin, c)]
    print(f"  After dual-apex:   {len(d_after_apex)}  (rejected {len(d_after_base)-len(d_after_apex)})")

    print("Valley check (3-group row in crossing region) ...")
    d_after_valley = [c for c in d_after_apex if d_passes_valley(symbols_bin, c)]
    print(f"  After valley:      {len(d_after_valley)}  (rejected {len(d_after_apex)-len(d_after_valley)})")

    print("Symmetry check ...")
    d_after_sym   = [c for c in d_after_valley if d_passes_symmetry(symbols_bin, c)]
    print(f"  After symmetry:    {len(d_after_sym)}  (rejected {len(d_after_valley)-len(d_after_sym)})")

    print("Isolation check ...")
    d_after_iso   = [c for c in d_after_sym if d_passes_isolation(symbols_bin, c)]
    print(f"  After isolation:   {len(d_after_iso)}  (rejected {len(d_after_sym)-len(d_after_iso)})")

    print("Valley density check ...")
    d_after_vden  = [c for c in d_after_iso if d_passes_valley_density(symbols_bin, c)]
    print(f"  After valley den.: {len(d_after_vden)}  (rejected {len(d_after_iso)-len(d_after_vden)})")

    print("Two-dot verification (soft metadata only) ...")
    d_confirmed   = [d_verify_two_dots(dot_bin, c) for c in d_after_vden]
    d_both_found  = sum(1 for c in d_confirmed if c["both_dots_found"])
    d_left_found  = sum(1 for c in d_confirmed if c["dot_left_found"])
    d_right_found = sum(1 for c in d_confirmed if c["dot_right_found"])
    print(f"  Both dots : {d_both_found}/{len(d_confirmed)}")
    print(f"  Left only : {d_left_found - d_both_found}/{len(d_confirmed)}")
    print(f"  Right only: {d_right_found - d_both_found}/{len(d_confirmed)}")

    for c in d_confirmed:
        c["centre_row"]   = c["row"] + c["scale_h"] // 2
        c["centre_col"]   = c["col"] + c["scale_w"] // 2
        c["symbol_class"] = "double_obstacle"

    print("\nSaving double-obstacle diagnostics ...")
    save_double_raw_peaks(symbols_bin, d_raw, double_dir)
    save_double_confirmed_map(symbols_bin, d_confirmed, double_dir)
    save_double_confirmed_crops(symbols_bin, d_confirmed, double_dir)
    save_double_dot_samples(dot_bin, d_confirmed, double_dir)

    print("\n=== DOUBLE RESULT ===")
    print(f"  NCC raw peaks   : {len(d_raw)}")
    print(f"  After NMS       : {len(d_after_nms)}")
    print(f"  After min scale : {len(d_after_scale)}")
    print(f"  After open-base : {len(d_after_base)}")
    print(f"  After dual-apex : {len(d_after_apex)}")
    print(f"  After valley    : {len(d_after_valley)}")
    print(f"  After symmetry  : {len(d_after_sym)}")
    print(f"  After isolation : {len(d_after_iso)}")
    print(f"  After valley den: {len(d_after_vden)}")
    print(f"  Confirmed       : {len(d_confirmed)}")
    print(f"  Both dots found : {d_both_found}/{len(d_confirmed)}")

    # =========================================================================
    # COMBINED OUTPUTS
    # =========================================================================
    print("\n" + "=" * 60)
    print("COMBINED OUTPUTS")
    print("=" * 60)

    # Cross-type NMS: remove single detections that overlap a confirmed double.
    # The double pipeline has stricter checks, so it wins any conflict.
    s_before_cross = len(s_confirmed)
    s_confirmed    = cross_type_nms(s_confirmed, d_confirmed)
    print(f"  Cross-type NMS: removed {s_before_cross - len(s_confirmed)} single "
          f"detection(s) overlapping a double obstacle")

    combined = {
        "total_single_confirmed": len(s_confirmed),
        "total_double_confirmed": len(d_confirmed),
        "single_ncc_threshold":  S_NCC_THRESHOLD,
        "double_ncc_threshold":  D_NCC_THRESHOLD,
        "single_template_widths": S_TEMPLATE_WIDTHS,
        "double_template_widths": D_TEMPLATE_WIDTHS,
        "single_confirmed": s_confirmed,
        "double_confirmed": d_confirmed,
    }
    json_path = out_dir / "detections_combined.json"
    with open(json_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"  detections_combined.json -> {json_path}")

    save_combined_rgb_overlay(RGB_PATH, s_confirmed, d_confirmed, H, W, out_dir)

    print("\n=== COMBINED SUMMARY ===")
    print(f"  Single obstacles confirmed : {len(s_confirmed)}")
    print(f"  Double obstacles confirmed : {len(d_confirmed)}")
    print(f"  Total                      : {len(s_confirmed) + len(d_confirmed)}")
    print(f"\nOutputs in: {out_dir.resolve()}")
    print("\nTuning guide:")
    print("  0 single raw peaks    : lower S_NCC_THRESHOLD or check SYMBOLS_PATH")
    print("  0 double raw peaks    : lower D_NCC_THRESHOLD or check SYMBOLS_PATH")
    print("  Single FP (letters)   : raise S_NCC_THRESHOLD or tighten S_BASE_INK_LIMIT")
    print("  Arm cont. too strict  : lower S_ARM_MIN_COVERAGE (currently %.2f)" % S_ARM_MIN_COVERAGE)
    print("  Double FP             : raise D_NCC_THRESHOLD or tighten D_APEX_MIN_INK")
    print("  Valley density strict : lower D_VALLEY_DENSITY_MIN (currently %.3f)" % D_VALLEY_DENSITY_MIN)
    print("  Single inside double  : cross-type NMS handles this automatically")
    print("  Inspect raw_peak_crops.png in each sub-folder first when count = 0")


if __name__ == "__main__":
    main()