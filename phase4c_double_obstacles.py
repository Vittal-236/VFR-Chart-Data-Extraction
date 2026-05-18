"""
Phase 4c — Double Obstacle Detection on Binary Map
VFR Chart Extraction Pipeline (FAA Base Model)

Detects the FAA double obstacle symbol: two open inverted-V triangles
sharing a common valley crossing in the lower half, forming an M-like
shape, with one dot below each triangle's outer base.

VERIFIED GEOMETRY (measured from Double_obstacle.png, 224x196 px reference)
----------------------------------------------------------------------------
  Ink bounding box: 167px wide x 129px tall  (H/W ratio = 0.772)

  Top row (row 0 of ink bbox):
    - TWO separate apex clusters: left at ~34% of width, right at ~65%
    - There is a GAP between them — the symbol starts with two apexes, not one

  Row 0-37% of height  (upper region):
    - 2 ink groups per row: left arm and right arm diverging outward

  Row 37-74% of height  (crossing region / M-valley):
    - 3 ink groups per row: outer-left arm, inner crossing arms, outer-right arm
    - The inner arms cross at horizontal centre — creates a 3-group pattern

  Row 74-100% of height  (lower region):
    - 2 ink groups again: outer arms heading to bottom corners
    - Dots appear at ~86% height, between the outer arms and the centre

  Bottom rows:
    - Outer arms land at the LEFT EDGE (0-12% of width) and RIGHT EDGE (88-100%)
    - The CENTRE 76% of the bottom is OPEN (no base edge)
    - Key difference from Phase 4a: open zone is CENTRE, arm ink is at OUTER EDGES

OPEN-BASE LOGIC (corrected from Phase 4a)
-----------------------------------------
  Phase 4a checks the CENTRE-THIRD for ink absence (correct for single ∧).
  For double obstacle the arms land at the OUTER edges — the centre IS open.
  Correct check: ink must be present in LEFT 15% and RIGHT 15% of bottom
  rows (arm landing zones), but absent in the CENTRE 50%.

DUAL-APEX LOGIC (corrected)
----------------------------
  Top rows must contain TWO separate ink clusters: one in left zone
  and one in right zone, with a GAP at horizontal centre.

HOW TO USE
----------
1. Set the three paths in CONFIG below.
2. Hit Run — no CLI needed.
3. Outputs saved to OUTPUT_DIR:
     detections_double.json   all confirmed detections
     confirmed_map.png        map annotated in blue
     confirmed_crops.png      contact sheet of cropped patches
     dot_samples.png          dot search windows for tuning
     raw_peak_crops.png       crops of RAW NCC peaks (before any filter)
                              — inspect this first when detections = 0

TILING RULE
-----------
The full map is NEVER processed at once. NCC runs on 2048x2048 tiles
with 100px overlap. Peak memory stays under 64 MB per tile.
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
# CONFIG — edit these paths, then hit Run
# =============================================================================

SYMBOLS_PATH = r"outputs/phase2_layer_segmentation/phase2c_symbols_only_binary/washington_symbols_only.png"
BINARY_PATH  = r"outputs/phase1_preprocessing/Washington_binary.png"
RGB_PATH     = r"outputs/phase1_preprocessing/Washington_rgb_150dpi.png"   # original colour map for overlay
OUTPUT_DIR   = r"outputs/phase4_symbol_detection/phase4c_double_obstacles"

# =============================================================================
# TUNING PARAMETERS
# =============================================================================

NCC_THRESHOLD = 0.50

NCC_THRESHOLD_PER_SCALE = {
    40: 0.52,
    48: 0.50,
    56: 0.50,
    64: 0.50,
}

PEAK_MIN_DISTANCE = 18

# Template widths in pixels on the binary map.
# Single obstacle: ~24-32px wide. Double is 167/118 = 1.41x wider.
# Derived: 24*1.41=34, 28*1.41=39, 32*1.41=45 -> rounded to 40, 48, 56, 64.
TEMPLATE_WIDTHS = [40, 48, 56, 64]
MIN_SCALE_W     = 40

# --- Open-base check (corrected for double obstacle) ---
# Outer arms land at the LEFT and RIGHT edges. CENTRE is open.
BASE_CHECK_FRAC     = 0.12   # inspect bottom 12% of bounding box rows
BASE_ARM_ZONE       = 0.15   # outer arm zone = left 15% and right 15% of width
BASE_ARM_MIN_INK    = 0.05   # arm zone must have at least this ink fraction
BASE_CENTRE_ZONE    = 0.25   # centre zone spans from 25% to 75% of width
BASE_CENTRE_MAX_INK = 0.08   # centre zone must be mostly empty

# --- Dual-apex check ---
# At top rows: left cluster in left zone, right cluster in right zone, gap at centre.
APEX_CHECK_FRAC   = 0.15
APEX_LEFT_ZONE    = 0.40   # check left 40% for left apex
APEX_RIGHT_ZONE   = 0.40   # check right 40% for right apex
APEX_MIN_INK      = 0.05   # each zone must have at least this ink density
APEX_GAP_ZONE_LO  = 0.40   # centre gap zone: 40%-60% of width
APEX_GAP_ZONE_HI  = 0.60
APEX_MAX_GAP_INK  = 0.10   # centre must be relatively empty at top

# --- Valley (3-group) check ---
# In the crossing region, at least one row must have 3+ ink groups.
VALLEY_ROW_LO   = 0.37
VALLEY_ROW_HI   = 0.74
VALLEY_GAP_PX   = 4      # minimum column gap between groups
VALLEY_N_GROUPS = 3

# --- Symmetry check ---
SYMMETRY_MIN = 0.40

# --- Isolation check ---
ISOLATION_MAX_W_FACTOR = 1.8

# --- Two-dot verification ---
# Dot positions measured from Double_obstacle.png pixel analysis:
#   Dots are INSIDE the bounding box at 86%-100% of template height.
#   Left dot centre:  34.4% of template width (inner left blob)
#   Right dot centre: 65.3% of template width (inner right blob)
#   Arm ends at outer 0-16% and 84-100% of width excluded by window placement.
#   At 48px template width each dot is ~5px wide x ~1-2px tall -> area ~5-10px.
#   Dots often erased by Phase 2c. Soft gate only.
DOT_BELOW_FRAC        = 0.86   # search starts at 86% of template height (dot row)
DOT_SEARCH_W_FRAC     = 0.12   # each dot window = 12% of template width
DOT_LEFT_CENTRE_FRAC  = 0.344  # left dot at 34.4% of template width
DOT_RIGHT_CENTRE_FRAC = 0.653  # right dot at 65.3% of template width
DOT_MIN_AREA          = 2
DOT_MAX_AREA          = 20
DOT_MAX_ECCENTRICITY  = 0.85

LEGEND_STRIP_WIDTH = 140

TILE_SIZE    = 2048
TILE_OVERLAP = 100


# =============================================================================
# STEP 1 — Build the double-obstacle template
# =============================================================================

def build_double_template(width_px: int) -> np.ndarray:
    """
    Build a binary float32 template of the FAA double obstacle symbol.

    Verified geometry (pixel analysis of Double_obstacle.png):
      - TWO separate apexes at top: left at ~34% of width, right at ~65%
      - Left outer arm:  left apex  -> bottom-left corner  (0% width)
      - Left inner arm:  left apex  -> bottom-centre       (50% width)
      - Right inner arm: right apex -> bottom-centre       (50% width)
      - Right outer arm: right apex -> bottom-right corner (100% width)
      - NO base edges. H/W ratio = 0.772.
    """
    height_px = int(round(width_px * 0.772))
    tmpl = np.zeros((height_px, width_px), dtype=np.float32)

    stroke = 2 if width_px >= 48 else 1

    apex_L_r, apex_L_c = 0, width_px * 0.344
    apex_R_r, apex_R_c = 0, width_px * 0.650

    base_L_r,   base_L_c   = height_px - 1, 0.0
    base_R_r,   base_R_c   = height_px - 1, float(width_px - 1)
    base_mid_r, base_mid_c = height_px - 1, (width_px - 1) * 0.5

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

    draw_arm(apex_L_r, apex_L_c, base_L_r,   base_L_c)
    draw_arm(apex_L_r, apex_L_c, base_mid_r, base_mid_c)
    draw_arm(apex_R_r, apex_R_c, base_mid_r, base_mid_c)
    draw_arm(apex_R_r, apex_R_c, base_R_r,   base_R_c)

    return tmpl


# =============================================================================
# STEP 2 — Multi-scale tiled NCC
# =============================================================================

def run_ncc(binary: np.ndarray, template_widths: list,
            ncc_threshold: float, peak_min_dist: int,
            legend_strip: int) -> list:
    """
    Tiled multi-scale NCC. 2048x2048 tiles with 100px overlap.
    Never processes the full map image at once.
    """
    H, W      = binary.shape
    all_peaks = []

    templates = {w: build_double_template(w) for w in template_widths}

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

            print(f"  Tile {tile_idx}/{n_tiles}  "
                  f"map[{inner_r0}:{inner_r1}, {inner_c0}:{inner_c1}]  "
                  f"padded {tile.shape[1]}x{tile.shape[0]}px", end="")

            tile_peaks_found = 0

            for w, tmpl in templates.items():
                th = tmpl.shape[0]
                if tile.shape[0] < th or tile.shape[1] < w:
                    continue

                corr = match_template(tile, tmpl, pad_input=False)

                peaks = peak_local_max(
                    corr,
                    min_distance=peak_min_dist,
                    threshold_abs=ncc_threshold,
                )

                for pr, pc in peaks:
                    map_r = pr + pad_r0
                    map_c = pc + pad_c0

                    scale_thresh = NCC_THRESHOLD_PER_SCALE.get(w, ncc_threshold)
                    if corr[pr, pc] < scale_thresh:
                        continue

                    if not (inner_r0 <= map_r < inner_r1 and
                            inner_c0 <= map_c < inner_c1):
                        continue

                    if map_c < legend_strip:
                        continue

                    all_peaks.append({
                        "row":       int(map_r),
                        "col":       int(map_c),
                        "scale_w":   w,
                        "scale_h":   th,
                        "ncc_score": float(corr[pr, pc]),
                    })
                    tile_peaks_found += 1

            print(f"  -> {tile_peaks_found} peaks")

    return all_peaks


# =============================================================================
# STEP 3 — Non-maximum suppression
# =============================================================================

def nms(candidates: list) -> list:
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
            dist = math.hypot(cy - dy, cx - dx)
            avg_w = (c["scale_w"] + d["scale_w"]) / 2
            if dist < avg_w * 0.6:
                suppressed.add(j)

    return kept


# =============================================================================
# STEP 4 — Structural checks
# =============================================================================

def passes_open_base(symbols_bin: np.ndarray, candidate: dict) -> bool:
    """
    Confirm open-base pattern for double obstacle.

    Outer arms land at LEFT EDGE and RIGHT EDGE. CENTRE is open.
    Checks:
      a) Left arm zone (leftmost 15%) has ink
      b) Right arm zone (rightmost 15%) has ink
      c) Centre zone (middle 50%) is mostly empty
    All three must pass.
    """
    r0, c0 = candidate["row"], candidate["col"]
    tw, th = candidate["scale_w"], candidate["scale_h"]
    H, W   = symbols_bin.shape

    r1 = min(H, r0 + th)
    c1 = min(W, c0 + tw)
    patch = symbols_bin[r0:r1, c0:c1]
    ph, pw = patch.shape
    if ph == 0 or pw == 0:
        return True

    bot_rows = max(1, int(round(ph * BASE_CHECK_FRAC)))
    bot      = patch[ph - bot_rows:, :]

    arm_w   = max(1, int(round(pw * BASE_ARM_ZONE)))
    left_z  = bot[:, :arm_w]
    right_z = bot[:, pw - arm_w:]

    cz_lo    = int(round(pw * BASE_CENTRE_ZONE))
    cz_hi    = pw - int(round(pw * BASE_CENTRE_ZONE))
    centre_z = bot[:, cz_lo:cz_hi]

    if left_z.size == 0 or right_z.size == 0:
        return True

    left_ink   = left_z.sum()   / left_z.size
    right_ink  = right_z.sum()  / right_z.size
    centre_ink = centre_z.sum() / centre_z.size if centre_z.size > 0 else 0.0

    return (left_ink  >= BASE_ARM_MIN_INK and
            right_ink >= BASE_ARM_MIN_INK and
            centre_ink <= BASE_CENTRE_MAX_INK)


def passes_dual_apex(symbols_bin: np.ndarray, candidate: dict) -> bool:
    """
    Confirm two separate ink clusters at the top (the two apexes).

    Left cluster in left 40% of width, right cluster in right 40%,
    with a gap at horizontal centre (40%-60%).
    """
    r0, c0 = candidate["row"], candidate["col"]
    tw, th = candidate["scale_w"], candidate["scale_h"]
    H, W   = symbols_bin.shape

    r1 = min(H, r0 + th)
    c1 = min(W, c0 + tw)
    patch = symbols_bin[r0:r1, c0:c1]
    ph, pw = patch.shape
    if ph == 0 or pw == 0:
        return True

    apex_rows = max(1, int(round(ph * APEX_CHECK_FRAC)))
    top       = patch[:apex_rows, :]

    left_w  = max(1, int(round(pw * APEX_LEFT_ZONE)))
    right_w = max(1, int(round(pw * APEX_RIGHT_ZONE)))
    gap_lo  = int(round(pw * APEX_GAP_ZONE_LO))
    gap_hi  = int(round(pw * APEX_GAP_ZONE_HI))

    left_zone  = top[:, :left_w]
    right_zone = top[:, pw - right_w:]
    gap_zone   = top[:, gap_lo:gap_hi]

    if left_zone.size == 0 or right_zone.size == 0:
        return True

    left_ink  = left_zone.sum()  / left_zone.size
    right_ink = right_zone.sum() / right_zone.size
    gap_ink   = gap_zone.sum()   / gap_zone.size if gap_zone.size > 0 else 0.0

    return (left_ink  >= APEX_MIN_INK and
            right_ink >= APEX_MIN_INK and
            gap_ink   <= APEX_MAX_GAP_INK)


def passes_valley_check(symbols_bin: np.ndarray, candidate: dict) -> bool:
    """
    Confirm the M-valley: at least one row in the crossing region has
    3 or more separate ink groups.

    This is the unique topological fingerprint of the double obstacle.
    A single triangle never has 3 groups. A letter M/W may have 2 at most.
    """
    r0, c0 = candidate["row"], candidate["col"]
    tw, th = candidate["scale_w"], candidate["scale_h"]
    H, W   = symbols_bin.shape

    r1 = min(H, r0 + th)
    c1 = min(W, c0 + tw)
    patch = symbols_bin[r0:r1, c0:c1]
    ph, pw = patch.shape
    if ph == 0 or pw == 0:
        return True

    row_lo = int(round(ph * VALLEY_ROW_LO))
    row_hi = int(round(ph * VALLEY_ROW_HI))
    if row_hi <= row_lo:
        return True

    for r in range(row_lo, row_hi):
        row_ink  = np.where(patch[r, :])[0]
        if len(row_ink) == 0:
            continue
        gaps     = np.where(np.diff(row_ink) > VALLEY_GAP_PX)[0]
        n_groups = len(gaps) + 1
        if n_groups >= VALLEY_N_GROUPS:
            return True

    return False


def passes_symmetry(symbols_bin: np.ndarray, candidate: dict) -> bool:
    r0, c0 = candidate["row"], candidate["col"]
    tw, th = candidate["scale_w"], candidate["scale_h"]
    H, W   = symbols_bin.shape

    r1 = min(H, r0 + th)
    c1 = min(W, c0 + tw)
    patch = symbols_bin[r0:r1, c0:c1]
    ph, pw = patch.shape
    if ph == 0 or pw < 2:
        return True

    mid       = pw // 2
    left_ink  = patch[:, :mid].sum()
    right_ink = patch[:, mid:].sum()

    if left_ink == 0 and right_ink == 0:
        return False

    balance = min(left_ink, right_ink) / max(left_ink, right_ink)
    return balance >= SYMMETRY_MIN


def passes_isolation(symbols_bin: np.ndarray, candidate: dict) -> bool:
    r0, c0 = candidate["row"], candidate["col"]
    tw, th = candidate["scale_w"], candidate["scale_h"]
    H, W   = symbols_bin.shape

    cr = min(H - 1, r0 + th // 2)
    cc = min(W - 1, c0 + tw // 2)

    if not symbols_bin[cr, cc]:
        search_r = slice(max(0, cr - 4), min(H, cr + 5))
        search_c = slice(max(0, cc - 4), min(W, cc + 5))
        window = symbols_bin[search_r, search_c]
        if window.sum() == 0:
            return True
        ink_positions = np.argwhere(window)
        nearest = ink_positions[0]
        cr = max(0, cr - 4) + nearest[0]
        cc = max(0, cc - 4) + nearest[1]

    pad  = tw * 3
    lr0  = max(0, r0 - pad)
    lc0  = max(0, c0 - pad)
    lr1  = min(H, r0 + th + pad)
    lc1  = min(W, c0 + tw + pad)

    local = symbols_bin[lr0:lr1, lc0:lc1]
    lbl   = label(local, connectivity=2)

    local_cr = max(0, min(lbl.shape[0] - 1, cr - lr0))
    local_cc = max(0, min(lbl.shape[1] - 1, cc - lc0))

    comp_id = lbl[local_cr, local_cc]
    if comp_id == 0:
        return True

    comp_mask = lbl == comp_id
    comp_cols = np.where(comp_mask.any(axis=0))[0]
    if len(comp_cols) == 0:
        return True

    comp_width = comp_cols[-1] - comp_cols[0] + 1
    return comp_width <= ISOLATION_MAX_W_FACTOR * tw


# =============================================================================
# STEP 5 — Two-dot verification
# =============================================================================

def verify_two_dots(dot_binary: np.ndarray, candidate: dict) -> dict:
    """
    Search for two dots INSIDE the bounding box in the bottom region.

    Measured from Double_obstacle.png pixel analysis:
      - Dots occupy rows 86%-100% of template height, INSIDE the bounding box.
        Searching below the bounding box is wrong — it lands on map text.
      - Left dot centre:  34.4% of template width (inner left blob)
      - Right dot centre: 65.3% of template width (inner right blob)
      - Arm ends occupy outer 0-16% and 84-100% of width at dot rows.
        Search windows are in the safe inner zone, clear of arm ends.
      - At 48px template width each dot is ~5px wide x ~1px tall.
        Dots often erased by Phase 2c. Soft gate — detections kept even
        when dots are not found.
    """
    r0, c0 = candidate["row"], candidate["col"]
    tw, th = candidate["scale_w"], candidate["scale_h"]
    H, W   = dot_binary.shape

    # Dot row band: DOT_BELOW_FRAC * th from top, down to template bottom
    win_r0   = max(0, r0 + int(round(th * DOT_BELOW_FRAC)))
    win_r1   = min(H, r0 + th)   # do not exceed template bottom
    search_w = max(4, int(round(tw * DOT_SEARCH_W_FRAC)))

    left_c  = int(round(c0 + tw * DOT_LEFT_CENTRE_FRAC))
    right_c = int(round(c0 + tw * DOT_RIGHT_CENTRE_FRAC))

    def search_dot(centre_c):
        wc0 = max(0, centre_c - search_w // 2)
        wc1 = min(W, centre_c + search_w // 2)
        win = (win_r0, wc0, win_r1, wc1)
        found = False
        area  = 0
        if win_r1 > win_r0 and wc1 > wc0:
            window = dot_binary[win_r0:win_r1, wc0:wc1]
            if window.sum() > 0:
                labelled = label(window, connectivity=2)
                for p in regionprops(labelled):
                    if (DOT_MIN_AREA <= p.area <= DOT_MAX_AREA and
                            p.eccentricity <= DOT_MAX_ECCENTRICITY):
                        found = True
                        area  = int(p.area)
                        break
        return found, area, win

    lf, la, lw = search_dot(left_c)
    rf, ra, rw = search_dot(right_c)

    candidate["dot_left_found"]  = lf
    candidate["dot_left_area"]   = la
    candidate["dot_left_win"]    = lw
    candidate["dot_right_found"] = rf
    candidate["dot_right_area"]  = ra
    candidate["dot_right_win"]   = rw
    candidate["both_dots_found"] = lf and rf

    return candidate


# =============================================================================
# STEP 6 — Debug: raw peak crops
# =============================================================================

def save_raw_peak_crops(symbols_bin: np.ndarray, raw_peaks: list,
                        out_dir: Path, max_show: int = 120):
    """
    Contact sheet of NCC peaks before any structural filtering.
    Inspect this first when confirmed=0.
    Sorted by NCC score descending (best matches first).
    NCC score and scale width printed on each cell.
    """
    sorted_peaks = sorted(raw_peaks, key=lambda x: -x["ncc_score"])
    H, W = symbols_bin.shape

    cell = 80
    pad  = 6
    cols = 12
    n    = min(len(sorted_peaks), max_show)
    rows = math.ceil(n / cols) if n > 0 else 1

    sheet = Image.new("RGB", (cell * cols, cell * rows + 20), (15, 15, 15))
    draw  = ImageDraw.Draw(sheet)
    draw.text((4, 2), "RAW NCC peaks (no filter)  sorted best->worst",
              fill=(200, 200, 200))

    for i, c in enumerate(sorted_peaks[:n]):
        r0, c0_px = c["row"], c["col"]
        tw, th    = c["scale_w"], c["scale_h"]

        crop_r0 = max(0, r0 - pad)
        crop_c0 = max(0, c0_px - pad)
        crop_r1 = min(H, r0 + th + pad)
        crop_c1 = min(W, c0_px + tw + pad)

        crop = symbols_bin[crop_r0:crop_r1, crop_c0:crop_c1]
        ch, cw = crop.shape
        if ch == 0 or cw == 0:
            continue

        scale = min((cell - 4) / ch, (cell - 4) / cw, 4.0)
        nw = max(1, int(cw * scale))
        nh = max(1, int(ch * scale))

        patch = Image.fromarray((crop * 255).astype(np.uint8)).resize(
            (nw, nh), Image.NEAREST).convert("RGB")

        col_i = i % cols
        row_i = i // cols
        ox    = col_i * cell + (cell - nw) // 2
        oy    = row_i * cell + (cell - nh) // 2 + 20
        sheet.paste(patch, (ox, oy))

        draw.rectangle(
            [col_i * cell, row_i * cell + 20,
             (col_i + 1) * cell - 1, (row_i + 1) * cell + 19],
            outline=(180, 180, 60), width=1)

        draw.text((col_i * cell + 2, row_i * cell + 22),
                  f"{c['ncc_score']:.2f} w{c['scale_w']}",
                  fill=(200, 200, 60))

    path = out_dir / "raw_peak_crops.png"
    sheet.save(str(path))
    print(f"  Raw peak crops  -> {path}  ({n} shown)")


# =============================================================================
# STEP 7 — Save confirmed outputs
# =============================================================================

def save_confirmed_map(symbols_bin: np.ndarray, confirmed: list,
                       out_dir: Path):
    H, W = symbols_bin.shape
    rgb  = np.stack([(symbols_bin * 255).astype(np.uint8)] * 3, axis=-1)
    img  = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)

    for c in confirmed:
        r0, c0_px = c["row"], c["col"]
        draw.rectangle(
            [c0_px, r0, c0_px + c["scale_w"], r0 + c["scale_h"]],
            outline=(0, 80, 255), width=2)
        wr0, wc0, wr1, wc1 = c["dot_left_win"]
        draw.rectangle([wc0, wr0, wc1, wr1], outline=(0, 220, 220), width=1)
        wr0, wc0, wr1, wc1 = c["dot_right_win"]
        draw.rectangle([wc0, wr0, wc1, wr1], outline=(255, 220, 0), width=1)

    path = out_dir / "confirmed_map.png"
    img.save(str(path))
    print(f"  Confirmed map   -> {path}")


def save_confirmed_crops(symbols_bin: np.ndarray, confirmed: list,
                         out_dir: Path):
    sorted_conf = sorted(confirmed, key=lambda x: x["ncc_score"])
    H, W  = symbols_bin.shape
    cell  = 80
    pad   = 8
    cols  = 12
    n     = min(len(sorted_conf), 300)
    rows  = math.ceil(n / cols) if n > 0 else 1

    sheet = Image.new("RGB", (cell * cols, cell * rows), (15, 15, 15))
    draw  = ImageDraw.Draw(sheet)

    for i, c in enumerate(sorted_conf[:n]):
        r0, c0_px = c["row"], c["col"]
        tw, th    = c["scale_w"], c["scale_h"]

        crop_r0 = max(0, r0 - pad)
        crop_c0 = max(0, c0_px - pad)
        crop_r1 = min(H, r0 + th + pad)
        crop_c1 = min(W, c0_px + tw + pad)

        crop = symbols_bin[crop_r0:crop_r1, crop_c0:crop_c1]
        ch, cw = crop.shape
        if ch == 0 or cw == 0:
            continue

        scale = min((cell - 4) / ch, (cell - 4) / cw, 4.0)
        nw = max(1, int(cw * scale))
        nh = max(1, int(ch * scale))

        patch = Image.fromarray((crop * 255).astype(np.uint8)).resize(
            (nw, nh), Image.NEAREST).convert("RGB")

        col_i = i % cols
        row_i = i // cols
        ox    = col_i * cell + (cell - nw) // 2
        oy    = row_i * cell + (cell - nh) // 2
        sheet.paste(patch, (ox, oy))

        if c["both_dots_found"]:
            border = (0, 200, 60)
        elif c["dot_left_found"] or c["dot_right_found"]:
            border = (255, 140, 0)
        else:
            border = (200, 40, 40)

        draw.rectangle(
            [col_i * cell, row_i * cell,
             (col_i + 1) * cell - 1, (row_i + 1) * cell - 1],
            outline=border, width=2)

    path = out_dir / "confirmed_crops.png"
    sheet.save(str(path))
    print(f"  Confirmed crops -> {path}  ({n} shown)")


def save_dot_samples(dot_binary: np.ndarray, confirmed: list,
                     out_dir: Path, max_samples: int = 60):
    cell    = 48
    cols    = 10
    samples = confirmed[:max_samples]
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
            win = dot_binary[wr0:wr1, wc0:wc1]
            wh, ww = win.shape
            if wh == 0 or ww == 0:
                continue
            scale = min((cell - 4) / wh, (cell - 4) / ww, 6.0)
            nw = max(1, int(ww * scale))
            nh = max(1, int(wh * scale))
            patch = Image.fromarray((win * 255).astype(np.uint8)).resize(
                (nw, nh), Image.NEAREST).convert("RGB")

            col_i = i % cols
            row_i = i // cols
            ox = col_i * cell + (cell - nw) // 2
            oy = row_i * cell + (cell - nh) // 2 + 24
            sheet.paste(patch, (ox, oy))

            colour = (0, 220, 80) if c[found_key] else (220, 60, 60)
            draw_obj.rectangle(
                [col_i * cell, row_i * cell + 24,
                 (col_i + 1) * cell - 1, (row_i + 1) * cell + 23],
                outline=colour, width=2)

    combined = Image.new("RGB", (W_sheet * 2 + 8, H_sheet), (10, 10, 10))
    combined.paste(left_sheet,  (0, 0))
    combined.paste(right_sheet, (W_sheet + 8, 0))
    path = out_dir / "dot_samples.png"
    combined.save(str(path))
    print(f"  Dot samples     -> {path}")



# =============================================================================
# STEP 8 — RGB overlay
# =============================================================================

def save_rgb_overlay(rgb_path: str, confirmed: list,
                     bin_h: int, bin_w: int, out_dir: Path):
    """
    Draw confirmed double obstacle detections onto the original RGB colour map.

    The binary map (Phase 2c) and the RGB map may differ in resolution.
    bin_h, bin_w are the binary map dimensions so coordinates can be
    scaled to match the RGB image size.

    Each detection is drawn as:
      - Magenta bounding box (2px) around the symbol
      - Small filled magenta circle at the centre point
      - Short cyan tick at the left dot column position
      - Short yellow tick at the right dot column position

    The overlay is skipped if RGB_PATH does not exist — all other outputs
    are saved normally.
    """
    rgb_file = Path(rgb_path)
    if not rgb_file.exists():
        print(f"  WARNING: RGB map not found at '{rgb_path}' — overlay skipped.")
        print( "  Set RGB_PATH in CONFIG to your original colour chart image.")
        return

    print(f"  Loading RGB map: {rgb_file.name} ...")
    img     = Image.open(str(rgb_file)).convert("RGB")
    draw    = ImageDraw.Draw(img)
    img_w, img_h = img.size

    # Scale factor from binary space to RGB space
    scale_x = img_w / bin_w
    scale_y = img_h / bin_h

    MAGENTA = (255,   0, 255)
    CYAN    = (  0, 220, 220)
    YELLOW  = (255, 220,   0)

    for c in confirmed:
        # Scale bounding box from binary coords to RGB coords
        r0    = int(round(c["row"]             * scale_y))
        c0_px = int(round(c["col"]             * scale_x))
        tw    = max(2, int(round(c["scale_w"]  * scale_x)))
        th    = max(2, int(round(c["scale_h"]  * scale_y)))
        cx    = int(round((c["col"] + c["scale_w"] / 2) * scale_x))
        cy    = int(round((c["row"] + c["scale_h"] / 2) * scale_y))

        # Bounding box
        draw.rectangle(
            [c0_px, r0, c0_px + tw, r0 + th],
            outline=MAGENTA, width=2)

        # Centre dot
        dot_r = max(3, tw // 10)
        draw.ellipse(
            [cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r],
            fill=MAGENTA)

        # Left dot tick (cyan): at DOT_LEFT_CENTRE_FRAC of symbol width
        ldc = int(round((c["col"] + c["scale_w"] * DOT_LEFT_CENTRE_FRAC) * scale_x))
        tick_top    = int(round((c["row"] + c["scale_h"] * 0.82) * scale_y))
        tick_bottom = int(round((c["row"] + c["scale_h"] * 0.98) * scale_y))
        draw.line([ldc, tick_top, ldc, tick_bottom], fill=CYAN, width=2)

        # Right dot tick (yellow): at DOT_RIGHT_CENTRE_FRAC of symbol width
        rdc = int(round((c["col"] + c["scale_w"] * DOT_RIGHT_CENTRE_FRAC) * scale_x))
        draw.line([rdc, tick_top, rdc, tick_bottom], fill=YELLOW, width=2)

    path = out_dir / "rgb_overlay.png"
    img.save(str(path))
    print(f"  RGB overlay     -> {path}  ({len(confirmed)} detections)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading symbols binary (for NCC): {SYMBOLS_PATH}")
    symbols_bin = np.array(Image.open(SYMBOLS_PATH).convert("L")) > 128
    H, W = symbols_bin.shape
    print(f"  Size: {W} x {H} px    Ink: {symbols_bin.mean()*100:.2f}%")

    print(f"Loading full binary  (for dots):  {BINARY_PATH}")
    dot_bin = np.array(Image.open(BINARY_PATH).convert("L")) > 128
    dH, dW  = dot_bin.shape
    print(f"  Size: {dW} x {dH} px    Ink: {dot_bin.mean()*100:.2f}%")

    if (H, W) != (dH, dW):
        print("  WARNING: size mismatch — resizing full binary to match ...")
        dot_pil = Image.fromarray((dot_bin * 255).astype(np.uint8)).resize(
            (W, H), Image.NEAREST)
        dot_bin = np.array(dot_pil) > 128

    print(f"\nNCC at scales {TEMPLATE_WIDTHS} px (threshold={NCC_THRESHOLD}) ...")
    raw_peaks = run_ncc(symbols_bin, TEMPLATE_WIDTHS, NCC_THRESHOLD,
                        PEAK_MIN_DISTANCE, LEGEND_STRIP_WIDTH)
    print(f"  Raw peaks: {len(raw_peaks)}")

    after_nms = nms(raw_peaks)
    print(f"  After NMS: {len(after_nms)}")

    after_scale = [c for c in after_nms if c["scale_w"] >= MIN_SCALE_W]
    print(f"  After min scale ({MIN_SCALE_W}px): {len(after_scale)}  "
          f"(rejected {len(after_nms)-len(after_scale)})")

    print("\nOpen-base check (arms at outer edges, centre open) ...")
    after_base = [c for c in after_scale if passes_open_base(symbols_bin, c)]
    print(f"  After open-base:   {len(after_base)}  "
          f"(rejected {len(after_scale)-len(after_base)})")

    print("Dual-apex check (two separate apexes at top, gap at centre) ...")
    after_apex = [c for c in after_base if passes_dual_apex(symbols_bin, c)]
    print(f"  After dual-apex:   {len(after_apex)}  "
          f"(rejected {len(after_base)-len(after_apex)})")

    print("Valley check (3-group row in crossing region) ...")
    after_valley = [c for c in after_apex if passes_valley_check(symbols_bin, c)]
    print(f"  After valley:      {len(after_valley)}  "
          f"(rejected {len(after_apex)-len(after_valley)})")

    print("Symmetry check ...")
    after_sym = [c for c in after_valley if passes_symmetry(symbols_bin, c)]
    print(f"  After symmetry:    {len(after_sym)}  "
          f"(rejected {len(after_valley)-len(after_sym)})")

    print("Isolation check ...")
    after_iso = [c for c in after_sym if passes_isolation(symbols_bin, c)]
    print(f"  After isolation:   {len(after_iso)}  "
          f"(rejected {len(after_sym)-len(after_iso)})")

    print("Two-dot verification (soft — metadata only) ...")
    confirmed = [verify_two_dots(dot_bin, c) for c in after_iso]
    both_found  = sum(1 for c in confirmed if c["both_dots_found"])
    left_found  = sum(1 for c in confirmed if c["dot_left_found"])
    right_found = sum(1 for c in confirmed if c["dot_right_found"])
    print(f"  Both dots : {both_found}/{len(confirmed)}")
    print(f"  Left only : {left_found - both_found}/{len(confirmed)}")
    print(f"  Right only: {right_found - both_found}/{len(confirmed)}")

    for c in confirmed:
        c["centre_row"]   = c["row"] + c["scale_h"] // 2
        c["centre_col"]   = c["col"] + c["scale_w"] // 2
        c["symbol_class"] = "double_obstacle"

    print("\nSaving outputs ...")

    with open(out_dir / "detections_double.json", "w") as f:
        json.dump({
            "total_confirmed": len(confirmed),
            "symbol_class":    "double_obstacle",
            "ncc_threshold":   NCC_THRESHOLD,
            "template_widths": TEMPLATE_WIDTHS,
            "filters": {
                "open_base_check_frac":  BASE_CHECK_FRAC,
                "base_arm_zone":         BASE_ARM_ZONE,
                "base_arm_min_ink":      BASE_ARM_MIN_INK,
                "base_centre_max_ink":   BASE_CENTRE_MAX_INK,
                "apex_check_frac":       APEX_CHECK_FRAC,
                "apex_min_ink":          APEX_MIN_INK,
                "apex_max_gap_ink":      APEX_MAX_GAP_INK,
                "valley_row_lo":         VALLEY_ROW_LO,
                "valley_row_hi":         VALLEY_ROW_HI,
                "valley_n_groups":       VALLEY_N_GROUPS,
                "symmetry_min":          SYMMETRY_MIN,
                "isolation_max_w":       ISOLATION_MAX_W_FACTOR,
            },
            "dot_params": {
                "min_area":          DOT_MIN_AREA,
                "max_area":          DOT_MAX_AREA,
                "max_eccentricity":  DOT_MAX_ECCENTRICITY,
                "left_centre_frac":  DOT_LEFT_CENTRE_FRAC,
                "right_centre_frac": DOT_RIGHT_CENTRE_FRAC,
            },
            "confirmed": confirmed,
        }, f, indent=2)
    print(f"  detections_double.json -> {out_dir / 'detections_double.json'}")

    save_raw_peak_crops(symbols_bin, raw_peaks, out_dir)
    save_confirmed_map(symbols_bin, confirmed, out_dir)
    save_confirmed_crops(symbols_bin, confirmed, out_dir)
    save_dot_samples(dot_bin, confirmed, out_dir)
    save_rgb_overlay(RGB_PATH, confirmed, H, W, out_dir)

    print("\n=== RESULT ===")
    print(f"  NCC raw peaks    : {len(raw_peaks)}")
    print(f"  After NMS        : {len(after_nms)}")
    print(f"  After min scale  : {len(after_scale)}")
    print(f"  After open-base  : {len(after_base)}")
    print(f"  After dual-apex  : {len(after_apex)}")
    print(f"  After valley     : {len(after_valley)}")
    print(f"  After symmetry   : {len(after_sym)}")
    print(f"  After isolation  : {len(after_iso)}")
    print(f"  Confirmed        : {len(confirmed)}")
    print(f"  Both dots found  : {both_found}/{len(confirmed)}")
    print(f"\nOutputs in: {out_dir.resolve()}")
    print("\nTuning guide:")
    print("  0 raw peaks          : lower NCC_THRESHOLD or check SYMBOLS_PATH")
    print("  Raw peaks but 0 conf : inspect raw_peak_crops.png first")
    print("  open-base killing all: lower BASE_ARM_MIN_INK or raise BASE_CENTRE_MAX_INK")
    print("  dual-apex killing all: lower APEX_MIN_INK or raise APEX_MAX_GAP_INK")
    print("  valley killing all   : lower VALLEY_N_GROUPS=2 or widen row range")


if __name__ == "__main__":
    main()