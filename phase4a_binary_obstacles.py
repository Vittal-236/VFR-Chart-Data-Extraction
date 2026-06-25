"""
Phase 4 — Obstacle Detection on Binary Map
VFR Chart Extraction Pipeline (FAA Base Model)

Two improvements over the previous colour-mask approach:

  IMPROVEMENT 1 — NCC on Phase 2b cleaned binary instead of the colour mask.
      The binary map has no colour ambiguity (yellow city backgrounds, water,
      terrain shading all gone). Template matching runs purely on ink geometry.
      Line intersections and arc fragments are already reduced by Phase 2b.

  IMPROVEMENT 2 — Dot verification as a mandatory gate.
      After NCC finds a triangle candidate, the code explicitly searches for
      a small circular dot blob in the window directly below and horizontally
      centred on the candidate triangle. If no dot is found, the candidate is
      rejected — regardless of how well the triangle matched.

      This is the strongest classical discriminator available:
        - Letter A has a horizontal crossbar, not a dot below.
        - Letters M, N, W have no dot below.
        - Line intersections have no dot below.
        - Only the FAA obstacle symbol has a compact dot sitting below the apex.

HOW TO USE
----------
1. Set the two paths in CONFIG below:
     BINARY_PATH  — output from phase2b (cleaned binary PNG, white ink on black)
     OUTPUT_DIR   — folder where results will be saved

2. Hit Run. No CLI needed.

3. Outputs saved to OUTPUT_DIR:
     detections.json          — all confirmed detections with pixel coords
     annotated.png            — map with green boxes (confirmed) +
                                red boxes (triangle found but no dot)
     dot_search_samples.png   — visual grid of dot-search windows for
                                the first 60 candidates (for tuning)

TEMPLATE GEOMETRY (measured from obstacle.png, 140x168 px reference image)
---------------------------------------------------------------------------
  Triangle ink: rows 19-144, cols 8-125  →  H=126px, W=118px
  Dot ink:      rows 145-151, cols 58-75 →  H=7px,   W=18px
  Both horizontally centred at col 66 (same centre column).
  Dot sits directly below triangle base. No horizontal offset.

  At binary map scale the triangle is ~20-30px wide (we test 5 sizes).
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

# Phase 2c symbols-only binary (white ink on black, dots likely missing).
# NCC template matching runs on this — lines are gone, less clutter.
SYMBOLS_PATH = r"outputs/phase2_layer_segmentation/phase2c_symbols_only_binary/washington_symbols_only.png"

# Phase 1 full binary (white ink on black, ALL ink including dots present).
# Dot verification searches this — dots were never filtered out here.
BINARY_PATH  = r"outputs/phase1_preprocessing/Washington_binary.png"

OUTPUT_DIR   = r"outputs/phase4_symbol_detection/phase4a_binary_obstacles"    # folder for all outputs

# =============================================================================
# TUNING PARAMETERS
# =============================================================================

# Global NCC threshold floor — no candidate below this is considered.
# Data analysis on 1402 confirmed detections showed:
#   NCC < 0.60: 213 detections — 71% are scale-24 noise (n, 0, 6, slashes)
#   NCC 0.60-0.62: 30 detections — still mixed FP
#   NCC >= 0.62: 1159 detections — 76% scale-32, predominantly clean ∧ shapes
# Clean cutoff is at 0.62.
NCC_THRESHOLD = 0.62

# Per-scale NCC minimum.
# Scale-24 is noisier — raise its floor higher to cut the small-template FP.
# Scale-28 and 32 are clean at the global floor.
NCC_THRESHOLD_PER_SCALE = {
    24: 0.65,   # scale-24 FP persist even at 0.62; raise further
    28: 0.62,
    32: 0.62,
}

# Minimum pixel distance between two NCC peaks (prevents double-counting).
PEAK_MIN_DISTANCE = 10

# Template widths tested in pixels on the binary map (multi-scale).
# Scales 16 and 20 were removed — at NCC >= 0.62 they produce zero peaks
# and only add compute time. Real obstacles match at 24-32px.
TEMPLATE_WIDTHS = [24, 28, 32]
MIN_SCALE_W = 24   # keep for safety, already matches TEMPLATE_WIDTHS floor

# --- Dot search window (relative to triangle bounding box) ---
DOT_BELOW_FRAC    = 0.0    # start right at the base row
DOT_SEARCH_H_FRAC = 0.30   # search window height = 30% of triangle height
DOT_SEARCH_W_FRAC = 0.40   # narrowed from 0.50: dot must be near the centre

# --- Dot blob acceptance criteria ---
DOT_MIN_AREA         = 3    # px²
DOT_MAX_AREA         = 25   # px² — real dot is tiny (~5-20px²)
DOT_MAX_ECCENTRICITY = 0.65 # nearly circular

# --- Open-base structural check ---
BASE_CHECK_FRAC = 0.15   # inspect bottom 15% of bounding box rows
BASE_INK_LIMIT  = 0.10   # reject if centre-base ink fraction exceeds this

# --- Apex sharpness check ---
# Real ∧ has a SHARP single-point apex at the top. The top few rows contain
# very few ink pixels, tightly clustered near the horizontal centre.
# The curve of '0', 'O', 'Q' etc. has broad ink spread across the top.
# APEX_CHECK_FRAC: fraction of bounding box height to inspect at the top.
# APEX_MAX_SPREAD: maximum fraction of bounding box width that ink may span
#   in the top rows. Real ∧ apex spans ~10-20% of width. '0' curve spans 60%+.
APEX_CHECK_FRAC  = 0.15   # inspect top 15% of rows
APEX_MAX_SPREAD  = 0.35   # reject if ink spans more than 35% of bbox width at apex

# Suppress detections inside the legend strip on the left of the chart.
LEGEND_STRIP_WIDTH = 140   # pixels from left edge


# =============================================================================
# STEP 1 — Build the open-V triangle template at a given pixel width
# =============================================================================

def build_triangle_template(width_px: int) -> np.ndarray:
    """
    Build a binary float32 template of the FAA obstacle open-V triangle.

    Shape:
      - Left arm: straight line from apex (top centre) to bottom-left corner
      - Right arm: straight line from apex (top centre) to bottom-right corner
      - NO base edge (FAA symbol is open at the bottom)
      - NO dot (dot is verified separately in Step 3)

    Stroke width is 1px for small templates, 2px for larger ones.
    Returns array shape (H, W), values 0.0 (background) or 1.0 (ink).
    """
    # Height proportional to reference template: H=126, W=118 → ratio 1.068
    height_px = int(round(width_px * 126 / 118))
    tmpl = np.zeros((height_px, width_px), dtype=np.float32)

    stroke = 1 if width_px < 24 else 2

    apex_r  = 0
    apex_c  = (width_px - 1) / 2.0   # horizontal centre
    base_l_r, base_l_c = height_px - 1, 0.0
    base_r_r, base_r_c = height_px - 1, float(width_px - 1)

    def draw_arm(r0, c0, r1, c1):
        steps = max(abs(r1 - r0), abs(c1 - c0), 1)
        for i in range(int(steps) + 1):
            t = i / steps
            r = r0 + t * (r1 - r0)
            c = c0 + t * (c1 - c0)
            for dr in range(-stroke + 1, stroke):
                for dc in range(-stroke + 1, stroke):
                    rr = int(round(r)) + dr
                    cc = int(round(c)) + dc
                    if 0 <= rr < height_px and 0 <= cc < width_px:
                        tmpl[rr, cc] = 1.0

    draw_arm(apex_r, apex_c, base_l_r, base_l_c)   # left arm
    draw_arm(apex_r, apex_c, base_r_r, base_r_c)   # right arm
    # Base edge intentionally omitted

    return tmpl


# =============================================================================
# STEP 2 — Multi-scale tiled NCC on the binary map
# =============================================================================

# Tile size for NCC. Each tile is processed independently.
# 2048 keeps peak FFT allocation ~32 MB per tile (safe on 8 GB RAM).
# Overlap must be >= max template height so detections at tile edges are not lost.
TILE_SIZE    = 2048
TILE_OVERLAP = 64    # px — must be >= largest template height (~29px for width=32)


def run_ncc(binary: np.ndarray, template_widths: list,
            ncc_threshold: float, peak_min_dist: int,
            legend_strip: int) -> list:
    """
    Tiled multi-scale NCC. Processes the map in TILE_SIZE x TILE_SIZE chunks
    with TILE_OVERLAP px overlap so symbols at tile boundaries are not missed.

    WHY TILED:
      match_template internally FFTs both the image and template.
      For a 16770x12345 map the FFT output is ~805 MB per scale — OOM on
      most workstations. Tiling caps peak allocation to ~32 MB per tile
      regardless of map size, at the cost of a small overlap bookkeeping step.

    OVERLAP HANDLING:
      Each tile is extracted with TILE_OVERLAP extra pixels on every side.
      After NCC, peaks are found in the full padded tile. Only peaks whose
      position (in map coordinates) falls inside the *inner* tile region are
      kept — this prevents the same peak being reported twice from adjacent tiles.

    binary: bool array (H, W), True = ink
    Returns list of dicts: {row, col, scale_w, scale_h, ncc_score}
      row, col = top-left corner of matched template in MAP coordinates
    """
    H, W      = binary.shape
    all_peaks = []

    # Pre-build all templates once (not per tile)
    templates = {w: build_triangle_template(w) for w in template_widths}

    # Tile grid
    tile_rows = list(range(0, H, TILE_SIZE))
    tile_cols = list(range(0, W, TILE_SIZE))
    n_tiles   = len(tile_rows) * len(tile_cols)
    tile_idx  = 0

    for tr in tile_rows:
        for tc in tile_cols:
            tile_idx += 1

            # Inner tile bounds (what we keep results from)
            inner_r0 = tr
            inner_c0 = tc
            inner_r1 = min(tr + TILE_SIZE, H)
            inner_c1 = min(tc + TILE_SIZE, W)

            # Padded tile bounds (what we feed to NCC)
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

                # Skip tile if it's too small for this template
                if tile.shape[0] < th or tile.shape[1] < w:
                    continue

                corr = match_template(tile, tmpl, pad_input=False)

                peaks = peak_local_max(
                    corr,
                    min_distance=peak_min_dist,
                    threshold_abs=ncc_threshold,
                )

                for pr, pc in peaks:
                    # Convert peak coords from padded-tile space → map space
                    map_r = pr + pad_r0
                    map_c = pc + pad_c0

                    # Apply per-scale NCC threshold (scale 24 is noisier)
                    scale_thresh = NCC_THRESHOLD_PER_SCALE.get(w, ncc_threshold)
                    if corr[pr, pc] < scale_thresh:
                        continue

                    # Only keep peaks whose top-left falls inside the inner tile.
                    # This deduplicates peaks that appear in the overlap region
                    # of two adjacent tiles.
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

            print(f"  → {tile_peaks_found} peaks")

    return all_peaks


# =============================================================================
# STEP 3 — Non-maximum suppression across scales
# =============================================================================

def nms(candidates: list) -> list:
    """
    Remove duplicate detections from overlapping scales.
    Keeps the highest NCC score when two bounding-box centres are closer
    than 60% of the average template width.
    """
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
# STEP 3b — Open-base structural check
# =============================================================================

def passes_open_base(symbols_bin: np.ndarray, candidate: dict) -> bool:
    """
    Reject candidates that have ink at the bottom-centre of their bounding box.

    The FAA obstacle symbol is an OPEN inverted-V — no horizontal ink at the
    base. Letter A has a crossbar there. Letters M, N, W have bottom strokes.
    This check is the single most powerful letter-rejection filter because it
    directly tests the defining structural difference.

    Method:
      1. Extract the bounding box of the candidate from the symbols binary.
      2. Look at the bottom BASE_CHECK_FRAC rows of the bounding box.
      3. Within those rows, look at only the centre third of the width
         (ignoring the left/right corners where the arms legitimately end).
      4. If ink fraction in that centre-base zone > BASE_INK_LIMIT → reject.

    Returns True if the candidate PASSES (open base confirmed), False to reject.
    """
    r0, c0 = candidate["row"], candidate["col"]
    tw, th = candidate["scale_w"], candidate["scale_h"]
    H, W   = symbols_bin.shape

    r1 = min(H, r0 + th)
    c1 = min(W, c0 + tw)

    patch = symbols_bin[r0:r1, c0:c1]
    ph, pw = patch.shape
    if ph == 0 or pw == 0:
        return True   # can't check, pass through

    # Bottom zone: last BASE_CHECK_FRAC of rows
    bot_rows = max(1, int(round(ph * BASE_CHECK_FRAC)))
    bot_zone = patch[ph - bot_rows:, :]

    # Centre third of width (arms touch left/right corners, we ignore those)
    c_lo = pw // 3
    c_hi = pw - pw // 3
    centre_zone = bot_zone[:, c_lo:c_hi]

    if centre_zone.size == 0:
        return True

    ink_frac = centre_zone.sum() / centre_zone.size
    return ink_frac <= BASE_INK_LIMIT


def passes_apex_sharpness(symbols_bin: np.ndarray, candidate: dict) -> bool:
    """
    Reject shapes with a broad/curved top — keeps only sharp-apex ∧ shapes.

    Real FAA obstacle ∧:
      Top rows contain a single-point apex — ink spans only 10-20% of the
      bounding box width at the very top, expanding as arms diverge downward.

    False positives with broad tops:
      '0', 'O', 'Q' curves: ink spans 60-80% of width at the top (full curve).
      'M', 'W' at top: wide flat ink across most of the width.
      Rounded arc segments: broad curved ink at the top.

    Method:
      Look at the top APEX_CHECK_FRAC rows of the bounding box.
      Find the column range (min to max) of all ink pixels in those rows.
      Compute spread = (max_col - min_col + 1) / bounding_box_width.
      If spread > APEX_MAX_SPREAD → broad top → reject.
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

    # Top zone rows
    apex_rows = max(1, int(round(ph * APEX_CHECK_FRAC)))
    apex_zone = patch[:apex_rows, :]

    ink_cols = np.where(apex_zone.any(axis=0))[0]
    if len(ink_cols) == 0:
        return True   # no ink at top — unusual but pass through

    spread = (ink_cols[-1] - ink_cols[0] + 1) / pw
    return spread <= APEX_MAX_SPREAD


def passes_arm_symmetry(symbols_bin: np.ndarray, candidate: dict) -> bool:
    """
    Reject shapes whose left and right ink halves are badly unbalanced.

    A real ∧ has near-equal ink weight on the left and right sides of the
    bounding box (both arms are the same length and stroke width).
    Letters N, single diagonal slashes, and line fragments are strongly
    asymmetric — one side has far more ink than the other.

    Method:
      Split the bounding box vertically down the centre.
      Count ink pixels in left half vs right half.
      Compute balance = min(left,right) / max(left,right).
      balance=1.0 is perfect symmetry. balance<SYMMETRY_MIN → reject.
    """
    SYMMETRY_MIN = 0.40   # reject if one arm has less than 40% of the other's ink

    r0, c0 = candidate["row"], candidate["col"]
    tw, th = candidate["scale_w"], candidate["scale_h"]
    H, W   = symbols_bin.shape

    r1 = min(H, r0 + th)
    c1 = min(W, c0 + tw)
    patch = symbols_bin[r0:r1, c0:c1]
    ph, pw = patch.shape
    if ph == 0 or pw < 2:
        return True

    mid = pw // 2
    left_ink  = patch[:, :mid].sum()
    right_ink = patch[:, mid:].sum()

    if left_ink == 0 and right_ink == 0:
        return False   # no ink at all — not a real symbol

    balance = min(left_ink, right_ink) / max(left_ink, right_ink)
    return balance >= SYMMETRY_MIN


def passes_isolation(symbols_bin: np.ndarray, candidate: dict) -> bool:
    """
    Reject candidates where the ∧ shape is fused with adjacent text.

    On the symbols-only binary, a real obstacle appears as an isolated
    compact blob. The letter A fused with "IA", "CA", "EA" etc. belongs
    to a connected component that is much wider than a single ∧.

    Method:
      1. Find the connected component that contains the centre pixel of
         the candidate bounding box.
      2. Measure that component's bounding box width.
      3. If the component width > ISOLATION_MAX_W_FACTOR * template_width
         → the symbol is fused with neighbours → reject.

    ISOLATION_MAX_W_FACTOR = 1.5 means: the blob containing this candidate
    may be at most 1.5× as wide as the template. A lone ∧ scores ~1.0–1.3×.
    Single adjacent letter (A∧) scores 1.6-1.9× — now rejected.
    "IA∧" fused blob scores ~2.5–4×.
    """
    ISOLATION_MAX_W_FACTOR = 1.5   # tightened from 2.0 — cuts single-adjacent-letter FP

    r0, c0 = candidate["row"], candidate["col"]
    tw, th = candidate["scale_w"], candidate["scale_h"]
    H, W   = symbols_bin.shape

    # Centre pixel of the candidate bounding box
    cr = min(H - 1, r0 + th // 2)
    cc = min(W - 1, c0 + tw // 2)

    # If the centre pixel is not ink, search nearby for the nearest ink pixel
    if not symbols_bin[cr, cc]:
        # Search in a small window around the centre
        search_r = slice(max(0, cr - 3), min(H, cr + 4))
        search_c = slice(max(0, cc - 3), min(W, cc + 4))
        window = symbols_bin[search_r, search_c]
        if window.sum() == 0:
            return True   # no ink nearby — isolated enough, pass through
        # Find nearest ink pixel
        ink_positions = np.argwhere(window)
        nearest = ink_positions[0]
        cr = max(0, cr - 3) + nearest[0]
        cc = max(0, cc - 3) + nearest[1]

    # Label connected components in a local region around the candidate
    # (labelling the full map is too slow; use a generous local window)
    pad   = tw * 3
    lr0   = max(0, r0 - pad)
    lc0   = max(0, c0 - pad)
    lr1   = min(H, r0 + th + pad)
    lc1   = min(W, c0 + tw + pad)

    local  = symbols_bin[lr0:lr1, lc0:lc1]
    lbl    = label(local, connectivity=2)

    # Component ID at the (adjusted) centre pixel
    local_cr = cr - lr0
    local_cc = cc - lc0
    local_cr = max(0, min(lbl.shape[0] - 1, local_cr))
    local_cc = max(0, min(lbl.shape[1] - 1, local_cc))

    comp_id = lbl[local_cr, local_cc]
    if comp_id == 0:
        return True   # background — pass through

    # Bounding box of this component
    comp_mask = lbl == comp_id
    comp_cols = np.where(comp_mask.any(axis=0))[0]
    if len(comp_cols) == 0:
        return True

    comp_width = comp_cols[-1] - comp_cols[0] + 1
    return comp_width <= ISOLATION_MAX_W_FACTOR * tw


# =============================================================================
# STEP 4 — Dot verification gate
# =============================================================================

def verify_dot(dot_binary: np.ndarray, candidate: dict) -> dict:
    """
    Search for a dot blob directly below and centred on the triangle candidate.

    WHY dot_binary IS THE PHASE 1 FULL BINARY (not the symbols-only binary):
      Phase 2c removes small blobs below its area minimum — the obstacle dot
      (~3-15 px² at map scale) is one of them. So the symbols-only binary has
      the ∧ triangle but no dot. The Phase 1 full binary retains all ink,
      including the dot. Searching there gives the dot its best chance of
      being found.

    Geometry (from obstacle.png measurements):
      - Dot is horizontally centred at the same column as the triangle centre
      - Dot sits directly at or just below the triangle base row
      - Dot is small and roughly circular

    Adds to candidate dict:
      dot_found: bool
      dot_area:  int (area of dot blob found, 0 if not found)
      dot_win:   (r0, c0, r1, c1) search window in map coordinates
    """
    r0, c0 = candidate["row"], candidate["col"]
    tw, th = candidate["scale_w"], candidate["scale_h"]
    H, W   = dot_binary.shape

    base_row   = r0 + th          # one row below the template bottom
    centre_col = c0 + tw // 2

    below_px    = int(round(th * DOT_BELOW_FRAC))
    search_h_px = max(4, int(round(th * DOT_SEARCH_H_FRAC)))
    search_w_px = max(6, int(round(tw * DOT_SEARCH_W_FRAC)))

    win_r0 = max(0, base_row + below_px)
    win_r1 = min(H, win_r0 + search_h_px)
    win_c0 = max(0, centre_col - search_w_px // 2)
    win_c1 = min(W, centre_col + search_w_px // 2)

    candidate["dot_win"]   = (win_r0, win_c0, win_r1, win_c1)
    candidate["dot_found"] = False
    candidate["dot_area"]  = 0

    if win_r1 <= win_r0 or win_c1 <= win_c0:
        return candidate

    window = dot_binary[win_r0:win_r1, win_c0:win_c1]
    if window.sum() == 0:
        return candidate

    labelled = label(window, connectivity=2)
    for p in regionprops(labelled):
        if DOT_MIN_AREA <= p.area <= DOT_MAX_AREA and p.eccentricity <= DOT_MAX_ECCENTRICITY:
            candidate["dot_found"] = True
            candidate["dot_area"]  = int(p.area)
            break

    return candidate


# =============================================================================
# STEP 5 — Save outputs
# =============================================================================

def save_annotated(binary: np.ndarray, confirmed: list, rejected: list,
                   output_path: str):
    """
    Annotated map image:
      Green box  = confirmed obstacle (NCC passed + dot found)
      Red box    = triangle matched but dot NOT found (false positive)
      Cyan box   = dot search window for confirmed detections
      Orange box = dot search window for rejected detections
    """
    rgb = np.stack([binary * 255] * 3, axis=-1).astype(np.uint8)
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)

    for c in confirmed:
        r0, c0_px = c["row"], c["col"]
        draw.rectangle([c0_px, r0, c0_px + c["scale_w"], r0 + c["scale_h"]],
                       outline=(0, 255, 0), width=1)
        wr0, wc0, wr1, wc1 = c["dot_win"]
        draw.rectangle([wc0, wr0, wc1, wr1], outline=(0, 220, 220), width=1)

    for c in rejected:
        r0, c0_px = c["row"], c["col"]
        draw.rectangle([c0_px, r0, c0_px + c["scale_w"], r0 + c["scale_h"]],
                       outline=(255, 0, 0), width=1)
        wr0, wc0, wr1, wc1 = c["dot_win"]
        draw.rectangle([wc0, wr0, wc1, wr1], outline=(255, 165, 0), width=1)

    img.save(output_path)
    print(f"  Annotated image → {output_path}")


def save_dot_samples(dot_binary: np.ndarray, candidates: list,
                     output_path: str, max_samples: int = 60):
    """
    Grid of dot-search windows for the first max_samples candidates.
    Green border = dot found (PASS). Red border = no dot (FAIL).
    Use this to tune DOT_MIN_AREA, DOT_MAX_AREA, DOT_MAX_ECCENTRICITY.
    """
    cell    = 48
    cols    = 10
    samples = candidates[:max_samples]
    rows    = math.ceil(len(samples) / cols)
    sheet   = Image.new("RGB", (cell * cols, cell * rows + 20), (20, 20, 20))
    draw    = ImageDraw.Draw(sheet)
    draw.text((4, 2), "Dot search windows  GREEN=dot found  RED=no dot",
              fill=(200, 200, 200))

    for i, c in enumerate(samples):
        wr0, wc0, wr1, wc1 = c["dot_win"]
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
        oy = row_i * cell + (cell - nh) // 2 + 20
        sheet.paste(patch, (ox, oy))

        colour = (0, 220, 80) if c["dot_found"] else (220, 60, 60)
        draw.rectangle(
            [col_i * cell, row_i * cell + 20,
             (col_i + 1) * cell - 1, (row_i + 1) * cell + 19],
            outline=colour, width=2)

    sheet.save(output_path)
    print(f"  Dot samples grid → {output_path}")


def save_confirmed_only(symbols_bin: np.ndarray, dot_bin: np.ndarray,
                        confirmed: list, out_dir: Path):
    """
    Two confirmed-only outputs:

    1. confirmed_map.png
       The full map image with ONLY confirmed detections drawn.
       Black background (no red rejected boxes cluttering the view).
       Each confirmed obstacle gets:
         - Green bounding box around the triangle
         - Cyan dot search window below it

    2. confirmed_crops.png
       Contact sheet of cropped symbol patches from the symbols binary.
       Each cell shows the actual ∧ shape that was detected.
       Sorted by NCC score descending so the best matches are top-left.
       Caption shows NCC score and scale width.
    """
    H, W = symbols_bin.shape

    # ── Output 1: confirmed_map.png ──────────────────────────────────────────
    rgb  = np.stack([symbols_bin * 255] * 3, axis=-1).astype(np.uint8)
    img  = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)

    for c in confirmed:
        r0, c0_px = c["row"], c["col"]
        # Green box around triangle
        draw.rectangle(
            [c0_px, r0, c0_px + c["scale_w"], r0 + c["scale_h"]],
            outline=(0, 255, 0), width=2)
        # Cyan box showing dot search window
        wr0, wc0, wr1, wc1 = c["dot_win"]
        draw.rectangle([wc0, wr0, wc1, wr1], outline=(0, 220, 220), width=1)

    map_path = out_dir / "confirmed_map.png"
    img.save(str(map_path))
    print(f"  Confirmed map → {map_path}")

    # ── Output 2: confirmed_crops.png ────────────────────────────────────────
    # Sort by NCC score ASCENDING — worst matches first.
    # This puts any remaining false positives at the top of the sheet
    # where they are immediately visible, rather than buried at the end.
    sorted_conf = sorted(confirmed, key=lambda x: x["ncc_score"])

    cell     = 56          # cell size in the contact sheet
    pad      = 8           # padding around each crop inside the cell
    cols     = 20
    max_show = min(len(sorted_conf), 400)   # cap at 400 cells
    rows     = math.ceil(max_show / cols)

    sheet = Image.new("RGB", (cell * cols, cell * rows), (15, 15, 15))
    draw2 = ImageDraw.Draw(sheet)

    for i, c in enumerate(sorted_conf[:max_show]):
        r0, c0_px = c["row"], c["col"]
        tw, th    = c["scale_w"], c["scale_h"]

        # Crop with a little context padding
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

        # Thin green border on every cell
        draw2.rectangle(
            [col_i * cell, row_i * cell,
             (col_i + 1) * cell - 1, (row_i + 1) * cell - 1],
            outline=(0, 180, 60), width=1)

    crops_path = out_dir / "confirmed_crops.png"
    sheet.save(str(crops_path))
    print(f"  Confirmed crops → {crops_path}  ({max_show} detections)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load symbols-only binary (Phase 2c) — used for NCC ───────────────────
    # Lines and background removed. Dots also removed (too small for Phase 2c
    # area filter), but that is fine — NCC only needs the triangle shape.
    print(f"Loading symbols binary (for NCC): {SYMBOLS_PATH}")
    symbols_bin = np.array(Image.open(SYMBOLS_PATH).convert("L")) > 128
    H, W = symbols_bin.shape
    print(f"  Size: {W} x {H} px    Ink coverage: {symbols_bin.mean() * 100:.2f}%")

    # ── Load full binary (Phase 1) — used for dot verification ───────────────
    # All ink present including the small dots that Phase 2c discarded.
    # Must be the same resolution as the symbols binary.
    print(f"Loading full binary  (for dots):  {BINARY_PATH}")
    dot_bin = np.array(Image.open(BINARY_PATH).convert("L")) > 128
    dH, dW = dot_bin.shape
    print(f"  Size: {dW} x {dH} px    Ink coverage: {dot_bin.mean() * 100:.2f}%")

    if (H, W) != (dH, dW):
        print(f"\n  WARNING: image sizes differ!")
        print(f"  Symbols binary: {W}x{H}    Full binary: {dW}x{dH}")
        print(f"  Resizing full binary to match symbols binary ...")
        dot_pil = Image.fromarray((dot_bin * 255).astype(np.uint8)).resize(
            (W, H), Image.NEAREST)
        dot_bin = np.array(dot_pil) > 128

    # ── NCC on symbols binary ─────────────────────────────────────────────────
    print(f"\nNCC at scales {TEMPLATE_WIDTHS} px (threshold={NCC_THRESHOLD}) ...")
    raw_peaks = run_ncc(symbols_bin, TEMPLATE_WIDTHS, NCC_THRESHOLD,
                        PEAK_MIN_DISTANCE, LEGEND_STRIP_WIDTH)
    print(f"  Raw peaks: {len(raw_peaks)}")

    # ── NMS across scales ─────────────────────────────────────────────────────
    after_nms = nms(raw_peaks)
    print(f"  After NMS: {len(after_nms)}")

    # ── Minimum scale filter ──────────────────────────────────────────────────
    # Real obstacles are ~90% the same size. Detections at small template widths
    # (16px, 20px) are almost always letters or number fragments — too small.
    after_scale = [c for c in after_nms if c["scale_w"] >= MIN_SCALE_W]
    print(f"  After min scale ({MIN_SCALE_W}px): {len(after_scale)}  "
          f"(rejected {len(after_nms) - len(after_scale)})")

    # ── Open-base structural check ────────────────────────────────────────────
    print("\nOpen-base structural check ...")
    after_base = [c for c in after_scale if passes_open_base(symbols_bin, c)]
    print(f"  After open-base: {len(after_base)}  "
          f"(rejected {len(after_scale) - len(after_base)})")

    # ── Apex sharpness check ──────────────────────────────────────────────────
    # Real ∧ has a sharp single-pixel apex. Rejects '0' curves, arc tops, M/W.
    print("Apex sharpness check ...")
    after_apex = [c for c in after_base if passes_apex_sharpness(symbols_bin, c)]
    print(f"  After apex:      {len(after_apex)}  "
          f"(rejected {len(after_base) - len(after_apex)})")

    # ── Arm symmetry check ────────────────────────────────────────────────────
    print("Arm symmetry check ...")
    after_sym = [c for c in after_apex if passes_arm_symmetry(symbols_bin, c)]
    print(f"  After symmetry:  {len(after_sym)}  "
          f"(rejected {len(after_apex) - len(after_sym)})")

    # ── Blob isolation check ──────────────────────────────────────────────────
    # Rejects candidates fused with adjacent text (e.g. "IA", "CA", "EA").
    # A real obstacle blob is compact; fused text blobs are 2-4x wider.
    print("Blob isolation check ...")
    after_iso = [c for c in after_sym if passes_isolation(symbols_bin, c)]
    print(f"  After isolation: {len(after_iso)}  "
          f"(rejected {len(after_sym) - len(after_iso)})")

    # ── Dot verification — soft score only, NOT a hard gate ──────────────────
    # The dot at map scale is too small and unreliable to use as a hard reject.
    # We run it to add dot_found/dot_area metadata to each detection for
    # inspection, but ALL after_iso candidates are treated as confirmed.
    print("Dot check (soft — metadata only, not gating) ...")
    confirmed = [verify_dot(dot_bin, c) for c in after_iso]
    dot_found_count = sum(1 for c in confirmed if c["dot_found"])
    print(f"  Dot found in {dot_found_count}/{len(confirmed)} confirmed detections")

    # Centre coordinates
    for c in confirmed:
        c["centre_row"] = c["row"] + c["scale_h"] // 2
        c["centre_col"] = c["col"] + c["scale_w"] // 2

    # ── Save ──────────────────────────────────────────────────────────────────
    print("\nSaving outputs ...")

    with open(out_dir / "detections.json", "w") as f:
        json.dump({
            "total_confirmed": len(confirmed),
            "ncc_threshold": NCC_THRESHOLD,
            "template_widths": TEMPLATE_WIDTHS,
            "filters": {
                "open_base_check_frac": BASE_CHECK_FRAC,
                "open_base_ink_limit": BASE_INK_LIMIT,
                "arm_symmetry_min": 0.40,
            },
            "dot_params": {
                "min_area": DOT_MIN_AREA,
                "max_area": DOT_MAX_AREA,
                "max_eccentricity": DOT_MAX_ECCENTRICITY,
            },
            "confirmed": confirmed,
        }, f, indent=2)
    print(f"  detections.json → {out_dir / 'detections.json'}")

    # Confirmed-only map (green boxes only)
    save_confirmed_only(symbols_bin, dot_bin, confirmed, out_dir)

    # Dot sample grid on confirmed only — shows dot search window metadata
    save_dot_samples(dot_bin, confirmed,
                     str(out_dir / "dot_search_samples.png"))

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n=== RESULT ===")
    print(f"  NCC raw peaks   : {len(raw_peaks)}")
    print(f"  After NMS       : {len(after_nms)}")
    print(f"  After min scale : {len(after_scale)}")
    print(f"  After open-base : {len(after_base)}")
    print(f"  After apex      : {len(after_apex)}")
    print(f"  After symmetry  : {len(after_sym)}")
    print(f"  After isolation : {len(after_iso)}")
    print(f"  Confirmed       : {len(confirmed)}")
    print(f"  Dot found in    : {dot_found_count}/{len(confirmed)} confirmed")
    print(f"\nOutputs in: {out_dir.resolve()}")
    print("\nTuning guide:")
    print("  Too many letters still: raise NCC_THRESHOLD (currently {NCC_THRESHOLD})")
    print("  Too few detections:     lower NCC_THRESHOLD or loosen BASE_INK_LIMIT")


if __name__ == "__main__":
    main()