"""
Phase 2e — Power Transmission Line Tower Detection
VFR Chart Extraction Pipeline

Detects the inverted-Y tower pylon glyphs that mark power
transmission lines on FAA VFR Sectional Charts.

The symbol is a Y-shape: one long stem stroke going into the power
line, and two short arm strokes diverging from the junction.
The symbol rotates freely with the line direction — no fixed angle.

Ground-truth validation (tile_6.png, 304x304):
  Tower 1 (Yellow Creek): stem=29  arms=5,3  gap=150°  err=12°  inner_dens=0.074
  Tower 2 (near Saxton) : stem=46  arms=3,3  gap=152°  err=5°   inner_dens=0.086
  FP1 (contour text)    : stem=28  arms=2,2  gap=116°  err=50°  → rejected by STEM_TOL
  FP2 (airport circle)  : stem=22  arms=5,3  gap=132°  err=53°  → rejected by STEM_TOL
  FP3 (text letter Y)   : stem=47  arms=5,2  gap=155°  err=2°   inner_dens=0.198
                                                                  → rejected by INK_DENSITY_MAX

Two key discriminating filters beyond Y-geometry:
  1. STEM_TOL = 45° — stem must closely oppose arm bisector
     (eliminates junctions where the "stem" is a pseudo-branch
      of a complex intersection, not a clean power-line stroke)
  2. INK_DENSITY_MAX = 0.14 — inner ink density ceiling
     (eliminates text characters which pack 3x more ink than
      isolated tower glyphs in the same bounding area)

Detection pipeline per tile:
  1. Extract black ink (L-channel threshold < INK_THRESHOLD).
  2. Upscale 2x LANCZOS, re-binarise at >80.
  3. Skeletonise.
  4. Build junction map (pixels with 3+ skeleton neighbours).
  5. Dilate junction map to merge sub-clusters → super-junctions.
  6. For each super-junction centroid, collect exit pixels
     (non-junction skeleton pixels within EXIT_RADIUS), cluster
     by direction, trace each branch outward along the skeleton.
  7. Stem = longest branch (STEM_MIN..STEM_MAX).
     Arm pair = the two non-stem branches (ARM_MIN..ARM_MAX each)
     whose bisector best opposes the stem (lowest stem_err).
     Filters applied:
       a. ARM_OPEN_MIN <= arm gap <= ARM_OPEN_MAX
       b. stem_err <= STEM_TOL
       c. stem >= STEM_MIN_RATIO * max(arm lengths)
       d. local ink density in INNER_RADIUS <= INK_DENSITY_MAX
  8. NMS in 300dpi coordinate space.
  9. Save overlay map, crop sheet, detections JSON.

Input : outputs\\phase1_preprocessing\\Washington_rgb_300dpi.png
Output: outputs\\phase4_symbol_detection\\phase2e_power_towers\\
"""

import json
import logging
import time
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from scipy import ndimage
from skimage.morphology import skeletonize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── CONFIG ───────────────────────────────────────────────────────────────────
RGB_PATH   = r"outputs\phase1_preprocessing\Washington_rgb_300dpi.png"
OUTPUT_DIR = r"outputs\phase4_symbol_detection\phase2e_power_towers"

INK_THRESHOLD  = 80     # L-channel threshold for black ink extraction
UPSCALE        = 2      # upscale factor applied per tile before skeletonisation

JUNC_DILATION  = 18     # px — dilation radius to merge junction sub-clusters
EXIT_RADIUS    = 22     # px — branch exit search radius from centroid
EXIT_ANGLE_TOL = 40     # degrees — angle tolerance to cluster exit pixels

# Branch lengths in UPSCALED pixel space
STEM_MIN       = 18     # upscaled px
STEM_MAX       = 100    # upscaled px
ARM_MIN        = 2      # upscaled px
ARM_MAX        = 22     # upscaled px

ARM_OPEN_MIN   = 50     # degrees — minimum V opening between the two arms
ARM_OPEN_MAX   = 175    # degrees — maximum V opening
STEM_TOL       = 45     # degrees — stem vs bisector tolerance (tighter = fewer FP)
STEM_MIN_RATIO = 2.5    # stem must be >= this * max(arm lengths)

# Local ink density filter — in ORIGINAL 300dpi space
# Tower glyphs are isolated; text/complex symbols are denser
INNER_RADIUS    = 22    # px — window radius around junction centroid
INK_DENSITY_MAX = 0.14  # maximum fraction of pixels that can be ink

NMS_RADIUS     = 18     # px — NMS merge radius in 300dpi space
TILE_SIZE      = 1024
TILE_OVL       = 64
CROP_HALF      = 35
CROPS_PER_ROW  = 20
# ─────────────────────────────────────────────────────────────────────────────


def _angle_between(v1, v2):
    a1 = np.degrees(np.arctan2(v1[0], v1[1]))
    a2 = np.degrees(np.arctan2(v2[0], v2[1]))
    d  = abs(a1 - a2) % 360
    return d if d <= 180 else 360 - d


def _trace_branch(skel, junc_map, UH, UW, sy, sx, from_y, from_x, max_len):
    cy, cx = sy, sx
    py, px = from_y, from_x
    length = 1
    while length < max_len:
        if not (0 <= cy < UH and 0 <= cx < UW) or not skel[cy, cx]:
            break
        nbrs = [
            (dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
            if not (dy == 0 and dx == 0)
            and 0 <= cy+dy < UH and 0 <= cx+dx < UW
            and skel[cy+dy, cx+dx]
            and (cy+dy != py or cx+dx != px)
        ]
        if not nbrs:
            break
        non_j = [(dy, dx) for dy, dx in nbrs if not junc_map[cy+dy, cx+dx]]
        if not non_j:
            if len(nbrs) > 1:
                break
            step = nbrs[0]
        else:
            step = non_j[0]
        py, px = cy, cx
        cy += step[0]; cx += step[1]
        length += 1
    return cy - from_y, cx - from_x, length


def _cluster_exits(exits, angle_tol):
    groups, used = [], set()
    for i, (dy0, dx0, sy0, sx0) in enumerate(exits):
        if i in used:
            continue
        grp = [(dy0, dx0, sy0, sx0)]
        used.add(i)
        for j, (dy1, dx1, sy1, sx1) in enumerate(exits):
            if j not in used and _angle_between((dy0, dx0), (dy1, dx1)) < angle_tol:
                grp.append((dy1, dx1, sy1, sx1))
                used.add(j)
        groups.append(grp)
    return groups


def _stem_bisector_err(a1v, a2v, sv):
    v1 = np.array(a1v, float)
    v2 = np.array(a2v, float)
    sv_ = np.array(sv,  float)
    for v in (v1, v2, sv_):
        n = np.linalg.norm(v)
        if n > 0:
            v /= n
    bis = v1 + v2
    n = np.linalg.norm(bis)
    if n > 0:
        bis /= n
    return np.degrees(np.arccos(np.clip(-float(np.dot(sv_, bis)), -1, 1)))


def detect_in_tile(rgb_tile):
    """
    Detect tower symbols in one RGB tile (300dpi numpy array).
    Returns list of (y, x) in tile-local 300dpi coordinates.
    """
    gray   = cv2.cvtColor(rgb_tile, cv2.COLOR_RGB2GRAY)
    binary = (gray < INK_THRESHOLD).astype(np.uint8)
    if binary.sum() == 0:
        return []

    th, tw = binary.shape

    # ── Ink density filter preparation (300dpi space) ────────────────────
    # Pre-build distance grid for the density check later
    # (done per candidate, not globally, to avoid memory overhead)

    # ── Upscale + skeletonise ────────────────────────────────────────────
    up_img = Image.fromarray(binary * 255).resize(
        (tw * UPSCALE, th * UPSCALE), Image.LANCZOS
    )
    up_bin = (np.array(up_img) > 80).astype(np.uint8)
    skel   = skeletonize(up_bin.astype(bool)).astype(np.uint8)
    UH, UW = skel.shape
    if skel.sum() == 0:
        return []

    # ── Junction map ─────────────────────────────────────────────────────
    junc_map = np.zeros_like(skel)
    for y in range(1, UH - 1):
        for x in range(1, UW - 1):
            if skel[y, x] and skel[y-1:y+2, x-1:x+2].sum() - 1 >= 3:
                junc_map[y, x] = 1
    if junc_map.sum() == 0:
        return []

    skel_nojunc = skel & ~junc_map

    # ── Super-junction clusters ───────────────────────────────────────────
    k_sz   = JUNC_DILATION * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_sz, k_sz))
    labeled, n_super = ndimage.label(cv2.dilate(junc_map, kernel))

    super_centroids = []
    for lbl in range(1, n_super + 1):
        mask = (labeled == lbl) & (junc_map == 1)
        ys, xs = np.where(mask)
        if len(ys) == 0:
            continue
        super_centroids.append((
            int(np.round(ys.mean())),
            int(np.round(xs.mean())),
        ))

    candidates = []
    for cy, cx in super_centroids:
        # ── Collect and cluster exit pixels ──────────────────────────────
        exits = [
            (dy, dx, cy+dy, cx+dx)
            for dy in range(-EXIT_RADIUS, EXIT_RADIUS + 1)
            for dx in range(-EXIT_RADIUS, EXIT_RADIUS + 1)
            if 0 <= cy+dy < UH and 0 <= cx+dx < UW
            and skel_nojunc[cy+dy, cx+dx]
        ]
        if not exits:
            continue
        groups = _cluster_exits(exits, EXIT_ANGLE_TOL)
        if len(groups) < 3:
            continue

        # ── Trace branches ────────────────────────────────────────────────
        raw_branches = []
        for grp in groups:
            best = max(grp, key=lambda e: e[0]**2 + e[1]**2)
            _, _, sy, sx = best
            edy, edx, ln = _trace_branch(
                skel, junc_map, UH, UW, sy, sx, cy, cx, STEM_MAX + 2
            )
            raw_branches.append((edy, edx, ln))

        raw_branches.sort(key=lambda b: b[2], reverse=True)
        stem = raw_branches[0]
        if not (STEM_MIN <= stem[2] <= STEM_MAX):
            continue

        # ── Find best arm pair by bisector alignment ──────────────────────
        non_stem = [b for b in raw_branches[1:] if ARM_MIN <= b[2] <= ARM_MAX]
        if len(non_stem) < 2:
            continue

        best_err  = float("inf")
        best_arms = None
        for a1, a2 in combinations(non_stem, 2):
            gap = _angle_between((a1[0], a1[1]), (a2[0], a2[1]))
            if not (ARM_OPEN_MIN <= gap <= ARM_OPEN_MAX):
                continue
            err = _stem_bisector_err(
                (a1[0], a1[1]), (a2[0], a2[1]), (stem[0], stem[1])
            )
            if err < best_err:
                best_err  = err
                best_arms = (a1, a2, gap)

        if best_arms is None or best_err > STEM_TOL:
            continue

        a1, a2, _ = best_arms
        if stem[2] < max(a1[2], a2[2]) * STEM_MIN_RATIO:
            continue

        # ── Local ink density filter (in 300dpi space) ───────────────────
        # Map centroid back to 300dpi tile coords
        loc_y = cy // UPSCALE
        loc_x = cx // UPSCALE
        r = INNER_RADIUS
        y0d = max(0, loc_y - r); y1d = min(th, loc_y + r + 1)
        x0d = max(0, loc_x - r); x1d = min(tw, loc_x + r + 1)
        patch = binary[y0d:y1d, x0d:x1d]
        # Only count pixels within the circular radius
        ys_p, xs_p = np.ogrid[y0d-loc_y:y1d-loc_y, x0d-loc_x:x1d-loc_x]
        circle = (ys_p**2 + xs_p**2 <= r**2)
        n_circle = circle.sum()
        if n_circle == 0:
            continue
        ink_density = float((patch * circle).sum()) / float(n_circle)
        if ink_density > INK_DENSITY_MAX:
            continue

        candidates.append((loc_y, loc_x))

    return candidates


def nms(detections, radius):
    if not detections:
        return []
    pts  = np.array(detections, float)
    kept = []
    supp = np.zeros(len(pts), bool)
    for i in range(len(pts)):
        if supp[i]:
            continue
        kept.append(detections[i])
        d     = np.linalg.norm(pts - pts[i], axis=1)
        supp |= (d < radius)
        supp[i] = False
    return kept


def build_crop_sheet(rgb, detections, half, per_row, path):
    if not detections:
        return
    H, W  = rgb.shape[:2]
    n     = len(detections)
    rows  = (n + per_row - 1) // per_row
    side  = half * 2 + 1
    sheet = np.full((rows * side, per_row * side, 3), 220, dtype=np.uint8)
    for idx, (cy, cx) in enumerate(detections):
        r, c = divmod(idx, per_row)
        y0 = max(0, cy-half); y1 = min(H, cy+half+1)
        x0 = max(0, cx-half); x1 = min(W, cx+half+1)
        patch = rgb[y0:y1, x0:x1]
        ph, pw = patch.shape[:2]
        sy = r*side + (side-ph)//2
        sx = c*side + (side-pw)//2
        sheet[sy:sy+ph, sx:sx+pw] = patch
    Image.fromarray(sheet).save(str(path))


def detect_power_towers(rgb_path, output_dir):
    t0  = time.time()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading RGB: {Path(rgb_path).name}")
    rgb  = np.array(Image.open(rgb_path).convert("RGB"))
    H, W = rgb.shape[:2]
    log.info(f"  {W} x {H} px")

    ys = list(range(0, H, TILE_SIZE))
    xs = list(range(0, W, TILE_SIZE))
    log.info(f"  Grid: {len(ys)}r x {len(xs)}c = {len(ys)*len(xs)} tiles")

    all_dets = []
    for ri, y0 in enumerate(ys):
        for ci, x0 in enumerate(xs):
            ty0 = max(0, y0-TILE_OVL); ty1 = min(H, y0+TILE_SIZE+TILE_OVL)
            tx0 = max(0, x0-TILE_OVL); tx1 = min(W, x0+TILE_SIZE+TILE_OVL)
            local_dets = detect_in_tile(rgb[ty0:ty1, tx0:tx1])
            cy0_l = y0-ty0; cy1_l = cy0_l + min(TILE_SIZE, H-y0)
            cx0_l = x0-tx0; cx1_l = cx0_l + min(TILE_SIZE, W-x0)
            for ly, lx in local_dets:
                if cy0_l <= ly < cy1_l and cx0_l <= lx < cx1_l:
                    all_dets.append((ty0+ly, tx0+lx))
        log.info(f"  Row {ri+1}/{len(ys)}  running: {len(all_dets)}")

    log.info(f"Raw: {len(all_dets)}")
    detections = nms(all_dets, NMS_RADIUS)
    log.info(f"NMS: {len(detections)}")

    overlay = rgb.copy()
    for cy, cx in detections:
        cv2.circle(overlay, (cx, cy), 14, (220, 30, 30), 2)
        cv2.drawMarker(overlay, (cx, cy), (220, 30, 30), cv2.MARKER_CROSS, 10, 2)
    Image.fromarray(overlay).save(
        str(out / "confirmed_map.png"), format="PNG", compress_level=3
    )
    build_crop_sheet(rgb, detections, CROP_HALF, CROPS_PER_ROW,
                     out / "confirmed_crops.png")

    elapsed = time.time() - t0
    with open(out / "detections.json", "w") as f:
        json.dump({
            "rgb_path": rgb_path, "img_width": W, "img_height": H,
            "parameters": {
                "INK_THRESHOLD": INK_THRESHOLD, "UPSCALE": UPSCALE,
                "JUNC_DILATION": JUNC_DILATION, "EXIT_RADIUS": EXIT_RADIUS,
                "EXIT_ANGLE_TOL": EXIT_ANGLE_TOL,
                "STEM_MIN": STEM_MIN, "STEM_MAX": STEM_MAX,
                "ARM_MIN": ARM_MIN,   "ARM_MAX": ARM_MAX,
                "ARM_OPEN_MIN": ARM_OPEN_MIN, "ARM_OPEN_MAX": ARM_OPEN_MAX,
                "STEM_TOL": STEM_TOL, "STEM_MIN_RATIO": STEM_MIN_RATIO,
                "INNER_RADIUS": INNER_RADIUS, "INK_DENSITY_MAX": INK_DENSITY_MAX,
                "NMS_RADIUS": NMS_RADIUS,
            },
            "raw_candidates": len(all_dets),
            "confirmed": len(detections),
            "detections": [{"y": int(y), "x": int(x)} for y, x in detections],
            "elapsed_sec": round(elapsed, 2),
        }, f, indent=2)

    log.info(f"  Towers: {len(detections)}  elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    detect_power_towers(RGB_PATH, OUTPUT_DIR)
