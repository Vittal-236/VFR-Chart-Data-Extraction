"""
ITCiD — Isosceles Triangle Circle Detection (Test Script)
VFR Chart Extraction Pipeline

Based on:
    Zhang, Wiklund & Andersson (2016).
    "A fast and robust circle detection method using isosceles triangles sampling."
    Pattern Recognition, 54:218-228.

Core idea
---------
For any two edge points A and B lying on the same circle, the centre O is
equidistant from both — forming an isosceles triangle AOB.
Geometric property: the perpendicular bisector of AB passes through O.
By checking that the gradient at A and B point symmetrically toward the
midpoint of AB (the IT criterion), irrelevant edge pairs are rejected
before any voting happens — this is the false-positive suppression mechanism.

Algorithm
---------
  1. Canny edge detection on greyscale tile
  2. Compute gradient orientation map (Sobel)
  3. Label connected edge components; skip tiny fragments
  4. For each component, randomly sample edge-point pairs
     a. Distance filter: chord length must be in [RADIUS_MIN, RADIUS_MAX*2]
     b. IT criterion: gradients at A and B must point toward the AB midpoint
        within GRADIENT_ANGLE_TOL degrees
     c. Estimate centre: perpendicular bisector of AB intersected with
        the gradient line from A
     d. Vote for that centre in a 2D accumulator grid
  5. Peak detection on the accumulator → candidate centres
  6. Radius estimation: sweep radii and pick the one with highest Canny coverage
  7. Coverage threshold: confirm only if enough of the circumference has edge pixels

How to use
----------
1. Set BINARY_PATH and CIRCLES_JSON below (from circle_detector_v4 outputs).
2. Optionally set TEST_TILES to define which map regions to test.
3. Run: python itcid_test.py
4. Outputs go to OUTPUT_DIR:
     itcid_<tile>.png — 4-panel visualisation per tile
       Panel TL: greyscale tile
       Panel TR: Canny edges
       Panel BL: vote accumulator (brighter = more votes)
       Panel BR: detections (green=ITCiD, yellow=Hough reference)
     summary.txt     — precision/recall vs Hough reference circles

Requirements
------------
pip install numpy pillow scikit-image scipy
"""

import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage import feature, measure
from skimage.feature import peak_local_max

Image.MAX_IMAGE_PIXELS = None


# =============================================================================
# CONFIG — set these paths
# =============================================================================

# The annotated_map.png or Washington_binary.png from circle_detector_v4
# If using annotated_map.png (greyscale chart image): set IS_BINARY = False
# If using the clean binary PNG:                       set IS_BINARY = True
IMAGE_PATH   = r"D:\Cognida Internship\VFR extraction\outputs\phase1_preprocessing\Washington_binary.png"
IS_BINARY    = True   # True if image is already a clean binary (0/255 only)

# circles.json from circle_detector_v4 — used as reference for recall measurement
CIRCLES_JSON = r"outputs/phase4_symbol_detection/circle_detector_v4/circles.json"

OUTPUT_DIR   = r"outputs/phase4_symbol_detection/itcid_test"

# Three map regions to test. Format: (label, row_start, col_start, row_end, col_end)
# Adjust these to match where confirmed circles appear in your image.
TEST_TILES = [
    ("tile_A",  700,  150, 1100,  750),   # dense known-circle region
    ("tile_B", 1300, 3450, 1700, 3950),   # mid-chart region
    ("tile_C",  400, 5400,  900, 5850),   # right-side region
]

RANDOM_SEED = 42


# =============================================================================
# TUNING PARAMETERS
# =============================================================================

# Radius search range in pixels (at 150 DPI, circle symbols are ~17-26px radius)
RADIUS_MIN = 15
RADIUS_MAX = 28

# Canny sigma — lower = sharper edges, better for thin binary strokes
# Higher = smoother, better for noisy/compressed images
CANNY_SIGMA = 0.8

# Minimum edge component length — tiny fragments are noise, skip them
MIN_CC_LEN = 15

# Number of random pairs sampled per edge component
# More samples = better coverage but slower
N_SAMPLES = 1000

# Isosceles triangle criterion — how closely the gradient at each edge point
# must align with the direction toward the AB midpoint.
# Lower = stricter (fewer IT-valid pairs, faster but may miss circles)
# Higher = more lenient (more pairs pass, better recall, more noise)
GRADIENT_ANGLE_TOL = 35  # degrees

# Accumulator grid cell size — coarser grid = faster but less precise centres
VOTE_GRID_CELL = 3  # pixels

# Minimum votes for a peak to be considered a circle candidate
MIN_VOTES = 5

# Minimum distance between peaks (prevents double-counting one circle)
CLUSTER_DIST = 20  # pixels

# Minimum fraction of sampled circumference points that must be on Canny edges
# to confirm a circle. Lower for compressed/blurry images, higher for clean binary.
# On clean binary: use 0.50+
# On compressed chart image: use 0.12-0.20
MIN_COVERAGE = 0.45

# Match radius: a detected circle is considered a true positive if its centre
# is within this many pixels of a reference circle from circles.json
MATCH_RADIUS = 15  # pixels


# =============================================================================
# UTILITIES
# =============================================================================

def load_image(path: str, is_binary: bool) -> np.ndarray:
    """
    Load image as float [0,1] for Canny and gradient computation.
    If is_binary: treat white pixels as ink (invert so ink=1.0).
    """
    img = Image.open(path).convert("L")
    arr = np.array(img).astype(float) / 255.0
    return arr


def compute_angle_map(gray: np.ndarray) -> np.ndarray:
    """Sobel gradient orientation in degrees, range [-180, 180]."""
    gx = ndimage.sobel(gray, axis=1)
    gy = ndimage.sobel(gray, axis=0)
    return np.degrees(np.arctan2(gy, gx))


def angle_diff(a1: float, a2: float) -> float:
    """Smallest unsigned angular difference between two angles in degrees."""
    d = abs(a1 - a2) % 360
    return min(d, 360 - d)


def perpendicular_bisector_centre(ax, ay, bx, by, ga_deg):
    """
    Estimate circle centre from two edge points A=(ax,ay) and B=(bx,by).

    The centre lies on the perpendicular bisector of AB.
    We additionally use the gradient direction at A (ga_deg) to resolve
    which point on the bisector is the centre:
      Centre = intersection of:
        - Perpendicular bisector of AB
        - Gradient line through A (gradient points toward centre for convex arcs)

    Returns (cx, cy) or None if lines are parallel (degenerate case).
    """
    mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return None

    # Perpendicular bisector direction (rotated 90° from AB)
    px, py = -dy / length, dx / length

    # Gradient direction at A
    gx = math.cos(math.radians(ga_deg))
    gy = math.sin(math.radians(ga_deg))

    # Solve intersection of perpendicular bisector and gradient line from A
    denom = px * gy - py * gx
    if abs(denom) < 1e-6:
        return None  # lines are parallel

    t = ((ax - mx) * gy - (ay - my) * gx) / denom
    return mx + t * px, my + t * py


def it_criterion(ax, ay, bx, by, angle_map, tol_deg):
    """
    Isosceles Triangle criterion.

    For points A and B on a circle, the gradients at A and B should point
    roughly toward the midpoint M of AB (i.e., inward toward the centre
    of curvature). This rejects edge pairs from straight lines, corners,
    and unrelated structures that happen to be at the right distance.

    Checks:
      - gradient at A vs direction A→M: must be within tol_deg
      - gradient at B vs direction B→M: must be within tol_deg
      (Both the gradient and its 180° opposite are accepted, since
       gradient direction depends on which side of the stroke we're on.)

    Returns True if the pair passes the IT criterion.
    """
    ga = angle_map[ay, ax]
    gb = angle_map[by, bx]
    mx, my = (ax + bx) / 2.0, (ay + by) / 2.0

    dir_a = math.degrees(math.atan2(my - ay, mx - ax))
    dir_b = math.degrees(math.atan2(my - by, mx - bx))

    # Allow gradient or its 180° opposite (stroke can be traversed either way)
    diff_a = angle_diff(ga, dir_a)
    diff_b = angle_diff(gb, dir_b)
    ok_a = min(diff_a, 180 - diff_a) < tol_deg
    ok_b = min(diff_b, 180 - diff_b) < tol_deg

    return ok_a and ok_b


# =============================================================================
# MAIN DETECTOR
# =============================================================================

def detect_circles_itcid(gray_tile: np.ndarray, label: str = "") -> tuple:
    """
    Run ITCiD on a greyscale tile.

    Returns:
        confirmed  list of dicts: {cx, cy, r, cov, votes}
        edges      bool array — Canny edge map
        acc        int array  — vote accumulator
    """
    H, W = gray_tile.shape
    print(f"\n  [{label}]  {W} × {H} px")

    # Step 1: Canny edge detection
    edges = feature.canny(gray_tile, sigma=CANNY_SIGMA)
    print(f"    Canny edges   : {edges.sum()} px  ({edges.mean()*100:.1f}%)")

    # Step 2: Gradient orientation map
    angle_map = compute_angle_map(gray_tile)

    # Step 3: Connected edge components (filter tiny fragments)
    lbl_map = measure.label(edges, connectivity=2)
    comps   = [p.coords for p in measure.regionprops(lbl_map)
               if len(p.coords) >= MIN_CC_LEN]
    print(f"    Edge comps    : {len(comps)}  (len >= {MIN_CC_LEN})")

    # Step 4: Random pair sampling + IT criterion + centre voting
    acc       = np.zeros((H // VOTE_GRID_CELL + 1,
                          W // VOTE_GRID_CELL + 1), dtype=np.int32)
    it_count  = 0
    pair_count = 0

    for comp in comps:
        n = len(comp)
        if n < 4:
            continue

        n_pairs = min(N_SAMPLES, n * (n - 1) // 2)
        for _ in range(n_pairs):
            pair_count += 1
            i, j   = random.sample(range(n), 2)
            ay, ax  = int(comp[i][0]), int(comp[i][1])
            by_, bx = int(comp[j][0]), int(comp[j][1])

            # Distance filter: chord must be plausible for our radius range
            dist = math.hypot(bx - ax, by_ - ay)
            if dist < RADIUS_MIN or dist > RADIUS_MAX * 2:
                continue

            # IT criterion: gradients must point toward midpoint
            if not it_criterion(ax, ay, bx, by_, angle_map, GRADIENT_ANGLE_TOL):
                continue

            it_count += 1

            # Estimate centre from perpendicular bisector
            result = perpendicular_bisector_centre(
                ax, ay, bx, by_, angle_map[ay, ax])
            if result is None:
                continue

            cx_est, cy_est = result

            # Bounds check
            if not (0 <= cx_est < W and 0 <= cy_est < H):
                continue

            # Accumulate vote
            gi = int(cy_est) // VOTE_GRID_CELL
            gj = int(cx_est) // VOTE_GRID_CELL
            if 0 <= gi < acc.shape[0] and 0 <= gj < acc.shape[1]:
                acc[gi, gj] += 1

    print(f"    Pairs sampled : {pair_count}")
    print(f"    IT-valid pairs: {it_count}  ({it_count/max(pair_count,1)*100:.1f}%)")
    print(f"    Acc max votes : {acc.max()}")

    # Step 5: Peak detection on accumulator
    peaks = peak_local_max(
        acc.astype(float),
        min_distance=max(1, CLUSTER_DIST // VOTE_GRID_CELL),
        threshold_abs=MIN_VOTES,
    )
    print(f"    Vote peaks    : {len(peaks)}")

    # Step 6: Radius estimation + coverage verification
    confirmed = []
    for peak in peaks:
        cy_a = int(peak[0] * VOTE_GRID_CELL + VOTE_GRID_CELL // 2)
        cx_a = int(peak[1] * VOTE_GRID_CELL + VOTE_GRID_CELL // 2)

        best_r, best_cov = RADIUS_MIN, 0.0
        for r in range(RADIUS_MIN, RADIUS_MAX + 1):
            N_circ = max(24, int(2 * math.pi * r))
            hits = sum(
                1 for k in range(N_circ)
                if 0 <= (rr := int(round(cy_a + r * math.sin(
                    2 * math.pi * k / N_circ)))) < H
                and 0 <= (cc := int(round(cx_a + r * math.cos(
                    2 * math.pi * k / N_circ)))) < W
                and edges[rr, cc]
            )
            cov = hits / N_circ
            if cov > best_cov:
                best_cov = cov
                best_r   = r

        if best_cov >= MIN_COVERAGE:
            confirmed.append({
                "cx":    cx_a,
                "cy":    cy_a,
                "r":     best_r,
                "cov":   round(best_cov, 3),
                "votes": int(acc[peak[0], peak[1]]),
            })

    print(f"    Confirmed     : {len(confirmed)}  (cov >= {MIN_COVERAGE})")
    return confirmed, edges, acc


# =============================================================================
# VISUALISATION
# =============================================================================

def visualise_tile(gray_tile, edges, acc, confirmed, ref_circles,
                   tile_r0, tile_c0):
    """
    4-panel visualisation:
      TL: original greyscale tile
      TR: Canny edge map
      BL: vote accumulator (normalised to 0-255)
      BR: detections — green=ITCiD, yellow=Hough reference from circles.json
    """
    H, W = gray_tile.shape

    pTL = Image.fromarray((gray_tile * 255).astype(np.uint8)).convert("RGB")
    pTR = Image.fromarray((edges * 255).astype(np.uint8)).convert("RGB")

    # Accumulator — normalise and upscale to tile size
    acc_norm = (acc / max(acc.max(), 1) * 255).astype(np.uint8)
    pBL = Image.fromarray(acc_norm).resize((W, H), Image.NEAREST).convert("RGB")

    # Detection overlay
    pBR = pTL.copy()
    draw = ImageDraw.Draw(pBR)

    # Yellow rings: Hough reference circles (from circles.json)
    for rc in ref_circles:
        ky = rc["row"] - tile_r0
        kx = rc["col"] - tile_c0
        kr = rc["radius"]
        draw.ellipse([kx-kr-1, ky-kr-1, kx+kr+1, ky+kr+1],
                     outline=(255, 220, 0), width=1)

    # Green rings: ITCiD detections
    for d in confirmed:
        cx, cy, r = d["cx"], d["cy"], d["r"]
        draw.ellipse([cx-r-2, cy-r-2, cx+r+2, cy+r+2],
                     outline=(0, 255, 0), width=2)
        draw.ellipse([cx-2, cy-2, cx+2, cy+2], fill=(255, 0, 0))
        draw.text((cx + r + 3, cy - 5),
                  f"v={d['votes']} c={d['cov']}",
                  fill=(100, 255, 100))

    # Combine 2×2
    combined = Image.new("RGB", (W * 2, H * 2), (20, 20, 20))
    combined.paste(pTL, (0, 0))
    combined.paste(pTR, (W, 0))
    combined.paste(pBL, (0, H))
    combined.paste(pBR, (W, H))

    # Panel labels
    draw_comb = ImageDraw.Draw(combined)
    for text, pos in [
        ("Original",    (4,       4)),
        ("Canny edges", (W + 4,   4)),
        ("Vote acc.",   (4,       H + 4)),
        ("Detections (green=ITCiD  yellow=Hough)", (W + 4, H + 4)),
    ]:
        draw_comb.text(pos, text, fill=(200, 200, 200))

    return combined


# =============================================================================
# MAIN
# =============================================================================

def main():
    random.seed(RANDOM_SEED)

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load image
    print(f"Loading image: {IMAGE_PATH}")
    gray_full = load_image(IMAGE_PATH, IS_BINARY)
    H, W = gray_full.shape
    print(f"  Size: {W} × {H} px\n")

    # Load reference circles from Hough detector
    with open(CIRCLES_JSON) as f:
        ref_data = json.load(f)
    ref_circles = ref_data["circles"]
    print(f"Reference circles (Hough): {len(ref_circles)}\n")

    # Run ITCiD on each test tile
    summary_lines = [
        "ITCiD Test Summary",
        "=" * 60,
        f"Image     : {IMAGE_PATH}",
        f"Reference : {CIRCLES_JSON}",
        f"Params    : CANNY_SIGMA={CANNY_SIGMA}  TOL={GRADIENT_ANGLE_TOL}°  "
        f"MIN_COV={MIN_COVERAGE}  MIN_VOTES={MIN_VOTES}",
        "",
    ]

    total_detected  = 0
    total_ref       = 0
    total_tp        = 0

    for tile_label, r0, c0, r1, c1 in TEST_TILES:
        tile_gray = gray_full[r0:r1, c0:c1]

        # Reference circles in this tile
        refs_in = [rc for rc in ref_circles
                   if r0 < rc["row"] < r1 and c0 < rc["col"] < c1]

        # Run ITCiD
        confirmed, edges, acc = detect_circles_itcid(tile_gray, tile_label)

        # Precision / Recall vs Hough reference
        tp = sum(
            1 for d in confirmed
            if any(
                math.hypot(d["cx"] - (rc["col"] - c0),
                           d["cy"] - (rc["row"] - r0)) <= MATCH_RADIUS
                for rc in refs_in
            )
        )
        fp       = len(confirmed) - tp
        fn       = len(refs_in)   - min(tp, len(refs_in))
        precision = tp / max(len(confirmed), 1)
        recall    = tp / max(len(refs_in),   1)

        total_detected += len(confirmed)
        total_ref      += len(refs_in)
        total_tp       += tp

        # Per-tile summary
        line = (f"{tile_label:12s}  "
                f"detected={len(confirmed):3d}  "
                f"ref={len(refs_in):2d}  "
                f"TP={tp}  FP={fp}  FN={fn}  "
                f"P={precision:.2f}  R={recall:.2f}")
        print(f"\n  {line}")
        summary_lines.append(line)

        # Missed reference circles
        missed = [rc for rc in refs_in if not any(
            math.hypot(d["cx"] - (rc["col"] - c0),
                       d["cy"] - (rc["row"] - r0)) <= MATCH_RADIUS
            for d in confirmed
        )]
        if missed:
            summary_lines.append(f"    Missed: {[(rc['row'],rc['col']) for rc in missed]}")

        # Save visualisation
        vis = visualise_tile(tile_gray, edges, acc, confirmed,
                             refs_in, r0, c0)
        vis_path = str(out_dir / f"itcid_{tile_label}.png")
        vis.save(vis_path)
        print(f"  Saved: {vis_path}")

    # Overall summary
    overall_p = total_tp / max(total_detected, 1)
    overall_r = total_tp / max(total_ref,      1)
    summary_lines += [
        "",
        "=" * 60,
        f"OVERALL   detected={total_detected}  ref={total_ref}  "
        f"TP={total_tp}  P={overall_p:.2f}  R={overall_r:.2f}",
        "",
        "Notes:",
        "  - Reference circles are from circle_detector_v4 (Hough method).",
        "  - ITCiD is run on a compressed chart image, not the clean binary.",
        "  - On the clean binary, coverage and recall should be significantly higher.",
        "  - Tuning: raise MIN_COVERAGE or MIN_VOTES to cut false positives.",
        "  - Tuning: lower GRADIENT_ANGLE_TOL to make IT criterion stricter.",
    ]

    summary_path = out_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))

    print("\n" + "=" * 55)
    print("ITCiD TEST COMPLETE")
    print("=" * 55)
    for line in summary_lines:
        print(" ", line)
    print(f"\n  Output: {out_dir.resolve()}")


if __name__ == "__main__":
    main()