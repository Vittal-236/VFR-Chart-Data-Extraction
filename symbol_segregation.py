"""
Symbol Segregation — Phase 2 Full
VFR Chart Extraction Pipeline (FAA Base Model)

Takes the Phase 1 binary mask (white ink on black) and Phase 1 RGB image
as input and produces separate binary mask files for each symbol category.

Each output mask is a PNG where:
    White (255) = pixels belonging to that symbol category
    Black  (0)  = everything else

Symbol categories produced
--------------------------
  01_obstacles.png          — single + double obstacle symbols (∧ + dot)
  02_airports_towered.png   — blue filled circle airports
  03_airports_nontowered.png— magenta filled circle airports
  04_airports_large.png     — large runway polygons (> 8069 ft)
  05_letter_in_circle.png   — R/H/U/F/X in circle symbols
  06_vor.png                — VOR hexagon symbols (blue)
  07_dme.png                — DME square symbols (blue)
  08_vortac.png             — VORTAC hexagon+spurs (blue)
  09_ndb.png                — NDB stippled circles (magenta)
  10_waypoints.png          — VFR waypoint four-pointed stars (magenta)
  11_airspace_classB.png    — Class B solid blue boundary lines
  12_airspace_classC.png    — Class C solid magenta boundary lines
  13_airspace_classD.png    — Class D dashed blue boundary lines
  14_airspace_classE.png    — Class E dashed magenta lines
  15_special_use.png        — MOA/Restricted/Prohibited hatched areas
  16_airways.png            — Victor airways and MTR route lines
  17_text_labels.png        — all text annotation blobs
  composite_overview.png    — colour-coded composite of all layers

Method per category
-------------------
  Colour-based  (uses RGB): airports, NAVAIDs, airspace boundaries, airways
  Shape-based   (uses binary): obstacles (NCC), letter-in-circle (PB3),
                               waypoints (star template), text blobs (CC filter)
  Combined      (colour + shape): airports (colour mask → shape filter)

HOW TO USE
----------
1. Set BINARY_PATH, RGB_PATH, OUTPUT_DIR below.
2. python symbol_segregation.py
3. Each output PNG is a mask ready to feed into the relevant detector.

Requirements: numpy, pillow, scikit-image, scipy
"""

import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage import feature, filters, measure, morphology
from skimage.color import rgb2hsv
from skimage.feature import match_template, peak_local_max

Image.MAX_IMAGE_PIXELS = None


# =============================================================================
# CONFIG
# =============================================================================

BINARY_PATH = r"outputs/phase1_preprocessing/Washington_binary.png"
RGB_PATH    = r"outputs/phase1_preprocessing/Washington_rgb_150dpi.png"
OUTPUT_DIR  = r"outputs/phase2_segregation"

DPI = 150   # must match Phase 1 DPI

# Legend strip — left side of chart, suppress all detections here
LEGEND_STRIP_WIDTH = 140   # px

# Tile size for all tiled operations
TILE_SIZE = 2048

RANDOM_SEED = 42


# =============================================================================
# HELPERS
# =============================================================================

def load_binary(path):
    """Load binary PNG → bool array (H, W). True = ink."""
    return np.array(Image.open(path).convert("L")) > 128

def load_rgb(path):
    """Load RGB PNG → uint8 array (H, W, 3)."""
    return np.array(Image.open(path).convert("RGB"))

def save_mask(mask_bool, path):
    """Save bool array as binary PNG (True=white)."""
    img = mask_bool.astype(np.uint8)
    img *= 255

    Image.fromarray(img, mode="L").save(path)  
    coverage = mask_bool.mean() * 100
    print(f"  Saved: {path}  ({coverage:.2f}% coverage)")

def colour_mask_tiled(rgb, hue_lo, hue_hi, sat_min, val_min,
                      hue_lo2=None, hue_hi2=None):
    """
    Build a binary mask for pixels in an HSV colour range.
    Processed in tiles to limit RAM.
    Optionally supports a second hue range (for magenta that wraps around red).
    """
    H, W = rgb.shape[:2]
    mask = np.zeros((H, W), dtype=bool)
    for tr in range(0, H, TILE_SIZE):
        for tc in range(0, W, TILE_SIZE):
            tile = rgb[tr:min(H, tr+TILE_SIZE),
                       tc:min(W, tc+TILE_SIZE)].astype(np.float64) / 255.0
            hsv = rgb2hsv(tile)
            h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
            hue_ok = (h >= hue_lo) & (h <= hue_hi)
            if hue_lo2 is not None:
                hue_ok |= (h >= hue_lo2) & (h <= hue_hi2)
            tile_mask = hue_ok & (s >= sat_min) & (v >= val_min)
            mask[tr:min(H,tr+TILE_SIZE), tc:min(W,tc+TILE_SIZE)] = tile_mask
    return mask

def blob_filter(mask, min_area, max_area, min_solidity=0.0,
                max_eccentricity=1.0):
    """
    Label connected components in a mask and keep only those that
    satisfy area, solidity, and eccentricity constraints.
    Processed in tiles with overlap to handle large images.
    """
    H, W = mask.shape
    out  = np.zeros((H, W), dtype=bool)
    seen = set()
    overlap = 60

    for tr in range(0, H, TILE_SIZE):
        for tc in range(0, W, TILE_SIZE):
            r0 = max(0, tr - overlap)
            c0 = max(0, tc - overlap)
            r1 = min(H, tr + TILE_SIZE + overlap)
            c1 = min(W, tc + TILE_SIZE + overlap)
            inner_r0, inner_c0 = tr, tc
            inner_r1 = min(tr + TILE_SIZE, H)
            inner_c1 = min(tc + TILE_SIZE, W)

            tile  = mask[r0:r1, c0:c1]
            lbl   = measure.label(tile, connectivity=2)

            for prop in measure.regionprops(lbl):
                if not (min_area <= prop.area <= max_area):
                    continue
                if prop.solidity < min_solidity:
                    continue
                if prop.eccentricity > max_eccentricity:
                    continue
                cr = int(round(prop.centroid[0])) + r0
                cc_cent = int(round(prop.centroid[1])) + c0
                if not (inner_r0 <= cr < inner_r1 and
                        inner_c0 <= cc_cent < inner_c1):
                    continue
                if cc_cent < LEGEND_STRIP_WIDTH:
                    continue
                key = (cr // 3, cc_cent // 3)
                if key in seen:
                    continue
                seen.add(key)
                # Paint the blob into the output
                for rr, cc in prop.coords:
                    out[rr + r0, cc + c0] = True
    return out


# =============================================================================
# 01 — OBSTACLES (open ∧ + dot)
# NCC template matching on binary map. Exact same logic as phase4a.
# =============================================================================

def _build_triangle_template(width_px):
    h = int(round(width_px * 126 / 118))
    tmpl = np.zeros((h, width_px), dtype=np.float32)
    stroke = 1 if width_px < 24 else 2
    apex_c = (width_px - 1) / 2.0
    def arm(r0, c0, r1, c1):
        steps = max(abs(r1-r0), abs(c1-c0), 1)
        for i in range(int(steps)+1):
            t = i/steps
            r = int(round(r0 + t*(r1-r0)))
            c = int(round(c0 + t*(c1-c0)))
            for dr in range(-stroke+1, stroke):
                for dc in range(-stroke+1, stroke):
                    rr, cc = r+dr, c+dc
                    if 0<=rr<h and 0<=cc<width_px:
                        tmpl[rr,cc] = 1.0
    arm(0, apex_c, h-1, 0.0)
    arm(0, apex_c, h-1, float(width_px-1))
    return tmpl

def segregate_obstacles(binary):
    """
    NCC template matching for obstacle ∧ symbols.
    Returns binary mask with pixels belonging to obstacle symbols.
    """
    print("  Obstacles: NCC template matching ...")
    H, W    = binary.shape
    widths  = [24, 28, 32]
    tmpls   = {w: _build_triangle_template(w) for w in widths}
    out     = np.zeros((H, W), dtype=bool)
    overlap = 64

    for tr in range(0, H, TILE_SIZE):
        for tc in range(0, W, TILE_SIZE):
            inner_r0, inner_c0 = tr, tc
            inner_r1 = min(tr+TILE_SIZE, H)
            inner_c1 = min(tc+TILE_SIZE, W)
            pad_r0 = max(0, tr-overlap); pad_c0 = max(0, tc-overlap)
            pad_r1 = min(H, tr+TILE_SIZE+overlap)
            pad_c1 = min(W, tc+TILE_SIZE+overlap)
            tile = binary[pad_r0:pad_r1, pad_c0:pad_c1].astype(np.float32)

            for w, tmpl in tmpls.items():
                th = tmpl.shape[0]
                if tile.shape[0] < th or tile.shape[1] < w:
                    continue
                corr  = match_template(tile, tmpl, pad_input=False)
                peaks = peak_local_max(corr, min_distance=10,
                                       threshold_abs=0.62)
                for pr, pc in peaks:
                    map_r = pr + pad_r0
                    map_c = pc + pad_c0
                    if not (inner_r0 <= map_r < inner_r1 and
                            inner_c0 <= map_c < inner_c1):
                        continue
                    if map_c < LEGEND_STRIP_WIDTH:
                        continue
                    # Paint a box around the detected symbol
                    r0b = max(0, map_r)
                    r1b = min(H, map_r + th + 6)   # +6 for dot below
                    c0b = max(0, map_c)
                    c1b = min(W, map_c + w)
                    out[r0b:r1b, c0b:c1b] |= binary[r0b:r1b, c0b:c1b]

    return out


# =============================================================================
# 02, 03, 04 — AIRPORTS
# Colour mask (blue/magenta) + compact blob filter.
# =============================================================================

def segregate_airports_towered(rgb):
    """Blue filled circle airports (towered)."""
    print("  Airports (towered): blue colour mask ...")
    blue = colour_mask_tiled(rgb,
        hue_lo=0.55, hue_hi=0.72, sat_min=0.30, val_min=0.25)
    # Compact blobs: area 20-2500 px², solidity >= 0.40, eccentricity <= 0.90
    return blob_filter(blue, min_area=20, max_area=2500,
                       min_solidity=0.40, max_eccentricity=0.90)

def segregate_airports_nontowered(rgb):
    """Magenta filled circle airports (non-towered)."""
    print("  Airports (non-towered): magenta colour mask ...")
    mag = colour_mask_tiled(rgb,
        hue_lo=0.82, hue_hi=1.00, sat_min=0.30, val_min=0.25,
        hue_lo2=0.00, hue_hi2=0.05)
    return blob_filter(mag, min_area=20, max_area=2500,
                       min_solidity=0.40, max_eccentricity=0.90)

def segregate_airports_large(rgb):
    """
    Large airports (runway > 8069 ft) — polygon shapes drawn to scale.
    These are larger, more irregular blobs in blue or magenta.
    """
    print("  Airports (large): polygon blobs ...")
    blue = colour_mask_tiled(rgb,
        hue_lo=0.55, hue_hi=0.72, sat_min=0.25, val_min=0.20)
    mag  = colour_mask_tiled(rgb,
        hue_lo=0.82, hue_hi=1.00, sat_min=0.25, val_min=0.20,
        hue_lo2=0.00, hue_hi2=0.05)
    combined = blue | mag
    return blob_filter(combined, min_area=2500, max_area=50000,
                       min_solidity=0.25, max_eccentricity=0.98)


# =============================================================================
# 05 — LETTER-IN-CIRCLE (R, H, U, F, X, etc.)
# PB3 geometry: 3-point perpendicular bisector clustering.
# Returns a mask with the ring + interior of each confirmed circle.
# =============================================================================

def _perp_bisect(ax, ay, bx, by, cx, cy):
    mx1, my1 = (ax+bx)/2.0, (ay+by)/2.0
    mx2, my2 = (bx+cx)/2.0, (by+cy)/2.0
    dx1, dy1 = -(by-ay), (bx-ax)
    dx2, dy2 = -(cy-by), (cx-bx)
    denom = dx1*(-dy2) - (-dx2)*dy1
    if abs(denom) < 1e-8:
        return None
    t = ((mx2-mx1)*(-dy2) - (-dx2)*(my2-my1)) / denom
    return mx1+t*dx1, my1+t*dy1

def _circ_coverage(edges, cx, cy, r, H, W):
    N = max(24, int(2*math.pi*r))
    hits = sum(1 for k in range(N)
               if 0 <= (rr:=int(round(cy+r*math.sin(2*math.pi*k/N)))) < H
               and 0 <= (cc:=int(round(cx+r*math.cos(2*math.pi*k/N)))) < W
               and edges[rr, cc])
    return hits/N

def segregate_letter_in_circle(binary):
    """
    PB3 detection of closed rings containing letters.
    Returns mask with the ring pixels of each confirmed circle.
    """
    print("  Letter-in-circle: PB3 detector ...")
    from collections import defaultdict
    random.seed(RANDOM_SEED)

    H, W     = binary.shape
    out      = np.zeros((H, W), dtype=bool)
    seen     = set()
    overlap  = 38   # RADIUS_MAX + 10

    RADIUS_MIN = 15; RADIUS_MAX = 28
    MIN_CHORD = 8; N_TRIPLETS = 1000
    CLUSTER_CELL = 4; MIN_CLUSTER = 80; MIN_COV = 0.25
    MIN_CC_LEN = 12; CANNY_SIGMA = 1.0

    for tr in range(0, H, TILE_SIZE):
        for tc in range(0, W, TILE_SIZE):
            inner_r0, inner_c0 = tr, tc
            inner_r1 = min(tr+TILE_SIZE, H)
            inner_c1 = min(tc+TILE_SIZE, W)
            pad_r0 = max(0, tr-overlap); pad_c0 = max(0, tc-overlap)
            pad_r1 = min(H, tr+TILE_SIZE+overlap)
            pad_c1 = min(W, tc+TILE_SIZE+overlap)

            tile_gray = binary[pad_r0:pad_r1,
                               pad_c0:pad_c1].astype(float)
            th, tw = tile_gray.shape
            edges = feature.canny(tile_gray, sigma=CANNY_SIGMA)

            lbl_map = measure.label(edges, connectivity=2)
            comps   = [p.coords for p in measure.regionprops(lbl_map)
                       if len(p.coords) >= MIN_CC_LEN]

            estimates = []
            for comp in comps:
                n = len(comp)
                if n < 6: continue
                n_s = min(N_TRIPLETS, n*(n-1)*(n-2)//6)
                for _ in range(n_s):
                    i,j,k = random.sample(range(n), 3)
                    ay,ax   = int(comp[i][0]), int(comp[i][1])
                    by_,bx  = int(comp[j][0]), int(comp[j][1])
                    cy_,cx_ = int(comp[k][0]), int(comp[k][1])
                    d_ab = math.hypot(bx-ax,   by_-ay)
                    d_bc = math.hypot(cx_-bx,  cy_-by_)
                    d_ac = math.hypot(cx_-ax,  cy_-ay)
                    if min(d_ab,d_bc,d_ac)<MIN_CHORD: continue
                    if max(d_ab,d_bc,d_ac)>RADIUS_MAX*2: continue
                    res = _perp_bisect(ax,ay,bx,by_,cx_,cy_)
                    if res is None: continue
                    ox,oy = res
                    r_est = math.hypot(ox-ax, oy-ay)
                    if not (RADIUS_MIN<=r_est<=RADIUS_MAX): continue
                    if not (0<=ox<tw and 0<=oy<th): continue
                    estimates.append((ox, oy, r_est))

            if not estimates: continue

            bins = defaultdict(list)
            for ox,oy,r_est in estimates:
                bins[(int(ox)//CLUSTER_CELL,
                      int(oy)//CLUSTER_CELL)].append((ox,oy,r_est))

            sorted_bins = sorted(bins.items(), key=lambda x: -len(x[1]))
            visited = set()
            for (gx,gy), pts in sorted_bins:
                if (gx,gy) in visited: continue
                if len(pts) < MIN_CLUSTER: break
                merged = list(pts)
                for dgx in [-1,0,1]:
                    for dgy in [-1,0,1]:
                        if dgx==0 and dgy==0: continue
                        nb = (gx+dgx, gy+dgy)
                        if nb in bins:
                            merged.extend(bins[nb])
                            visited.add(nb)
                visited.add((gx,gy))
                if len(merged) < MIN_CLUSTER: continue

                cx_r = float(np.median([p[0] for p in merged]))
                cy_r = float(np.median([p[1] for p in merged]))
                r_r  = float(np.median([p[2] for p in merged]))
                cx_i, cy_i, r_i = (int(round(cx_r)),
                                    int(round(cy_r)),
                                    int(round(r_r)))

                cov = _circ_coverage(edges, cx_i, cy_i, r_i, th, tw)
                if cov < MIN_COV: continue

                # Map coords
                map_cx = cx_i + pad_c0
                map_cy = cy_i + pad_r0
                if not (inner_c0 <= map_cx < inner_c1 and
                        inner_r0 <= map_cy < inner_r1): continue
                if map_cx < LEGEND_STRIP_WIDTH: continue

                key = (map_cx//4, map_cy//4)
                if key in seen: continue
                seen.add(key)

                # Paint ring pixels into output mask
                r_map = int(round(r_r))
                N = max(24, int(2*math.pi*r_map))
                for k in range(N):
                    a = 2*math.pi*k/N
                    rr = int(round(map_cy + r_map*math.sin(a)))
                    cc = int(round(map_cx + r_map*math.cos(a)))
                    for dr in range(-2, 3):
                        for dc in range(-2, 3):
                            nr, nc = rr+dr, cc+dc
                            if 0<=nr<H and 0<=nc<W:
                                out[nr, nc] = True
                # Also paint interior
                inn = max(2, r_map - 4)
                for dr in range(-inn, inn+1):
                    for dc in range(-inn, inn+1):
                        if dr*dr+dc*dc <= inn*inn:
                            nr = map_cy+dr; nc = map_cx+dc
                            if 0<=nr<H and 0<=nc<W:
                                out[nr, nc] |= binary[nr, nc]

    return out


# =============================================================================
# 06, 07, 08 — VOR / DME / VORTAC
# Colour mask (blue) + size/shape filter.
# VOR: hexagon-ish blobs, larger + spikier than airport circles.
# DME: square-ish blobs.
# VORTAC: same area as VOR but has spurs → lower solidity.
# =============================================================================

def segregate_vor(rgb):
    print("  VOR: blue blobs (hexagon, larger/spikier than airports) ...")
    blue = colour_mask_tiled(rgb,
        hue_lo=0.55, hue_hi=0.72, sat_min=0.30, val_min=0.20)
    # VOR rosette: area 200-2000 px², solidity 0.20-0.55 (spiky hexagon)
    return blob_filter(blue, min_area=200, max_area=2000,
                       min_solidity=0.20, max_eccentricity=0.80)

def segregate_dme(rgb, binary):
    print("  DME: blue square blobs ...")
    blue = colour_mask_tiled(rgb,
        hue_lo=0.55, hue_hi=0.72, sat_min=0.30, val_min=0.20)
    H, W = binary.shape
    out  = np.zeros((H, W), dtype=bool)
    seen = set()

    for tr in range(0, H, TILE_SIZE):
        for tc in range(0, W, TILE_SIZE):
            inner_r0, inner_c0 = tr, tc
            inner_r1 = min(tr+TILE_SIZE, H)
            inner_c1 = min(tc+TILE_SIZE, W)
            tile = blue[tr:inner_r1, tc:inner_c1]
            lbl  = measure.label(tile, connectivity=2)
            for prop in measure.regionprops(lbl):
                if not (30 <= prop.area <= 600): continue
                if prop.solidity < 0.55: continue  # DME is filled square
                # Aspect ratio close to 1.0 (square)
                bb = prop.bbox
                h_bb = bb[2]-bb[0]; w_bb = bb[3]-bb[1]
                if w_bb == 0: continue
                aspect = h_bb / w_bb
                if not (0.60 <= aspect <= 1.60): continue
                cr = int(round(prop.centroid[0])) + tr
                cc_c = int(round(prop.centroid[1])) + tc
                if not (inner_r0<=cr<inner_r1 and inner_c0<=cc_c<inner_c1): continue
                if cc_c < LEGEND_STRIP_WIDTH: continue
                key = (cr//3, cc_c//3)
                if key in seen: continue
                seen.add(key)
                for rr, cc in prop.coords:
                    out[rr+tr, cc+tc] = True
    return out

def segregate_vortac(rgb):
    print("  VORTAC: blue blobs with spurs (low solidity, hexagon area) ...")
    blue = colour_mask_tiled(rgb,
        hue_lo=0.55, hue_hi=0.72, sat_min=0.30, val_min=0.20)
    # VORTAC has spurs → solidity 0.10-0.30, larger area than DME
    return blob_filter(blue, min_area=150, max_area=800,
                       min_solidity=0.10, max_eccentricity=0.70)


# =============================================================================
# 09 — NDB (stippled magenta circle)
# Hough circle on magenta binary at smaller radius range.
# =============================================================================

def segregate_ndb(rgb, binary):
    print("  NDB: Hough circles on magenta channel ...")
    from skimage.transform import hough_circle, hough_circle_peaks

    H, W = binary.shape
    out  = np.zeros((H, W), dtype=bool)
    seen = set()

    mag = colour_mask_tiled(rgb,
        hue_lo=0.82, hue_hi=1.00, sat_min=0.25, val_min=0.20,
        hue_lo2=0.00, hue_hi2=0.05).astype(np.uint8)

    radii   = np.arange(12, 23)
    overlap = 32

    for tr in range(0, H, TILE_SIZE):
        for tc in range(0, W, TILE_SIZE):
            inner_r0, inner_c0 = tr, tc
            inner_r1 = min(tr+TILE_SIZE, H)
            inner_c1 = min(tc+TILE_SIZE, W)
            pad_r0 = max(0, tr-overlap); pad_c0 = max(0, tc-overlap)
            pad_r1 = min(H, tr+TILE_SIZE+overlap)
            pad_c1 = min(W, tc+TILE_SIZE+overlap)
            tile = mag[pad_r0:pad_r1, pad_c0:pad_c1]
            hspaces = hough_circle(tile, radii)
            accums, cx_arr, cy_arr, rad_arr = hough_circle_peaks(
                hspaces, radii, min_xdistance=12, min_ydistance=12,
                threshold=0.50, num_peaks=200)
            for acc, cx, cy, rad in zip(accums, cx_arr, cy_arr, rad_arr):
                map_r = int(cy) + pad_r0
                map_c = int(cx) + pad_c0
                if not (inner_r0<=map_r<inner_r1 and inner_c0<=map_c<inner_c1): continue
                if map_c < LEGEND_STRIP_WIDTH: continue
                # Check circumference coverage (stippled → lower threshold)
                N = max(24, int(2*math.pi*rad))
                hits = sum(1 for k in range(N)
                           if 0<=(rr:=int(round(map_r+rad*math.sin(2*math.pi*k/N))))<H
                           and 0<=(cc:=int(round(map_c+rad*math.cos(2*math.pi*k/N))))<W
                           and mag[rr,cc])
                if hits/N < 0.35: continue  # stippled ring — lower threshold
                key = (map_r//4, map_c//4)
                if key in seen: continue
                seen.add(key)
                # Paint ring
                r_i = int(rad)
                for k in range(N):
                    a = 2*math.pi*k/N
                    rr = int(round(map_r+r_i*math.sin(a)))
                    cc = int(round(map_c+r_i*math.cos(a)))
                    for dr in range(-2,3):
                        for dc in range(-2,3):
                            nr,nc = rr+dr, cc+dc
                            if 0<=nr<H and 0<=nc<W: out[nr,nc] = True
    return out


# =============================================================================
# 10 — VFR WAYPOINTS (four-pointed star, magenta)
# NCC star template on magenta binary.
# =============================================================================

def _build_star_template(size):
    tmpl = np.zeros((size, size), dtype=np.float32)
    cx = cy = (size-1)/2.0
    arm = size//2
    def line(r0,c0,r1,c1):
        steps = max(abs(r1-r0),abs(c1-c0),1)
        for i in range(int(steps)+1):
            t=i/steps
            r=int(round(r0+t*(r1-r0))); c=int(round(c0+t*(c1-c0)))
            if 0<=r<size and 0<=c<size: tmpl[r,c]=1.0
    line(cy,cx,cy-arm,cx); line(cy,cx,cy+arm,cx)
    line(cy,cx,cy,cx-arm); line(cy,cx,cy,cx+arm)
    diag=int(arm*0.55)
    line(cy,cx,cy-diag,cx-diag); line(cy,cx,cy-diag,cx+diag)
    line(cy,cx,cy+diag,cx-diag); line(cy,cx,cy+diag,cx+diag)
    return tmpl

def segregate_waypoints(rgb, binary):
    print("  VFR Waypoints: NCC star template on magenta ...")
    H, W = binary.shape
    out  = np.zeros((H, W), dtype=bool)

    mag_bin = colour_mask_tiled(rgb,
        hue_lo=0.82, hue_hi=1.00, sat_min=0.30, val_min=0.25,
        hue_lo2=0.00, hue_hi2=0.05).astype(np.float32)

    sizes   = [14, 18, 22]
    tmpls   = {s: _build_star_template(s) for s in sizes}
    overlap = 32

    for tr in range(0, H, TILE_SIZE):
        for tc in range(0, W, TILE_SIZE):
            inner_r0, inner_c0 = tr, tc
            inner_r1 = min(tr+TILE_SIZE, H)
            inner_c1 = min(tc+TILE_SIZE, W)
            pad_r0=max(0,tr-overlap); pad_c0=max(0,tc-overlap)
            pad_r1=min(H,tr+TILE_SIZE+overlap)
            pad_c1=min(W,tc+TILE_SIZE+overlap)
            tile = mag_bin[pad_r0:pad_r1, pad_c0:pad_c1]
            for sz, tmpl in tmpls.items():
                if tile.shape[0]<sz or tile.shape[1]<sz: continue
                corr  = match_template(tile, tmpl, pad_input=False)
                peaks = peak_local_max(corr, min_distance=14,
                                       threshold_abs=0.45)
                for pr, pc in peaks:
                    map_r = pr+pad_r0; map_c = pc+pad_c0
                    if not (inner_r0<=map_r<inner_r1 and
                            inner_c0<=map_c<inner_c1): continue
                    if map_c < LEGEND_STRIP_WIDTH: continue
                    r0b=max(0,map_r); r1b=min(H,map_r+sz)
                    c0b=max(0,map_c); c1b=min(W,map_c+sz)
                    out[r0b:r1b, c0b:c1b] |= binary[r0b:r1b, c0b:c1b]
    return out


# =============================================================================
# 11-14 — AIRSPACE BOUNDARIES
# Colour mask → keep only long, thin, line-like blobs.
# =============================================================================

def _airspace_line_mask(colour_mask):
    """Keep only long thin blobs (lines) from a colour mask."""
    H, W = colour_mask.shape
    out  = np.zeros((H, W), dtype=bool)
    seen = set()
    for tr in range(0, H, TILE_SIZE):
        for tc in range(0, W, TILE_SIZE):
            inner_r0,inner_c0 = tr, tc
            inner_r1=min(tr+TILE_SIZE,H); inner_c1=min(tc+TILE_SIZE,W)
            tile = colour_mask[tr:inner_r1, tc:inner_c1]
            lbl  = measure.label(tile, connectivity=2)
            for prop in measure.regionprops(lbl):
                if prop.area < 50: continue          # too small
                bb = prop.bbox
                max_dim = max(bb[2]-bb[0], bb[3]-bb[1])
                if max_dim < 40: continue            # too short to be a line
                if prop.eccentricity < 0.80: continue  # not elongated enough
                cr=int(round(prop.centroid[0]))+tr
                cc_c=int(round(prop.centroid[1]))+tc
                if not (inner_r0<=cr<inner_r1 and inner_c0<=cc_c<inner_c1): continue
                if cc_c < LEGEND_STRIP_WIDTH: continue
                key = (cr//5, cc_c//5)
                if key in seen: continue
                seen.add(key)
                for rr, cc in prop.coords:
                    out[rr+tr, cc+tc] = True
    return out

def segregate_airspace_classB(rgb):
    print("  Class B: solid blue lines ...")
    blue = colour_mask_tiled(rgb,
        hue_lo=0.55, hue_hi=0.70, sat_min=0.35, val_min=0.25)
    return _airspace_line_mask(blue)

def segregate_airspace_classC(rgb):
    print("  Class C: solid magenta lines ...")
    mag = colour_mask_tiled(rgb,
        hue_lo=0.82, hue_hi=1.00, sat_min=0.35, val_min=0.25,
        hue_lo2=0.00, hue_hi2=0.05)
    return _airspace_line_mask(mag)

def segregate_airspace_classD(rgb):
    print("  Class D: dashed blue lines ...")
    blue = colour_mask_tiled(rgb,
        hue_lo=0.55, hue_hi=0.72, sat_min=0.30, val_min=0.20)
    return _airspace_line_mask(blue)

def segregate_airspace_classE(rgb):
    print("  Class E: dashed magenta lines ...")
    mag = colour_mask_tiled(rgb,
        hue_lo=0.80, hue_hi=1.00, sat_min=0.25, val_min=0.20,
        hue_lo2=0.00, hue_hi2=0.06)
    return _airspace_line_mask(mag)


# =============================================================================
# 15 — SPECIAL USE AIRSPACE (hatched boundaries)
# Both blue-hatched (Restricted/Prohibited) and magenta-hatched (MOA/Alert).
# =============================================================================

def segregate_special_use(rgb):
    print("  Special use airspace: hatched boundaries ...")
    blue = colour_mask_tiled(rgb,
        hue_lo=0.55, hue_hi=0.72, sat_min=0.25, val_min=0.20)
    mag  = colour_mask_tiled(rgb,
        hue_lo=0.80, hue_hi=1.00, sat_min=0.25, val_min=0.20,
        hue_lo2=0.00, hue_hi2=0.06)
    combined = blue | mag
    # Hatched areas are dense clusters of short diagonal line segments
    # Keep blobs that are large area but low solidity (hatch pattern)
    H, W = combined.shape
    out  = np.zeros((H, W), dtype=bool)
    seen = set()
    for tr in range(0, H, TILE_SIZE):
        for tc in range(0, W, TILE_SIZE):
            inner_r0,inner_c0=tr,tc
            inner_r1=min(tr+TILE_SIZE,H); inner_c1=min(tc+TILE_SIZE,W)
            tile=combined[tr:inner_r1,tc:inner_c1]
            lbl =measure.label(tile,connectivity=2)
            for prop in measure.regionprops(lbl):
                if prop.area < 200: continue
                if prop.solidity > 0.40: continue  # hatched = low solidity
                cr=int(round(prop.centroid[0]))+tr
                cc_c=int(round(prop.centroid[1]))+tc
                if not (inner_r0<=cr<inner_r1 and inner_c0<=cc_c<inner_c1): continue
                if cc_c<LEGEND_STRIP_WIDTH: continue
                key=(cr//5,cc_c//5)
                if key in seen: continue
                seen.add(key)
                for rr,cc in prop.coords: out[rr+tr,cc+tc]=True
    return out


# =============================================================================
# 16 — AIRWAYS (Victor airways + MTR lines)
# Long thin continuous lines — keep very long elongated blobs from binary.
# =============================================================================

def segregate_airways(binary):
    print("  Airways: long thin line blobs from binary ...")
    H, W = binary.shape
    out  = np.zeros((H, W), dtype=bool)
    seen = set()
    for tr in range(0, H, TILE_SIZE):
        for tc in range(0, W, TILE_SIZE):
            inner_r0,inner_c0=tr,tc
            inner_r1=min(tr+TILE_SIZE,H); inner_c1=min(tc+TILE_SIZE,W)
            tile=binary[tr:inner_r1,tc:inner_c1]
            lbl =measure.label(tile,connectivity=2)
            for prop in measure.regionprops(lbl):
                bb=prop.bbox
                max_dim=max(bb[2]-bb[0],bb[3]-bb[1])
                if max_dim < 100: continue           # must be very long
                if prop.eccentricity < 0.95: continue  # must be very elongated
                if prop.solidity > 0.30: continue    # lines are thin
                cr=int(round(prop.centroid[0]))+tr
                cc_c=int(round(prop.centroid[1]))+tc
                if not (inner_r0<=cr<inner_r1 and inner_c0<=cc_c<inner_c1): continue
                if cc_c<LEGEND_STRIP_WIDTH: continue
                key=(cr//5,cc_c//5)
                if key in seen: continue
                seen.add(key)
                for rr,cc in prop.coords: out[rr+tr,cc+tc]=True
    return out


# =============================================================================
# 17 — TEXT LABELS
# Small compact blobs with enclosed holes (text characters) from binary.
# =============================================================================

def segregate_text(binary):
    print("  Text labels: compact CC blobs with character-like metrics ...")
    H, W = binary.shape
    out  = np.zeros((H, W), dtype=bool)
    seen = set()
    for tr in range(0, H, TILE_SIZE):
        for tc in range(0, W, TILE_SIZE):
            inner_r0,inner_c0=tr,tc
            inner_r1=min(tr+TILE_SIZE,H); inner_c1=min(tc+TILE_SIZE,W)
            tile=binary[tr:inner_r1,tc:inner_c1]
            lbl =measure.label(tile,connectivity=2)
            for prop in measure.regionprops(lbl):
                if not (4 <= prop.area <= 500): continue
                bb=prop.bbox
                h_bb=bb[2]-bb[0]; w_bb=bb[3]-bb[1]
                if h_bb==0 or w_bb==0: continue
                # Text chars: roughly portrait, moderate solidity
                aspect=h_bb/w_bb
                if not (0.3<=aspect<=4.0): continue
                if prop.solidity < 0.25: continue
                cr=int(round(prop.centroid[0]))+tr
                cc_c=int(round(prop.centroid[1]))+tc
                if not (inner_r0<=cr<inner_r1 and inner_c0<=cc_c<inner_c1): continue
                if cc_c<LEGEND_STRIP_WIDTH: continue
                key=(cr//2,cc_c//2)
                if key in seen: continue
                seen.add(key)
                for rr,cc in prop.coords: out[rr+tr,cc+tc]=True
    return out


# =============================================================================
# COMPOSITE OVERVIEW
# Colour-code all layers into a single RGB image for visual inspection.
# =============================================================================

LAYER_COLOURS = {
    "obstacles":          (255, 100,   0),   # orange
    "airports_towered":   (  0, 150, 255),   # blue
    "airports_nontowered":(220,   0, 220),   # magenta
    "airports_large":     (  0, 200, 200),   # cyan
    "letter_in_circle":   (255, 255,   0),   # yellow
    "vor":                (  0, 255, 100),   # green
    "dme":                (  0, 200,  80),   # dark green
    "vortac":             (100, 255, 100),   # light green
    "ndb":                (255,   0, 150),   # pink
    "waypoints":          (255, 200,   0),   # gold
    "airspace_classB":    ( 80, 120, 255),   # light blue
    "airspace_classC":    (255,  80, 180),   # light magenta
    "airspace_classD":    ( 60,  80, 200),   # mid blue
    "airspace_classE":    (180,  60, 160),   # mid magenta
    "special_use":        (255,  50,  50),   # red
    "airways":            (180, 180, 180),   # grey
    "text":               (200, 200, 200),   # light grey
}

def save_composite(masks_dict, H, W, out_path):
    """Colour-coded composite of all layers."""
    composite = np.zeros((H, W, 3), dtype=np.uint8)
    for name, mask in masks_dict.items():
        if name not in LAYER_COLOURS: continue
        colour = LAYER_COLOURS[name]
        for ch, val in enumerate(colour):
            composite[:,:,ch] = np.where(mask,
                np.maximum(composite[:,:,ch], val),
                composite[:,:,ch])
    Image.fromarray(composite).save(out_path)
    print(f"  Composite overview → {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    random.seed(RANDOM_SEED)

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading images ...")
    binary = load_binary(BINARY_PATH)
    rgb    = load_rgb(RGB_PATH)
    H, W   = binary.shape
    print(f"  Binary: {W}×{H} px  ink={binary.mean()*100:.1f}%")
    print(f"  RGB:    {rgb.shape[1]}×{rgb.shape[0]} px\n")

    # Resize RGB to match binary if needed
    if rgb.shape[:2] != (H, W):
        print("  Resizing RGB to match binary ...")
        rgb = np.array(Image.fromarray(rgb).resize((W, H), Image.LANCZOS))

    masks = {}

    print("=== SEGREGATING SYMBOLS ===\n")

    # --- Point symbols ---
    print("--- POINT SYMBOLS ---")
    masks["obstacles"]           = segregate_obstacles(binary)
    masks["airports_towered"]    = segregate_airports_towered(rgb)
    masks["airports_nontowered"] = segregate_airports_nontowered(rgb)
    masks["airports_large"]      = segregate_airports_large(rgb)
    masks["letter_in_circle"]    = segregate_letter_in_circle(binary)
    masks["vor"]                 = segregate_vor(rgb)
    masks["dme"]                 = segregate_dme(rgb, binary)
    masks["vortac"]              = segregate_vortac(rgb)
    masks["ndb"]                 = segregate_ndb(rgb, binary)
    masks["waypoints"]           = segregate_waypoints(rgb, binary)

    # --- Line symbols ---
    print("\n--- LINE / BOUNDARY SYMBOLS ---")
    masks["airspace_classB"] = segregate_airspace_classB(rgb)
    masks["airspace_classC"] = segregate_airspace_classC(rgb)
    masks["airspace_classD"] = segregate_airspace_classD(rgb)
    masks["airspace_classE"] = segregate_airspace_classE(rgb)
    masks["special_use"]     = segregate_special_use(rgb)
    masks["airways"]         = segregate_airways(binary)

    # --- Text ---
    print("\n--- TEXT LABELS ---")
    masks["text"] = segregate_text(binary)

    # --- Save individual masks ---
    print("\n=== SAVING MASKS ===\n")
    filenames = {
        "obstacles":           "01_obstacles.png",
        "airports_towered":    "02_airports_towered.png",
        "airports_nontowered": "03_airports_nontowered.png",
        "airports_large":      "04_airports_large.png",
        "letter_in_circle":    "05_letter_in_circle.png",
        "vor":                 "06_vor.png",
        "dme":                 "07_dme.png",
        "vortac":              "08_vortac.png",
        "ndb":                 "09_ndb.png",
        "waypoints":           "10_waypoints.png",
        "airspace_classB":     "11_airspace_classB.png",
        "airspace_classC":     "12_airspace_classC.png",
        "airspace_classD":     "13_airspace_classD.png",
        "airspace_classE":     "14_airspace_classE.png",
        "special_use":         "15_special_use.png",
        "airways":             "16_airways.png",
        "text":                "17_text_labels.png",
    }

    for key, fname in filenames.items():
        save_mask(masks[key], str(out_dir / fname))

    # --- Composite overview ---
    print()
    save_composite(masks, H, W, str(out_dir / "composite_overview.png"))

    # --- Coverage summary ---
    print("\n=== COVERAGE SUMMARY ===")
    print(f"  {'Layer':<25}  {'Coverage':>8}  {'Ink pixels':>12}")
    print(f"  {'-'*25}  {'-'*8}  {'-'*12}")
    for key, fname in filenames.items():
        m = masks[key]
        pct = m.mean()*100
        npx = m.sum()
        print(f"  {key:<25}  {pct:7.3f}%  {npx:12,}")

    print(f"\n  Output: {out_dir.resolve()}")
    print("\nNext steps:")
    print("  01_obstacles.png      → phase4a (NCC + dot verification)")
    print("  05_letter_in_circle   → letter_reader.py (Tesseract)")
    print("  02/03_airports_*.png  → airport attribute extractor + OCR")
    print("  06-09_navaid*.png     → NAVAID coordinate extractor")
    print("  11-15_airspace*.png   → airspace boundary tracer")


if __name__ == "__main__":
    main()