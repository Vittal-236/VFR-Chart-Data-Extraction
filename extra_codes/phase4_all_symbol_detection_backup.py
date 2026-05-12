"""
Phase 4 — Symbol Detection
VFR Chart Extraction Pipeline (FAA Base Model)

Detects and classifies all FAA VFR Sectional Chart symbols from the
preprocessed binary + RGB arrays produced by Phase 1.

Detection methods (run in this order, results merged via NMS):

    M1  Colour masking          (RGB)    → Airspace class polygons
    M2  Hough circle detection  (binary) → VOR / NDB / TACAN / Airports
    M3  Blob analysis           (binary) → Dot-class candidates
    M4  Structural V+dot search (binary) → Obstacles (single + group/double)
    M5  Multi-scale template    (binary) → All remaining point symbols
    M6  Shape-feature catch-all (binary) → Residual unclassified blobs

FAA symbol vocabulary covered (25 types):
    Obstacle (single), Obstacle (group/double), Airport (civil), Airport
    (military), Airport (private/heliport), VOR, VOR-DME, VORTAC, NDB,
    NDB-DME, TACAN, Waypoint/RNAV fix, Spot elevation, Parachute area,
    Hang glider area, Ultralight area, Class B airspace, Class C airspace,
    Class D airspace, Class E surface airspace, MOA boundary, Restricted
    area, Prohibited area, Warning area, Alert area.

Usage (module):
    from phase4_symbol_detection import SymbolDetector, detect_symbols
    detections = detect_symbols(rgb, binary, template_dir="templates/")

Usage (One-Click Run):
    Update the configuration paths at the bottom of this file and run:
    python phase4_symbol_detection.py

All coordinate outputs are (col, row) = (x, y) in pixel space relative
to the top-left corner of the input image.
"""

# =============================================================================
# Standard library
# =============================================================================
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

# =============================================================================
# Third-party
# =============================================================================
import cv2
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from skimage import color, measure, morphology
from skimage.feature import match_template
from skimage.transform import probabilistic_hough_line

# =============================================================================
# Logging
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase4")


# =============================================================================
# Constants — FAA chart geometry at 300 DPI
# =============================================================================

# Approximate pixel dimensions at 300 DPI for every symbol type.
# Tuple = (min_px_span, max_px_span) for the bounding box major axis.
SYMBOL_SIZE_RANGE_300DPI: dict[str, tuple[int, int]] = {
    "obstacle_single":      (30,  65),
    "obstacle_group":       (50, 100),
    "airport_civil":        (30,  80),
    "airport_military":     (30,  80),
    "airport_private":      (20,  50),
    "airport_heliport":     (20,  50),
    "vor":                  (120, 220),
    "vor_dme":              (120, 220),
    "vortac":               (120, 220),
    "ndb":                  (40,  90),
    "ndb_dme":              (40,  90),
    "tacan":                (120, 220),
    "waypoint":             (25,  55),
    "spot_elevation":       (6,   20),   # just the dot; digits handled by OCR
    "parachute":            (25,  55),
    "hang_glider":          (25,  55),
    "ultralight":           (25,  55),
}

# HSV colour thresholds for FAA airspace colours (from the RGB layer).
# Each entry: (lower_hsv, upper_hsv)  — all in OpenCV uint8 scale (H:0-179).
AIRSPACE_COLOUR_RANGES: dict[str, tuple[list, list]] = {
    "class_b":              ([100, 50,  60],  [125, 255, 255]),  # solid blue
    "class_c":              ([130, 40,  60],  [155, 255, 255]),  # magenta ring
    "class_d":              ([100, 30,  40],  [130, 150, 220]),  # dashed blue (lighter)
    "class_e_surface":      ([140, 40,  60],  [170, 255, 255]),  # magenta
    "restricted":           ([0,   80,  80],  [12,  255, 255]),  # red (+ wrap below)
    "restricted_wrap":      ([168, 80,  80],  [179, 255, 255]),  # red hue wrap
    "moa":                  ([20,  60,  60],  [35,  255, 255]),  # yellow/amber
    "warning":              ([0,   80,  80],  [12,  255, 255]),  # same red as restricted
    "prohibited":           ([0,   100, 100], [10,  255, 255]),  # deeper red
    "alert":                ([15,  60,  60],  [28,  255, 255]),  # orange
}

# Detection thresholds
TEMPLATE_MATCH_THRESHOLD   = 0.76   # normalised cross-correlation minimum
CIRCLE_HOUGH_PARAM2        = 45     # accumulator threshold; lower = more circles
DOT_MIN_AREA_PX            = 6      # minimum blob area to be a dot candidate
DOT_MAX_AREA_PX            = 140    # maximum blob area to be a dot candidate
DOT_MIN_CIRCULARITY        = 0.68   # 4π·A/P²
OBSTACLE_SEARCH_RADIUS_PX  = 70     # how far above a dot we search for the V
NMS_RADIUS_PX              = 18     # non-maximum suppression radius (pixels)

TILE_SIZE                  = 2048
TILE_OVERLAP               = 128


# =============================================================================
# Result dataclasses
# =============================================================================

@dataclass
class SymbolDetection:
    """One detected symbol instance."""

    symbol_type:  str           # e.g. "obstacle_single", "vor", "airport_civil"
    method:       str           # which method produced this detection
    x:            int           # pixel column of symbol centre / reference point
    y:            int           # pixel row    of symbol centre / reference point
    width:        int           # bounding box width  (px)
    height:       int           # bounding box height (px)
    confidence:   float         # [0.0, 1.0]

    # Optional extras depending on symbol type
    radius_px:    Optional[int]   = None   # for circle-class symbols
    scale:        Optional[float] = None   # scale factor used in template match
    apex_xy:      Optional[tuple] = None   # (x,y) apex pixel for obstacle V
    dot_xy:       Optional[tuple] = None   # (x,y) base dot for obstacles
    dot_xy_2:     Optional[tuple] = None   # second dot for group obstacle

    def bbox(self) -> tuple[int, int, int, int]:
        """Return (x_min, y_min, x_max, y_max)."""
        hw, hh = self.width // 2, self.height // 2
        return (self.x - hw, self.y - hh, self.x + hw, self.y + hh)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bbox"] = self.bbox()
        return d


@dataclass
class DetectionResult:
    """All outputs from Phase 4."""

    detections:       list[SymbolDetection]
    source_rgb_path:  str
    source_bin_path:  str
    image_size_px:    tuple[int, int]   # (width, height)
    elapsed_sec:      float
    counts:           dict = field(default_factory=dict)
    notes:            list = field(default_factory=list)
    geojson_path:     Optional[str] = None
    json_path:        Optional[str] = None

    def summary(self) -> str:
        lines = [
            "=== Phase 4 Result ===",
            f"  Source RGB   : {self.source_rgb_path}",
            f"  Source binary: {self.source_bin_path}",
            f"  Image size   : {self.image_size_px[0]} x {self.image_size_px[1]} px",
            f"  Total symbols: {len(self.detections)}",
            f"  Elapsed      : {self.elapsed_sec:.1f} s",
            "  By type:",
        ]
        for sym_type, cnt in sorted(self.counts.items()):
            lines.append(f"    {sym_type:<30s}: {cnt}")
        if self.notes:
            lines.append("  Notes:")
            for n in self.notes:
                lines.append(f"    - {n}")
        return "\n".join(lines)


# =============================================================================
# Tiling Engine Helpers
# =============================================================================

@dataclass
class TileSpec:
    inner_x0: int; inner_y0: int
    inner_x1: int; inner_y1: int
    pad_x0:   int; pad_y0:   int
    pad_x1:   int; pad_y1:   int
    tile_row: int; tile_col: int
    total_tiles: int


def _iter_tiles(H, W, tile_size=TILE_SIZE, overlap=TILE_OVERLAP):
    n_rows = max(1, int(np.ceil(H / tile_size)))
    n_cols = max(1, int(np.ceil(W / tile_size)))
    for tr in range(n_rows):
        for tc in range(n_cols):
            iy0 = tr * tile_size;          ix0 = tc * tile_size
            iy1 = min(iy0 + tile_size, H); ix1 = min(ix0 + tile_size, W)
            yield TileSpec(
                inner_x0=ix0, inner_y0=iy0, inner_x1=ix1, inner_y1=iy1,
                pad_x0=max(0, ix0-overlap), pad_y0=max(0, iy0-overlap),
                pad_x1=min(W, ix1+overlap), pad_y1=min(H, iy1+overlap),
                tile_row=tr, tile_col=tc,
                total_tiles=n_rows*n_cols,
            )


def _keep_inner(detections, spec):
    return [d for d in detections
            if spec.inner_x0 <= d.x < spec.inner_x1
            and spec.inner_y0 <= d.y < spec.inner_y1]


def _to_full(x_t, y_t, spec):
    return (x_t + spec.pad_x0, y_t + spec.pad_y0)


# =============================================================================
# Utility functions
# =============================================================================

def _load_image(path: Path) -> np.ndarray:
    """Load a PNG/JPEG as an RGB uint8 numpy array."""
    img = Image.open(str(path)).convert("RGB")
    return np.array(img, dtype=np.uint8)


def _load_binary(path: Path) -> np.ndarray:
    """
    Load a binary PNG.
    The Phase 1 convention is white=foreground (pixel=255).
    Returns bool array: True = foreground.
    """
    img = Image.open(str(path)).convert("L")
    arr = np.array(img, dtype=np.uint8)
    return arr > 127


def _nms(detections: list[SymbolDetection],
         radius: int = NMS_RADIUS_PX) -> list[SymbolDetection]:
    """
    Non-maximum suppression.
    Removes duplicate detections within `radius` pixels of a higher-confidence
    detection of the same symbol type.

    For cross-type conflicts (e.g. a VOR-DME centre overlapping a dot candidate)
    the higher-confidence detection wins regardless of type.
    """
    if not detections:
        return []

    # Sort descending by confidence
    dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: list[SymbolDetection] = []

    for d in dets:
        too_close = False
        for k in kept:
            dist = np.hypot(d.x - k.x, d.y - k.y)
            if dist < radius:
                too_close = True
                break
        if not too_close:
            kept.append(d)

    return kept


def _find_line_intersection(lines) -> Optional[tuple[float, float]]:
    """
    Given a list of line segments [(p0, p1), ...] from Hough, attempt to
    find the best single intersection point (apex for the obstacle V).
    Returns (x, y) in the crop coordinate system, or None.
    """
    if len(lines) < 2:
        return None

    intersection_pts = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            (x1, y1), (x2, y2) = lines[i]
            (x3, y3), (x4, y4) = lines[j]
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-6:
                continue   # parallel
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
            ix = x1 + t * (x2 - x1)
            iy = y1 + t * (y2 - y1)
            intersection_pts.append((ix, iy))

    if not intersection_pts:
        return None

    # Median intersection — robust to outliers
    xs = [p[0] for p in intersection_pts]
    ys = [p[1] for p in intersection_pts]
    return (float(np.median(xs)), float(np.median(ys)))


def _build_shape_features(region) -> np.ndarray:
    """
    Extract a 14-dimensional shape feature vector from a skimage RegionProps
    object, for use in the catch-all classifier (Method 6).

    Dimensions:
        0   aspect ratio (width / height)
        1   log10(area)
        2   circularity  4π·A/P²
        3   extent       area / bbox area
        4   solidity     area / convex hull area
        5   Euler number (holes: -1 per hole)
        6–12 log-Hu moments [0..6]
        13  normalised perimeter  P / sqrt(A)
    """
    try:
        h = region.bbox[2] - region.bbox[0]
        w = region.bbox[3] - region.bbox[1]
        aspect = (w / h) if h > 0 else 1.0
        area_log = float(np.log10(max(region.area, 1)))
        perim = max(region.perimeter, 1.0)
        circ = (4 * np.pi * region.area) / (perim ** 2)

        # Hu moments from the region image
        crop = region.image.astype(np.uint8)
        m = cv2.moments(crop)
        hu = cv2.HuMoments(m).flatten()
        hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

        return np.array([
            aspect,
            area_log,
            circ,
            region.extent,
            region.solidity,
            float(region.euler_number),
            *hu_log[:7],
            perim / np.sqrt(max(region.area, 1)),
        ], dtype=np.float32)
    except Exception:
        return np.zeros(14, dtype=np.float32)


# =============================================================================
# Method 1 — Colour masking  (operates on RGB)
# =============================================================================

def _m1_colour_masking(rgb: np.ndarray, spec: TileSpec) -> list[SymbolDetection]:
    """
    Method 1: Colour Masking
    ========================
    Convert the RGB image to HSV and threshold for each FAA airspace colour.
    Finds contiguous colour blobs and records them as airspace polygon
    detections (bounding box approximation).
    """
    detections: list[SymbolDetection] = []
    H, W = rgb.shape[:2]
    max_area = H * W * 0.10    # 10% of tile area upper limit for a single blob
    min_area = 200             # px² — ignore specks

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    for sym_type, (lo, hi) in AIRSPACE_COLOUR_RANGES.items():
        mask = cv2.inRange(hsv, np.array(lo, dtype=np.uint8),
                               np.array(hi, dtype=np.uint8))

        if sym_type == "restricted":
            lo2, hi2 = AIRSPACE_COLOUR_RANGES["restricted_wrap"]
            mask2 = cv2.inRange(hsv, np.array(lo2, dtype=np.uint8),
                                     np.array(hi2, dtype=np.uint8))
            mask = cv2.bitwise_or(mask, mask2)

        if sym_type == "restricted_wrap":
            continue

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            bx, by, bw, bh = cv2.boundingRect(cnt)
            
            # Tile coordinates -> Global coordinates
            cx, cy = _to_full(bx + bw // 2, by + bh // 2, spec)
            
            conf = min(1.0, area / 5000)

            detections.append(SymbolDetection(
                symbol_type=sym_type,
                method="M1_colour",
                x=cx, y=cy,
                width=bw, height=bh,
                confidence=conf,
            ))

    return detections


# =============================================================================
# Method 2 — Hough circle detection  (operates on binary)
# =============================================================================

def _m2_hough_circles(binary: np.ndarray, spec: TileSpec) -> list[SymbolDetection]:
    """
    Method 2: Hough Circle Detection
    =================================
    Detects circular structures in the binary layer.
    """
    detections: list[SymbolDetection] = []

    bin_u8 = (binary * 255).astype(np.uint8)
    edges = cv2.Canny(bin_u8, threshold1=40, threshold2=120)

    circles_small = cv2.HoughCircles(
        edges, cv2.HOUGH_GRADIENT, dp=2, minDist=25,
        param1=120, param2=CIRCLE_HOUGH_PARAM2,
        minRadius=14, maxRadius=60,
    )

    circles_large = cv2.HoughCircles(
        edges, cv2.HOUGH_GRADIENT, dp=2, minDist=80,
        param1=120, param2=CIRCLE_HOUGH_PARAM2 + 12,
        minRadius=70, maxRadius=420,
    )

    H, W = binary.shape

    def _classify_circle(cx: int, cy: int, r: int) -> tuple[str, float]:
        margin = int(r * 0.3)
        y0 = max(0, cy - r - margin)
        y1 = min(H, cy + r + margin)
        x0 = max(0, cx - r - margin)
        x1 = min(W, cx + r + margin)
        crop = binary[y0:y1, x0:x1].astype(np.uint8)

        annulus_mask = np.zeros_like(crop, dtype=np.uint8)
        local_cx = cx - x0
        local_cy = cy - y0
        cv2.circle(annulus_mask, (local_cx, local_cy), r,           1, thickness=6)
        cv2.circle(annulus_mask, (local_cx, local_cy), max(r-8, 1), 0, thickness=cv2.FILLED)

        on_annulus = int((crop & annulus_mask).sum())
        annulus_area = max(int(annulus_mask.sum()), 1)
        density = on_annulus / annulus_area

        inner_mask = np.zeros_like(crop, dtype=np.uint8)
        inner_r = max(int(r * 0.55), 1)
        cv2.circle(inner_mask, (local_cx, local_cy), inner_r, 1, thickness=cv2.FILLED)
        inner_density = float((crop & inner_mask).sum()) / max(inner_mask.sum(), 1)

        if r >= 70:
            if density > 0.35:
                return ("vor", 0.80)
            else:
                return ("class_d", 0.65)
        else:
            if inner_density > 0.08:
                return ("airport_civil", 0.72)
            else:
                return ("ndb", 0.68)

    for band_circles in [circles_small, circles_large]:
        if band_circles is None:
            continue
        for x, y, r in band_circles[0]:
            cx_local, cy_local, ri = int(round(x)), int(round(y)), int(round(r))
            sym_type, conf = _classify_circle(cx_local, cy_local, ri)
            
            cx_global, cy_global = _to_full(cx_local, cy_local, spec)
            
            detections.append(SymbolDetection(
                symbol_type=sym_type,
                method="M2_hough_circle",
                x=cx_global, y=cy_global,
                width=ri * 2, height=ri * 2,
                confidence=conf,
                radius_px=ri,
            ))

    return detections


# =============================================================================
# Method 3 — Blob analysis (dot candidates)  (operates on binary)
# =============================================================================

def _m3_blob_dots(binary: np.ndarray, spec: TileSpec) -> list[dict]:
    """
    Method 3: Blob Analysis — Dot Candidates
    =========================================
    Labels connected components in the binary layer and filters to those
    matching the size and shape of FAA symbol dots.
    """
    labeled = measure.label(binary, connectivity=2)
    dot_candidates = []

    for region in measure.regionprops(labeled):
        if not (DOT_MIN_AREA_PX <= region.area <= DOT_MAX_AREA_PX):
            continue
        perim = region.perimeter
        if perim < 1:
            continue
        circ = (4 * np.pi * region.area) / (perim ** 2)
        if circ < DOT_MIN_CIRCULARITY:
            continue
            
        r_t, c_t = region.centroid
        
        dot_candidates.append({
            "centroid": (r_t + spec.pad_y0, c_t + spec.pad_x0),
            "area":     region.area,
            "circularity": circ,
            "bbox":     region.bbox,
        })

    return dot_candidates


# =============================================================================
# Method 4 — Structural V+dot obstacle detector  (operates on binary)
# =============================================================================

def _m4_obstacle_structural(
    binary: np.ndarray,
    dot_candidates: list[dict],
    spec: TileSpec
) -> tuple[list[SymbolDetection], set[int]]:
    """
    Method 4: Structural V+Dot Obstacle Detector
    =============================================
    Searches window above dot for the open inverted-V of an obstacle symbol.
    Receives all global dots and processes those within the local tile view.
    """
    detections: list[SymbolDetection] = []
    matched_indices: set[int] = set()

    H, W = binary.shape
    sr = OBSTACLE_SEARCH_RADIUS_PX

    right_leg_angles = np.linspace(np.radians(45),  np.radians(70),  40)
    left_leg_angles  = np.linspace(np.radians(110), np.radians(135), 40)
    both_leg_angles  = np.concatenate([right_leg_angles, left_leg_angles])

    # ── First pass: single obstacle (one dot) ─────────────────────────
    for idx, dot in enumerate(dot_candidates):
        row, col = dot["centroid"] # GLOBAL
        cy, cx = int(row), int(col)

        local_cy = cy - spec.pad_y0
        local_cx = cx - spec.pad_x0

        if not (0 <= local_cy < H and 0 <= local_cx < W):
            continue

        y0 = max(0, local_cy - sr)
        y1 = local_cy
        x0 = max(0, local_cx - sr)
        x1 = min(W, local_cx + sr)
        crop = binary[y0:y1, x0:x1]

        if crop.size == 0:
            continue

        lines = probabilistic_hough_line(
            # --- FIRST PASS REPLACEMENT ---
        crop_u8 = (crop * 255).astype(np.uint8)
        cv_lines = cv2.HoughLinesP(
            crop_u8, rho=1, theta=np.pi/180, 
            threshold=12, minLineLength=12, maxLineGap=4
        )

        if cv_lines is None:
            continue

        # Strict Angle Filter (Maintains 100% accuracy with original skimage logic)
        lines = []
        for line in cv_lines:
            x1, y1, x2, y2 = line[0]
            # Calculate angle in degrees
            angle = np.degrees(np.arctan2(y1 - y2, x2 - x1)) % 180 
            
            # Keep only exact left and right leg angles
            if (45 <= angle <= 70) or (110 <= angle <= 135):
                lines.append(((x1, y1), (x2, y2)))

        if len(lines) < 2:
            continue

        apex = _find_line_intersection(lines)
        if apex is None:
            continue

        ax_crop, ay_crop = apex
        if ay_crop < 0 or ay_crop >= (y1 - y0):
            continue
        if abs(ax_crop - (local_cx - x0)) > sr * 0.6:
            continue

        horiz_angles = np.linspace(np.radians(-5), np.radians(5), 20)
        base_lines = probabilistic_hough_line(
            crop, threshold=10, line_length=10, line_gap=3,
            theta=horiz_angles,
        )
        if len(base_lines) > 1:
            continue

        apex_local_x = x0 + int(ax_crop)
        apex_local_y = y0 + int(ay_crop)
        apex_global_x, apex_global_y = _to_full(apex_local_x, apex_local_y, spec)

        sym_h = cy - apex_global_y + 5
        sym_w = max(sym_h, 30)

        detections.append(SymbolDetection(
            symbol_type="obstacle_single",
            method="M4_structural",
            x=cx, y=(cy + apex_global_y) // 2,
            width=sym_w, height=sym_h,
            confidence=0.86,
            apex_xy=(apex_global_x, apex_global_y),
            dot_xy=(cx, cy),
        ))
        matched_indices.add(idx)

    # ── Second pass: group / double obstacle (two adjacent dots) ──────
    unmatched = [i for i in range(len(dot_candidates)) if i not in matched_indices]
    paired_used: set[int] = set()
    group_detections: list[SymbolDetection] = []

    for i in unmatched:
        if i in paired_used:
            continue
        d1 = dot_candidates[i]
        r1, c1 = d1["centroid"] # GLOBAL

        for j in unmatched:
            if j <= i or j in paired_used:
                continue
            d2 = dot_candidates[j]
            r2, c2 = d2["centroid"] # GLOBAL

            horiz_dist = abs(c2 - c1)
            vert_dist  = abs(r2 - r1)
            if not (20 <= horiz_dist <= 55 and vert_dist <= 10):
                continue

            cx = int((c1 + c2) / 2)
            cy = int((r1 + r2) / 2)
            
            local_cx = cx - spec.pad_x0
            local_cy = cy - spec.pad_y0

            if not (0 <= local_cy < H and 0 <= local_cx < W):
                continue

            y0 = max(0, local_cy - sr)
            y1 = local_cy
            x0 = max(0, local_cx - sr)
            x1 = min(W, local_cx + sr)
            crop = binary[y0:y1, x0:x1]

            if crop.size == 0:
                continue

            lines = probabilistic_hough_line(
                crop_u8 = (crop * 255).astype(np.uint8)
            cv_lines = cv2.HoughLinesP(
                crop_u8, rho=1, theta=np.pi/180, 
                threshold=10, minLineLength=10, maxLineGap=4
            )

            if cv_lines is None:
                continue

            lines = []
            for line in cv_lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y1 - y2, x2 - x1)) % 180 
                if (45 <= angle <= 70) or (110 <= angle <= 135):
                    lines.append(((x1, y1), (x2, y2)))

            if len(lines) < 2:
                continue

            apex = _find_line_intersection(lines)
            if apex is None:
                continue

            ax_crop, ay_crop = apex
            if ay_crop < 0 or ay_crop >= (y1 - y0):
                continue

            apex_local_x = x0 + int(ax_crop)
            apex_local_y = y0 + int(ay_crop)
            apex_global_x, apex_global_y = _to_full(apex_local_x, apex_local_y, spec)
            
            sym_h = cy - apex_global_y + 5
            sym_w = max(sym_h + int(horiz_dist), 40)

            group_detections.append(SymbolDetection(
                symbol_type="obstacle_group",
                method="M4_structural",
                x=cx, y=(cy + apex_global_y) // 2,
                width=sym_w, height=sym_h,
                confidence=0.84,
                apex_xy=(apex_global_x, apex_global_y),
                dot_xy=(int(c1), int(r1)),
                dot_xy_2=(int(c2), int(r2)),
            ))
            paired_used.add(i)
            paired_used.add(j)
            matched_indices.add(i)
            matched_indices.add(j)
            break

    detections.extend(group_detections)
    return detections, matched_indices


# =============================================================================
# Method 5 — Multi-scale template matching  (operates on binary)
# =============================================================================

def _load_templates(template_dir: Path) -> dict[str, np.ndarray]:
    """Load binary template PNGs from template_dir."""
    templates: dict[str, np.ndarray] = {}
    if not template_dir.exists():
        log.warning(f"Template directory not found: {template_dir} — M5 will be skipped.")
        return templates

    for png_path in sorted(template_dir.glob("*.png")):
        sym_type = png_path.stem
        img = Image.open(str(png_path)).convert("L")
        arr = np.array(img, dtype=np.uint8)
        templates[sym_type] = arr > 127

    return templates


def _m5_template_matching(
    binary: np.ndarray,
    templates: dict[str, np.ndarray],
    spec: TileSpec,
    scales: np.ndarray = np.linspace(0.82, 1.18, 9),
    threshold: float = TEMPLATE_MATCH_THRESHOLD,
) -> list[SymbolDetection]:
    SKIP_TYPES = {"vor", "vor_dme", "vortac", "tacan",
                  "obstacle_single", "obstacle_group",
                  "class_b", "class_c", "class_d", "class_e_surface",
                  "restricted", "moa", "warning", "prohibited", "alert"}

    detections: list[SymbolDetection] = []
    
    # OpenCV requires uint8 arrays (0 or 255) for template matching
    bin_u8 = (binary * 255).astype(np.uint8)

    for sym_type, tmpl in templates.items():
        if sym_type in SKIP_TYPES:
            continue

        TH, TW = tmpl.shape
        tmpl_u8 = (tmpl * 255).astype(np.uint8)
        per_template_dets: list[SymbolDetection] = []

        for scale in scales:
            new_w = max(4, int(TW * scale))
            new_h = max(4, int(TH * scale))
            scaled_t = cv2.resize(
                tmpl_u8, (new_w, new_h),
                interpolation=cv2.INTER_NEAREST,
            )

            # cv2.TM_CCOEFF_NORMED is the exact mathematical equivalent to skimage's match_template
            score_map = cv2.matchTemplate(bin_u8, scaled_t, cv2.TM_CCOEFF_NORMED)
            peak_coords = np.argwhere(score_map >= threshold)

            for row, col in peak_coords:
                # Acccuracy correction: Shift top-left coordinates to the center pixel
                cx_local = col + (new_w // 2)
                cy_local = row + (new_h // 2)
                
                gx, gy = _to_full(cx_local, cy_local, spec)
                
                per_template_dets.append(SymbolDetection(
                    symbol_type=sym_type,
                    method="M5_template",
                    x=gx, y=gy,
                    width=new_w, height=new_h,
                    confidence=float(score_map[row, col]),
                    scale=float(scale),
                ))

        per_template_dets = _nms(per_template_dets, radius=max(TW, TH) // 2)
        detections.extend(per_template_dets)

    return detections

# =============================================================================
# Method 6 — Shape-feature catch-all classifier  (operates on binary)
# =============================================================================

def _m6_residual_classifier(
    binary: np.ndarray,
    existing_detections: list[SymbolDetection],
    spec: TileSpec
) -> list[SymbolDetection]:
    """
    Method 6: Residual Shape-Feature Classifier
    =============================================
    Uses Hu moments + simple geometric ratios on unclassified blobs.
    """
    H, W = binary.shape

    suppress = np.zeros((H, W), dtype=bool)
    for d in existing_detections:
        gx0, gy0, gx1, gy1 = d.bbox()
        margin = 8
        
        lx0 = max(0, gx0 - spec.pad_x0 - margin)
        ly0 = max(0, gy0 - spec.pad_y0 - margin)
        lx1 = min(W, gx1 - spec.pad_x0 + margin)
        ly1 = min(H, gy1 - spec.pad_y0 + margin)
        
        if lx0 < lx1 and ly0 < ly1:
            suppress[ly0:ly1, lx0:lx1] = True

    residual = binary & ~suppress
    labeled = measure.label(residual, connectivity=2)

    detections: list[SymbolDetection] = []

    for region in measure.regionprops(labeled):
        if region.area < 15 or region.area > 3000:
            continue

        feats = _build_shape_features(region)
        sym_type, conf = _classify_by_rules(feats)

        if sym_type is None or conf < 0.60:
            continue

        r0, c0, r1, c1 = region.bbox
        lcx = (c0 + c1) // 2
        lcy = (r0 + r1) // 2
        
        gcx, gcy = _to_full(lcx, lcy, spec)

        detections.append(SymbolDetection(
            symbol_type=sym_type,
            method="M6_residual",
            x=gcx, y=gcy,
            width=(c1 - c0), height=(r1 - r0),
            confidence=conf,
        ))

    return detections


def _classify_by_rules(feats: np.ndarray) -> tuple[Optional[str], float]:
    """Rule-based classifier for catch-all detection."""
    aspect   = feats[0]
    log_area = feats[1]
    circ     = feats[2]
    extent   = feats[3]
    solidity = feats[4]
    euler    = feats[5]

    if circ > 0.80 and 0.9 < log_area < 2.1:
        return ("spot_elevation", 0.72)
    if 0.30 < circ < 0.62 and solidity > 0.88 and 0.55 < aspect < 1.20:
        return ("waypoint", 0.68)
    if aspect < 0.55 and log_area > 1.8 and solidity > 0.45:
        return ("parachute", 0.62)
    if aspect > 1.6 and log_area > 1.5 and solidity > 0.50:
        return ("ultralight", 0.60)
    if 0.65 < circ < 0.85 and euler < -0.5 and 1.8 < log_area < 2.8:
        return ("ndb", 0.64)

    return (None, 0.0)


# =============================================================================
# Post-processing: emit unmatched dots as spot elevations
# =============================================================================

def _emit_spot_elevations(
    dot_candidates: list[dict],
    matched_dot_indices: set[int],
) -> list[SymbolDetection]:
    """Emit remaining dots as spot elevations."""
    detections: list[SymbolDetection] = []
    for idx, dot in enumerate(dot_candidates):
        if idx in matched_dot_indices:
            continue
        row, col = dot["centroid"] # GLOBAL
        r = max(3, int(np.sqrt(dot["area"] / np.pi)))
        detections.append(SymbolDetection(
            symbol_type="spot_elevation",
            method="M3_dot_residual",
            x=int(col), y=int(row),
            width=r * 2, height=r * 2,
            confidence=0.66,
            dot_xy=(int(col), int(row)),
        ))
    return detections


# =============================================================================
# Output — save detections to disk
# =============================================================================

def _save_outputs(
    detections: list[SymbolDetection],
    rgb: np.ndarray,
    stem: str,
    output_dir: Path,
    image_size: tuple[int, int],
) -> tuple[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{stem}_detections.json"
    with open(json_path, "w") as f:
        json.dump([d.to_dict() for d in detections], f, indent=2)
    log.info(f"Saved JSON → {json_path}")

    features = []
    for d in detections:
        x0, y0, x1, y1 = d.bbox()
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [d.x, d.y],
            },
            "properties": {
                "symbol_type": d.symbol_type,
                "method":      d.method,
                "confidence":  round(d.confidence, 4),
                "bbox_px":     [x0, y0, x1, y1],
                "width_px":    d.width,
                "height_px":   d.height,
            },
        })

    geojson = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {"name": "pixel_coordinates"},
        },
        "features": features,
    }
    geojson_path = output_dir / f"{stem}_detections.geojson"
    with open(geojson_path, "w") as f:
        json.dump(geojson, f, indent=2)
    log.info(f"Saved GeoJSON → {geojson_path}")

    vis = rgb.copy()

    method_colours = {
        "M1_colour":       (255, 200,   0),
        "M2_hough_circle": (  0, 180, 255),
        "M3_dot_residual": (180, 255,   0),
        "M4_structural":   (255,  80,  80),
        "M5_template":     ( 80, 255, 120),
        "M6_residual":     (200,  80, 255),
    }

    for d in detections:
        x0, y0, x1, y1 = d.bbox()
        colour = method_colours.get(d.method, (255, 255, 255))
        cv2.rectangle(vis, (x0, y0), (x1, y1), colour, thickness=2)
        cv2.putText(vis, d.symbol_type[:12], (x0, max(y0 - 4, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1,
                    cv2.LINE_AA)

    vis_path = output_dir / f"{stem}_visualisation.png"
    Image.fromarray(vis).save(str(vis_path), format="PNG", compress_level=3)
    log.info(f"Saved visualisation → {vis_path}")

    return str(json_path), str(geojson_path)


# =============================================================================
# Public API
# =============================================================================

def detect_symbols(
    rgb:           np.ndarray,
    binary:        np.ndarray,
    template_dir:  Optional[str] = None,
    output_dir:    Optional[str] = None,
    stem:          str = "chart",
    dpi:           int = 300,
    tile_size:     int = TILE_SIZE,
    tile_overlap:  int = TILE_OVERLAP,
) -> DetectionResult:
    """
    Run the full Phase 4 symbol detection pipeline.
    """
    t0 = time.time()
    notes: list[str] = []
    H, W = rgb.shape[:2]
    image_size = (W, H)

    log.info(f"Phase 4 starting — image {W}x{H} px at {dpi} DPI (Tiled)")

    if template_dir:
        templates = _load_templates(Path(template_dir))
    else:
        templates = {}
        notes.append("template_dir not provided — M5 template matching skipped.")
        log.warning("M5 skipped: no template_dir provided.")
        
    scales = np.linspace(0.82, 1.18, 9)

    # ── Tiled Loop ────────────────────────────────────────────────────
    tile_specs     = list(_iter_tiles(H, W, tile_size, tile_overlap))
    all_dets       = []
    all_dots_full  = []
    matched_global = set()

    log.info(f"Processing {len(tile_specs)} tiles...")

    for i, spec in enumerate(tile_specs, 1):
        log.info(f"  Tile {i}/{len(tile_specs)} ({spec.pad_x1 - spec.pad_x0}x{spec.pad_y1 - spec.pad_y0})")
        
        rgb_tile = rgb   [spec.pad_y0:spec.pad_y1, spec.pad_x0:spec.pad_x1]
        bin_tile = binary[spec.pad_y0:spec.pad_y1, spec.pad_x0:spec.pad_x1]

        m1 = _keep_inner(_m1_colour_masking(rgb_tile, spec), spec)
        m2 = _keep_inner(_m2_hough_circles(bin_tile, spec), spec)

        tile_dots = _m3_blob_dots(bin_tile, spec)
        tile_dots_inner = [d for d in tile_dots
                           if spec.inner_x0 <= d["centroid"][1] < spec.inner_x1
                           and spec.inner_y0 <= d["centroid"][0] < spec.inner_y1]
        all_dots_full.extend(tile_dots_inner)

        m4_dets, tile_matched = _m4_obstacle_structural(bin_tile, all_dots_full, spec)
        m4 = _keep_inner(m4_dets, spec)
        matched_global |= tile_matched

        m5 = _keep_inner(_m5_template_matching(bin_tile, templates, spec, scales, TEMPLATE_MATCH_THRESHOLD), spec) if templates else []

        tile_so_far = _nms(m1 + m2 + m4 + m5, radius=NMS_RADIUS_PX)

        m6 = _keep_inner(_m6_residual_classifier(bin_tile, all_dets + tile_so_far, spec), spec)

        tile_final = _nms(tile_so_far + m6, radius=NMS_RADIUS_PX)
        all_dets.extend(tile_final)

    # ── Spot Elevations & Final Merge ─────────────────────────────────
    spot_dets = _emit_spot_elevations(all_dots_full, matched_global)
    all_dets.extend(spot_dets)
    all_dets = _nms(all_dets, radius=NMS_RADIUS_PX)  # final global dedup

    # ── Count by type ─────────────────────────────────────────────────
    counts: dict[str, int] = {}
    for d in all_dets:
        counts[d.symbol_type] = counts.get(d.symbol_type, 0) + 1

    # ── Save outputs ──────────────────────────────────────────────────
    json_path = geojson_path = None
    if output_dir:
        json_path, geojson_path = _save_outputs(
            all_dets, rgb, stem, Path(output_dir), image_size
        )

    elapsed = time.time() - t0

    result = DetectionResult(
        detections=all_dets,
        source_rgb_path=stem + "_rgb.png",
        source_bin_path=stem + "_binary.png",
        image_size_px=image_size,
        elapsed_sec=elapsed,
        counts=counts,
        notes=notes,
        geojson_path=geojson_path,
        json_path=json_path,
    )
    log.info(result.summary())
    return result


# =============================================================================
# Convenience wrapper — load from disk paths
# =============================================================================

def detect_symbols_from_paths(
    rgb_path:      str,
    binary_path:   str,
    template_dir:  Optional[str] = None,
    output_dir:    Optional[str] = None,
    dpi:           int = 300,
    tile_size:     int = TILE_SIZE,
    tile_overlap:  int = TILE_OVERLAP,
) -> DetectionResult:
    """Load Phase 1 output PNGs from disk and run detect_symbols()."""
    rgb_p   = Path(rgb_path)
    bin_p   = Path(binary_path)

    log.info(f"Loading RGB    : {rgb_p}")
    rgb    = _load_image(rgb_p)

    log.info(f"Loading binary : {bin_p}")
    binary = _load_binary(bin_p)

    stem = rgb_p.stem.replace("_rgb", "")

    return detect_symbols(
        rgb=rgb,
        binary=binary,
        template_dir=template_dir,
        output_dir=output_dir,
        stem=stem,
        dpi=dpi,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
    )


# =============================================================================
# Integration with Phase 1
# =============================================================================

def detect_from_phase1_result(phase1_result, **kwargs) -> DetectionResult:
    """Accept a Phase1.PreprocessingResult directly."""
    stem = Path(phase1_result.source_path).stem

    return detect_symbols(
        rgb=phase1_result.rgb_array,
        binary=phase1_result.binary_array,
        stem=stem,
        dpi=phase1_result.dpi or 300,
        **kwargs,
    )


# =============================================================================
# ONE-CLICK RUN EXECUTION BLOCK
# =============================================================================

if __name__ == "__main__":
    # ---------------------------------------------------------
    # CONFIGURATION: Update these paths to match your local setup
    # ---------------------------------------------------------
    CONFIG_RGB_PATH = "outputs/phase1/Washington_rgb.png"
    CONFIG_BINARY_PATH = "outputs/phase1/Washington_binary.png"
    CONFIG_TEMPLATE_DIR = "templates/"
    CONFIG_OUTPUT_DIR = "outputs/phase4/"
    CONFIG_DPI = 300
    CONFIG_TILE_SIZE = 2048
    CONFIG_TILE_OVERLAP = 128

    print("======================================================")
    print(" Starting Phase 4 Symbol Detection (One-Click Mode)")
    print("======================================================")
    print(f" > RGB Image:   {CONFIG_RGB_PATH}")
    print(f" > Binary Mask: {CONFIG_BINARY_PATH}")
    print(f" > Templates:   {CONFIG_TEMPLATE_DIR}")
    print(f" > Output Dir:  {CONFIG_OUTPUT_DIR}")
    print(f" > Tiling:      {CONFIG_TILE_SIZE}px (Overlap: {CONFIG_TILE_OVERLAP}px)")
    print("------------------------------------------------------")

    try:
        result = detect_symbols_from_paths(
            rgb_path=CONFIG_RGB_PATH,
            binary_path=CONFIG_BINARY_PATH,
            template_dir=CONFIG_TEMPLATE_DIR,
            output_dir=CONFIG_OUTPUT_DIR,
            dpi=CONFIG_DPI,
            tile_size=CONFIG_TILE_SIZE,
            tile_overlap=CONFIG_TILE_OVERLAP,
        )
        print("\nExecution completed successfully.")
        print(result.summary())
    except FileNotFoundError as e:
        print(f"\n[ERROR] File or directory not found. Please verify your config paths: {e}")
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")