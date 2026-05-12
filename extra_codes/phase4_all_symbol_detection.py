"""
Phase 4 — Symbol Detection (Fully Optimized & Bounded)
VFR Chart Extraction Pipeline
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# =============================================================================
# Logging
# =============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s  [%(levelname)s]  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("phase4")

# =============================================================================
# Constants
# =============================================================================
AIRSPACE_COLOUR_RANGES: dict[str, tuple[list, list]] = {
    "class_b":              ([100, 50,  60],  [125, 255, 255]),
    "class_c":              ([130, 40,  60],  [155, 255, 255]),
    "class_d":              ([100, 30,  40],  [130, 150, 220]),
    "class_e_surface":      ([140, 40,  60],  [170, 255, 255]),
    "restricted":           ([0,   80,  80],  [12,  255, 255]),
    "restricted_wrap":      ([168, 80,  80],  [179, 255, 255]),
    "moa":                  ([20,  60,  60],  [35,  255, 255]),
    "warning":              ([0,   80,  80],  [12,  255, 255]),
    "prohibited":           ([0,   100, 100], [10,  255, 255]),
    "alert":                ([15,  60,  60],  [28,  255, 255]),
}

TEMPLATE_MATCH_THRESHOLD   = 0.76   
DOT_MIN_AREA_PX            = 6      
DOT_MAX_AREA_PX            = 140    
DOT_MIN_CIRCULARITY        = 0.68   
OBSTACLE_SEARCH_RADIUS_PX  = 70     
NMS_RADIUS_PX              = 18     

# =============================================================================
# Dataclasses
# =============================================================================
@dataclass
class SymbolDetection:
    symbol_type:  str           
    method:       str           
    x:            int           
    y:            int           
    width:        int           
    height:       int           
    confidence:   float         

    radius_px:    Optional[int]   = None   
    scale:        Optional[float] = None   
    apex_xy:      Optional[tuple] = None   
    dot_xy:       Optional[tuple] = None   
    dot_xy_2:     Optional[tuple] = None   

    def bbox(self) -> tuple[int, int, int, int]:
        hw, hh = self.width // 2, self.height // 2
        return (self.x - hw, self.y - hh, self.x + hw, self.y + hh)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bbox"] = self.bbox()
        return d

@dataclass
class DetectionResult:
    detections:       list[SymbolDetection]
    source_rgb_path:  str
    source_bin_path:  str
    image_size_px:    tuple[int, int]   
    elapsed_sec:      float
    counts:           dict = field(default_factory=dict)
    notes:            list = field(default_factory=list)
    geojson_path:     Optional[str] = None
    json_path:        Optional[str] = None

    def summary(self) -> str:
        lines = [
            "=== Phase 4 Result ===",
            f"  Total symbols: {len(self.detections)}",
            f"  Elapsed      : {self.elapsed_sec:.1f} s",
            "  By type:",
        ]
        for sym_type, cnt in sorted(self.counts.items()):
            lines.append(f"    {sym_type:<30s}: {cnt}")
        return "\n".join(lines)

@dataclass
class TileSpec:
    inner_x0: int; inner_y0: int
    inner_x1: int; inner_y1: int
    pad_x0:   int; pad_y0:   int
    pad_x1:   int; pad_y1:   int
    tile_row: int; tile_col: int

# =============================================================================
# Helpers
# =============================================================================
def _iter_tiles(H, W, tile_size, overlap):
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
            )

def _keep_inner(detections, spec):
    return [d for d in detections if spec.inner_x0 <= d.x < spec.inner_x1 and spec.inner_y0 <= d.y < spec.inner_y1]

def _to_full(x_t, y_t, spec):
    return (x_t + spec.pad_x0, y_t + spec.pad_y0)

def _nms(detections: list[SymbolDetection], radius: int) -> list[SymbolDetection]:
    """Extremely fast Vectorized NMS using pure Numpy."""
    if not detections: return []
    dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
    
    # HARD CAP: Prevent NMS lockup on data explosions
    if len(dets) > 3000: dets = dets[:3000]

    coords = np.array([(d.x, d.y) for d in dets], dtype=np.float32)
    keep = []
    suppressed = np.zeros(len(dets), dtype=bool)
    r2 = radius * radius

    for i in range(len(dets)):
        if suppressed[i]: continue
        keep.append(dets[i])
        dists_sq = (coords[i+1:, 0] - coords[i, 0])**2 + (coords[i+1:, 1] - coords[i, 1])**2
        suppress_idx = np.where(dists_sq < r2)[0] + (i + 1)
        suppressed[suppress_idx] = True

    return keep

# =============================================================================
# Detectors
# =============================================================================
def _m1_colour_masking(rgb: np.ndarray, spec: TileSpec, sf: float) -> list[SymbolDetection]:
    detections = []
    H, W = rgb.shape[:2]
    max_area, min_area = H * W * 0.10, 200 * (sf ** 2)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    for sym_type, (lo, hi) in AIRSPACE_COLOUR_RANGES.items():
        if sym_type == "restricted_wrap": continue
        mask = cv2.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
        if sym_type == "restricted":
            lo2, hi2 = AIRSPACE_COLOUR_RANGES["restricted_wrap"]
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lo2, dtype=np.uint8), np.array(hi2, dtype=np.uint8)))

        if cv2.countNonZero(mask) == 0: continue # Skip empty colors instantly

        mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2), cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (min_area <= area <= max_area): continue
            bx, by, bw, bh = cv2.boundingRect(cnt)
            cx, cy = _to_full(bx + bw // 2, by + bh // 2, spec)
            detections.append(SymbolDetection(sym_type, "M1_colour", cx, cy, bw, bh, min(1.0, area / 5000)))
    return detections


def _m2_hough_circles(binary: np.ndarray, spec: TileSpec, sf: float) -> list[SymbolDetection]:
    detections = []
    # Phase 1 binary: True=foreground=symbol ink (white on black).
    # Canny finds edges at transitions; it works on both polarities, but
    # FAA symbols are thin strokes — inverting gives dark-on-light which
    # produces sharper Canny responses on stroke edges.
    bin_u8 = (binary * 255).astype(np.uint8)
    edges = cv2.Canny(bin_u8, 40, 120)

    # dp=1 is mandatory — dp=2 halves the accumulator resolution and misses
    # small airport/NDB circles entirely, especially at sf < 1.0 (150 DPI).
    # param2 scales with sf so detection sensitivity is DPI-invariant.
    p2_small = max(18, int(22 * sf))
    p2_large = max(25, int(32 * sf))

    circles_small = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, max(1, int(25 * sf)), param1=100, param2=p2_small, minRadius=max(1, int(14 * sf)), maxRadius=max(5, int(60 * sf)))
    circles_large = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, max(1, int(80 * sf)), param1=100, param2=p2_large, minRadius=max(5, int(70 * sf)), maxRadius=max(10, int(420 * sf)))

    for band in [circles_small, circles_large]:
        if band is None: continue
        # CAP: Take only the top 100 strongest circles per band
        for x, y, r in band[0][:100]:
            cx_local, cy_local, ri = int(round(x)), int(round(y)), int(round(r))
            
            y0, y1 = max(0, cy_local - ri - int(ri*0.3)), min(binary.shape[0], cy_local + ri + int(ri*0.3))
            x0, x1 = max(0, cx_local - ri - int(ri*0.3)), min(binary.shape[1], cx_local + ri + int(ri*0.3))
            if y1 <= y0 or x1 <= x0: continue
            crop = binary[y0:y1, x0:x1].astype(np.uint8)

            ann_mask = np.zeros_like(crop)
            cv2.circle(ann_mask, (cx_local-x0, cy_local-y0), ri, 1, max(1, int(6*sf)))
            cv2.circle(ann_mask, (cx_local-x0, cy_local-y0), max(ri-int(8*sf), 1), 0, cv2.FILLED)
            density = int((crop & ann_mask).sum()) / max(int(ann_mask.sum()), 1)

            inn_mask = np.zeros_like(crop)
            cv2.circle(inn_mask, (cx_local-x0, cy_local-y0), max(int(ri * 0.55), 1), 1, cv2.FILLED)
            inn_density = float((crop & inn_mask).sum()) / max(inn_mask.sum(), 1)

            large_threshold = int(70 * sf)
            if ri >= large_threshold:
                sym_type = "vor" if density > 0.35 else "class_d"
            else:
                sym_type = "airport_civil" if inn_density > 0.08 else "ndb"
            gx, gy = _to_full(cx_local, cy_local, spec)
            detections.append(SymbolDetection(sym_type, "M2_hough", gx, gy, ri*2, ri*2, 0.75, radius_px=ri))
    return detections


def _m3_blob_dots(binary: np.ndarray, spec: TileSpec, sf: float) -> list[dict]:
    contours, _ = cv2.findContours((binary * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dots = []
    min_area, max_area = DOT_MIN_AREA_PX * (sf ** 2), DOT_MAX_AREA_PX * (sf ** 2)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (min_area <= area <= max_area): continue
        perim = cv2.arcLength(cnt, True)
        if perim < 1 or (4 * np.pi * area) / (perim ** 2) < DOT_MIN_CIRCULARITY: continue
        M = cv2.moments(cnt)
        if M["m00"] == 0: continue
        
        dots.append({
            "centroid": ((M["m01"] / M["m00"]) + spec.pad_y0, (M["m10"] / M["m00"]) + spec.pad_x0),
            "area": area
        })
    return dots


def _m4_obstacle_structural(binary: np.ndarray, dot_candidates: list[dict], spec: TileSpec, sf: float):
    detections, matched_xy = [], set()
    sr = int(OBSTACLE_SEARCH_RADIUS_PX * sf)

    for dot in dot_candidates:
        cy, cx = int(dot["centroid"][0]), int(dot["centroid"][1])
        local_cy, local_cx = cy - spec.pad_y0, cx - spec.pad_x0
        if not (0 <= local_cy < binary.shape[0] and 0 <= local_cx < binary.shape[1]): continue

        crop = binary[max(0, local_cy - sr):local_cy, max(0, local_cx - sr):min(binary.shape[1], local_cx + sr)]
        if crop.size == 0: continue

        cv_lines = cv2.HoughLinesP((crop * 255).astype(np.uint8), 1, np.pi/180, max(4, int(12*sf)), minLineLength=max(2, int(12*sf)), maxLineGap=max(1, int(4*sf)))
        if cv_lines is None: continue

        # arctan2(dy, dx) — dy = y2-y1, dx = x2-x1 for each line segment.
        # FAA obstacle V legs: right leg rises left-to-right at 45-75°,
        # left leg rises right-to-left at 105-140° (both measured 0-180°).
        valid_lines = []
        for l in cv_lines:
            x1_l, y1_l, x2_l, y2_l = l[0]
            angle = np.degrees(np.arctan2(y2_l - y1_l, x2_l - x1_l)) % 180
            if (40 <= angle <= 75) or (105 <= angle <= 140):
                valid_lines.append(l[0])
        lines = valid_lines
        if len(lines) < 2: continue

        ax_crop = np.mean([x for l in lines for x in (l[0], l[2])])
        ay_crop = min([y for l in lines for y in (l[1], l[3])])
        if ay_crop < 0 or abs(ax_crop - (local_cx - max(0, local_cx - sr))) > sr * 0.6: continue

        gx, gy = _to_full(max(0, local_cx - sr) + int(ax_crop), max(0, local_cy - sr) + int(ay_crop), spec)
        sym_h = cy - gy + int(5*sf)
        detections.append(SymbolDetection("obstacle_single", "M4_structural", cx, (cy + gy) // 2, max(sym_h, int(30*sf)), sym_h, 0.86, apex_xy=(gx, gy), dot_xy=(cx, cy)))
        matched_xy.add((cx, cy))

    return detections, matched_xy


def _m5_template_matching(binary: np.ndarray, templates: dict, spec: TileSpec, sf: float) -> list[SymbolDetection]:
    detections = []
    bin_u8 = (binary * 255).astype(np.uint8)
    scales = np.linspace(0.82, 1.18, 9) * sf

    for sym_type, tmpl in templates.items():
        if sym_type in {"vor", "class_b", "class_c", "class_d", "restricted"}: continue
        tmpl_u8 = (tmpl * 255).astype(np.uint8)
        per_template = []

        for scale in scales:
            new_w, new_h = max(4, int(tmpl.shape[1] * scale)), max(4, int(tmpl.shape[0] * scale))
            
            # --- BUG FIX: Skip if the edge tile is smaller than the template ---
            if bin_u8.shape[0] < new_h or bin_u8.shape[1] < new_w:
                continue
            # -------------------------------------------------------------------
            
            score_map = cv2.matchTemplate(bin_u8, cv2.resize(tmpl_u8, (new_w, new_h), interpolation=cv2.INTER_NEAREST), cv2.TM_CCOEFF_NORMED)
            peak_coords = np.argwhere(score_map >= TEMPLATE_MATCH_THRESHOLD)

            # CAP: Take top 100 to prevent template explosion on blank/dense areas
            if len(peak_coords) > 100:
                peak_coords = peak_coords[np.argsort(score_map[peak_coords[:, 0], peak_coords[:, 1]])[-100:]]

            for row, col in peak_coords:
                gx, gy = _to_full(col + (new_w // 2), row + (new_h // 2), spec)
                per_template.append(SymbolDetection(sym_type, "M5_template", gx, gy, new_w, new_h, float(score_map[row, col])))
                
        detections.extend(_nms(per_template, max(tmpl.shape[1], tmpl.shape[0]) // 2))
    return detections

def _m6_residual_classifier(binary: np.ndarray, tile_so_far: list, spec: TileSpec, sf: float) -> list[SymbolDetection]:
    suppress = np.zeros(binary.shape, dtype=bool)
    margin = int(8 * sf)

    # We ONLY suppress based on symbols found locally in this tile!
    for d in tile_so_far:
        gx0, gy0, gx1, gy1 = d.bbox()
        lx0, ly0 = max(0, gx0 - spec.pad_x0 - margin), max(0, gy0 - spec.pad_y0 - margin)
        lx1, ly1 = min(binary.shape[1], gx1 - spec.pad_x0 + margin), min(binary.shape[0], gy1 - spec.pad_y0 + margin)
        if lx0 < lx1 and ly0 < ly1: suppress[ly0:ly1, lx0:lx1] = True

    contours, hierarchy = cv2.findContours((((binary & ~suppress) * 255).astype(np.uint8)), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return []

    detections = []
    processed_count = 0
    min_area, max_area = 15 * (sf ** 2), 3000 * (sf ** 2)

    for i, cnt in enumerate(contours):
        if hierarchy[0][i][3] != -1: continue 
        
        # CAP: Prevent Python loop locking on text margins
        processed_count += 1
        if processed_count > 500: break

        area = cv2.contourArea(cnt)
        if not (min_area <= area <= max_area): continue

        x, y, w, h = cv2.boundingRect(cnt)
        perim = cv2.arcLength(cnt, True)
        circ = (4 * np.pi * area) / (perim ** 2) if perim > 0 else 0
        solidity = area / cv2.contourArea(cv2.convexHull(cnt)) if cv2.contourArea(cv2.convexHull(cnt)) > 0 else 0
        
        holes, child_idx = 0, hierarchy[0][i][2]
        while child_idx != -1: holes, child_idx = holes + 1, hierarchy[0][child_idx][0]

        aspect, log_area, euler = w / max(h, 1), float(np.log10(max(area, 1))), 1.0 - holes
        sym_type = "spot_elevation" if circ > 0.80 and 0.9 < log_area < 2.1 else "waypoint" if 0.30 < circ < 0.62 and solidity > 0.88 and 0.55 < aspect < 1.20 else "parachute" if aspect < 0.55 and log_area > 1.8 and solidity > 0.45 else "ultralight" if aspect > 1.6 and log_area > 1.5 and solidity > 0.50 else "ndb" if 0.65 < circ < 0.85 and euler < -0.5 and 1.8 < log_area < 2.8 else None
        
        if sym_type:
            gx, gy = _to_full(x + w // 2, y + h // 2, spec)
            detections.append(SymbolDetection(sym_type, "M6_residual", gx, gy, w, h, 0.65))

    return detections

# =============================================================================
# Execution
# =============================================================================
def detect_symbols_from_paths(rgb_path, binary_path, template_dir, output_dir, dpi=150, tile_size=1024, tile_overlap=64):
    t0 = time.time()
    rgb = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
    binary = np.array(Image.open(binary_path).convert("L"), dtype=np.uint8) > 127
    H, W = rgb.shape[:2]
    
    sf = dpi / 300.0
    nms_radius = int(NMS_RADIUS_PX * sf)

    log.info(f"Phase 4 starting — {W}x{H} px | {dpi} DPI | Tiled Memory Safe Mode")

    templates = {}
    if template_dir and Path(template_dir).exists():
        for p in Path(template_dir).glob("*.png"): templates[p.stem] = np.array(Image.open(p).convert("L"), dtype=np.uint8) > 127

    tile_specs = list(_iter_tiles(H, W, tile_size, tile_overlap))
    all_dets, all_dots_full, matched_global_xy = [], [], set()

    for i, spec in enumerate(tile_specs, 1):
        log.info(f"  Processing Tile {i}/{len(tile_specs)}...")
        rgb_tile = rgb[spec.pad_y0:spec.pad_y1, spec.pad_x0:spec.pad_x1]
        bin_tile = binary[spec.pad_y0:spec.pad_y1, spec.pad_x0:spec.pad_x1]

        m1 = _keep_inner(_m1_colour_masking(rgb_tile, spec, sf), spec)
        m2 = _keep_inner(_m2_hough_circles(bin_tile, spec, sf), spec)

        tile_dots = _m3_blob_dots(bin_tile, spec, sf)
        tile_dots_inner = [d for d in tile_dots if spec.inner_x0 <= d["centroid"][1] < spec.inner_x1 and spec.inner_y0 <= d["centroid"][0] < spec.inner_y1]
        all_dots_full.extend(tile_dots_inner)

        m4_dets, tile_matched_xy = _m4_obstacle_structural(bin_tile, tile_dots, spec, sf)
        matched_global_xy |= tile_matched_xy

        m5 = _keep_inner(_m5_template_matching(bin_tile, templates, spec, sf), spec) if templates else []
        tile_so_far = _nms(m1 + m2 + _keep_inner(m4_dets, spec) + m5, nms_radius)

        m6 = _keep_inner(_m6_residual_classifier(bin_tile, tile_so_far, spec, sf), spec)
        all_dets.extend(_nms(tile_so_far + m6, nms_radius))

    for dot in all_dots_full:
        cx, cy = int(dot["centroid"][1]), int(dot["centroid"][0])
        if (cx, cy) not in matched_global_xy:
            r = max(int(3*sf), int(np.sqrt(dot["area"] / np.pi)))
            all_dets.append(SymbolDetection("spot_elevation", "M3_residual", cx, cy, r*2, r*2, 0.66, dot_xy=(cx, cy)))

    all_dets = _nms(all_dets, nms_radius)
    
    counts = {}
    for d in all_dets: counts[d.symbol_type] = counts.get(d.symbol_type, 0) + 1

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(output_dir) / f"chart_detections.json", "w") as f: json.dump([d.to_dict() for d in all_dets], f, indent=2)

    result = DetectionResult(detections=all_dets, source_rgb_path=rgb_path, source_bin_path=binary_path, image_size_px=(W, H), elapsed_sec=time.time()-t0, counts=counts)
    log.info(result.summary())
    return result

if __name__ == "__main__":
    detect_symbols_from_paths(
        rgb_path="outputs/phase1/Washington_rgb.png",
        binary_path="outputs/phase1/Washington_binary.png",
        template_dir="templates/",
        output_dir="outputs/phase4_fixed_v2/",
        dpi=150, tile_size=1024, tile_overlap=64
    )