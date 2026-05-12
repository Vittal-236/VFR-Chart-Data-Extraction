"""
Phase 2c — Symbol-Only Binary Mask
VFR Chart Extraction Pipeline (FAA Base Model)

Consumes the Phase 1 binary mask and produces a new binary containing
ONLY the aeronautical symbols and text annotations drawn from the
Washington Sectional legend:

    Airports            — compact ring / dot glyphs, 5–20 px
    Radio aids          — VOR rosettes, NDB circles, DME squares
    Obstacle symbols    — inverted-V + dot pairs  (single & group)
    Airspace labels     — WARNING / CLASS text clusters
    Altitude labels     — numeric annotations near symbols

Everything else (airspace boundary arcs, airways, coastline, lat/lon
grid ticks, topo contours, noise) is removed.

═══════════════════════════════════════════════════════════════════════
ALGORITHM  — four-stage connected-component pipeline
═══════════════════════════════════════════════════════════════════════

  STAGE 1 — REFERENCE-TEMPLATE RESPONSE MAP
  ──────────────────────────────────────────
  The two FAA obstacle reference images (obstacle.png / Double_obstacle.png)
  are converted to binary templates and scaled to every integer height in
  [TMPL_H_MIN, TMPL_H_MAX].  Normalised cross-correlation (NCC) is run
  against the binary map for every template; the pixel-wise maximum across
  all scales and both references forms a "template response map" R(y,x).

  Pixels of R are used as a BOOST signal: connected components whose
  centroid falls on a local NCC peak above TMPL_THRESHOLD receive a bonus
  that relaxes the stage-3 chord/solidity criteria, making it easier for
  true obstacle symbols to survive even when their shape metrics are
  borderline.

  STAGE 2 — NOISE / LINE REMOVAL  (five sequential CC filters)
  ─────────────────────────────────────────────────────────────
  F1  Area ≤ NOISE_MAX_AREA                  → noise (isolated px, JPG dirt)
  F2  aspect > LONG_RATIO AND max_dim > LONG_DIM → full-width straight lines
  F3  min_dim == 1 AND area < THIN_MAX        → 1-px-wide dash fragments
  F4  Local ink density > DENSITY_THRESH
        (Gaussian σ=DENSITY_SIGMA, neighbourhood r=2σ)
        AND area < DENSITY_MAX_AREA           → airspace arc / circle pieces
  F5  max_dim > CHORD_MIN_DIM
        AND chord (= area/perimeter) < CHORD_THRESH
        AND solidity < SOLID_THRESH
        AND euler ≥ 0   (euler<0 = has holes = text char)  → curved lines

  STAGE 3 — TEMPLATE-BOOST OVERRIDE
  ───────────────────────────────────
  Any CC removed by F4 or F5 whose centroid sits on a template response
  peak > TMPL_THRESHOLD is REINSTATED.  This prevents the density and
  chord filters from deleting genuine obstacle symbols that happen to sit
  in a slightly dense neighbourhood.

  STAGE 4 — SIZE-GATE  (final pass)
  ──────────────────────────────────
  CCs that survived stages 2–3 but are implausibly large for any symbol
  (max_dim > SYMBOL_MAX_DIM) are removed.  This catches coastline or
  boundary arc fragments that survived all earlier filters.

═══════════════════════════════════════════════════════════════════════
PARAMETERS
═══════════════════════════════════════════════════════════════════════

  --obstacle-ref       path(s) to reference PNG files (space-separated)
                       default: obstacle.png  Double_obstacle.png
  --tmpl-h-min         7     template scale range low bound
  --tmpl-h-max         14    template scale range high bound
  --tmpl-threshold     0.45  NCC score to count as obstacle hit
  --nms-radius         12    non-max-suppression window for NCC peaks

  --noise-max-area     3
  --long-ratio         50
  --long-dim           50
  --thin-max           20
  --density-sigma      8.0
  --density-thresh     0.10  (tighter than phase2b to remove more arcs)
  --density-max-area   100
  --chord-min-dim      25
  --chord-thresh       1.20  (tighter than phase2b)
  --solid-thresh       0.45
  --symbol-max-dim     80    final size gate

═══════════════════════════════════════════════════════════════════════
USAGE
═══════════════════════════════════════════════════════════════════════

  CLI:
      python phase2c_symbol_mask.py \\
          --input   outputs/phase1_preprocessing/Washington_binary.png \\
          --output  outputs/phase2_layer_segmentation/phase2c_only_symbols_binary/Washington_symbols_only.png \\
          --obstacle-ref obstacle.png Double_obstacle.png

  Module:
      from phase2c_symbol_mask import build_symbol_mask
      result = build_symbol_mask(
          binary_path  = "Washington_binary.png",
          ref_paths    = ["obstacle.png", "Double_obstacle.png"],
          output_path  = "Washington_symbols_only.png",
      )
      # result.mask       → (H, W) bool
      # result.ncc_map    → (H, W) float32 — template response map
      # result.detections → list of (y, x, score) NCC peaks
      # result.stats      → filter counts
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from scipy.ndimage import maximum_filter
from skimage import feature, filters, measure

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase2c")


# ── defaults ─────────────────────────────────────────────────────────────────
class Cfg:
    TMPL_H_MIN:       int   = 7
    TMPL_H_MAX:       int   = 14
    TMPL_THRESHOLD:   float = 0.45
    NMS_RADIUS:       int   = 12

    NOISE_MAX_AREA:   int   = 3
    LONG_RATIO:       float = 50.0
    LONG_DIM:         int   = 50
    THIN_MAX:         int   = 20
    DENSITY_SIGMA:    float = 8.0
    DENSITY_THRESH:   float = 0.10
    DENSITY_MAX_AREA: int   = 100
    CHORD_MIN_DIM:    int   = 25
    CHORD_THRESH:     float = 1.20
    SOLID_THRESH:     float = 0.45
    SYMBOL_MAX_DIM:   int   = 80


# ── result dataclass ──────────────────────────────────────────────────────────
@dataclass
class SymbolMaskResult:
    mask:       np.ndarray               # (H,W) bool — True = symbol/text ink
    ncc_map:    np.ndarray               # (H,W) float32 — template response map
    detections: List[Tuple[int,int,float]]  # (y, x, score) NCC peaks

    input_path:  str
    output_path: Optional[str]
    log_path:    Optional[str]

    stats: dict = field(default_factory=dict)
    input_fg_pct:  float = 0.0
    output_fg_pct: float = 0.0
    elapsed_sec:   float = 0.0

    def summary(self) -> str:
        removed = sum(v for k,v in self.stats.items() if k != "kept")
        lines = [
            "=== Phase 2c Result ===",
            f"  Input          : {self.input_path}",
            f"  Input fg       : {self.input_fg_pct:.2f}%",
            f"  Output fg      : {self.output_fg_pct:.2f}%",
            f"  Total CCs in   : {sum(self.stats.values())}",
            f"  Removed        : {removed}",
            f"  Kept           : {self.stats.get('kept', 0)}",
            f"  NCC detections : {len(self.detections)}",
            "  Stage breakdown:",
        ]
        labels = {
            "noise":        "    F1 noise",
            "long_line":    "    F2 long-line",
            "thin":         "    F3 thin-frag",
            "density":      "    F4 density-arc",
            "chord":        "    F5 chord/solid",
            "reinstated":   "    S3 NCC-reinstated",
            "size_gate":    "    S4 size-gated",
            "kept":         "    KEPT",
        }
        for k, label in labels.items():
            v = self.stats.get(k, 0)
            if v:
                lines.append(f"{label:<28}: {v}")
        lines.append(f"  Elapsed        : {self.elapsed_sec:.1f} s")
        if self.output_path:
            lines.append(f"  Output         : {self.output_path}")
        return "\n".join(lines)


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_binary(path: Path) -> np.ndarray:
    """Load Phase-1 binary PNG → bool array  (True = foreground ink)."""
    log.info(f"Loading binary mask: {path.name}")
    arr = np.array(Image.open(str(path)).convert("L"))
    fg  = arr > 128
    log.info(f"  Size: {arr.shape[1]} × {arr.shape[0]} px  |  "
             f"fg: {fg.mean()*100:.2f}%")
    return fg


def _build_templates(
    ref_paths: List[str],
    h_min: int,
    h_max: int,
) -> List[np.ndarray]:
    """
    For each reference image, create binary templates at every integer height
    in [h_min, h_max] using LANCZOS downscaling for smoothness.

    Returns a flat list of 2-D float32 arrays.
    """
    templates: List[np.ndarray] = []
    for rp in ref_paths:
        ref = Image.open(rp).convert("L")
        ra  = np.array(ref)
        rb  = ra < 200          # dark blue ink → True
        # trim to ink bounding box
        rows = np.any(rb, axis=1); cols = np.any(rb, axis=0)
        if not rows.any():
            log.warning(f"Reference {rp} has no ink — skipping.")
            continue
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        rt = rb[r0:r1+1, c0:c1+1].astype(np.uint8) * 255
        rh, rw = rt.shape[:2]
        for h_t in range(h_min, h_max + 1):
            w_t = max(4, round(h_t * rw / rh))
            t   = Image.fromarray(rt).resize((w_t, h_t), Image.LANCZOS)
            ta  = (np.array(t) > 100).astype(np.float32)
            templates.append(ta)
        log.info(f"  {Path(rp).name}: {h_max-h_min+1} templates "
                 f"(h={h_min}–{h_max})")
    log.info(f"  Total templates: {len(templates)}")
    return templates


def _compute_ncc_map(
    fg: np.ndarray,
    templates: List[np.ndarray],
) -> np.ndarray:
    """
    Run NCC for every template; return pixel-wise maximum response.
    Only foreground pixels receive meaningful scores (background is forced
    to −1 after each template to avoid empty-region false positives).
    """
    log.info("Computing multi-scale NCC template response map …")
    fa = fg.astype(np.float32)
    max_resp = np.full(fg.shape, -1.0, dtype=np.float32)

    for i, tmpl in enumerate(templates):
        resp = feature.match_template(fa, tmpl, pad_input=True)
        # Suppress response at non-foreground locations
        resp[~fg] = -1.0
        np.maximum(max_resp, resp, out=max_resp)
        if (i + 1) % 4 == 0 or i == len(templates) - 1:
            log.info(f"  {i+1}/{len(templates)} templates processed  "
                     f"current max={max_resp.max():.3f}")

    log.info(f"  NCC map range: [{max_resp.min():.3f}, {max_resp.max():.3f}]")
    return max_resp


def _find_ncc_peaks(
    ncc_map: np.ndarray,
    threshold: float,
    nms_radius: int,
) -> Tuple[np.ndarray, List[Tuple[int,int,float]]]:
    """
    Non-maximum suppression → set of peak locations above threshold.

    Returns
    -------
    peak_mask   : (H,W) bool — True at NCC peak pixels
    detections  : list of (y, x, score)
    """
    local_max = (
        (ncc_map == maximum_filter(ncc_map, size=nms_radius))
        & (ncc_map > threshold)
    )
    ys, xs  = np.where(local_max)
    scores  = ncc_map[ys, xs]
    order   = np.argsort(scores)[::-1]

    detections: List[Tuple[int,int,float]] = [
        (int(ys[i]), int(xs[i]), float(scores[i])) for i in order
    ]
    log.info(f"  NCC peaks > {threshold}: {len(detections)}")
    return local_max, detections


def _filter_components(
    fg: np.ndarray,
    density: np.ndarray,
    ncc_peak_mask: np.ndarray,
    *,
    # F1
    noise_max_area: int,
    # F2
    long_ratio: float,
    long_dim:   int,
    # F3
    thin_max: int,
    # F4
    density_thresh:   float,
    density_max_area: int,
    density_sigma:    float,
    # F5
    chord_min_dim: int,
    chord_thresh:  float,
    solid_thresh:  float,
    # S4
    symbol_max_dim: int,
) -> Tuple[np.ndarray, set, dict]:
    """
    Label CCs and apply all filters + NCC-boost override.

    Returns (labeled, keep_labels, stats).
    """
    log.info("Labelling connected components …")
    labeled = measure.label(fg, connectivity=2)
    props   = measure.regionprops(labeled)
    log.info(f"  Total CCs: {len(props)}")

    H, W = fg.shape
    r_dens = int(density_sigma * 2)

    keep_labels:  set = set()
    boost_labels: set = set()   # CCs sitting on NCC peaks (may be reinstated)
    stats = dict(noise=0, long_line=0, thin=0, density=0,
                 chord=0, reinstated=0, size_gate=0, kept=0)

    # Identify which labels have a centroid on an NCC peak
    for p in props:
        cy, cx = int(p.centroid[0]), int(p.centroid[1])
        if 0 <= cy < H and 0 <= cx < W and ncc_peak_mask[cy, cx]:
            boost_labels.add(p.label)

    for p in props:
        bb  = p.bbox
        h, w       = bb[2]-bb[0], bb[3]-bb[1]
        max_d      = max(h, w)
        min_d      = max(min(h, w), 1)
        asp        = max_d / min_d
        area       = p.area
        euler      = p.euler_number
        sol        = p.solidity
        chord      = area / max(p.perimeter, 1.0)
        boosted    = p.label in boost_labels

        # ── F1: noise ─────────────────────────────────────────────────────
        if area <= noise_max_area:
            stats["noise"] += 1
            continue

        # ── F2: long straight line ────────────────────────────────────────
        if asp > long_ratio and max_d > long_dim:
            stats["long_line"] += 1
            continue

        # ── F3: 1-px-wide thin fragment ───────────────────────────────────
        if min_d == 1 and area < thin_max:
            stats["thin"] += 1
            continue

        # ── F4: local density (airspace arcs) ─────────────────────────────
        removed_f4 = False
        if area < density_max_area:
            cy = int(np.clip(p.centroid[0], 0, H-1))
            cx = int(np.clip(p.centroid[1], 0, W-1))
            y0, y1 = max(0, cy-r_dens), min(H, cy+r_dens)
            x0, x1 = max(0, cx-r_dens), min(W, cx+r_dens)
            ld = float(density[y0:y1, x0:x1].mean())
            if ld > density_thresh:
                if boosted:                 # ← NCC says "keep this"
                    stats["reinstated"] += 1
                    keep_labels.add(p.label)
                    continue
                stats["density"] += 1
                removed_f4 = True

        if removed_f4:
            continue

        # ── F5: chord/solidity (coastline, large arcs) ────────────────────
        if (max_d > chord_min_dim
                and chord < chord_thresh
                and sol < solid_thresh
                and euler >= 0):
            if boosted:
                stats["reinstated"] += 1
                keep_labels.add(p.label)
                continue
            stats["chord"] += 1
            continue

        # ── S4: final size gate ───────────────────────────────────────────
        if max_d > symbol_max_dim and not boosted:
            stats["size_gate"] += 1
            continue

        keep_labels.add(p.label)
        stats["kept"] += 1

    return labeled, keep_labels, stats


def _reconstruct(labeled: np.ndarray, keep_labels: set) -> np.ndarray:
    mask = np.isin(labeled, list(keep_labels))
    log.info(f"  Clean fg coverage: {mask.mean()*100:.2f}%")
    return mask


def _save(
    mask: np.ndarray,
    output_path: Path,
    metadata: dict,
) -> Tuple[str, str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = output_path.with_suffix(".json")

    log.info(f"Saving symbol mask → {output_path}")
    Image.fromarray((mask * 255).astype(np.uint8)).save(
        str(output_path), format="PNG", compress_level=3
    )
    log.info(f"Saving log         → {log_path}")
    with open(log_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    return str(output_path), str(log_path)


# ── public API ────────────────────────────────────────────────────────────────

def build_symbol_mask(
    binary_path: str,
    ref_paths:   List[str],
    output_path: Optional[str] = None,
    *,
    tmpl_h_min:       int   = Cfg.TMPL_H_MIN,
    tmpl_h_max:       int   = Cfg.TMPL_H_MAX,
    tmpl_threshold:   float = Cfg.TMPL_THRESHOLD,
    nms_radius:       int   = Cfg.NMS_RADIUS,
    noise_max_area:   int   = Cfg.NOISE_MAX_AREA,
    long_ratio:       float = Cfg.LONG_RATIO,
    long_dim:         int   = Cfg.LONG_DIM,
    thin_max:         int   = Cfg.THIN_MAX,
    density_sigma:    float = Cfg.DENSITY_SIGMA,
    density_thresh:   float = Cfg.DENSITY_THRESH,
    density_max_area: int   = Cfg.DENSITY_MAX_AREA,
    chord_min_dim:    int   = Cfg.CHORD_MIN_DIM,
    chord_thresh:     float = Cfg.CHORD_THRESH,
    solid_thresh:     float = Cfg.SOLID_THRESH,
    symbol_max_dim:   int   = Cfg.SYMBOL_MAX_DIM,
) -> SymbolMaskResult:
    """
    Build a binary mask containing only aeronautical symbols and text.

    Parameters
    ----------
    binary_path : str
        Phase-1 binary PNG (white foreground on black background).
    ref_paths : list[str]
        Paths to FAA obstacle reference images
        (e.g. ["obstacle.png", "Double_obstacle.png"]).
    output_path : str, optional
        Where to save the output PNG.  If None, only the array is returned.

    All filter thresholds are keyword-only; see module docstring for details.

    Returns
    -------
    SymbolMaskResult
    """
    t0  = time.time()
    src = Path(binary_path)
    if not src.exists():
        raise FileNotFoundError(f"Binary mask not found: {binary_path}")

    # ── 1. Load ───────────────────────────────────────────────────────────
    fg            = _load_binary(src)
    input_fg_pct  = float(fg.mean() * 100)

    # ── 2. Build templates ────────────────────────────────────────────────
    templates = _build_templates(ref_paths, tmpl_h_min, tmpl_h_max)

    # ── 3. NCC response map ───────────────────────────────────────────────
    if templates:
        ncc_map = _compute_ncc_map(fg, templates)
    else:
        log.warning("No valid templates — NCC boost disabled.")
        ncc_map = np.full(fg.shape, -1.0, dtype=np.float32)

    # ── 4. NCC peaks (NMS) ────────────────────────────────────────────────
    ncc_peak_mask, detections = _find_ncc_peaks(
        ncc_map, threshold=tmpl_threshold, nms_radius=nms_radius
    )

    # ── 5. Density map ────────────────────────────────────────────────────
    log.info(f"Computing local density map (σ={density_sigma}) …")
    density = filters.gaussian(fg.astype(np.float32), sigma=density_sigma)

    # ── 6. CC filter + NCC boost ──────────────────────────────────────────
    labeled, keep_labels, stats = _filter_components(
        fg, density, ncc_peak_mask,
        noise_max_area   = noise_max_area,
        long_ratio       = long_ratio,
        long_dim         = long_dim,
        thin_max         = thin_max,
        density_thresh   = density_thresh,
        density_max_area = density_max_area,
        density_sigma    = density_sigma,
        chord_min_dim    = chord_min_dim,
        chord_thresh     = chord_thresh,
        solid_thresh     = solid_thresh,
        symbol_max_dim   = symbol_max_dim,
    )

    # ── 7. Reconstruct ───────────────────────────────────────────────────
    log.info(f"Reconstructing mask from {len(keep_labels)} kept CCs …")
    mask = _reconstruct(labeled, keep_labels)
    output_fg_pct = float(mask.mean() * 100)

    # ── 8. Save ───────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    metadata = {
        "input":       str(src),
        "ref_paths":   ref_paths,
        "parameters": {
            "tmpl_h_min":       tmpl_h_min,
            "tmpl_h_max":       tmpl_h_max,
            "tmpl_threshold":   tmpl_threshold,
            "nms_radius":       nms_radius,
            "noise_max_area":   noise_max_area,
            "long_ratio":       long_ratio,
            "long_dim":         long_dim,
            "thin_max":         thin_max,
            "density_sigma":    density_sigma,
            "density_thresh":   density_thresh,
            "density_max_area": density_max_area,
            "chord_min_dim":    chord_min_dim,
            "chord_thresh":     chord_thresh,
            "solid_thresh":     solid_thresh,
            "symbol_max_dim":   symbol_max_dim,
        },
        "stats":          stats,
        "detections":     detections[:200],   # cap to 200 for JSON size
        "input_fg_pct":   round(input_fg_pct,  4),
        "output_fg_pct":  round(output_fg_pct, 4),
        "elapsed_sec":    round(elapsed, 2),
    }

    out_path_str = log_path_str = None
    if output_path:
        out_path_str, log_path_str = _save(mask, Path(output_path), metadata)

    result = SymbolMaskResult(
        mask        = mask,
        ncc_map     = ncc_map,
        detections  = detections,
        input_path  = str(src),
        output_path = out_path_str,
        log_path    = log_path_str,
        stats       = stats,
        input_fg_pct  = input_fg_pct,
        output_fg_pct = output_fg_pct,
        elapsed_sec   = elapsed,
    )
    log.info(result.summary())
    return result


if __name__ == "__main__":
    # ── CONFIG ──────────────────────────────────────────────────────────────────
    INPUT_PATH    = "outputs/phase1_preprocessing/chart_binary.png"  # Phase 1 binary PNG
    OUTPUT_PATH   = None  # None = <stem>_symbols_only.png next to input
    OBSTACLE_REFS = ["obstacle.png", "Double_obstacle.png"]  # FAA obstacle reference PNGs
    # ────────────────────────────────────────────────────────────────────────────

    out = OUTPUT_PATH
    if out is None:
        src = Path(INPUT_PATH)
        out = str(src.parent / f"{src.stem}_symbols_only.png")

    build_symbol_mask(
        binary_path      = INPUT_PATH,
        ref_paths        = OBSTACLE_REFS,
        output_path      = out,
        tmpl_h_min       = Cfg.TMPL_H_MIN,
        tmpl_h_max       = Cfg.TMPL_H_MAX,
        tmpl_threshold   = Cfg.TMPL_THRESHOLD,
        nms_radius       = Cfg.NMS_RADIUS,
        noise_max_area   = Cfg.NOISE_MAX_AREA,
        long_ratio       = Cfg.LONG_RATIO,
        long_dim         = Cfg.LONG_DIM,
        thin_max         = Cfg.THIN_MAX,
        density_sigma    = Cfg.DENSITY_SIGMA,
        density_thresh   = Cfg.DENSITY_THRESH,
        density_max_area = Cfg.DENSITY_MAX_AREA,
        chord_min_dim    = Cfg.CHORD_MIN_DIM,
        chord_thresh     = Cfg.CHORD_THRESH,
        solid_thresh     = Cfg.SOLID_THRESH,
        symbol_max_dim   = Cfg.SYMBOL_MAX_DIM,
    )