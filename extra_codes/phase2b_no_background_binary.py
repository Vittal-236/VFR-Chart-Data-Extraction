"""
Phase 2b — Symbol & Text Extraction
VFR Chart Extraction Pipeline (FAA Base Model)

Strips all background infrastructure (airways, airspace boundaries, coastlines,
lat/lon grid, topo contours) from the Phase 1 binary mask, leaving only
aeronautical symbols and text annotation blobs ready for Phase 4 (symbol
detection) and Phase 5 (OCR).

─────────────────────────────────────────────────────────────────────────────
DESIGN RATIONALE
─────────────────────────────────────────────────────────────────────────────
At 300 DPI, a VFR sectional binary mask contains four classes of foreground
ink that must be separated:

  CLASS A — Background lines/curves  (REMOVE)
      Airspace boundary arcs, airways, coastline, ADIZ boundary,
      lat/lon grid tick marks, topo contour lines.
      Signature: long/thin, low fill-ratio, large bounding box.

  CLASS B — Noise                    (REMOVE)
      Isolated 1–3 px salt pixels from JPEG compression artifacts,
      Sauvola threshold noise, and print grain.

  CLASS C — Aeronautical symbols     (KEEP)
      Obstacle triangles, airport icons, VOR/NAVAID rosettes,
      restricted-area ticks, waypoint marks, etc.
      Signature: compact 2-D blob, bounded size, moderate fill.

  CLASS D — Text annotations         (KEEP)
      Altitude labels, identifier strings, frequency boxes,
      WARNING area names, legend text.
      Signature: multi-character clusters with letter holes (Euler < 0)
      or individually small character glyphs.

─────────────────────────────────────────────────────────────────────────────
ALGORITHM  (5 sequential filters on connected components)
─────────────────────────────────────────────────────────────────────────────

  F1 — NOISE REMOVAL
       Remove any CC with area ≤ NOISE_MAX_AREA (default 3 px).
       Catches single-pixel and 2–3 px salt grains.

  F2 — LONG-LINE REMOVAL
       Remove CCs whose bounding-box aspect ratio > ASPECT_LINE_RATIO
       AND max bounding-box dimension > LINE_MIN_DIM.
       Catches full-width tick-mark lines, long straight airways that
       survived as a single CC.

  F3 — THIN-FRAGMENT REMOVAL
       Remove CCs whose minimum bounding-box dimension == 1 px
       AND area < THIN_MAX_AREA.
       Catches short 1-px-wide dashes that are broken airway/grid fragments.

  F4 — LOCAL DENSITY FILTER
       Convolve the binary mask with a Gaussian (σ = DENSITY_SIGMA) to
       produce a local ink-density map.  A CC whose centroid sits in a
       region with density > DENSITY_THRESHOLD is almost certainly part of
       a continuous line network (airspace circle, coastline arc, etc.).
       Only small CCs (area < DENSITY_MAX_AREA) are subject to this filter;
       large unique symbols are immune.

  F5 — CHORD / SOLIDITY FILTER  (thin-stroke geometry)
       For CCs that survived F4, compute:
           chord  = area / perimeter          (mean stroke half-width)
           euler  = euler_number              (< 0 when the CC has holes)
       Remove if: max_dim > CHORD_MIN_DIM
                  AND chord < CHORD_THRESHOLD
                  AND solidity < SOLIDITY_THRESHOLD
                  AND euler ≥ 0
       Rationale:
         • Lines/arcs have chord ≈ 0.8–1.3 (thin single-pixel strokes).
         • Filled symbol blobs have chord > 2.
         • Text characters with holes (O, B, R, A, P …) have euler < 0;
           protecting them prevents over-removal of annotation text.

─────────────────────────────────────────────────────────────────────────────
PARAMETERS  (all tunable via CLI or module API)
─────────────────────────────────────────────────────────────────────────────

  --noise-max-area        3       F1 area ceiling for pure noise
  --aspect-line-ratio    50       F2 minimum aspect ratio for long lines
  --line-min-dim         50       F2 minimum max-dim for long lines
  --thin-max-area        20       F3 area ceiling for thin fragments
  --density-sigma         8       F4 Gaussian σ (px) for local density map
  --density-threshold     0.12    F4 ink-density threshold
  --density-max-area    100       F4 only apply to CCs smaller than this
  --chord-min-dim        25       F5 min max-dim to apply chord filter
  --chord-threshold       1.30    F5 chord upper bound for line removal
  --solidity-threshold    0.45    F5 solidity upper bound for line removal

─────────────────────────────────────────────────────────────────────────────
USAGE
─────────────────────────────────────────────────────────────────────────────

  CLI:
      python phase2b_symbol_extraction.py \\
          --input  outputs/phase1_preprocessing/Washington_binary.png \\
          --output outputs/phase2_layer_segmentation/phase2b_no_background_binary/Washington_symbols.png

  Module:
      from phase2b_symbol_extraction import extract_symbols
      result = extract_symbols(
          binary_path="outputs/phase1_preprocessing/Washington_binary.png",
          output_path="outputs/phase2_layer_segmentation/phase2b_no_background_binary/Washington_symbols.png",
      )
      # result.clean_array  → (H, W) bool, True = symbol/text ink
      # result.stats        → dict of removal counts per filter
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from skimage import filters, measure, morphology

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase2b")


# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------
class Defaults:
    # F1 — noise
    NOISE_MAX_AREA: int = 3

    # F2 — long straight lines
    ASPECT_LINE_RATIO: float = 50.0
    LINE_MIN_DIM: int = 50

    # F3 — 1-px-wide thin fragments
    THIN_MAX_AREA: int = 20

    # F4 — local density (airspace arcs, circles)
    DENSITY_SIGMA: float = 8.0
    DENSITY_THRESHOLD: float = 0.12
    DENSITY_MAX_AREA: int = 100

    # F5 — chord/solidity (coastline, large arcs that survive F4)
    CHORD_MIN_DIM: int = 25
    CHORD_THRESHOLD: float = 1.30
    SOLIDITY_THRESHOLD: float = 0.45


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class ExtractionResult:
    """All outputs produced by Phase 2b."""

    clean_array: np.ndarray        # (H, W) bool — symbols + text only

    # Paths
    input_path: str
    output_path: Optional[str]
    log_path: Optional[str]

    # Filter statistics
    stats: dict = field(default_factory=dict)

    # Pixel coverage
    input_fg_pct: float = 0.0
    output_fg_pct: float = 0.0

    # Timing
    elapsed_sec: float = 0.0

    def summary(self) -> str:
        removed = sum(v for k, v in self.stats.items() if k != "kept")
        lines = [
            "=== Phase 2b Result ===",
            f"  Input          : {self.input_path}",
            f"  Input fg       : {self.input_fg_pct:.2f}%",
            f"  Output fg      : {self.output_fg_pct:.2f}%",
            f"  Total CCs in   : {sum(self.stats.values())}",
            f"  Total removed  : {removed}",
            f"  Total kept     : {self.stats.get('kept', 0)}",
            "  Breakdown:",
        ]
        labels = {
            "noise":        "  F1 noise",
            "long_line":    "  F2 long-line",
            "thin_frag":    "  F3 thin-frag",
            "density_line": "  F4 density",
            "chord_line":   "  F5 chord/solid",
            "kept":         "  KEPT",
        }
        for key, label in labels.items():
            lines.append(f"    {label:<20}: {self.stats.get(key, 0)}")
        lines.append(f"  Elapsed        : {self.elapsed_sec:.1f} s")
        if self.output_path:
            lines.append(f"  Output         : {self.output_path}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core extraction logic
# ---------------------------------------------------------------------------

def _load_binary(path: Path) -> np.ndarray:
    """
    Load Phase 1 binary PNG.

    Phase 1 saves foreground as white (255) on black (0).
    We threshold at >128 to produce a bool array where True = ink.
    """
    log.info(f"Loading binary mask: {path.name}")
    arr = np.array(Image.open(str(path)).convert("L"))
    fg = arr > 128
    log.info(f"  Size: {arr.shape[1]} x {arr.shape[0]} px  |  "
             f"Foreground: {fg.mean() * 100:.2f}%")
    return fg


def _compute_density_map(fg: np.ndarray, sigma: float) -> np.ndarray:
    """
    Gaussian-smoothed binary → local ink-density map.

    High values indicate pixels surrounded by many foreground neighbours
    (typical of continuous line networks).  Low values indicate isolated
    blobs (typical of symbols and text in open chart areas).
    """
    log.info(f"Computing local density map (σ={sigma}) …")
    density = filters.gaussian(fg.astype(np.float32), sigma=sigma)
    log.info(f"  Density range: [{density.min():.4f}, {density.max():.4f}]")
    return density


def _classify_components(
    fg: np.ndarray,
    density: np.ndarray,
    # F1
    noise_max_area: int,
    # F2
    aspect_line_ratio: float,
    line_min_dim: int,
    # F3
    thin_max_area: int,
    # F4
    density_threshold: float,
    density_max_area: int,
    density_sigma: float,
    # F5
    chord_min_dim: int,
    chord_threshold: float,
    solidity_threshold: float,
) -> tuple:
    """
    Label every connected component and apply filters F1–F5.

    Returns
    -------
    labeled      : (H, W) int32 label array
    keep_labels  : set of integer labels to retain
    stats        : dict with per-filter removal counts
    """
    log.info("Labelling connected components …")
    labeled = measure.label(fg, connectivity=2)
    props = measure.regionprops(labeled)
    log.info(f"  Total CCs: {len(props)}")

    keep_labels: set = set()
    stats = {
        "noise": 0,
        "long_line": 0,
        "thin_frag": 0,
        "density_line": 0,
        "chord_line": 0,
        "kept": 0,
    }

    H, W = fg.shape
    r_density = int(density_sigma * 2)  # neighbourhood radius for centroid density

    for p in props:
        bb = p.bbox                          # (min_row, min_col, max_row, max_col)
        h = bb[2] - bb[0]
        w = bb[3] - bb[1]
        max_dim = max(h, w)
        min_dim = max(min(h, w), 1)
        aspect = max_dim / min_dim
        area = p.area
        euler = p.euler_number
        solidity = p.solidity
        perimeter = max(p.perimeter, 1.0)
        chord = area / perimeter            # mean stroke half-width proxy

        # ── F1: Noise ────────────────────────────────────────────────────
        if area <= noise_max_area:
            stats["noise"] += 1
            continue

        # ── F2: Long straight line ───────────────────────────────────────
        if aspect > aspect_line_ratio and max_dim > line_min_dim:
            stats["long_line"] += 1
            continue

        # ── F3: 1-px-wide thin fragment ──────────────────────────────────
        if min_dim == 1 and area < thin_max_area:
            stats["thin_frag"] += 1
            continue

        # ── F4: Local density (airspace circles / arc clusters) ──────────
        if area < density_max_area:
            cy = int(np.clip(p.centroid[0], 0, H - 1))
            cx = int(np.clip(p.centroid[1], 0, W - 1))
            y0, y1 = max(0, cy - r_density), min(H, cy + r_density)
            x0, x1 = max(0, cx - r_density), min(W, cx + r_density)
            local_density = float(density[y0:y1, x0:x1].mean())
            if local_density > density_threshold:
                stats["density_line"] += 1
                continue

        # ── F5: Chord / solidity (coastline, large arcs) ─────────────────
        # Only apply to components with large bounding boxes; protects
        # small symbols.  Euler ≥ 0 guard spares text with character holes.
        if (max_dim > chord_min_dim
                and chord < chord_threshold
                and solidity < solidity_threshold
                and euler >= 0):
            stats["chord_line"] += 1
            continue

        # ── Passed all filters — keep ─────────────────────────────────────
        keep_labels.add(p.label)
        stats["kept"] += 1

    return labeled, keep_labels, stats


def _reconstruct(labeled: np.ndarray, keep_labels: set) -> np.ndarray:
    """Rebuild a bool array from the set of kept CC labels."""
    log.info(f"Reconstructing clean mask from {len(keep_labels)} kept CCs …")
    clean = np.isin(labeled, list(keep_labels))
    log.info(f"  Clean fg coverage: {clean.mean() * 100:.2f}%")
    return clean


def _save_outputs(
    clean: np.ndarray,
    output_path: Path,
    metadata: dict,
) -> tuple:
    """
    Save clean binary PNG and JSON log.
    Returns (output_path_str, log_path_str).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = output_path.with_suffix(".json")

    log.info(f"Saving clean binary → {output_path}")
    Image.fromarray((clean * 255).astype(np.uint8)).save(
        str(output_path), format="PNG", compress_level=3
    )

    log.info(f"Saving processing log → {log_path}")
    with open(log_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return str(output_path), str(log_path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_symbols(
    binary_path: str,
    output_path: Optional[str] = None,
    # F1
    noise_max_area: int = Defaults.NOISE_MAX_AREA,
    # F2
    aspect_line_ratio: float = Defaults.ASPECT_LINE_RATIO,
    line_min_dim: int = Defaults.LINE_MIN_DIM,
    # F3
    thin_max_area: int = Defaults.THIN_MAX_AREA,
    # F4
    density_sigma: float = Defaults.DENSITY_SIGMA,
    density_threshold: float = Defaults.DENSITY_THRESHOLD,
    density_max_area: int = Defaults.DENSITY_MAX_AREA,
    # F5
    chord_min_dim: int = Defaults.CHORD_MIN_DIM,
    chord_threshold: float = Defaults.CHORD_THRESHOLD,
    solidity_threshold: float = Defaults.SOLIDITY_THRESHOLD,
) -> ExtractionResult:
    """
    Extract aeronautical symbols and text from a Phase 1 binary mask.

    Parameters
    ----------
    binary_path : str
        Path to Phase 1 binary PNG (white foreground on black background).
    output_path : str, optional
        Destination PNG for the cleaned binary.  If None, no file is saved.

    Filter parameters (see module docstring for full explanation):
    noise_max_area, aspect_line_ratio, line_min_dim, thin_max_area,
    density_sigma, density_threshold, density_max_area,
    chord_min_dim, chord_threshold, solidity_threshold.

    Returns
    -------
    ExtractionResult
        .clean_array  → (H, W) bool, True = symbol/text ink
        .stats        → per-filter removal counts
        .output_path  → saved PNG path (or None)
    """
    t0 = time.time()
    src = Path(binary_path)
    if not src.exists():
        raise FileNotFoundError(f"Binary mask not found: {binary_path}")

    # ── Step 1: Load ──────────────────────────────────────────────────────
    fg = _load_binary(src)
    input_fg_pct = float(fg.mean() * 100)

    # ── Step 2: Density map for F4 ────────────────────────────────────────
    density = _compute_density_map(fg, sigma=density_sigma)

    # ── Step 3: CC classification + filtering ─────────────────────────────
    labeled, keep_labels, stats = _classify_components(
        fg=fg,
        density=density,
        noise_max_area=noise_max_area,
        aspect_line_ratio=aspect_line_ratio,
        line_min_dim=line_min_dim,
        thin_max_area=thin_max_area,
        density_threshold=density_threshold,
        density_max_area=density_max_area,
        density_sigma=density_sigma,
        chord_min_dim=chord_min_dim,
        chord_threshold=chord_threshold,
        solidity_threshold=solidity_threshold,
    )

    # ── Step 4: Reconstruct clean mask ────────────────────────────────────
    clean = _reconstruct(labeled, keep_labels)
    output_fg_pct = float(clean.mean() * 100)

    # ── Step 5: Persist ───────────────────────────────────────────────────
    elapsed = time.time() - t0
    out_path_str = log_path_str = None

    metadata = {
        "input": str(src),
        "filters": {
            "F1_noise_max_area": noise_max_area,
            "F2_aspect_line_ratio": aspect_line_ratio,
            "F2_line_min_dim": line_min_dim,
            "F3_thin_max_area": thin_max_area,
            "F4_density_sigma": density_sigma,
            "F4_density_threshold": density_threshold,
            "F4_density_max_area": density_max_area,
            "F5_chord_min_dim": chord_min_dim,
            "F5_chord_threshold": chord_threshold,
            "F5_solidity_threshold": solidity_threshold,
        },
        "stats": stats,
        "input_fg_pct": round(input_fg_pct, 4),
        "output_fg_pct": round(output_fg_pct, 4),
        "elapsed_sec": round(elapsed, 2),
    }

    if output_path:
        out_path_str, log_path_str = _save_outputs(
            clean, Path(output_path), metadata
        )

    result = ExtractionResult(
        clean_array=clean,
        input_path=str(src),
        output_path=out_path_str,
        log_path=log_path_str,
        stats=stats,
        input_fg_pct=input_fg_pct,
        output_fg_pct=output_fg_pct,
        elapsed_sec=elapsed,
    )
    log.info(result.summary())
    return result


if __name__ == "__main__":
    # ── CONFIG ──────────────────────────────────────────────────────────────────
    INPUT_PATH  = "outputs/phase1_preprocessing/chart_binary.png"  # Phase 1 binary PNG
    OUTPUT_PATH = None  # None = <input_stem>_symbols.png next to input
    # ────────────────────────────────────────────────────────────────────────────

    out = OUTPUT_PATH
    if out is None:
        src = Path(INPUT_PATH)
        out = str(src.parent / f"{src.stem}_symbols.png")

    extract_symbols(
        binary_path=INPUT_PATH,
        output_path=out,
        noise_max_area=Defaults.NOISE_MAX_AREA,
        aspect_line_ratio=Defaults.ASPECT_LINE_RATIO,
        line_min_dim=Defaults.LINE_MIN_DIM,
        thin_max_area=Defaults.THIN_MAX_AREA,
        density_sigma=Defaults.DENSITY_SIGMA,
        density_threshold=Defaults.DENSITY_THRESHOLD,
        density_max_area=Defaults.DENSITY_MAX_AREA,
        chord_min_dim=Defaults.CHORD_MIN_DIM,
        chord_threshold=Defaults.CHORD_THRESHOLD,
        solidity_threshold=Defaults.SOLIDITY_THRESHOLD,
    )