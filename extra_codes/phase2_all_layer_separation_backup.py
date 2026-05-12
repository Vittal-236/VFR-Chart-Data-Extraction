"""
Phase 2 — Synthetic Layer Separation
VFR Chart Extraction Pipeline (FAA Base Model)

Decomposes the preprocessed RGB chart into four binary masks:
    - text       : all printed labels, frequencies, identifiers, numbers
    - routes     : airways, VFR corridors, lat/lon grid lines
    - symbols    : airport icons, NAVAIDs, waypoints, obstacles
    - boundaries : airspace rings (Class B/C/D/E), Warning/Restricted area polygons

Method: Tier 1 — HSV decomposition (FAA-specific hardcoded palette).
FAA sectionals use a stable, standardised colour encoding that makes
HSV-range masking reliable without any machine learning.

FAA Colour Encoding
-------------------
  Magenta / Pink  → untowered airports, NDB NAVAIDs, Class E airspace boundaries
  Blue            → towered airports, VOR NAVAIDs, water, Class B/C/D boundaries
  Grey/Black      → text, obstacles, lat/lon grid, general linework
  Brown/Tan       → terrain contours and elevation shading (background — excluded)
  Green           → terrain elevation bands (background — excluded)

Pipeline
--------
  Phase 1 RGB  →  HSV conversion (tiled, memory-safe)
                →  per-channel boolean masks
                →  morphological cleanup
                →  self-validation (coverage checks)
                →  LayerResult

Usage (module):
    from phase2_layer_separation import separate_layers
    from phase1_preprocessing import preprocess_chart

    p1 = preprocess_chart("Washington.pdf", output_dir="outputs/phase1", dpi=300)
    result = separate_layers(p1.rgb_array, output_dir="outputs/phase2")

Usage (CLI):
    python phase2_layer_separation.py --input outputs/phase1/Washington_rgb.png --output-dir outputs/phase2
"""

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from skimage import color, morphology
from skimage.morphology import disk, remove_small_objects, skeletonize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase2")


# ---------------------------------------------------------------------------
# FAA HSV Palette Definition
# ---------------------------------------------------------------------------
# All ranges are in skimage HSV space: H in [0,1], S in [0,1], V in [0,1]
# Each entry: (h_min, h_max, s_min, s_max, v_min, v_max)
#
# Derivation: sampled from the Washington sectional RGB image and converted.
# These are intentionally slightly wide to handle print variation and
# JPEG compression artefacts. The morphological cleanup step handles any
# noise introduced by the loose bounds.
#
# FAA sectionals are printed on cyan-tinted paper; the scanner/PDF export
# can shift hues slightly. If a layer mask is unexpectedly empty, widen
# the S or V bounds first before touching H.

FAA_HSV_RANGES = {

    # ── MAGENTA layer ─────────────────────────────────────────────────────
    # Captures: untowered airport circles (magenta), NDB compass roses,
    #           Class E surface area dashed boundaries, special use airspace.
    # H wraps around 0/360°: needs two ranges.
    # Widened S_min 0.25→0.15 to catch faded/compressed magenta ink.
    "magenta": [
        (0.78, 1.00, 0.15, 1.00, 0.25, 1.00),   # H: 280–360°
        (0.00, 0.08, 0.15, 1.00, 0.25, 1.00),   # H: 0–29° (wrap-around)
    ],

    # ── BLUE layer ────────────────────────────────────────────────────────
    # Captures: towered airport circles, VOR symbols, Class B/C/D rings,
    #           water bodies, lat/lon grid ticks.
    # Widened slightly for cyan-tinted water areas.
    "blue": [
        (0.55, 0.72, 0.15, 1.00, 0.20, 1.00),
    ],

    # ── BLACK / DARK layer ────────────────────────────────────────────────
    # Captures: all text, obstacle symbols, airways, lat/lon grid, linework.
    # V < 0.45 = dark pixels regardless of hue.
    # Neutral grey (V 0.45–0.70, S < 0.15) catches grey routes and grid.
    "black": [
        (0.00, 1.00, 0.00, 1.00, 0.00, 0.45),   # true dark / black
        (0.00, 1.00, 0.00, 0.15, 0.45, 0.70),   # neutral grey (routes, grid)
    ],

    # ── TERRAIN layer (background exclusion mask) ─────────────────────────
    # Captured ONLY to subtract from the black mask so brown/green terrain
    # contour lines and elevation shading don't pollute text/routes.
    # Widened to cover brown (H 0.04–0.16) AND green (H 0.16–0.42)
    # elevation bands in the western Appalachian section.
    "terrain": [
        (0.05, 0.15, 0.08, 0.55, 0.55, 1.00),   # brown / tan contours
    ],
}

# ---------------------------------------------------------------------------
# Coverage validation bounds
# After separation, each extraction layer (not terrain) must cover
# between COVERAGE_MIN and COVERAGE_MAX percent of chart pixels.
# Outside these bounds → separation is suspect → log a warning.
# ---------------------------------------------------------------------------
COVERAGE_BOUNDS = {
    "magenta":    (0.5,  8.0),
    "blue":       (1.0, 15.0),
    "black":      (5.0, 40.0),
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class LayerResult:
    """All outputs produced by Phase 2."""

    # Binary masks — True = pixel belongs to this layer
    text_mask:       np.ndarray    # (H, W) bool
    routes_mask:     np.ndarray    # (H, W) bool
    symbols_mask:    np.ndarray    # (H, W) bool
    boundaries_mask: np.ndarray    # (H, W) bool

    # Raw HSV masks before morphological processing (useful for debugging)
    magenta_raw: np.ndarray        # (H, W) bool
    blue_raw:    np.ndarray        # (H, W) bool
    black_raw:   np.ndarray        # (H, W) bool

    # Coverage stats (% of total pixels)
    coverage: dict = field(default_factory=dict)

    # Disk paths
    text_path:       Optional[str] = None
    routes_path:     Optional[str] = None
    symbols_path:    Optional[str] = None
    boundaries_path: Optional[str] = None
    log_path:        Optional[str] = None

    elapsed_sec: float = 0.0
    notes: list = field(default_factory=list)

    def summary(self) -> str:
        lines = ["=== Phase 2 Result ==="]
        for name, pct in self.coverage.items():
            lines.append(f"  {name:<12}: {pct:.1f}% coverage")
        lines.append(f"  Elapsed      : {self.elapsed_sec:.1f} s")
        if self.notes:
            lines.append("  Notes:")
            for n in self.notes:
                lines.append(f"    - {n}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step helpers
# ---------------------------------------------------------------------------

def _hsv_mask(hsv: np.ndarray, ranges: list) -> np.ndarray:
    """
    Build a boolean mask for pixels matching any of the given HSV ranges.

    Each range is a tuple: (h_min, h_max, s_min, s_max, v_min, v_max).
    The mask is the union (OR) of all ranges — useful when a colour wraps
    around the 0/360° hue boundary (e.g., magenta = 300–360° OR 0–18°).

    Parameters
    ----------
    hsv    : (H, W, 3) float array in [0, 1] — output of skimage.color.rgb2hsv
    ranges : list of 6-tuples

    Returns
    -------
    (H, W) bool array
    """
    mask = np.zeros(hsv.shape[:2], dtype=bool)
    for (h0, h1, s0, s1, v0, v1) in ranges:
        m = (
            (hsv[..., 0] >= h0) & (hsv[..., 0] <= h1) &
            (hsv[..., 1] >= s0) & (hsv[..., 1] <= s1) &
            (hsv[..., 2] >= v0) & (hsv[..., 2] <= v1)
        )
        mask |= m
    return mask


def _rgb_to_hsv_tiled(rgb: np.ndarray, tile_size: int = 2048) -> np.ndarray:
    """
    Convert RGB to HSV in tiles to avoid a large float64 allocation.

    skimage.color.rgb2hsv converts to float64 internally. At 300 DPI
    (16 770 × 12 345 px) the output is ~3.1 GiB. Tiling keeps peak RAM
    near tile_size² × 3 × 8 bytes ≈ 96 MB for tile_size=2048.

    Returns an (H, W, 3) float32 HSV array (sufficient precision for
    boolean range comparisons; saves ~50% RAM vs float64).
    """
    log.info("Converting RGB → HSV (tiled) …")
    H, W = rgb.shape[:2]
    hsv = np.empty((H, W, 3), dtype=np.float32)

    n_ty = int(np.ceil(H / tile_size))
    n_tx = int(np.ceil(W / tile_size))
    total = n_ty * n_tx
    done = 0

    for ty in range(n_ty):
        for tx in range(n_tx):
            y0, y1 = ty * tile_size, min((ty + 1) * tile_size, H)
            x0, x1 = tx * tile_size, min((tx + 1) * tile_size, W)
            tile_rgb = rgb[y0:y1, x0:x1].astype(np.float64) / 255.0
            hsv[y0:y1, x0:x1] = color.rgb2hsv(tile_rgb).astype(np.float32)
            done += 1
            log.info(f"  HSV tile {done}/{total}")

    log.info("  HSV conversion complete.")
    return hsv


def _build_raw_masks(hsv: np.ndarray, notes: list) -> dict:
    """
    Apply FAA HSV palette ranges to produce raw boolean masks.

    The 'terrain' mask is used only to subtract from 'black' — terrain
    contour lines are dark brown and would otherwise pollute the text layer.
    """
    log.info("Applying FAA HSV palette masks …")

    raw = {}
    for layer, ranges in FAA_HSV_RANGES.items():
        raw[layer] = _hsv_mask(hsv, ranges)
        pct = raw[layer].mean() * 100
        log.info(f"  Raw {layer:<10}: {pct:.1f}% coverage")

    # Subtract terrain from black to remove brown contour lines
    terrain_dilated = morphology.dilation(raw["terrain"], disk(2))
    raw["black"] = raw["black"] & ~terrain_dilated
    log.info(f"  Black after terrain subtraction: {raw['black'].mean()*100:.1f}%")

    # Coverage validation
    for layer, (lo, hi) in COVERAGE_BOUNDS.items():
        pct = raw[layer].mean() * 100
        if pct < lo:
            msg = (f"Layer '{layer}' coverage {pct:.1f}% < minimum {lo}% — "
                   "HSV ranges may need widening.")
            log.warning(f"  WARN: {msg}")
            notes.append(msg)
        elif pct > hi:
            msg = (f"Layer '{layer}' coverage {pct:.1f}% > maximum {hi}% — "
                   "HSV ranges may be too broad.")
            log.warning(f"  WARN: {msg}")
            notes.append(msg)

    return raw


def _derive_semantic_masks(raw: dict, notes: list) -> dict:
    """
    Derive the four semantic output masks from raw HSV colour masks.

    Derivation logic
    ----------------

    TEXT mask
        Source: black layer (all dark pixels)
        Processing:
          - remove_small_objects(min_size=8) to drop noise specks
          - binary_opening(disk(1)) to disconnect weakly-joined blobs
          - We do NOT skeletonize — text strokes need their full width for OCR

    ROUTES mask
        Source: black layer (thin black airways) + blue layer (blue routes)
        Processing:
          - Keep only thin structures: skeletonize to 1-px centre lines
          - remove_small_objects(min_size=50) to drop isolated symbol pixels
          - binary_closing(disk(1)) to bridge 1-px gaps in dashed airways
        Note: the skeleton of a dashed line has isolated segments — that's
        expected and handled in Phase 7 (raster-to-vector).

    SYMBOLS mask
        Source: magenta + blue (coloured airport/NAVAID icons)
        Processing:
          - binary_closing(disk(3)) to fill internal gaps in symbol outlines
          - remove_small_objects(min_size=20) to drop noise
          - remove_small_objects(min_size=500, invert) to drop large filled
            regions (water bodies, large boundary fills) — keep only blobs
            in the symbol size range (20–500 px at 150 DPI)

    BOUNDARIES mask
        Source: magenta + blue (airspace boundary arcs)
        Processing:
          - binary_closing(disk(5)) to bridge gaps in dashed/dotted boundary lines
          - remove_small_objects(min_size=200) to keep only large structures
          - The result includes both the boundary line and the coloured
            airport/NAVAID symbols — Phase 4 will separate them by shape.
    """
    log.info("Deriving semantic masks …")
    sem = {}

    # ── TEXT ──────────────────────────────────────────────────────────────
    log.info("  Building text mask …")
    text = raw["black"].copy()
    text = remove_small_objects(text, min_size=8)
    text = morphology.opening(text, disk(1))
    sem["text"] = text
    log.info(f"    text coverage: {text.mean()*100:.1f}%")

    # ── ROUTES ────────────────────────────────────────────────────────────
    log.info("  Building routes mask …")
    routes_src = raw["black"] | raw["blue"]
    routes_src = remove_small_objects(routes_src, min_size=50)
    routes_src = morphology.closing(routes_src, disk(1))
    routes_skel = skeletonize(routes_src)
    sem["routes"] = routes_skel
    log.info(f"    routes coverage: {routes_skel.mean()*100:.2f}%")

    # ── SYMBOLS ───────────────────────────────────────────────────────────
    log.info("  Building symbols mask …")
    symbols_src = raw["magenta"] | raw["blue"]
    symbols_src = morphology.closing(symbols_src, disk(3))
    symbols_src = remove_small_objects(symbols_src, min_size=20)
    from skimage.measure import label, regionprops
    labelled = label(symbols_src)
    symbol_mask = np.zeros_like(symbols_src)
    for region in regionprops(labelled):
        if 20 <= region.area <= 8000:
            symbol_mask[labelled == region.label] = True
    sem["symbols"] = symbol_mask
    log.info(f"    symbols coverage: {symbol_mask.mean()*100:.2f}%")

    # ── BOUNDARIES ────────────────────────────────────────────────────────
    # Fix: instead of closing which fills entire water/warning regions,
    # extract only the EDGES of coloured regions using morphological gradient.
    # gradient = dilation - erosion = 1-px thick outline of every colour blob.
    # This keeps the boundary ring lines without filling their interiors.
    log.info("  Building boundaries mask …")
    boundaries_src = raw["magenta"] | raw["blue"]
    # Morphological gradient = edge of each coloured region
    dilated   = morphology.dilation(boundaries_src, disk(3))
    eroded    = morphology.erosion(boundaries_src,  disk(3))
    edges     = dilated & ~eroded
    # Remove small isolated noise blobs; keep only long arc/line structures
    edges     = remove_small_objects(edges, min_size=150)
    sem["boundaries"] = edges
    log.info(f"    boundaries coverage: {edges.mean()*100:.2f}%")

    return sem


def _save_mask(mask: np.ndarray, path: Path):
    """Save a boolean mask as a white-on-black PNG."""
    Image.fromarray((mask * 255).astype(np.uint8)).save(
        str(path), format="PNG", compress_level=3
    )


def _save_outputs(sem: dict, raw: dict, stem: str,
                  output_dir: Path, metadata: dict) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in ["text", "routes", "symbols", "boundaries"]:
        p = output_dir / f"{stem}_{name}.png"
        _save_mask(sem[name], p)
        paths[name] = str(p)
        log.info(f"Saved {name} mask → {p}")

    log_path = output_dir / f"{stem}_phase2_log.json"
    with open(log_path, "w") as f:
        json.dump(metadata, f, indent=2)
    paths["log"] = str(log_path)
    return paths


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def separate_layers(
    rgb: np.ndarray,
    output_dir: Optional[str] = None,
    stem: str = "Washington",
    hsv_tile_size: int = 2048,
) -> LayerResult:
    """
    Run Phase 2 layer separation on a preprocessed RGB chart array.

    Parameters
    ----------
    rgb : np.ndarray
        (H, W, 3) uint8 — output of Phase 1 preprocess_chart().
    output_dir : str, optional
        Directory to write mask PNGs and log JSON. If None, in-memory only.
    stem : str
        Output filename prefix (e.g. "Washington").
    hsv_tile_size : int
        Tile size for memory-safe HSV conversion. Default 2048.

    Returns
    -------
    LayerResult
        Dataclass with all four masks, raw masks, coverage stats, and paths.
    """
    t0 = time.time()
    notes = []

    # ── Step 1: RGB → HSV (tiled) ─────────────────────────────────────────
    hsv = _rgb_to_hsv_tiled(rgb, tile_size=hsv_tile_size)

    # ── Step 2: Raw colour masks ──────────────────────────────────────────
    raw = _build_raw_masks(hsv, notes)
    del hsv   # free ~3 GB

    # ── Step 3: Semantic masks ────────────────────────────────────────────
    sem = _derive_semantic_masks(raw, notes)

    # ── Step 4: Coverage stats ────────────────────────────────────────────
    coverage = {name: round(float(mask.mean() * 100), 3)
                for name, mask in sem.items()}

    # ── Step 5: Save ──────────────────────────────────────────────────────
    elapsed = time.time() - t0
    metadata = {
        "stem": stem,
        "image_shape": list(rgb.shape),
        "hsv_tile_size": hsv_tile_size,
        "coverage_pct": coverage,
        "notes": notes,
        "elapsed_sec": round(elapsed, 2),
    }

    paths = {}
    if output_dir:
        paths = _save_outputs(sem, raw, stem, Path(output_dir), metadata)

    result = LayerResult(
        text_mask=sem["text"],
        routes_mask=sem["routes"],
        symbols_mask=sem["symbols"],
        boundaries_mask=sem["boundaries"],
        magenta_raw=raw["magenta"],
        blue_raw=raw["blue"],
        black_raw=raw["black"],
        coverage=coverage,
        text_path=paths.get("text"),
        routes_path=paths.get("routes"),
        symbols_path=paths.get("symbols"),
        boundaries_path=paths.get("boundaries"),
        log_path=paths.get("log"),
        elapsed_sec=elapsed,
        notes=notes,
    )

    log.info(result.summary())
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Phase 2 — VFR Chart Layer Separation (FAA HSV, Tier 1)"
    )
    p.add_argument("--input", required=True,
                   help="Path to Phase 1 RGB output PNG (or any RGB image).")
    p.add_argument("--output-dir", default="outputs/phase2",
                   help="Directory for output mask PNGs and log.")
    p.add_argument("--stem", default=None,
                   help="Output filename prefix. Defaults to input filename stem.")
    p.add_argument("--hsv-tile-size", type=int, default=2048,
                   help="Tile size for HSV conversion. (default: 2048)")
    return p.parse_args()


if __name__ == "__main__":
    # Allow PIL to load very large images (300 DPI sectionals are 200+ MP)
    Image.MAX_IMAGE_PIXELS = None

    args = _parse_args()
    src = Path(args.input)
    if not src.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    log.info(f"Loading RGB image: {src}")
    rgb = np.array(Image.open(str(src)).convert("RGB"), dtype=np.uint8)
    log.info(f"  Loaded: {rgb.shape[1]} x {rgb.shape[0]} px")

    stem = args.stem or src.stem
    separate_layers(
        rgb=rgb,
        output_dir=args.output_dir,
        stem=stem,
        hsv_tile_size=args.hsv_tile_size,
    )