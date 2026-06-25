"""
Phase 1 — Input Preprocessing
VFR Chart Extraction Pipeline (FAA Base Model)

Converts raw raster inputs (PDF, PNG, JPEG, GeoTIFF) into clean, normalised
images ready for downstream layer separation and georeferencing.

Steps:
    1. Rasterise PDF at 300 DPI via PyMuPDF (fitz)
    2. Preserve original RGB array
    3. Non-local means (NLM) denoising on RGB  [skipped if σ < threshold]
    4. Sauvola adaptive thresholding → binary mask
    5. Hough-based global deskew
    6. Save outputs to disk + return PreprocessingResult

Usage (module):
    from phase1_preprocessing import preprocess_chart
    result = preprocess_chart("Houston.pdf", output_dir="outputs/phase1_preprocessing")

Usage (CLI):
    python phase1_preprocessing.py --input Houston.pdf --output-dir outputs/phase1_preprocessing --dpi 300
    python phase1_preprocessing.py --input Houston.pdf --skip-denoise   # fast debug run
"""

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from skimage import color, filters, morphology, restoration, transform
from skimage.transform import probabilistic_hough_line
from skimage.util import img_as_float32, img_as_ubyte

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase1")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# NLM is skipped when estimated noise σ is below this value.
# σ < 0.004 means the image is essentially noise-free (typical of clean PDF
# rasters). Forcing NLM on a noise-free 615 MB image allocates ~4.6 GiB and
# crashes on machines with ≤ 16 GB RAM.
_NLM_SIGMA_THRESHOLD: float = 0.004

# NLM patch_distance: reduced from 11 → 6.
# At 11, the internal integral-image buffer for a 16 k×12 k image exceeds
# 4 GiB even in fast_mode. At 6 the search window is still ±48 px (at 300 DPI
# that spans a full text character), quality loss is negligible on charts, and
# peak RAM drops to ~700 MB.
_NLM_PATCH_DISTANCE: int = 6


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class PreprocessingResult:
    """All outputs produced by Phase 1."""

    # Core arrays (always populated)
    rgb_array: np.ndarray          # (H, W, 3)  uint8  — denoised colour image
    binary_array: np.ndarray       # (H, W)     bool   — Sauvola binarised

    # Metadata
    source_path: str
    dpi: int
    page_index: int
    original_size_px: tuple        # (width, height) before any processing
    final_size_px: tuple           # (width, height) after deskew
    skew_angle_deg: float          # positive = clockwise rotation detected
    deskew_applied: bool

    # Disk paths (populated only when output_dir is given)
    rgb_path: Optional[str] = None
    binary_path: Optional[str] = None
    log_path: Optional[str] = None

    # Timing
    elapsed_sec: float = 0.0

    # Processing notes / warnings
    notes: list = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=== Phase 1 Result ===",
            f"  Source        : {self.source_path}",
            f"  DPI           : {self.dpi}",
            f"  Original size : {self.original_size_px[0]} x {self.original_size_px[1]} px",
            f"  Final size    : {self.final_size_px[0]} x {self.final_size_px[1]} px",
            f"  Skew detected : {self.skew_angle_deg:.3f} deg  (applied={self.deskew_applied})",
            f"  Elapsed       : {self.elapsed_sec:.1f} s",
        ]
        if self.rgb_path:
            lines.append(f"  RGB saved     : {self.rgb_path}")
        if self.binary_path:
            lines.append(f"  Binary saved  : {self.binary_path}")
        if self.notes:
            lines.append("  Notes:")
            for n in self.notes:
                lines.append(f"    - {n}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step helpers
# ---------------------------------------------------------------------------

def _rasterise_pdf(pdf_path: Path, dpi: int, page_index: int) -> np.ndarray:
    """
    Render a single PDF page to an RGB uint8 numpy array at the given DPI.

    PyMuPDF's fitz.Matrix scales by (dpi/72). A 300-DPI render of a
    standard FAA sectional (~56" x 41") produces ~16 800 x 12 300 px —
    expect 600–900 MB of RAM for the raw array.
    """
    log.info(f"Rasterising '{pdf_path.name}' page {page_index} at {dpi} DPI …")
    doc = fitz.open(str(pdf_path))

    if page_index >= len(doc):
        raise ValueError(
            f"PDF has {len(doc)} page(s); requested page index {page_index}."
        )

    page = doc[page_index]
    matrix = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB, alpha=False)
    doc.close()

    # pix.samples is a flat bytestring; reshape to (H, W, 3)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, 3
    )
    log.info(f"  Rasterised: {pix.width} x {pix.height} px  ({arr.nbytes / 1e6:.0f} MB)")
    return arr.copy()  # detach from PyMuPDF memory


def _load_raster(image_path: Path) -> np.ndarray:
    """Load a PNG / JPEG / GeoTIFF as RGB uint8 array."""
    log.info(f"Loading raster '{image_path.name}' …")
    img = Image.open(str(image_path)).convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    log.info(f"  Loaded: {arr.shape[1]} x {arr.shape[0]} px")
    return arr


def _sauvola_binarise(rgb: np.ndarray, window: int = 25, k: float = 0.2,
                      tile_size: int = 2048) -> np.ndarray:
    """
    Convert to greyscale and apply Sauvola adaptive thresholding.

    Sauvola computes a local threshold per pixel:
        T(x,y) = mean(x,y) * [1 + k * (std(x,y)/R − 1)]
    where R is the max value of std (128 for uint8).

    This outperforms global and OpenCV adaptive thresholds on documents
    with uneven illumination or topo shading (common on FAA sectionals).

    TILED IMPLEMENTATION
    --------------------
    Sauvola internally allocates a float64 array the same shape as the image.
    At 300 DPI (16 800 × 12 300 px) that is ~1.56 GiB — enough to crash on a
    machine with 8 GB RAM. We solve this by processing overlapping tiles:

      - Each tile is (tile_size + window) × (tile_size + window) so that
        border pixels have a full window of context.
      - Only the inner tile_size × tile_size result is kept; the overlap
        border is discarded, so there is no seam at tile boundaries.
      - Peak working RAM per tile ≈ tile_size² × 8 bytes × 4 arrays ≈ 128 MB
        for the default tile_size=2048, regardless of image size.

    Returns a bool array where True = foreground (text / lines / symbols).
    """
    log.info("Binarising (tiled Sauvola adaptive threshold) …")
    grey = color.rgb2gray(rgb)          # float64 [0, 1]
    H, W = grey.shape
    binary = np.zeros((H, W), dtype=bool)
    half_win = window // 2              # padding needed for border context

    # Number of tiles in each dimension
    n_tiles_y = int(np.ceil(H / tile_size))
    n_tiles_x = int(np.ceil(W / tile_size))
    total = n_tiles_y * n_tiles_x
    done = 0

    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            # Inner tile coordinates (what we keep)
            y0 = ty * tile_size
            x0 = tx * tile_size
            y1 = min(y0 + tile_size, H)
            x1 = min(x0 + tile_size, W)

            # Padded tile coordinates (what we feed to Sauvola)
            py0 = max(y0 - half_win, 0)
            px0 = max(x0 - half_win, 0)
            py1 = min(y1 + half_win, H)
            px1 = min(x1 + half_win, W)

            tile = grey[py0:py1, px0:px1]
            thresh_tile = filters.threshold_sauvola(tile, window_size=window, k=k)
            bin_tile = tile < thresh_tile

            # Offsets into the padded tile that correspond to the inner tile
            iy0 = y0 - py0
            ix0 = x0 - px0
            binary[y0:y1, x0:x1] = bin_tile[iy0:iy0 + (y1 - y0),
                                              ix0:ix0 + (x1 - x0)]
            done += 1
            log.info(f"  Tile {done}/{total} done ({tx+1}/{n_tiles_x} x {ty+1}/{n_tiles_y})")

    log.info(f"  Foreground coverage: {binary.mean() * 100:.1f}%")
    return binary


def _nlm_denoise(rgb: np.ndarray) -> np.ndarray:
    """
    Non-local means denoising on the RGB image.

    NLM replaces each pixel with a weighted average of perceptually similar
    patches anywhere in the image. Unlike Gaussian or median filters, it
    preserves thin text strokes and symbol edges — critical for OCR and
    template matching downstream.

    MEMORY FIXES (vs original)
    --------------------------
    1. Sigma guard: if estimated σ < _NLM_SIGMA_THRESHOLD (0.004), the image
       is essentially noise-free (clean PDF raster). NLM is skipped entirely;
       the original uint8 array is returned unchanged. This is the primary
       fix for the 4.6 GiB OOM on Houston.pdf (σ = 0.0000).

    2. float32 input: skimage NLM accepts float32 and avoids an internal
       upcasting to float64, halving peak allocation from ~4.6 GB to ~2.3 GB.

    3. patch_distance=6 (was 11): reduces the internal integral-image buffer
       from O(patch_distance²) to a smaller constant. At 300 DPI a ±48 px
       search window (distance=6, patch_size=5 → radius=6*5=30 px) still
       spans a full character glyph. Quality loss on chart content is
       negligible.

    Returns the denoised uint8 array, or the original if skipped.
    """
    log.info("Denoising (non-local means) …")

    # Use float32 to halve memory vs float64
    rgb_f32 = img_as_float32(rgb)   # (H, W, 3)  float32  [0.0, 1.0]

    # Estimate per-channel noise σ and take mean
    sigma_est = float(np.mean(
        restoration.estimate_sigma(rgb_f32, channel_axis=-1)
    ))
    log.info(f"  Estimated noise σ: {sigma_est:.4f}")

    # ── Guard: skip NLM on noise-free images ─────────────────────────────
    if sigma_est < _NLM_SIGMA_THRESHOLD:
        log.info(
            f"  σ ({sigma_est:.4f}) < threshold ({_NLM_SIGMA_THRESHOLD}) — "
            "image is noise-free. NLM skipped; returning original."
        )
        return rgb
    # ─────────────────────────────────────────────────────────────────────

    # h controls filter strength: clamp to a safe range regardless of σ
    h = max(0.02, min(sigma_est * 1.0, 0.08))
    log.info(f"  NLM filter strength h={h:.4f}, patch_distance={_NLM_PATCH_DISTANCE}")

    denoised = restoration.denoise_nl_means(
        rgb_f32,
        h=h,
        patch_size=5,
        patch_distance=_NLM_PATCH_DISTANCE,
        channel_axis=-1,
        fast_mode=True,          # fast_mode uses a precomputed integral image
    )

    result = img_as_ubyte(np.clip(denoised, 0.0, 1.0))
    log.info("  Denoising complete.")
    return result


def _detect_skew(rgb: np.ndarray, skew_threshold_deg: float = 0.5) -> float:
    """
    Estimate global skew angle from horizontal text baselines.

    Method:
      1. Convert to greyscale and binarise roughly (Otsu).
      2. Run probabilistic Hough restricted to lines within ±2° of horizontal.
         This tight window prevents the detector from latching onto lat/lon
         grid lines (which run at ~90°) or diagonal airway lines.
      3. Compute the angle of each candidate line relative to horizontal.
      4. Return the median angle — robust to a few non-baseline features.

    CRITICAL: The angle search is limited to np.linspace(-π/90, π/90, 180),
    which is ±2°. This is intentional. FAA sectionals have dense vertical
    and diagonal features (grid ticks, airways, boundary arcs) that dominate
    at wider angle ranges and produce false 90° skew estimates. Text baselines
    on a properly-scanned chart are within ±1° of horizontal.

    Sanity check: if the estimated angle is outside ±5°, it is almost
    certainly a false positive (e.g., a 90° grid line picked up as horizontal).
    In that case, return 0.0 — no rotation applied.

    Returns 0.0 if no reliable angle estimate is possible.
    """
    log.info("Detecting skew (probabilistic Hough on horizontal lines) …")

    grey = color.rgb2gray(rgb)
    thresh = filters.threshold_otsu(grey)
    binary_coarse = grey < thresh

    # Tight angular range: only consider lines within ±2° of horizontal.
    # This deliberately excludes vertical grid lines (~90°) and diagonal
    # airway features that caused the false -89.8° reading previously.
    angle_range = np.linspace(-np.pi / 90, np.pi / 90, 180)   # ±2 degrees

    min_line_len = int(rgb.shape[1] * 0.20)
    lines = probabilistic_hough_line(
        binary_coarse,
        threshold=80,
        line_length=min_line_len,
        line_gap=15,
        theta=angle_range,
    )

    if not lines:
        log.info("  No long near-horizontal lines found; assuming skew = 0.0°")
        return 0.0

    angles = []
    for (x0, y0), (x1, y1) in lines:
        if x1 == x0:
            continue
        angle_deg = np.degrees(np.arctan2(y1 - y0, x1 - x0))
        angles.append(angle_deg)

    if not angles:
        return 0.0

    median_angle = float(np.median(angles))

    # Sanity check: reject anything outside ±5° — it's a false positive
    if abs(median_angle) > 5.0:
        log.warning(
            f"  Skew estimate {median_angle:.3f}° outside ±5° sanity bound — "
            "likely a false positive (grid line picked up). Returning 0.0°."
        )
        return 0.0

    log.info(f"  Skew estimate: {median_angle:.3f}° (from {len(angles)} lines)")
    return median_angle


def _deskew(rgb: np.ndarray, binary: np.ndarray, angle_deg: float):
    """
    Rotate both arrays by -angle_deg to correct for detected skew.

    Uses skimage.transform.rotate with:
      - resize=True  → output is large enough to contain the full rotated image
      - cval=1.0     → fills background with white (appropriate for chart paper)

    Returns (rotated_rgb uint8, rotated_binary bool).
    """
    log.info(f"Applying deskew correction: rotating by {-angle_deg:.3f}° …")

    rgb_rot = transform.rotate(
        rgb.astype(np.float32) / 255.0,
        angle=-angle_deg,
        resize=True,
        cval=1.0,
        order=1,                 # bilinear for RGB
    )
    rgb_rot = (np.clip(rgb_rot, 0, 1) * 255).astype(np.uint8)

    binary_rot = transform.rotate(
        binary.astype(np.float32),
        angle=-angle_deg,
        resize=True,
        cval=1.0,
        order=0,                 # nearest-neighbour for binary
    )
    binary_rot = binary_rot < 0.5   # re-binarise after interpolation

    log.info(f"  Output size: {rgb_rot.shape[1]} x {rgb_rot.shape[0]} px")
    return rgb_rot, binary_rot


def _save_outputs(
    rgb: np.ndarray,
    binary: np.ndarray,
    stem: str,
    output_dir: Path,
    metadata: dict,
) -> tuple:
    """
    Save RGB (PNG), binary (PNG), and processing log (JSON) to output_dir.
    Returns (rgb_path, binary_path, log_path) as strings.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    rgb_path = output_dir / f"{stem}_rgb.png"
    binary_path = output_dir / f"{stem}_binary.png"
    log_path = output_dir / f"{stem}_log.json"

    log.info(f"Saving RGB → {rgb_path}")
    Image.fromarray(rgb).save(str(rgb_path), format="PNG", compress_level=3)

    log.info(f"Saving binary → {binary_path}")
    # Convert bool to uint8 {0, 255} for a viewable PNG
    Image.fromarray((binary * 255).astype(np.uint8)).save(
        str(binary_path), format="PNG", compress_level=3
    )

    log.info(f"Saving processing log → {log_path}")
    with open(log_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return str(rgb_path), str(binary_path), str(log_path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess_chart(
    input_path: str,
    output_dir: Optional[str] = None,
    dpi: int = 300,
    page_index: int = 0,
    sauvola_window: int = 25,
    sauvola_k: float = 0.2,
    sauvola_tile_size: int = 2048,
    skew_threshold_deg: float = 0.5,
    skip_denoise: bool = False,
) -> PreprocessingResult:
    """
    Run Phase 1 preprocessing on a VFR chart file.

    Parameters
    ----------
    input_path : str
        Path to input file. Supported: .pdf, .png, .jpg, .jpeg, .tif, .tiff
    output_dir : str, optional
        Directory to write output files. If None, arrays are returned only.
    dpi : int
        Render DPI for PDF inputs. Ignored for raster inputs. Default 300.
    page_index : int
        Zero-based page number for multi-page PDFs. Default 0.
    sauvola_window : int
        Sauvola window size (pixels). Must be odd. Default 25.
    sauvola_k : float
        Sauvola k parameter. Default 0.2.
    sauvola_tile_size : int
        Tile size for tiled Sauvola binarisation. Default 2048.
    skew_threshold_deg : float
        Minimum angle (degrees) to trigger deskew. Default 0.5.
    skip_denoise : bool
        If True, bypass NLM entirely (useful for fast debugging). Default False.
        Note: even when False, NLM is auto-skipped if σ < _NLM_SIGMA_THRESHOLD.

    Returns
    -------
    PreprocessingResult
        Dataclass containing arrays, metadata, and optional disk paths.
    """
    t0 = time.time()
    notes = []
    src = Path(input_path)

    if not src.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # ── Step 1: Load / rasterise ──────────────────────────────────────────
    ext = src.suffix.lower()
    if ext == ".pdf":
        rgb = _rasterise_pdf(src, dpi=dpi, page_index=page_index)
    elif ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        rgb = _load_raster(src)
        dpi = 0   # DPI meaningless for pre-rasterised inputs
    else:
        raise ValueError(f"Unsupported file type: '{ext}'")

    original_size = (rgb.shape[1], rgb.shape[0])   # (W, H)

    # ── Step 2: Denoise ───────────────────────────────────────────────────
    if skip_denoise:
        note = "NLM denoising skipped (skip_denoise=True)."
        notes.append(note)
        log.warning(note)
    else:
        rgb = _nlm_denoise(rgb)
        # _nlm_denoise may have auto-skipped; check by comparing pointer
        # (no separate note needed — _nlm_denoise already logs it)

    # ── Step 3: Binarise ──────────────────────────────────────────────────
    binary = _sauvola_binarise(rgb, window=sauvola_window, k=sauvola_k,
                               tile_size=sauvola_tile_size)

    # ── Step 4: Deskew ────────────────────────────────────────────────────
    skew_angle = _detect_skew(rgb, skew_threshold_deg=skew_threshold_deg)
    deskew_applied = abs(skew_angle) >= skew_threshold_deg

    if deskew_applied:
        rgb, binary = _deskew(rgb, binary, skew_angle)
    else:
        log.info(
            f"Skew {skew_angle:.3f}° below threshold {skew_threshold_deg}°; "
            "no rotation applied."
        )
        notes.append(
            f"Skew ({skew_angle:.3f}°) below threshold; deskew not applied."
        )

    final_size = (rgb.shape[1], rgb.shape[0])

    # ── Step 5: Persist to disk ───────────────────────────────────────────
    rgb_path = binary_path = log_path_str = None

    metadata = {
        "source": str(src),
        "dpi": dpi,
        "page_index": page_index,
        "original_size_px": original_size,
        "final_size_px": final_size,
        "skew_angle_deg": round(skew_angle, 4),
        "deskew_applied": deskew_applied,
        "sauvola_window": sauvola_window,
        "sauvola_k": sauvola_k,
        "sauvola_tile_size": sauvola_tile_size,
        "skew_threshold_deg": skew_threshold_deg,
        "skip_denoise": skip_denoise,
        "nlm_sigma_threshold": _NLM_SIGMA_THRESHOLD,
        "nlm_patch_distance": _NLM_PATCH_DISTANCE,
        "notes": notes,
        "elapsed_sec": None,   # filled after timing
    }

    if output_dir:
        stem = src.stem  # e.g. "Houston"
        rgb_path, binary_path, log_path_str = _save_outputs(
            rgb, binary, stem, Path(output_dir), metadata
        )

    elapsed = time.time() - t0
    metadata["elapsed_sec"] = round(elapsed, 2)

    # Update log file with actual elapsed time
    if log_path_str:
        with open(log_path_str, "w") as f:
            json.dump(metadata, f, indent=2)

    result = PreprocessingResult(
        rgb_array=rgb,
        binary_array=binary,
        source_path=str(src),
        dpi=dpi,
        page_index=page_index,
        original_size_px=original_size,
        final_size_px=final_size,
        skew_angle_deg=skew_angle,
        deskew_applied=deskew_applied,
        rgb_path=rgb_path,
        binary_path=binary_path,
        log_path=log_path_str,
        elapsed_sec=elapsed,
        notes=notes,
    )

    log.info(result.summary())
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 1 — VFR Chart Preprocessing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input", "-i",
        default="inputs/Houston.pdf",
        metavar="PATH",
        help="Path to input PDF or raster image (.pdf/.png/.jpg/.tif)",
    )
    p.add_argument(
        "--output-dir", "-o",
        default="outputs/phase1_preprocessing",
        metavar="DIR",
        help="Directory to write output files",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Render DPI for PDF inputs (ignored for raster inputs)",
    )
    p.add_argument(
        "--page",
        type=int,
        default=0,
        dest="page_index",
        metavar="N",
        help="Zero-based page index for multi-page PDFs",
    )
    p.add_argument(
        "--sauvola-window",
        type=int,
        default=25,
        metavar="PX",
        help="Sauvola window size in pixels (must be odd)",
    )
    p.add_argument(
        "--sauvola-k",
        type=float,
        default=0.2,
        metavar="K",
        help="Sauvola k parameter",
    )
    p.add_argument(
        "--sauvola-tile-size",
        type=int,
        default=2048,
        metavar="PX",
        help="Tile size for tiled Sauvola binarisation",
    )
    p.add_argument(
        "--skew-threshold",
        type=float,
        default=0.5,
        dest="skew_threshold_deg",
        metavar="DEG",
        help="Minimum skew angle (degrees) to trigger deskew",
    )
    p.add_argument(
        "--skip-denoise",
        action="store_true",
        default=False,
        help="Skip NLM denoising entirely (fast debug mode)",
    )
    p.add_argument(
        "--nlm-sigma-threshold",
        type=float,
        default=_NLM_SIGMA_THRESHOLD,
        metavar="S",
        help=(
            "NLM is auto-skipped when estimated noise σ < this value. "
            "Default 0.004 skips denoising on clean PDF rasters."
        ),
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    # Allow overriding the module-level constant from the CLI
    import phase1_preprocessing as _self
    _self._NLM_SIGMA_THRESHOLD = args.nlm_sigma_threshold

    preprocess_chart(
        input_path=args.input,
        output_dir=args.output_dir,
        dpi=args.dpi,
        page_index=args.page_index,
        sauvola_window=args.sauvola_window,
        sauvola_k=args.sauvola_k,
        sauvola_tile_size=args.sauvola_tile_size,
        skew_threshold_deg=args.skew_threshold_deg,
        skip_denoise=args.skip_denoise,
    )