"""
HSV Colour Segregation Test — Single Map Tile
VFR Chart Extraction Pipeline

Takes a single tile from the RGB chart image and produces:
  - One binary mask PNG per colour class (white = that colour, black = rest)
  - One composite PNG showing all colours overlaid on the original tile
  - One side-by-side comparison PNG (original vs each mask)
  - A printed coverage % for each colour class

Colour classes detected
-----------------------
  FAA Blue    — towered airports, obstacles, VOR, DME, VORTAC, Class B,
                airways, MEF numbers   (H: 0.53-0.72)
  FAA Magenta — non-towered airports, NDB, Class C/E, MOA, waypoints
                (H: 0.82-1.0 AND 0.0-0.05, wraps around red)
  Black/Dark  — text labels, obstacle triangles, route lines, contours
                (V < 0.25, any hue)
  Brown/Tan   — terrain elevation contours                (H: 0.05-0.12)
  Green       — terrain/vegetation shading               (H: 0.25-0.45)
  Yellow/Tan  — urban/city areas                         (H: 0.10-0.18,
                                                          S: 0.15-0.60)
  White/Light — background, open water labels            (S < 0.10, V > 0.80)
  Cyan/Water  — water body fill                          (H: 0.48-0.55)

HOW TO USE
----------
1. Set RGB_PATH and TILE below.
   TILE = (row_start, col_start, row_end, col_end) in pixels.
   Set TILE = None to use the whole image (slow for large charts).
2. Set OUTPUT_DIR.
3. python hsv_colour_test.py
4. Open the output folder — inspect each mask PNG.

Requirements: numpy, pillow, scikit-image
"""

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.color import rgb2hsv

Image.MAX_IMAGE_PIXELS = None


# =============================================================================
# CONFIG — set these
# =============================================================================

# Path to the full-resolution RGB chart image (Phase 1 output or raw chart)
RGB_PATH = r"outputs/phase1_preprocessing/Washington_rgb_150dpi.png"

# Tile to extract: (row_start, col_start, row_end, col_end) in pixels
# Adjust to a region that has a good mix of airports, NAVAIDs, boundaries.
# Example: a 1000x1000 tile from the middle of the chart.
# Set to None to process the entire image.
TILE = (1000, 1000, 2000, 2000)   # (r0, c0, r1, c1)

OUTPUT_DIR = r"outputs/hsv_colour_test"


# =============================================================================
# HSV COLOUR CLASSES
# Each entry: (label, colour_for_overlay_RGB, hsv_conditions_as_dict)
#
# HSV ranges (skimage convention, all normalised 0.0-1.0):
#   H = 0.0 (red) → 0.17 (yellow) → 0.33 (green) → 0.50 (cyan)
#      → 0.67 (blue) → 0.83 (magenta) → 1.0 (red again)
#   S = 0.0 (grey/white) → 1.0 (fully saturated)
#   V = 0.0 (black) → 1.0 (brightest)
#
# hue_wrap=True means the hue range wraps around 1.0→0.0 (used for magenta/red)
# =============================================================================

COLOUR_CLASSES = [
    {
        "label":      "faa_blue",
        "description":"Towered airports, obstacles, VOR, DME, VORTAC, Class B, airways",
        "overlay":    (0, 120, 255),          # bright blue
        "h_lo":  0.53, "h_hi":  0.72,
        "s_min": 0.20, "v_min": 0.25,
        "hue_wrap": False,
    },
    {
        "label":      "faa_magenta",
        "description":"Non-towered airports, NDB, Class C/E, MOA, waypoints",
        "overlay":    (220, 0, 220),          # magenta
        # Magenta wraps: hue 0.82-1.0 AND 0.0-0.05
        "h_lo":  0.82, "h_hi":  1.00,
        "h_lo2": 0.00, "h_hi2": 0.05,
        "s_min": 0.25, "v_min": 0.20,
        "hue_wrap": True,
    },
    {
        "label":      "black_ink",
        "description":"Text, route lines, contour lines, obstacle triangles",
        "overlay":    (200, 200, 200),        # light grey (so visible on black bg)
        "h_lo":  0.00, "h_hi":  1.00,        # any hue
        "s_min": 0.00, "v_min": 0.00,
        "v_max": 0.28,                        # very dark pixels only
        "hue_wrap": False,
    },
    {
        "label":      "terrain_brown",
        "description":"Elevation contour lines, terrain shading",
        "overlay":    (160, 80, 20),          # brown
        "h_lo":  0.04, "h_hi":  0.12,
        "s_min": 0.20, "v_min": 0.30,
        "hue_wrap": False,
    },
    {
        "label":      "terrain_green",
        "description":"Vegetation / terrain green shading (pale desaturated green)",
        "overlay":    (0, 200, 60),           # green
        "h_lo":  0.25, "h_hi":  0.45,
        # s_min 0.15→0.05: FAA terrain green is pale, very low saturation
        # v_min 0.30→0.50: keep lighter shading only, exclude dark blobs
        "s_min": 0.05, "v_min": 0.50,
        "hue_wrap": False,
    },
    {
        "label":      "urban_yellow",
        "description":"Urban / city area fill (cream-yellow blobs)",
        "overlay":    (255, 220, 0),          # yellow
        # h_hi 0.18→0.20: cream-yellow skews slightly more orange
        # s_min 0.15→0.08: urban fill is pale cream, very low saturation
        "h_lo":  0.09, "h_hi":  0.20,
        "s_min": 0.08, "v_min": 0.65,
        "hue_wrap": False,
    },
    {
        "label":      "water_cyan",
        "description":"Water body fill (pale blue-grey: Chesapeake Bay, rivers)",
        "overlay":    (0, 200, 220),          # cyan
        # s_min 0.15→0.22: was catching everything — exclude near-white pixels
        # v_max 0.90 added: exclude the brightest whites that bled in
        # v_min 0.40→0.55: water is mid-to-bright, not very dark
        "h_lo":  0.48, "h_hi":  0.56,
        "s_min": 0.22, "v_min": 0.55,
        "v_max": 0.90,
        "hue_wrap": False,
    },
    {
        "label":      "white_background",
        "description":"Background, paper white, open areas",
        "overlay":    (240, 240, 240),        # near-white
        "h_lo":  0.00, "h_hi":  1.00,        # any hue
        "s_min": 0.00, "v_min": 0.80,
        "s_max": 0.12,                        # very desaturated (white/grey)
        "hue_wrap": False,
    },
]


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def load_tile(path: str, tile) -> np.ndarray:
    """Load the RGB image and extract the specified tile."""
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    H, W = arr.shape[:2]
    print(f"  Full image: {W} × {H} px")

    if tile is None:
        print("  Using full image as tile")
        return arr

    r0, c0, r1, c1 = tile
    r0 = max(0, r0);  r1 = min(H, r1)
    c0 = max(0, c0);  c1 = min(W, c1)
    out = arr[r0:r1, c0:c1]
    print(f"  Tile: [{r0}:{r1}, {c0}:{c1}]  →  {out.shape[1]} × {out.shape[0]} px")
    return out


def build_mask(hsv: np.ndarray, cls: dict) -> np.ndarray:
    """
    Build a boolean mask for pixels matching the colour class.

    hsv: float array (H, W, 3) with values in [0, 1]
    Returns bool array (H, W): True = pixel belongs to this colour class.
    """
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # Hue condition
    if cls.get("hue_wrap", False):
        # Wrapping hue (e.g. magenta: 0.82-1.0 OR 0.0-0.05)
        hue_ok = ((h >= cls["h_lo"]) & (h <= cls["h_hi"])) | \
                 ((h >= cls.get("h_lo2", 0.0)) & (h <= cls.get("h_hi2", 0.0)))
    else:
        hue_ok = (h >= cls["h_lo"]) & (h <= cls["h_hi"])

    # Saturation condition
    s_ok = s >= cls.get("s_min", 0.0)
    if "s_max" in cls:
        s_ok = s_ok & (s <= cls["s_max"])

    # Value condition
    v_ok = v >= cls.get("v_min", 0.0)
    if "v_max" in cls:
        v_ok = v_ok & (v <= cls["v_max"])

    return hue_ok & s_ok & v_ok


def save_binary_mask(mask: np.ndarray, path: str, label: str, pct: float):
    """Save boolean mask as white-on-black PNG."""
    arr = (mask * 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L")

    # Add label text at top-left
    draw = ImageDraw.Draw(img)
    draw.text((4, 4), f"{label}  ({pct:.2f}%)", fill=200)

    img.save(path)


def save_coloured_mask(tile_rgb: np.ndarray, mask: np.ndarray,
                       colour: tuple, path: str, label: str, pct: float):
    """
    Save the original tile with the detected pixels tinted in the overlay colour.
    Non-matching pixels are shown in dark grey.
    """
    H, W = tile_rgb.shape[:2]
    out  = np.full((H, W, 3), 30, dtype=np.uint8)  # dark grey background

    # Show matching pixels in overlay colour
    for ch, val in enumerate(colour):
        out[:, :, ch] = np.where(mask, val, out[:, :, ch])

    img  = Image.fromarray(out)
    draw = ImageDraw.Draw(img)
    draw.text((4, 4), f"{label}  ({pct:.2f}%)", fill=(255, 255, 255))
    img.save(path)


def save_overlay_composite(tile_rgb: np.ndarray, masks_colours: list, path: str):
    """
    Composite: original tile with all colour classes overlaid simultaneously.
    Each detected pixel is tinted with its class colour.
    Undetected pixels show the original RGB.
    """
    composite = tile_rgb.copy()
    for mask, colour in masks_colours:
        for ch, val in enumerate(colour):
            composite[:, :, ch] = np.where(mask,
                np.clip(val * 0.7 + composite[:, :, ch] * 0.3, 0, 255),
                composite[:, :, ch])
    Image.fromarray(composite.astype(np.uint8)).save(path)


def save_side_by_side(tile_rgb: np.ndarray, masks: list, out_dir: Path):
    """
    For each colour class save a side-by-side PNG:
      Left:  original tile
      Right: binary mask
    """
    H, W = tile_rgb.shape[:2]
    orig = Image.fromarray(tile_rgb)

    for mask, cls, pct in masks:
        binary = Image.fromarray((mask * 255).astype(np.uint8), mode="L").convert("RGB")
        combined = Image.new("RGB", (W * 2 + 4, H), (50, 50, 50))
        combined.paste(orig, (0, 0))
        combined.paste(binary, (W + 4, 0))

        draw = ImageDraw.Draw(combined)
        draw.text((4, 4), "Original", fill=(255, 255, 255))
        draw.text((W + 8, 4),
                  f"{cls['label']}  ({pct:.2f}%)",
                  fill=(255, 255, 255))

        combined.save(str(out_dir / f"compare_{cls['label']}.png"))


# =============================================================================
# MAIN
# =============================================================================

def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== HSV COLOUR SEGREGATION TEST ===\n")

    # Load tile
    print(f"Loading: {RGB_PATH}")
    tile_rgb = load_tile(RGB_PATH, TILE)
    H, W = tile_rgb.shape[:2]
    total_px = H * W

    # Convert to HSV once (float [0,1])
    print("\nConverting to HSV ...")
    tile_float = tile_rgb.astype(np.float64) / 255.0
    hsv = rgb2hsv(tile_float)
    print("  Done\n")

    # Save original tile for reference
    Image.fromarray(tile_rgb).save(str(out_dir / "00_original_tile.png"))
    print(f"  Saved: 00_original_tile.png\n")

    # Build masks for all colour classes
    print("Building colour masks ...")
    print(f"  {'Class':<22}  {'Coverage':>9}  {'Pixels':>12}  Description")
    print(f"  {'-'*22}  {'-'*9}  {'-'*12}  {'-'*40}")

    results = []   # (mask, cls, pct)
    masks_colours = []  # for composite

    for cls in COLOUR_CLASSES:
        mask = build_mask(hsv, cls)
        n_px = int(mask.sum())
        pct  = n_px / total_px * 100

        print(f"  {cls['label']:<22}  {pct:8.3f}%  {n_px:12,}  {cls['description']}")

        results.append((mask, cls, pct))
        masks_colours.append((mask, cls["overlay"]))

        # Binary mask PNG (white on black)
        idx = COLOUR_CLASSES.index(cls) + 1
        fname = f"{idx:02d}_{cls['label']}_binary.png"
        save_binary_mask(mask, str(out_dir / fname), cls["label"], pct)

        # Coloured mask PNG (colour on dark background)
        fname2 = f"{idx:02d}_{cls['label']}_coloured.png"
        save_coloured_mask(tile_rgb, mask, cls["overlay"],
                           str(out_dir / fname2), cls["label"], pct)

    # Composite overview
    print("\nSaving composite overlay ...")
    save_overlay_composite(tile_rgb, masks_colours,
                           str(out_dir / "00_composite_all_colours.png"))
    print("  Saved: 00_composite_all_colours.png")

    # Side-by-side comparisons
    print("Saving side-by-side comparisons ...")
    save_side_by_side(tile_rgb, results, out_dir)
    print("  Saved: compare_<label>.png for each class")

    # Coverage summary
    print(f"\n=== COVERAGE SUMMARY ===")
    total_classified = sum(mask.sum() for mask, cls, pct in results)
    unclassified = total_px - min(total_classified, total_px)
    print(f"  Total pixels      : {total_px:,}")
    print(f"  Classified pixels : {total_classified:,}  "
          f"({total_classified/total_px*100:.1f}%)")
    print(f"  Unclassified      : {unclassified:,}  "
          f"({unclassified/total_px*100:.1f}%)")
    print(f"\n  Output: {out_dir.resolve()}")

    print("\n=== FILES PRODUCED ===")
    print("  00_original_tile.png           — original tile for reference")
    print("  00_composite_all_colours.png   — all masks overlaid on tile")
    print("  01-08_<label>_binary.png       — white-on-black mask per class")
    print("  01-08_<label>_coloured.png     — colour-tinted mask per class")
    print("  compare_<label>.png            — side-by-side original vs mask")

    print("\n=== HOW TO INTERPRET ===")
    print("  Good:  the mask covers ONLY the expected symbol type")
    print("  Issue: the mask bleeds into other symbol types or background")
    print("  → If bleeding: tighten s_min or v_min for that class in COLOUR_CLASSES")
    print("  → If missing:  loosen h_lo/h_hi or lower s_min for that class")
    print("  → Change TILE to test a different map region")


if __name__ == "__main__":
    main()