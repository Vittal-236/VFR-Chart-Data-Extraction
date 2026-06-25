from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import json

Image.MAX_IMAGE_PIXELS = None

# =============================================================================
# CONFIG
# =============================================================================

INPUT_DIR = Path("rendered_png")
OUTPUT_DIR = Path("tiled_dataset/images")
META_DIR = Path("tiled_dataset/metadata")

TILE_SIZE = 1280
OVERLAP = 0.15

EMPTY_THRESHOLD = 245

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)

stride = int(TILE_SIZE * (1 - OVERLAP))

# =============================================================================
# TILE FUNCTION
# =============================================================================

def is_informative(tile_np):

    gray = np.mean(tile_np, axis=2)

    dark_ratio = np.mean(gray < EMPTY_THRESHOLD)

    return dark_ratio > 0.02

# =============================================================================
# MAIN
# =============================================================================

image_files = list(INPUT_DIR.glob("*/*.png"))

print(f"\nFound {len(image_files)} rendered images")

for img_path in tqdm(image_files):

    img = Image.open(img_path).convert("RGB")

    arr = np.array(img)

    H, W, _ = arr.shape

    chart_name = img_path.parent.name

    tile_metadata = []

    tile_count = 0

    for y in range(0, H, stride):

        for x in range(0, W, stride):

            y2 = min(y + TILE_SIZE, H)
            x2 = min(x + TILE_SIZE, W)

            tile = arr[y:y2, x:x2]

            if tile.shape[0] < TILE_SIZE:
                pad = TILE_SIZE - tile.shape[0]
                tile = np.pad(
                    tile,
                    ((0,pad),(0,0),(0,0)),
                    mode="constant",
                    constant_values=255
                )

            if tile.shape[1] < TILE_SIZE:
                pad = TILE_SIZE - tile.shape[1]
                tile = np.pad(
                    tile,
                    ((0,0),(0,pad),(0,0)),
                    mode="constant",
                    constant_values=255
                )

            if not is_informative(tile):
                continue

            tile_name = f"{chart_name}_r{y}_c{x}.png"

            out_path = OUTPUT_DIR / tile_name

            Image.fromarray(tile).save(out_path)

            tile_metadata.append({
                "tile_name": tile_name,
                "source_chart": chart_name,
                "x_offset": x,
                "y_offset": y,
                "tile_size": TILE_SIZE,
            })

            tile_count += 1

    meta_path = META_DIR / f"{chart_name}.json"

    with open(meta_path, "w") as f:
        json.dump(tile_metadata, f, indent=2)

    print(f"\n{chart_name}: {tile_count} tiles")

print("\nDONE")