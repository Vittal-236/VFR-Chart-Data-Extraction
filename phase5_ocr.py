"""
Phase 5 — OCR: Elevation Number Extraction
VFR Chart Extraction Pipeline (FAA Base Model)

Extracts MSL/AGL elevation numbers printed adjacent to obstacle symbols
using Tesseract OCR on patches cropped from the full-resolution chart image.

FAA obstacle symbol annotation format:
    MSL elevation (ft) printed above/right of symbol
    AGL height (ft) printed in smaller font below MSL
    Example: "1234" or "1234/456" (MSL/AGL)

Steps:
    1. Load high-resolution chart image (PDF rendered at DPI)
    2. For each detected obstacle, crop a patch above the symbol centroid
    3. Upscale patch 3x, invert if dark background
    4. Run Tesseract with digit-only whitelist (PSM 7)
    5. Parse MSL and AGL from OCR output
    6. Merge with georeferenced detection CSV

Usage:
    python phase5_ocr.py \
        --chart-image Washington_150dpi.png \
        --georef-csv outputs/phase3_georeferencing/detections_georef.csv \
        --output-dir outputs/phase5_ocr \
        --confidence HIGH MEDIUM

Requirements:
    pip install pytesseract pillow
    apt-get install tesseract-ocr  (or equivalent)
"""

import csv
import re
from pathlib import Path

import numpy as np
import pytesseract
from PIL import Image


TESS_CONFIG = '--psm 7 -c tessedit_char_whitelist=0123456789/'


def load_detections_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def crop_elevation_patch(rgb_arr, px, py, pad_x=35, pad_y_above=40):
    """
    Crop the text patch above and around the obstacle symbol.

    The MSL elevation label on FAA charts is printed directly above the
    symbol centroid, within ~40px vertically and ~35px horizontally at 150 DPI.
    """
    H, W = rgb_arr.shape[:2]
    x0 = max(0, int(px) - pad_x)
    y0 = max(0, int(py) - pad_y_above)
    x1 = min(W, int(px) + pad_x + 20)   # slight right bias for label
    y1 = max(0, int(py) - 2)
    if y1 <= y0 or x1 <= x0:
        return None
    return rgb_arr[y0:y1, x0:x1]


def run_ocr(patch_rgb):
    """
    Run Tesseract on a patch.
    Returns raw string (digits and '/' only).
    """
    pil = Image.fromarray(patch_rgb)
    # Upscale for better OCR accuracy
    pil = pil.resize((pil.width * 3, pil.height * 3), Image.LANCZOS)
    # Invert dark-background patches
    arr = np.array(pil)
    if arr.mean() < 128:
        pil = Image.fromarray(255 - arr)
    return pytesseract.image_to_string(pil, config=TESS_CONFIG).strip()


def parse_elevation(raw_text):
    """
    Parse MSL and AGL from raw OCR text.

    FAA format: "1234" = MSL only; "1234/456" = MSL/AGL
    Returns (msl_str, agl_str) — empty string if not found.
    """
    nums = re.findall(r'\d+', raw_text)
    if not nums:
        return '', ''
    # Heuristic: first number ≥ 3 digits is MSL; second is AGL
    valid = [n for n in nums if len(n) >= 2]
    msl = valid[0] if valid else ''
    agl = valid[1] if len(valid) > 1 else ''
    return msl, agl


def run(chart_image_path, georef_csv_path, output_dir, confidence_filter=None, max_detections=None):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading chart image: {chart_image_path}")
    rgb = np.array(Image.open(chart_image_path).convert('RGB'))
    print(f"Chart: {rgb.shape[1]} x {rgb.shape[0]} px")

    detections = load_detections_csv(georef_csv_path)
    print(f"Loaded {len(detections)} detections from CSV")

    if confidence_filter:
        detections = [d for d in detections if d['confidence'] in confidence_filter]
        print(f"After confidence filter ({confidence_filter}): {len(detections)}")

    if max_detections:
        detections = detections[:max_detections]

    # OCR each detection
    results = []
    success = 0
    for det in detections:
        px = float(det['pixel_x'])
        py = float(det['pixel_y'])

        patch = crop_elevation_patch(rgb, px, py)
        msl, agl = '', ''

        if patch is not None:
            try:
                raw = run_ocr(patch)
                msl, agl = parse_elevation(raw)
                if msl:
                    success += 1
            except Exception:
                pass

        r = dict(det)
        r['elevation_msl_ft'] = msl
        r['elevation_agl_ft'] = agl
        results.append(r)

    print(f"OCR: {success}/{len(results)} returned elevation ({100*success//max(len(results),1)}%)")

    # Save output CSV
    out_csv = out / 'detections_with_elevation.csv'
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"Saved: {out_csv}")

    # Sample results
    with_elev = [r for r in results if r['elevation_msl_ft']][:10]
    print("\nSample detections with elevation:")
    for r in with_elev:
        elev = r['elevation_msl_ft']
        agl = f"/{r['elevation_agl_ft']}" if r['elevation_agl_ft'] else ''
        print(f"  ID {r['detection_id']:5s} ({r['lat_deg']}N, {r['lon_deg']}W) "
              f"tri={r['tri_overlap']} elev={elev}{agl}ft")

    return results


if __name__ == '__main__':
    # ── CONFIG ──────────────────────────────────────────────────────────────────
    CHART_IMAGE = "outputs/phase1_preprocessing/chart_rgb.png"   # Full-res chart PNG at DPI
    GEOREF_CSV  = "outputs/phase3_georeferencing/georef.csv"     # Phase 3 georef CSV
    OUTPUT_DIR  = "outputs/phase5_ocr"
    CONFIDENCE  = ['HIGH', 'MEDIUM']  # Confidence tiers to process
    MAX         = None                # Max detections (set int for debug)
    # ────────────────────────────────────────────────────────────────────────────

    run(CHART_IMAGE, GEOREF_CSV, OUTPUT_DIR,
        confidence_filter=CONFIDENCE, max_detections=MAX)
