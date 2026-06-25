"""
Phase 3 — Georeferencing
VFR Chart Extraction Pipeline (FAA Base Model)

Converts pixel coordinates to geographic coordinates (lat/lon)
using known chart corner GCPs and a bilinear affine model.

Steps:
    1. Define Ground Control Points (GCPs) from known chart extents
    2. Fit bilinear affine model: lon = f(px, py), lat = g(px, py)
    3. Apply to all detection pixel coordinates
    4. Save georef model and georeferenced detection CSV
    5. Render detection overlay on chart image

Usage:
    python phase3_georeferencing.py \
        --detections Washington_phase4_log.json \
        --output-dir outputs/phase3_georeferencing \
        --dpi 150

Notes:
    - Washington VFR Sectional (Ed.94 Mar 2026) extent:
        Lat 36°N–40°N, Lon 72°W–79°W
        (Verified from printed tick marks on physical chart corners)
    - Chart is Lambert Conformal Conic; this v1 uses a bilinear
      affine model. Lambert residual correction planned for v2.
    - DPI must match the DPI used to generate the phase4 image.
    - The legend/margin area (x < ~1512 at 150 DPI) is excluded
      by the phase4 pipeline; all detections fall in the map data area.
"""

import json
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
Image.MAX_IMAGE_PIXELS = None


# ── Known chart extent (Washington VFR Sectional) ────────────────────────────
# Coordinates read directly from printed degree tick marks
# on the four corners of the Washington sectional chart border.
# Image coordinate system: x increases east, y increases south (image convention).

CHART_GCPS_150DPI = [
    # (pixel_x, pixel_y, lon_deg, lat_deg)
    (1512,    0, -79.0, 40.0),   # NW corner of data area
    (8385,    0, -72.0, 40.0),   # NE corner
    (1512, 6173, -79.0, 36.0),   # SW corner
    (8385, 6173, -72.0, 36.0),   # SE corner
]

OVERLAY_RADIUS = 1   # circle radius in pixels


def fit_affine_model(gcps):
    """
    Fit a bilinear affine georeferencing model from GCPs.

    Model:
        lon = a0 + a1*px + a2*py
        lat = b0 + b1*px + b2*py

    Returns (lon_coeffs, lat_coeffs) as numpy arrays of shape (3,).
    """
    A = np.array([[1, g[0], g[1]] for g in gcps], dtype=float)
    lon_vals = np.array([g[2] for g in gcps], dtype=float)
    lat_vals = np.array([g[3] for g in gcps], dtype=float)

    lon_c, _, _, _ = np.linalg.lstsq(A, lon_vals, rcond=None)
    lat_c, _, _, _ = np.linalg.lstsq(A, lat_vals, rcond=None)
    return lon_c, lat_c


def pixel_to_latlon(px, py, lon_c, lat_c):
    """Convert pixel (px, py) to (lon_deg, lat_deg) using fitted coefficients."""
    row = np.array([1.0, float(px), float(py)])
    return float(row @ lon_c), float(row @ lat_c)


def compute_reprojection_error_km(gcps, lon_c, lat_c):
    """Return mean reprojection error in km across GCPs."""
    errors = []
    for g in gcps:
        lon_pred, lat_pred = pixel_to_latlon(g[0], g[1], lon_c, lat_c)
        lon_err_km = abs(lon_pred - g[2]) * 111.0 * np.cos(np.radians(g[3]))
        lat_err_km = abs(lat_pred - g[3]) * 111.0
        errors.append(np.sqrt(lon_err_km**2 + lat_err_km**2))
    return float(np.mean(errors))


def georeference_detections(detections, lon_c, lat_c):
    """
    Apply georef model to a list of detection dicts.

    Adds 'lat_deg', 'lon_deg', 'confidence' fields to each detection.
    Returns list of enriched records.
    """
    def tier(tri):
        if tri >= 0.90: return 'HIGH'
        if tri >= 0.75: return 'MEDIUM'
        if tri >= 0.55: return 'LOW'
        return 'MARGINAL'

    records = []
    for i, d in enumerate(detections, start=1):
        lon, lat = pixel_to_latlon(d['pixel_x'], d['pixel_y'], lon_c, lat_c)
        records.append({
            'detection_id': i,
            'pixel_x': d['pixel_x'],
            'pixel_y': d['pixel_y'],
            'lat_deg': round(lat, 6),
            'lon_deg': round(lon, 6),
            'symbol_class': 'obstacle',
            'tri_overlap': d['tri_overlap'],
            'dot_overlap': d['dot_overlap'],
            'confidence': tier(d['tri_overlap']),
            'blob_area_px2': d['blob_area_px2'],
            'solidity': d['solidity'],
        })
    return records


def load_detections(detections_path):
    """
    Load the detections list from a Phase 4 JSON file.

    The key name is detected automatically to handle variation between
    Phase 4 script versions. Raises KeyError if no list is found.
    """
    with open(detections_path) as f:
        phase4 = json.load(f)

    # Try known key names first
    for key in ['detections', 'obstacles', 'results', 'data']:
        if key in phase4:
            print(f"Detections loaded from key: '{key}'")
            return phase4[key]

    # Fall back to the first non-empty list value in the JSON
    for key, val in phase4.items():
        if isinstance(val, list) and len(val) > 0:
            print(f"Detections loaded from key: '{key}' (auto-detected)")
            return val

    raise KeyError(
        f"No detections list found in JSON. "
        f"Available keys: {list(phase4.keys())}"
    )


def render_overlay(records, chart_image_path, output_path):
    """
    Draw black filled circles with bold coordinate labels on the chart image.

    All detections are drawn regardless of confidence tier.
    Labels show latitude and longitude to four decimal places.

    If the chart image is not found at chart_image_path, the overlay
    is skipped and a warning is printed rather than raising an error.
    """
    chart_path = Path(chart_image_path)
    if not chart_path.exists():
        print(f"  Warning: chart image not found at '{chart_image_path}' — overlay skipped.")
        print(  "  Set CHART_IMAGE in the CONFIG block to your Phase 1 or rendered PDF PNG.")
        return

    print(f"Rendering overlay on '{chart_path.name}' ...")
    img = Image.open(chart_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Scale pixel coordinates from Phase 4 space (150 DPI) to actual image size
    img_w, img_h = img.size
    coord_w, coord_h = 8385, 6173
    scale_x = img_w / coord_w
    scale_y = img_h / coord_h
    r = max(int(OVERLAY_RADIUS * scale_x), 6)

    # Load bold font; fall back to default if arial bold is not found
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", size=13)
    except IOError:
        try:
            font = ImageFont.truetype("arialbd.ttf", size=13)
        except IOError:
            font = ImageFont.load_default()

    BLACK = (0, 0, 0, 255)

    for rec in records:
        sx = int(float(rec['pixel_x']) * scale_x)
        sy = int(float(rec['pixel_y']) * scale_y)

        # Thick outer ring for visibility against busy chart background
        draw.ellipse(
            [sx - r - 2, sy - r - 2, sx + r + 2, sy + r + 2],
            outline=BLACK, width=3
        )
        # Filled black circle
        draw.ellipse(
            [sx - r, sy - r, sx + r, sy + r],
            fill=BLACK
        )

        # Bold coordinate label to the right of the circle
        label = f"{float(rec['lat_deg']):.4f}N  {abs(float(rec['lon_deg'])):.4f}W"
        draw.text((sx + r + 6, sy - 11), label, fill=BLACK, font=font)

    result = Image.alpha_composite(img, overlay).convert("RGB")
    result.save(str(output_path), format="PNG", compress_level=3)
    print(f"Overlay saved: {output_path}")


def run(detections_path, output_dir, dpi=150, chart_image_path=None):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Adjust GCPs if DPI differs from 150
    scale = dpi / 150.0
    gcps = [(g[0]*scale, g[1]*scale, g[2], g[3]) for g in CHART_GCPS_150DPI]

    # Fit model
    lon_c, lat_c = fit_affine_model(gcps)
    rms_km = compute_reprojection_error_km(gcps, lon_c, lat_c)
    print(f"Georef model fitted. GCP reprojection RMS: {rms_km:.4f} km")

    # Save model
    model = {
        'method': 'bilinear_affine',
        'chart': 'Washington VFR Sectional (Ed.94 Mar 2026)',
        'dpi': dpi,
        'gcps': gcps,
        'lon_coeffs': lon_c.tolist(),
        'lat_coeffs': lat_c.tolist(),
        'gcp_rms_km': rms_km,
        'notes': (
            'Corner-based affine model. GCPs read from printed tick marks on '
            'physical chart borders. True extent: 36-40N, 72-79W. '
            'Lambert Conformal Conic residual not corrected in v1. '
            'Precision: ~1-3 NM at chart interior.'
        ),
    }
    model_path = out / 'georef_model.json'
    with open(model_path, 'w') as f:
        json.dump(model, f, indent=2)
    print(f"Model saved: {model_path}")

    # Load detections
    dets = load_detections(detections_path)
    print(f"Georeferencing {len(dets)} detections...")

    # Georeference
    records = georeference_detections(dets, lon_c, lat_c)

    # Save CSV
    csv_path = out / 'detections_georef.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=records[0].keys())
        w.writeheader()
        w.writerows(records)
    print(f"Georeferenced CSV: {csv_path} ({len(records)} records)")

    # Save GeoJSON (HIGH + MEDIUM only)
    features = []
    for rec in records:
        if rec['confidence'] in ('HIGH', 'MEDIUM'):
            features.append({
                'type': 'Feature',
                'geometry': {'type': 'Point', 'coordinates': [rec['lon_deg'], rec['lat_deg']]},
                'properties': {k: v for k, v in rec.items() if k not in ('lat_deg', 'lon_deg')},
            })
    gj = {'type': 'FeatureCollection', 'features': features}
    gj_path = out / 'detections_high_medium.geojson'
    with open(gj_path, 'w') as f:
        json.dump(gj, f)
    print(f"GeoJSON: {gj_path} ({len(features)} features)")

    # Render overlay
    if chart_image_path:
        overlay_path = out / 'detections_overlay.png'
        render_overlay(records, chart_image_path, overlay_path)
    else:
        print("No CHART_IMAGE set — overlay skipped.")

    # Summary
    by_tier = {}
    for rec in records:
        by_tier[rec['confidence']] = by_tier.get(rec['confidence'], 0) + 1
    print("Confidence breakdown:", by_tier)
    return records, model


if __name__ == '__main__':
    # ── CONFIG ──────────────────────────────────────────────────────────────────
    DETECTIONS_PATH = "outputs/phase4_symbol_detection/phase4a_rgb_obstacles/Washington_phase4_log.json"
    OUTPUT_DIR      = "outputs/phase3_georeferencing"
    DPI             = 150
    CHART_IMAGE     = "outputs/phase1_preprocessing/Washington_rgb_300dpi.png"   # set to None to skip overlay
    # ────────────────────────────────────────────────────────────────────────────

    run(DETECTIONS_PATH, OUTPUT_DIR, DPI, CHART_IMAGE)