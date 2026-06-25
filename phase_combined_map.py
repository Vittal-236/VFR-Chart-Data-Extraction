"""
Combined Detection Map
VFR Chart Extraction Pipeline -- Cognida.ai

Overlays every detection result onto the full-resolution 150 DPI Washington
chart as colour-coded bounding squares:

  Green   -- Single obstacles   (phase4_combined_obstacles)
  Magenta -- Double obstacles   (phase4_combined_obstacles)
  Red     -- Private / Restricted Area circles (phase4b_binary_private)
  Blue    -- Airspace circles   (phase4d_binary_circles)
  Orange  -- Power-line towers  (phase4f_towers)

Output: outputs/combined_detection_map.png  (~8385 x 6173 px, 150 DPI)
"""

import json
import os
import cv2

# ── paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(BASE_DIR, "outputs")
PHASE4_DIR = os.path.join(OUT_DIR, "phase4_symbol_detection")

# Full-resolution 150 DPI base image (8385 x 6173 px, 155 MB in RAM)
RGB_IMAGE  = os.path.join(OUT_DIR, "phase1_preprocessing", "Washington_rgb_150dpi.png")

# All detections were produced on the 300 DPI image; scale coords by 0.5
COORD_SCALE = 0.5

DETECTION_FILES = {
    "combined": os.path.join(PHASE4_DIR, "phase4_combined_obstacles", "detections_combined.json"),
    "private":  os.path.join(PHASE4_DIR, "phase4b_binary_private",    "detections.json"),
    "airspace": os.path.join(PHASE4_DIR, "phase4d_binary_circles",    "detections.json"),
    "tower":    os.path.join(PHASE4_DIR, "phase4f_towers",            "detections.json"),
}

OUTPUT_MAP = os.path.join(OUT_DIR, "combined_detection_map.png")

# ── colours (BGR) ──────────────────────────────────────────────────────────
COLOURS = {
    "single_obstacle": (  0, 210,   0),   # green
    "double_obstacle": (255,   0, 240),   # magenta
    "private":         (  0,   0, 220),   # red
    "airspace":        (200,  30,  10),   # blue
    "tower":           (  0, 140, 255),   # orange
}

LABELS = {
    "single_obstacle": "Single Obstacle",
    "double_obstacle": "Double Obstacle",
    "private":         "Private / Restricted Area",
    "airspace":        "Airspace Circle",
    "tower":           "Power-Line Tower",
}

BOX_THICKNESS = 2


# ── helper: draw a square box centred at (cx, cy) with half-side = hs ──────
def box(img, cx, cy, hs, colour, thickness=BOX_THICKNESS):
    hs = max(hs, 4)
    cv2.rectangle(img,
                  (cx - hs, cy - hs),
                  (cx + hs, cy + hs),
                  colour, thickness, cv2.LINE_AA)


# ── legend ─────────────────────────────────────────────────────────────────
def draw_legend(img, labels, colours, counts):
    x0, y0   = 30, 30
    font     = cv2.FONT_HERSHEY_SIMPLEX
    fs       = 1.1
    lh       = 50
    pad      = 18
    swatch_w = 32
    box_w    = 560
    box_h    = len(labels) * lh + 2 * pad

    overlay = img.copy()
    cv2.rectangle(overlay,
                  (x0 - pad, y0 - pad),
                  (x0 + box_w, y0 + box_h),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.70, img, 0.30, 0, img)

    for i, key in enumerate(labels):
        cy     = y0 + i * lh + lh // 2
        colour = colours[key]
        cv2.rectangle(img,
                      (x0, cy - 12), (x0 + swatch_w, cy + 12),
                      colour, -1)
        text = f"{labels[key]}  ({counts[key]})"
        cv2.putText(img, text,
                    (x0 + swatch_w + 12, cy + 10),
                    font, fs, (230, 230, 230), 2, cv2.LINE_AA)


# ── main ───────────────────────────────────────────────────────────────────
def main():
    print("Loading base image (150 DPI) ...")
    canvas = cv2.imread(RGB_IMAGE)
    if canvas is None:
        raise FileNotFoundError(f"Base image not found:\n  {RGB_IMAGE}")
    h, w = canvas.shape[:2]
    print(f"  Size: {w} x {h} px")

    # ── load detections ────────────────────────────────────────────────────
    print("Loading detections ...")
    combined    = json.load(open(DETECTION_FILES["combined"]))
    single_list = combined["single_confirmed"]   # centre_col, centre_row, scale_w, scale_h
    double_list = combined["double_confirmed"]   # centre_col, centre_row, scale_w, scale_h

    priv_list  = json.load(open(DETECTION_FILES["private"]))["confirmed"]   # col, row, radius
    air_list   = json.load(open(DETECTION_FILES["airspace"]))["confirmed"]  # col, row, radius
    tower_list = json.load(open(DETECTION_FILES["tower"]))["confirmed"]     # map_c, map_r

    counts = {
        "single_obstacle": len(single_list),
        "double_obstacle": len(double_list),
        "private":         len(priv_list),
        "airspace":        len(air_list),
        "tower":           len(tower_list),
    }
    for k, n in counts.items():
        print(f"  {LABELS[k]:<35} {n:>5}")

    CS = COORD_SCALE   # shorthand

    # ── airspace circles (largest boxes -- draw first) ─────────────────────
    print("Drawing airspace squares ...")
    for d in air_list:
        cx = int(d["col"]    * CS)
        cy = int(d["row"]    * CS)
        hs = int(d["radius"] * CS)
        box(canvas, cx, cy, hs, COLOURS["airspace"])

    # ── private / restricted circles ───────────────────────────────────────
    print("Drawing private squares ...")
    for d in priv_list:
        cx = int(d["col"]    * CS)
        cy = int(d["row"]    * CS)
        hs = int(d["radius"] * CS)
        box(canvas, cx, cy, hs, COLOURS["private"], thickness=3)

    # ── single obstacles ───────────────────────────────────────────────────
    print("Drawing single-obstacle squares ...")
    for d in single_list:
        cx = int(d["centre_col"] * CS)
        cy = int(d["centre_row"] * CS)
        hs = max(int(d["scale_w"] / 2 * CS), 6)
        box(canvas, cx, cy, hs, COLOURS["single_obstacle"])

    # ── double obstacles ───────────────────────────────────────────────────
    print("Drawing double-obstacle squares ...")
    for d in double_list:
        cx = int(d["centre_col"] * CS)
        cy = int(d["centre_row"] * CS)
        hs = max(int(d["scale_w"] / 2 * CS), 8)
        box(canvas, cx, cy, hs, COLOURS["double_obstacle"])

    # ── power-line towers ──────────────────────────────────────────────────
    print("Drawing tower squares ...")
    for d in tower_list:
        cx = int(d["map_c"] * CS)
        cy = int(d["map_r"] * CS)
        box(canvas, cx, cy, 12, COLOURS["tower"], thickness=3)

    # ── legend + title ─────────────────────────────────────────────────────
    draw_legend(canvas, LABELS, COLOURS, counts)

    cv2.putText(
        canvas,
        "VFR Sectional Chart -- Combined Detection Map  (Washington, 150 DPI)",
        (30, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (240, 240, 240), 2, cv2.LINE_AA,
    )

    # ── save ───────────────────────────────────────────────────────────────
    print("Saving ...")
    cv2.imwrite(OUTPUT_MAP, canvas, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    print(f"Saved: {OUTPUT_MAP}")
    print(f"Size : {os.path.getsize(OUTPUT_MAP) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
