import os
import json
import cv2
from collections import defaultdict

# ============================================================
# CONFIG
# ============================================================

LABELS_DIR = "runs/detect/predict/labels"

TEST_IMAGES_DIR = "tiled_dataset/images/test"

OUTPUT_DIR = "."

CLASS_NAMES = {
    0: "unpaved_airport",
    1: "paved_airport",
    2: "unpaved_control_airport",
    3: "paved_control_airport",
    4: "VORTAC",
    5: "VORDME",
    6: "hard_runway",
    7: "hard_control_runway",
    8: "APA",
    9: "space_launch",
    10: "aircraft"
}

# ============================================================
# IOU CALCULATION
# ============================================================

def compute_iou(box1, box2):

    x_left = max(box1["x1"], box2["x1"])
    y_top = max(box1["y1"], box2["y1"])
    x_right = min(box1["x2"], box2["x2"])
    y_bottom = min(box1["y2"], box2["y2"])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection = (
        (x_right - x_left) *
        (y_bottom - y_top)
    )

    area1 = (
        (box1["x2"] - box1["x1"]) *
        (box1["y2"] - box1["y1"])
    )

    area2 = (
        (box2["x2"] - box2["x1"]) *
        (box2["y2"] - box2["y1"])
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union

# ============================================================
# GLOBAL CONFIDENCE-AWARE NMS
# ============================================================

def apply_global_nms(detections, iou_threshold=0.5):

    final_detections = []

    # ========================================================
    # GROUP BY CLASS
    # ========================================================

    class_groups = defaultdict(list)

    for det in detections:

        class_groups[det["class"]].append(det)

    # ========================================================
    # PROCESS EACH CLASS SEPARATELY
    # ========================================================

    for cls_name, cls_detections in class_groups.items():

        # sort by confidence descending
        cls_detections = sorted(
            cls_detections,
            key=lambda x: x["confidence"],
            reverse=True
        )

        kept = []

        while cls_detections:

            best = cls_detections.pop(0)

            kept.append(best)

            remaining = []

            for det in cls_detections:

                iou = compute_iou(
                    best["bbox_global"],
                    det["bbox_global"]
                )

                # keep detections with LOW overlap
                if iou < iou_threshold:

                    remaining.append(det)

            cls_detections = remaining

        final_detections.extend(kept)

    return final_detections

# ============================================================
# STORAGE
# ============================================================

chart_data = defaultdict(list)

chart_sizes = {}

# ============================================================
# GET IMAGE SIZE
# ============================================================

print("Reading tile image sizes...")

for file in os.listdir(TEST_IMAGES_DIR):

    if not file.endswith(".png"):
        continue

    chart_name = "_".join(
        file.replace(".png", "").split("_")[:-2]
    )

    img_path = os.path.join(TEST_IMAGES_DIR, file)

    img = cv2.imread(img_path)

    if img is None:
        continue

    h, w = img.shape[:2]

    chart_sizes[chart_name] = {
        "tile_width": w,
        "tile_height": h
    }

# ============================================================
# PROCESS LABEL FILES
# ============================================================

print("\nProcessing YOLO label files...")

label_files = [
    f for f in os.listdir(LABELS_DIR)
    if f.endswith(".txt")
]

print(f"Found {len(label_files)} label files")

for label_file in label_files:

    # ========================================================
    # PARSE TILE NAME
    # Example:
    # Washington_r2176_c7616.txt
    # ========================================================

    base = label_file.replace(".txt", "")

    parts = base.split("_")

    chart_name = "_".join(parts[:-2])

    row_part = parts[-2]
    col_part = parts[-1]

    tile_y = int(row_part.replace("r", ""))
    tile_x = int(col_part.replace("c", ""))

    # ========================================================
    # LOAD TILE IMAGE
    # ========================================================

    tile_img_path = os.path.join(
        TEST_IMAGES_DIR,
        base + ".png"
    )

    tile_img = cv2.imread(tile_img_path)

    if tile_img is None:

        print(f"WARNING: Missing tile image: {tile_img_path}")
        continue

    tile_h, tile_w = tile_img.shape[:2]

    # ========================================================
    # READ LABEL FILE
    # ========================================================

    label_path = os.path.join(LABELS_DIR, label_file)

    with open(label_path, "r") as f:

        lines = f.readlines()

    for line in lines:

        line = line.strip()

        if not line:
            continue

        parts = line.split()

        # ====================================================
        # YOLO FORMAT
        # class xc yc w h conf
        # ====================================================

        if len(parts) < 6:
            continue

        cls = int(parts[0])

        x_center = float(parts[1])

        y_center = float(parts[2])

        width = float(parts[3])

        height = float(parts[4])

        confidence = float(parts[5])

        # ====================================================
        # NORMALIZED -> TILE PIXELS
        # ====================================================

        box_w = width * tile_w
        box_h = height * tile_h

        center_x_tile = x_center * tile_w
        center_y_tile = y_center * tile_h

        x1_tile = center_x_tile - box_w / 2
        y1_tile = center_y_tile - box_h / 2
        x2_tile = center_x_tile + box_w / 2
        y2_tile = center_y_tile + box_h / 2

        # ====================================================
        # TILE -> GLOBAL MAP COORDS
        # ====================================================

        x1_global = x1_tile + tile_x
        y1_global = y1_tile + tile_y
        x2_global = x2_tile + tile_x
        y2_global = y2_tile + tile_y

        center_x_global = center_x_tile + tile_x
        center_y_global = center_y_tile + tile_y

        detection = {

            "class": CLASS_NAMES[cls],

            "confidence": round(confidence, 4),

            "tile": {
                "x": tile_x,
                "y": tile_y
            },

            "bbox_tile": {
                "x1": round(x1_tile, 2),
                "y1": round(y1_tile, 2),
                "x2": round(x2_tile, 2),
                "y2": round(y2_tile, 2)
            },

            "bbox_global": {
                "x1": round(x1_global, 2),
                "y1": round(y1_global, 2),
                "x2": round(x2_global, 2),
                "y2": round(y2_global, 2)
            },

            "center_global": {
                "x": round(center_x_global, 2),
                "y": round(center_y_global, 2)
            }
        }

        chart_data[chart_name].append(detection)

# ============================================================
# APPLY GLOBAL NMS
# ============================================================

print("\nApplying confidence-aware global NMS...")

for chart_name in chart_data:

    before_count = len(chart_data[chart_name])

    chart_data[chart_name] = apply_global_nms(
        chart_data[chart_name],
        iou_threshold=0.5
    )

    after_count = len(chart_data[chart_name])

    print(
        f"{chart_name}: "
        f"{before_count} -> {after_count}"
    )

# ============================================================
# SAVE JSON FILES
# ============================================================

print("\nSaving JSON files...")

for chart_name, detections in chart_data.items():

    output = {

        "chart": chart_name,

        "detections": detections
    }
    
    output_path = f"{chart_name}.json"

    with open(output_path, "w") as f:

        json.dump(output, f, indent=2)

    print(f"Saved: {output_path}")

print("\nDONE")