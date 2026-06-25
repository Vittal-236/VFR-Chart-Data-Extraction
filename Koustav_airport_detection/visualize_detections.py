import os
import json
import cv2

# ============================================================
# CONFIG
# ============================================================

JSON_FILES = [
    "Washington.json",
    "Detroit.json"
]

RENDERED_DIR = "rendered_png"

OUTPUT_DIR = "."

# ============================================================
# COLORS
# ============================================================

CLASS_COLORS = {

    "unpaved_airport": (0, 255, 255),

    "paved_airport": (0, 255, 0),

    "unpaved_control_airport": (255, 0, 255),

    "paved_control_airport": (0, 0, 255),

    "VORTAC": (255, 255, 0),

    "VORDME": (255, 128, 0),

    "hard_runway": (128, 255, 0),

    "hard_control_runway": (0, 128, 255),

    "APA": (128, 128, 255),

    "space_launch": (255, 0, 128),

    "aircraft": (255, 255, 255)
}

# ============================================================
# PROCESS EACH JSON
# ============================================================

for json_file in JSON_FILES:

    print(f"\nProcessing: {json_file}")

    # ========================================================
    # LOAD JSON
    # ========================================================

    with open(json_file, "r") as f:

        data = json.load(f)

    chart_name = data["chart"]

    detections = data["detections"]

    # ========================================================
    # LOAD ORIGINAL MAP PNG
    # ========================================================

    image_path = os.path.join(
        RENDERED_DIR,
        chart_name,
        "page_1.png"
    )

    image = cv2.imread(image_path)

    if image is None:

        print(f"ERROR: Could not load {image_path}")
        continue

    print(f"Loaded map: {image_path}")

    # ========================================================
    # DRAW DETECTIONS
    # ========================================================

    for det in detections:

        cls_name = det["class"]

        conf = det["confidence"]

        bbox = det["bbox_global"]

        x1 = int(bbox["x1"])
        y1 = int(bbox["y1"])
        x2 = int(bbox["x2"])
        y2 = int(bbox["y2"])

        color = CLASS_COLORS.get(
            cls_name,
            (255, 255, 255)
        )

        # ====================================================
        # DRAW BOX
        # ====================================================

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        # ====================================================
        # LABEL
        # ====================================================

        label = f"{cls_name} {conf:.2f}"

        cv2.putText(
            image,
            label,
            (x1, max(y1 - 5, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA
        )

    # ========================================================
    # SAVE OUTPUT
    # ========================================================

    output_path = f"{chart_name}_overlay.png"

    cv2.imwrite(output_path, image)

    print(f"Saved: {output_path}")

print("\nDONE")