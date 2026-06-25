from ultralytics import YOLO

# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = YOLO(
    "runs/detect/aviation_training/faa_multiclass/weights/best.pt"
)

# ============================================================
# RUN INFERENCE
# ============================================================

results = model.predict(
    source="tiled_dataset/images/test",
    imgsz=640,
    conf=0.25,
    save=True,
    save_txt=True,
    save_conf=True,
    # project="runs/detect",
    # name="predict"
)

print("Inference complete.")