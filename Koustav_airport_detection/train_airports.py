from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="tiled_dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=4,
    device="cpu",
    workers=4,
    project="aviation_training",
    name="faa_multiclass"
)