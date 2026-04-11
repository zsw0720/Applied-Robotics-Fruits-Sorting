from ultralytics import YOLO

model = YOLO("yolo26n.pt")

model.train(
    data="dataset/dataset.yaml",
    epochs=100,
    imgsz=768,
    batch=4
)
