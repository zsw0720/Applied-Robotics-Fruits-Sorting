from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

results = model(
    "dataset/images/val",
    save=True,
    conf=0.05
)

for i, r in enumerate(results, start=1):
    print(f"image {i}: boxes =", 0 if r.boxes is None else len(r.boxes))

print("测试完成")