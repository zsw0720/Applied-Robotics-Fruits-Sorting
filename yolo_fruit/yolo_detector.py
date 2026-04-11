from ultralytics import YOLO
import torch


class YoloFruitDetector:
    def __init__(self, model_path="yolo26n.pt"):
        self.device = self._get_device()
        self.model = YOLO(model_path)

    def _get_device(self):
        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def detect(self, frame, conf=0.30):
        results = self.model.predict(
            source=frame,
            conf=conf,
            device=self.device,
            verbose=False
        )

        result = results[0]
        annotated = result.plot()

        detections = []
        if result.boxes is not None:
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            clss = result.boxes.cls.cpu().numpy()

            for box, score, cls_id in zip(xyxy, confs, clss):
                cls_id = int(cls_id)
                name = result.names[cls_id]
                x1, y1, x2, y2 = box.astype(int).tolist()

                detections.append({
                    "label": name,
                    "score": float(score),
                    "bbox": (x1, y1, x2, y2),
                })

        return annotated, detections
