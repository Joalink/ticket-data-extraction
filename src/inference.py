from pathlib import Path

from PIL import Image
from ultralytics import YOLO

WEIGHTS_PATH = Path("models/weights/best.onnx")


class YOLOInference:
    def __init__(self, weights_path: Path = WEIGHTS_PATH):
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights not found at {weights_path}")
        self.model = YOLO(str(weights_path), task="detect")
        self.class_names = self.model.names

    def predict(
        self,
        image: Image.Image,
        confidence: float = 0.5,
        iou: float = 0.45,
    ) -> list[dict]:
        results = self.model.predict(
            source=image,
            conf=confidence,
            iou=iou,
            verbose=False,
        )

        detections = []

        for result in results:
            boxes = result.boxes

            if boxes is None:
                continue

            for box in boxes:
                detection = {
                    "class_id": int(box.cls[0]),
                    "class_name": self.class_names[int(box.cls[0])],
                    "confidence": round(float(box.conf[0]), 4),
                    "bbox": {
                        "x1": round(float(box.xyxy[0][0]), 2),
                        "y1": round(float(box.xyxy[0][1]), 2),
                        "x2": round(float(box.xyxy[0][2]), 2),
                        "y2": round(float(box.xyxy[0][3]), 2),
                    },
                }
                detections.append(detection)

        return detections
