from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

WEIGHTS_PATH = Path("models/weights/best.onnx")

CLASS_NAMES = {
    0: "Address",
    1: "Date",
    2: "Item",
    3: "OrderId",
    4: "Subtotal",
    5: "Tax",
    6: "Title",
    7: "TotalPrice",
}


class YOLOInference:
    def __init__(self, weights_path: Path = WEIGHTS_PATH):
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights not found at {weights_path}")

        self.session = ort.InferenceSession(
            str(weights_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.class_names = CLASS_NAMES

    def preprocess(self, image: Image.Image) -> np.ndarray:
        image = image.resize((640, 640))
        img_array = np.array(image).astype(np.float32)
        img_array = img_array / 255.0
        img_array = img_array.transpose(2, 0, 1)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def postprocess(
        self,
        outputs: np.ndarray,
        orig_width: int,
        orig_height: int,
        confidence: float,
    ) -> list[dict]:
        predictions = outputs[0]
        detections = []

        if predictions.ndim == 3:
            predictions = predictions[0]

        predictions = predictions.T

        for pred in predictions:
            class_scores = pred[4:]
            class_id = int(np.argmax(class_scores))
            conf = float(class_scores[class_id])

            if conf < confidence:
                continue

            cx, cy, w, h = pred[:4]

            x1 = float((cx - w / 2) * orig_width / 640)
            y1 = float((cy - h / 2) * orig_height / 640)
            x2 = float((cx + w / 2) * orig_width / 640)
            y2 = float((cy + h / 2) * orig_height / 640)

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": self.class_names.get(class_id, "unknown"),
                    "confidence": round(conf, 4),
                    "bbox": {
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2),
                    },
                }
            )

        return detections

    def predict(
        self,
        image: Image.Image,
        confidence: float = 0.5,
        iou: float = 0.45,
    ) -> list[dict]:
        orig_width, orig_height = image.size
        input_array = self.preprocess(image)

        outputs = self.session.run(None, {self.input_name: input_array})

        return self.postprocess(outputs[0], orig_width, orig_height, confidence)
