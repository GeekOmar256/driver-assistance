import cv2
from ultralytics import YOLO

import config


class SignDetector:
    """Wraps a YOLO model and draws bounding boxes for detected classes."""

    def __init__(self, model_path=config.MODEL_PATH, confidence=config.CONFIDENCE_THRESHOLD,
                 target_classes=config.TARGET_CLASS_NAMES):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.target_classes = set(target_classes) if target_classes else None

    def annotate(self, frame):
        results = self.model.predict(frame, conf=self.confidence, verbose=False)[0]
        names = results.names

        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = names[class_id]
            if self.target_classes and class_name not in self.target_classes:
                continue

            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
            label = f"{class_name} {conf:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - text_h - 8)), (x1 + text_w + 4, y1), (0, 220, 0), -1)
            cv2.putText(frame, label, (x1 + 2, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 1, cv2.LINE_AA)

        return frame
