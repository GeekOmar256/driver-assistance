import cv2
from ultralytics import YOLO

import config

_UNSET = object()


class SignDetector:
    """Wraps a YOLO model and draws bounding boxes for detected classes.

    `scope` is a hard ceiling: classes outside it are never detected, because
    they are excluded from YOLO's own `classes=` postprocessing rather than
    just hidden after the fact. Per-request selections (from the UI) can only
    narrow within the scope, never widen past it.
    """

    def __init__(self, model_path=config.MODEL_PATH, confidence=config.CONFIDENCE_THRESHOLD,
                 scope=config.TARGET_CLASS_NAMES, image_size=640):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.image_size = image_size
        self.name_to_id = {name: idx for idx, name in self.model.names.items()}
        self.scope = self._known(scope) if scope else None

    def _known(self, names):
        """Keep only names the loaded model actually has, as a sorted list."""
        return sorted(n for n in set(names) if n in self.name_to_id)

    def class_names(self):
        """Class names selectable in the UI: the configured scope, or every
        class the model knows if no scope is set."""
        if self.scope is not None:
            return list(self.scope)
        return sorted(self.model.names.values())

    @staticmethod
    def _direction(x1, x2, frame_width):
        """Rough left/ahead/right split of the frame, by box center."""
        center_x = (x1 + x2) / 2
        if center_x < frame_width / 3:
            return "left"
        if center_x > 2 * frame_width / 3:
            return "right"
        return "ahead"

    def detect_and_annotate(self, frame, target_classes=_UNSET, confidence=None, image_size=None):
        """Draw boxes for detected objects and return (frame, detections).

        target_classes, if given, selects within the scope for this call: a
        (possibly empty) set/list of names to show. Omit the argument to show
        the full scope. Each detection dict has class_name, confidence and a
        left/ahead/right direction estimated from the box's horizontal
        position in the frame.
        """
        if target_classes is _UNSET:
            requested = self.scope
        else:
            requested = list(target_classes) if target_classes else []

        if requested is None:
            effective = self.scope
        elif self.scope is None:
            effective = requested
        else:
            effective = [n for n in requested if n in self.scope]

        class_ids = None
        if effective is not None:
            class_ids = [self.name_to_id[n] for n in effective if n in self.name_to_id]

        results = self.model.predict(frame, imgsz=image_size or self.image_size,
                                      conf=self.confidence if confidence is None else confidence, classes=class_ids,
                                      verbose=False)[0]
        names = results.names
        frame_width = frame.shape[1]
        detections = []

        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = names[class_id]

            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            direction = self._direction(x1, x2, frame_width)
            detections.append({"class_name": class_name, "confidence": conf, "direction": direction})

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
            label = f"{class_name} {conf:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - text_h - 8)), (x1 + text_w + 4, y1), (0, 220, 0), -1)
            cv2.putText(frame, label, (x1 + 2, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 1, cv2.LINE_AA)

        return frame, detections
