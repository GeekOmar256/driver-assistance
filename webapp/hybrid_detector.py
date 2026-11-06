"""Two-stage road-object localization and recognition on enlarged crops."""
import cv2
import numpy as np
from ultralytics import YOLO

import config
from detector import SignDetector, _UNSET

LIGHT_TYPES = {"red traffic light", "yellow traffic light", "green traffic light"}
SIGN_TYPES = {"stop sign", "yield sign", "no entry", "speed limit 40", "speed limit 60", "speed limit 80"}


def overlap(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = max(0, a[2]-a[0])*max(0, a[3]-a[1]) + max(0, b[2]-b[0])*max(0, b[3]-b[1]) - intersection
    return intersection / union if union else 0


def light_color(crop):
    """Conservative bright-lamp HSV fallback. Returns a color, not a probability.

    Require one dominant connected bright/saturated region. Ambiguous colors,
    small crops and dark lamps remain unknown rather than implying green.
    """
    if crop.size == 0 or min(crop.shape[:2]) < 8:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    bright = (s >= 100) & (v >= 180)
    scores = {}
    for color, hue in {
        "red": (h <= 10) | (h >= 170),
        "yellow": (h >= 16) & (h <= 38),
        "green": (h >= 40) & (h <= 95),
    }.items():
        mask = np.uint8(bright & hue)
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        areas = [int(row[cv2.CC_STAT_AREA]) for row in stats[1:count]]
        scores[color] = max(areas, default=0)
    ranked = sorted(scores, key=scores.get, reverse=True)
    best, second = ranked[:2]
    area = crop.shape[0] * crop.shape[1]
    if scores[best] < max(4, area * .003) or scores[best] > area * .55:
        return None
    if scores[best] < max(1, scores[second]) * 1.8:
        return None
    return best + " traffic light"


def red_sign_candidates(frame):
    """Shape/color proposals supplement COCO's missing non-stop sign classes.

    Proposals are never emitted as a typed sign without second-stage recognition.
    Work at bounded resolution; verify compact circular/triangular red contours.
    """
    height, width = frame.shape[:2]
    scale = min(1.0, 960 / max(height, width))
    image = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 90, 55), (12, 255, 255)) | cv2.inRange(hsv, (160, 90, 55), (179, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if min(w, h) < 10 or area < 60 or not .55 <= w / h <= 1.6:
            continue
        if w*h > image.shape[0]*image.shape[1]*.4 or not perimeter:
            continue
        hull = cv2.convexHull(contour)
        hull_perimeter = cv2.arcLength(hull, True)
        hull_area = cv2.contourArea(hull)
        vertices = len(cv2.approxPolyDP(hull, .035*hull_perimeter, True))
        circularity = 4*np.pi*hull_area/(hull_perimeter*hull_perimeter) if hull_perimeter else 0
        if (vertices == 3 or circularity >= .65) and hull_area/(w*h) >= .5 and area/(w*h) >= .08:
            candidates.append({"box": [x/scale, y/scale, (x+w)/scale, (y+h)/scale],
                               "family": "sign", "source": "red-shape", "confidence": None})
    return sorted(candidates, key=lambda p: (p["box"][2]-p["box"][0])*(p["box"][3]-p["box"][1]), reverse=True)[:12]


def stop_has_red_face(frame, bounds):
    """Reject circular white speed signs that COCO sometimes calls stop signs."""
    x1, y1, x2, y2 = bounds
    w, h = x2-x1, y2-y1
    center = frame[max(0,int(y1+.2*h)):min(frame.shape[0],int(y2-.2*h)),
                   max(0,int(x1+.2*w)):min(frame.shape[1],int(x2-.2*w))]
    if center.size == 0:
        return False
    hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
    red = ((hsv[:,:,0] <= 12) | (hsv[:,:,0] >= 160)) & (hsv[:,:,1] >= 80) & (hsv[:,:,2] >= 45)
    return float(red.mean()) >= .18


class HybridDetector:
    def __init__(self, pretrained_path, trained_path, color_fallback=False):
        self.localizer = YOLO(pretrained_path)
        self.recognizer = YOLO(trained_path)
        self.color_fallback = color_fallback
        self.localizer_ids = [i for i, name in self.localizer.names.items() if name in {"traffic light", "stop sign", "person"}]
        self.names = sorted((SIGN_TYPES | LIGHT_TYPES) & set(self.recognizer.names.values()) | {"stop sign"} | ({"person"} & set(self.localizer.names.values())))

    def class_names(self):
        return list(self.names)

    def _recognize(self, frame, proposal, threshold):
        height, width = frame.shape[:2]
        bx1, by1, bx2, by2 = proposal["box"]
        padding = max(12, (bx2-bx1)*.25, (by2-by1)*.25)
        x1, y1 = max(0, int(bx1-padding)), max(0, int(by1-padding))
        x2, y2 = min(width, int(bx2+padding)), min(height, int(by2+padding))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        allowed = LIGHT_TYPES if proposal["family"] == "light" else SIGN_TYPES
        ids = [i for i, name in self.recognizer.names.items() if name in allowed]
        # Shape proposals have no learned objectness score; require stronger
        # second-stage evidence to reject colored background clutter.
        recognition_threshold = max(.65, threshold) if proposal["source"] == "red-shape" else threshold
        result = self.recognizer.predict(crop, imgsz=640, conf=recognition_threshold, classes=ids, verbose=False)[0]
        matches = []
        for box in result.boxes:
            coords = box.xyxy[0].tolist()
            absolute = [coords[0]+x1, coords[1]+y1, coords[2]+x1, coords[3]+y1]
            cx, cy = (absolute[0]+absolute[2])/2, (absolute[1]+absolute[3])/2
            if bx1 <= cx <= bx2 and by1 <= cy <= by2 and overlap(absolute, proposal["box"]) >= .1:
                matches.append((float(box.conf.item()), result.names[int(box.cls.item())]))
        if matches:
            score, name = max(matches)
            return name, min(score, proposal["confidence"]) if proposal["confidence"] is not None else score, "crop-model"
        if self.color_fallback and proposal["family"] == "light":
            tight = frame[max(0,int(by1)):min(height,int(by2)), max(0,int(bx1)):min(width,int(bx2))]
            name = light_color(tight)
            if name:
                # Confidence belongs to the localized object, not a calibrated color score.
                return name, proposal["confidence"], "HSV-color"
        if proposal["source"] == "pretrained":
            if proposal["family"] == "sign":
                return "stop sign", proposal["confidence"], "pretrained"
        return None

    def detect_and_annotate(self, frame, target_classes=_UNSET, confidence=None, image_size=None):
        threshold = config.CONFIDENCE_THRESHOLD if confidence is None else confidence
        selected = set(self.names) if target_classes is _UNSET else set(target_classes or []) & set(self.names)
        if not selected:
            return frame, []
        result = self.localizer.predict(frame, imgsz=image_size or 1280, conf=threshold,
                                        classes=self.localizer_ids, iou=.45, verbose=False)[0]
        proposals = [{"box": b.xyxy[0].tolist(), "confidence": float(b.conf.item()),
                      "family": {"traffic light": "light", "person": "person"}.get(result.names[int(b.cls.item())], "sign"),
                      "source": "pretrained"} for b in result.boxes]
        if selected & SIGN_TYPES:
            for candidate in red_sign_candidates(frame):
                if all(overlap(candidate["box"], p["box"]) < .35 for p in proposals if p["family"] == "sign"):
                    proposals.append(candidate)
        detections = []
        people = [p for p in proposals if p["family"] == "person"]
        crops = [p for p in proposals if p["family"] != "person"]
        for proposal in people + crops[:24]:
            if proposal["family"] == "person":
                recognized = ("person", proposal["confidence"], "pretrained")
            else:
                if not selected & (LIGHT_TYPES if proposal["family"] == "light" else SIGN_TYPES):
                    continue
                recognized = self._recognize(frame, proposal, threshold)
            if recognized is None:
                continue
            name, score, method = recognized
            if name not in selected:
                continue
            box = proposal["box"]
            if name == "stop sign" and not stop_has_red_face(frame, box):
                continue
            if any((name == "person") == (d["class_name"] == "person") and overlap(box, d["bbox"]) > .5 for d in detections):
                continue
            detections.append({"class_name": name, "confidence": score,
                               "direction": SignDetector._direction(box[0], box[2], frame.shape[1]),
                               "bbox": box, "recognition_method": method})
        # Only annotate after recognition; drawings must never enter another crop.
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection["bbox"])
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,220,0), 2)
            label = f"{detection['class_name']} {detection['confidence']:.2f}"
            if detection["recognition_method"] == "HSV-color":
                label = f"{detection['class_name']} (color analysis)"
            cv2.putText(frame, label, (x1, max(18,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, .55, (0,220,0), 2, cv2.LINE_AA)
        return frame, detections
