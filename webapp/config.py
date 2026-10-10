import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}

# Use the fine-tuned checkpoint when installed, with the existing pretrained
# model available for stop signs only until then (it has no yield/no-entry/
# speed-limit classes, and no notion of traffic-light colour).
CUSTOM_MODEL_PATH = os.path.join(BASE_DIR, "models", "road_signs_lights.pt")
MODEL_PATH = os.environ.get("VISIONGUARD_MODEL_PATH") or (
    CUSTOM_MODEL_PATH if os.path.isfile(CUSTOM_MODEL_PATH)
    else os.path.join(BASE_DIR, "yolov8n.pt")
)
CONFIDENCE_THRESHOLD = 0.4

# Exactly the nine classes defined in training/VisionGuard_9_class_training.ipynb
# (CLASSES there). The detector intersects this with the loaded checkpoint's
# own class names, so the pretrained fallback only ever matches "stop sign"
# until the fine-tuned checkpoint is installed.
ROAD_CLASSES = [
    "stop sign",
    "yield sign",
    "no entry",
    "speed limit 40",
    "speed limit 60",
    "speed limit 80",
    "red traffic light",
    "yellow traffic light",
    "green traffic light",
]

TARGET_CLASS_NAMES = ROAD_CLASSES

# Icon + friendly labels for the driver-alerts panel, one per dataset class.
# A class without an entry here still works, it just gets a generic icon and
# a title-cased name.
CLASS_DISPLAY = {
    "stop sign": {"icon": "\U0001F6D1", "label": "Stop sign", "plural": "Stop signs", "kind": "signal"},
    "yield sign": {"icon": "\u26A0\uFE0F", "label": "Yield sign", "plural": "Yield signs", "kind": "signal"},
    "no entry": {"icon": "\u26D4", "label": "No entry", "plural": "No-entry signs", "kind": "signal"},
    "speed limit 40": {"icon": "\u26A0\uFE0F", "label": "Speed limit 40 km/h", "plural": "Speed limit 40 km/h signs", "kind": "signal"},
    "speed limit 60": {"icon": "\u26A0\uFE0F", "label": "Speed limit 60 km/h", "plural": "Speed limit 60 km/h signs", "kind": "signal"},
    "speed limit 80": {"icon": "\u26A0\uFE0F", "label": "Speed limit 80 km/h", "plural": "Speed limit 80 km/h signs", "kind": "signal"},
    "red traffic light": {"icon": "\U0001F534", "label": "Red traffic light", "plural": "Red traffic lights", "kind": "signal"},
    "yellow traffic light": {"icon": "\U0001F7E1", "label": "Yellow traffic light", "plural": "Yellow traffic lights", "kind": "signal"},
    "green traffic light": {"icon": "\U0001F7E2", "label": "Green traffic light", "plural": "Green traffic lights", "kind": "signal"},
}
