import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}

# Use the fine-tuned sign/light checkpoint when installed, with the stock
# pretrained COCO model (cars, buses, traffic lights, etc.) available until
# then.
CUSTOM_MODEL_PATH = os.path.join(BASE_DIR, "models", "road_signs_lights.pt")
MODEL_PATH = os.environ.get("VISIONGUARD_MODEL_PATH") or (
    CUSTOM_MODEL_PATH if os.path.isfile(CUSTOM_MODEL_PATH)
    else os.path.join(BASE_DIR, "yolov8n.pt")
)
CONFIDENCE_THRESHOLD = 0.4

# Pretrained-COCO-compatible road classes: the set the stock yolov8n.pt
# checkpoint can actually detect, used until the fine-tuned nine-class
# checkpoint (see training/VisionGuard_9_class_training.ipynb) is installed
# at CUSTOM_MODEL_PATH. The detector intersects this with the loaded
# checkpoint's own class names, so switching MODEL_PATH later to that
# checkpoint should go with switching this list to its nine sign/light classes.
ROAD_CLASSES = [
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic light",
    "stop sign",
]

TARGET_CLASS_NAMES = ROAD_CLASSES

# Icon + friendly labels for the driver-alerts panel. A class without an
# entry here still works, it just gets a generic icon and a title-cased name.
CLASS_DISPLAY = {
    "bicycle": {"icon": "\U0001F6B2", "label": "Bicycle", "plural": "Bicycles", "kind": "vehicle"},
    "car": {"icon": "\U0001F697", "label": "Car", "plural": "Cars", "kind": "vehicle"},
    "motorcycle": {"icon": "\U0001F3CD\uFE0F", "label": "Motorcycle", "plural": "Motorcycles", "kind": "vehicle"},
    "bus": {"icon": "\U0001F68C", "label": "Bus", "plural": "Buses", "kind": "vehicle"},
    "truck": {"icon": "\U0001F69A", "label": "Truck", "plural": "Trucks", "kind": "vehicle"},
    "traffic light": {"icon": "\U0001F6A6", "label": "Traffic light", "plural": "Traffic lights", "kind": "signal"},
    "stop sign": {"icon": "\U0001F6D1", "label": "Stop sign", "plural": "Stop signs", "kind": "signal"},
}
