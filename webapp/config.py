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
DEFAULT_MODEL = "hybrid"
AUTO_IMAGE_SIZE = int(os.environ.get("VISIONGUARD_AUTO_IMGSZ", "640"))
CAMERA_SOURCE = os.environ.get("VISIONGUARD_CAMERA", "0")

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

PROJECT_DIR = os.path.dirname(BASE_DIR)
MODEL_OPTIONS = {
    "pretrained": {"label": "Pretrained YOLOv8n — road objects", "path": os.path.join(BASE_DIR, "yolov8n.pt"), "scope": ROAD_CLASSES},
    "trained": {"label": "Trained VisionGuard — signs and lights", "path": CUSTOM_MODEL_PATH if os.path.isfile(CUSTOM_MODEL_PATH) else os.path.join(PROJECT_DIR, "training_runs", "local_roboflow_v2", "road_signs_lights", "weights", "best.pt"), "scope": None},
}

for _model_id, _label, _color in [
    ("hybrid", "Two-stage — pretrained + cropped trained model", False),
    ("hybrid_color", "Two-stage — cropped model + light-color fallback", True),
]:
    MODEL_OPTIONS[_model_id] = {
        "label": _label, "path": MODEL_OPTIONS["trained"]["path"],
        "pretrained_path": MODEL_OPTIONS["pretrained"]["path"],
        "pipeline": "hybrid", "color_fallback": _color,
    }

# Icon + friendly labels for the driver-alerts panel. A class without an
# entry here still works, it just gets a generic icon and a title-cased name.
CLASS_DISPLAY = {
    "person": {"icon": "\U0001F6B6", "label": "Person", "plural": "People", "kind": "signal"},
    "bicycle": {"icon": "\U0001F6B2", "label": "Bicycle", "plural": "Bicycles", "kind": "vehicle"},
    "car": {"icon": "\U0001F697", "label": "Car", "plural": "Cars", "kind": "vehicle"},
    "motorcycle": {"icon": "\U0001F3CD\uFE0F", "label": "Motorcycle", "plural": "Motorcycles", "kind": "vehicle"},
    "bus": {"icon": "\U0001F68C", "label": "Bus", "plural": "Buses", "kind": "vehicle"},
    "truck": {"icon": "\U0001F69A", "label": "Truck", "plural": "Trucks", "kind": "vehicle"},
    "traffic light": {"icon": "\U0001F6A6", "label": "Traffic light", "plural": "Traffic lights", "kind": "signal"},
    "stop sign": {"icon": "\U0001F6D1", "label": "Stop sign", "plural": "Stop signs", "kind": "signal"},
}

for _name, _icon in {
    "yield sign": "⚠️", "no entry": "⛔", "speed limit 40": "④⓪",
    "speed limit 60": "⑥⓪", "speed limit 80": "⑧⓪",
    "red traffic light": "🔴", "yellow traffic light": "🟡", "green traffic light": "🟢",
}.items():
    CLASS_DISPLAY[_name] = {"icon": _icon, "label": _name.capitalize(),
                            "plural": _name.capitalize() + "s", "kind": "signal"}
