import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}

# Placeholder pretrained model (auto-downloaded by ultralytics on first run).
# Swap this for your custom traffic-sign weights (e.g. trained on GTSRB) once ready.
MODEL_PATH = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.4

# Restrict drawn boxes to specific class names, e.g. ["stop sign"].
# Left as None for now so any detection is boxed, since the pretrained COCO
# model has no general "traffic sign" class. Once a custom sign-detection
# model is loaded, set this to its class names (or keep None to show all).
TARGET_CLASS_NAMES = None
