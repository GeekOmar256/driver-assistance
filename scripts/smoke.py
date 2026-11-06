"""Load the bundled weights, run CPU inference and check the Flask routes."""
import sys
from pathlib import Path

import numpy as np
from ultralytics import YOLO

webapp = Path(__file__).resolve().parents[1] / "webapp"
sys.path.insert(0, str(webapp))
from app import app

for file, expected in [("yolov8n.pt", 80), ("models/road_signs_lights.pt", 9)]:
    model = YOLO(str(webapp / file))
    assert len(model.names) == expected, model.names
    model.predict(np.zeros((320, 320, 3), dtype=np.uint8),
                  imgsz=640, device="cpu", verbose=False)
    print(f"CPU inference OK: {file} ({expected} classes)")
with app.test_client() as client:
    assert client.get("/").status_code == 200
    modes = client.get("/api/models").get_json()
    assert len(modes) == 4 and all(mode["available"] for mode in modes), modes
    assert client.get("/static/audio/sound_enabled.mp3").status_code == 200
    print("Flask, all four model options, and audio asset OK.")
