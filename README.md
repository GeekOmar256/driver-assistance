# VisionGuard — Driver Assistance

A real-time assistive driving web app that detects traffic signs in an uploaded
video or live webcam feed using YOLO, drawing bounding boxes around detections.

This is the implementation prototype for the VisionGuard FYP (see
`FYP1ProjectPlan_DrivingAssistance_Complete.docx` for the full project plan).

## Status

Uses a pretrained YOLOv8 model (`yolov8n.pt`, auto-downloaded on first run) as
a placeholder, since no custom traffic-sign model has been trained yet. It
currently boxes any detected object to prove out the pipeline end-to-end.
Swap in a custom model trained on GTSRB/LISA by editing `webapp/config.py`
(`MODEL_PATH` and `TARGET_CLASS_NAMES`) — no other code changes needed.

## Requirements

- Python 3.12 (also tested on 3.10/3.11)
- A webcam (optional, for live detection)

## Setup

```bash
cd webapp
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000 in your browser.

## Project layout

```
webapp/
├── app.py          # Flask routes: upload, video stream, webcam stream
├── detector.py      # YOLO wrapper, draws bounding boxes
├── config.py         # Model path, confidence threshold, target classes
├── templates/        # HTML UI
├── static/           # CSS
└── requirements.txt
```
