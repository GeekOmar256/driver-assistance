# VisionGuard — Driver Assistance

A real-time assistive driving web app that detects traffic signs in an uploaded
video or live webcam feed using YOLO, drawing bounding boxes around detections.

This is the implementation prototype for the VisionGuard FYP (see
`FYP1ProjectPlan_DrivingAssistance_Complete.docx` for the full project plan).

## Status

Uses the existing pretrained YOLOv8n checkpoint for stop signs and generic
traffic lights. The detection scope excludes vehicles and unrelated objects.
The pretrained checkpoint does not distinguish light colours or additional sign types.

`training/VisionGuard_9_class_training.ipynb` prepares fine-tuning in Google
Colab for stop, yield, no entry, speed limits 40/60/80 km/h, and red/yellow/green
traffic lights. It needs a compact, annotated nine-class YOLO detection export
with separate training, validation and test splits. A local 50-epoch fine-tuning
run has now been started on Roboflow version 2, using the nine-class subset at
`datasets/roboflow-v2/selected-nine-clean/`. Its logs and checkpoints are in
`training_runs/local_roboflow_v2/`; training completion and final test metrics
must be checked before treating the checkpoint as the finished model.
The large original dataset downloads remain stopped.

Install the trained checkpoint at `webapp/models/road_signs_lights.pt` and
restart the app. It is selected automatically when present. Alternatively,
set `VISIONGUARD_MODEL_PATH` to a checkpoint path. Use the class names produced
by the notebook so the app's scope and alert labels match.

## Hosted Roboflow image inference

The separate `inference/roboflow_detect.py` adapter uses
`road-and-traffic-signs-recognition/2` through Roboflow's Serverless API.
It leaves the local Flask detector separate. It reads `ROBOFLOW_API_KEY` from
the process environment and authenticates using only a Bearer header.
`inference/.env.example` documents the variable; the script does not load
`.env` files automatically.

Run from the project root in PowerShell:

```powershell
& ./.venv-roboflow/Scripts/python.exe -m pip install -r inference/requirements.txt
$roboflowSecret = Read-Host 'Roboflow API key' -AsSecureString
$env:ROBOFLOW_API_KEY = [System.Net.NetworkCredential]::new('', $roboflowSecret).Password
& ./.venv-roboflow/Scripts/python.exe inference/roboflow_detect.py inference/examples/traffic_scene.jpg
Remove-Item Env:ROBOFLOW_API_KEY
```

Outputs are saved in the ignored `inference/results/` directory: raw predictions,
selected-class detections in the project schema, and an annotated PNG.
The bundled public sample is from the noncommercial
[Bosch traffic-light dataset](https://zenodo.org/records/12706046).
`inference/examples/dataset_image.jpg` is a CC BY 4.0 test image from
[the selected Roboflow dataset](https://universe.roboflow.com/ibtissammouaad/road-and-traffic-signs-recognition/dataset/2).
This uses an existing hosted model; it does not train a new model or download a dataset.

## Local web requirements

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
