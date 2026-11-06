import os
import math
import threading
import time
import uuid

import cv2
from flask import Flask, Response, abort, jsonify, redirect, render_template, request, url_for

import config
from alerts import RecentDetections, build_alerts
from detector import SignDetector, _UNSET
from hybrid_detector import HybridDetector

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = config.UPLOAD_DIR
app.config["MAX_CONTENT_LENGTH"] = 256 * 1024 * 1024
os.makedirs(config.UPLOAD_DIR, exist_ok=True)

_detectors = {}
_model_lock = threading.Lock()


def model_available(option):
    return os.path.isfile(option["path"]) and (
        "pretrained_path" not in option or os.path.isfile(option["pretrained_path"]))


def default_model():
    if model_available(config.MODEL_OPTIONS[config.DEFAULT_MODEL]):
        return config.DEFAULT_MODEL
    return next((key for key, option in config.MODEL_OPTIONS.items()
                 if model_available(option)), config.DEFAULT_MODEL)


def requested_detector():
    model_id = request.args.get("model", default_model())
    option = config.MODEL_OPTIONS.get(model_id)
    if not option or not model_available(option):
        abort(400, "Model is unavailable")
    with _model_lock:
        if model_id not in _detectors:
            if option.get("pipeline") == "hybrid":
                _detectors[model_id] = HybridDetector(option["pretrained_path"], option["path"],
                                                     color_fallback=option["color_fallback"])
            else:
                _detectors[model_id] = SignDetector(option["path"], scope=option["scope"],
                                                   image_size=1280 if model_id == "trained" else 640)
        return _detectors[model_id]

# Shared playback/detection state for the single active stream. This app is a
# one-viewer local tool (see README), so a couple of module-level globals are
# enough; a multi-user deployment would need this keyed per session instead.
_state_lock = threading.Lock()
_recent_detections = RecentDetections()
_stream_active = False
_stream_version = 0
_stream_client_id = None
playback_paused = threading.Event()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_VIDEO_EXTENSIONS


def parse_requested_classes():
    """Read the ?classes=a,b,c query param. Absent -> detector default (the
    configured road-class scope). Present but empty -> show nothing."""
    if "classes" not in request.args:
        return _UNSET
    raw = request.args.get("classes", "")
    return {c.strip() for c in raw.split(",") if c.strip()}


def parse_requested_confidence():
    try:
        confidence = float(request.args.get("confidence", config.CONFIDENCE_THRESHOLD))
    except (ValueError, TypeError):
        abort(400, "Confidence must be a number between 0.01 and 1.")
    if not math.isfinite(confidence) or not 0.01 <= confidence <= 1:
        abort(400, "Confidence must be a number between 0.01 and 1.")
    return confidence


def parse_requested_image_size():
    raw = request.args.get("resolution", "auto")
    if raw == "auto":
        return config.AUTO_IMAGE_SIZE
    if raw not in {"640", "960", "1280"}:
        abort(400, "Unsupported inference resolution")
    return int(raw)


def generate_frames(capture, target_classes, selected_detector, confidence, image_size, stream_id=None):
    global _stream_version, _stream_active, _stream_client_id
    with _state_lock:
        _stream_version += 1
        version = _stream_version
        _stream_client_id = stream_id
        _recent_detections.clear()
        _stream_active = True
    try:
        while True:
            with _state_lock:
                if version != _stream_version:
                    break
            if playback_paused.is_set():
                time.sleep(0.1)
                continue
            success, frame = capture.read()
            if not success:
                break
            frame, detections = selected_detector.detect_and_annotate(
                frame, target_classes=target_classes, confidence=confidence, image_size=image_size)
            with _state_lock:
                if version != _stream_version:
                    break
                _recent_detections.update(detections)
            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
    finally:
        capture.release()
        with _state_lock:
            if version == _stream_version:
                _recent_detections.finish()
                _stream_active = False


@app.route("/")
def index():
    return render_template("index.html", uploaded_filename=request.args.get("filename"),
                           default_confidence=round(config.CONFIDENCE_THRESHOLD * 100),
                           default_model=default_model(), auto_image_size=config.AUTO_IMAGE_SIZE)


@app.route("/api/classes")
def api_classes():
    return jsonify(requested_detector().class_names())


@app.route("/api/models")
def api_models():
    return jsonify([{"id": key, "label": option["label"], "available": model_available(option)}
                    for key, option in config.MODEL_OPTIONS.items()])


@app.route("/api/latest_detections")
def api_latest_detections():
    with _state_lock:
        if request.args.get("stream_id") and request.args["stream_id"] != _stream_client_id:
            return jsonify([])
        detections = _recent_detections.snapshot()
    return jsonify(build_alerts(detections))


@app.route("/api/stream_status")
def api_stream_status():
    """Whether a frame generator is currently running. The frontend polls
    this to notice when an uploaded video reaches its end on its own (as
    opposed to the user clicking Stop), so it can offer Replay."""
    with _state_lock:
        active = _stream_active
        started = not request.args.get("stream_id") or request.args["stream_id"] == _stream_client_id
    return jsonify({"active": active if started else False, "started": started})


@app.route("/api/toggle_pause", methods=["POST"])
def api_toggle_pause():
    if playback_paused.is_set():
        playback_paused.clear()
    else:
        playback_paused.set()
    return jsonify({"paused": playback_paused.is_set()})


@app.route("/api/resume", methods=["POST"])
def api_resume():
    """Force-clear pause. Called when stopping a stream, so a generator that
    was mid-pause wakes up, notices the client is gone, and cleans up instead
    of sitting in its sleep loop forever."""
    playback_paused.clear()
    return jsonify({"paused": False})


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("video")
    if not file or file.filename == "":
        return redirect(url_for("index"))
    if not allowed_file(file.filename):
        abort(400, "Unsupported file type. Use mp4, avi, mov, or mkv.")

    ext = file.filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    return redirect(url_for("index", filename=filename))


@app.route("/video_feed/<filename>")
def video_feed(filename):
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if not os.path.isfile(filepath):
        abort(404)
    classes = parse_requested_classes()
    confidence = parse_requested_confidence()
    image_size = parse_requested_image_size()
    selected_detector = requested_detector()
    playback_paused.clear()
    capture = cv2.VideoCapture(filepath)
    return Response(generate_frames(capture, classes, selected_detector, confidence, image_size,
                                    request.args.get("stream_id")), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/webcam_feed")
def webcam_feed():
    classes = parse_requested_classes()
    confidence = parse_requested_confidence()
    image_size = parse_requested_image_size()
    selected_detector = requested_detector()
    playback_paused.clear()
    source = config.CAMERA_SOURCE
    if source == "picamera2":
        from pi_camera import PiCameraCapture
        try:
            capture = PiCameraCapture()
        except (ImportError, RuntimeError, OSError):
            app.logger.exception("Could not open Raspberry Pi Camera Module")
            abort(503, "Camera unavailable. Check Picamera2 installation and camera connection.")
    else:
        capture = cv2.VideoCapture(int(source) if source.isdecimal() else source)
        if not capture.isOpened():
            capture.release()
            abort(503, "Camera unavailable. Check the configured V4L2 device.")
    return Response(generate_frames(capture, classes, selected_detector, confidence, image_size,
                                    request.args.get("stream_id")), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(host="127.0.0.1", debug=False, threaded=True)
