import os
import threading
import time
import uuid

import cv2
from flask import Flask, Response, abort, jsonify, redirect, render_template, request, url_for

import config
from alerts import build_alerts
from detector import SignDetector, _UNSET

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = config.UPLOAD_DIR
os.makedirs(config.UPLOAD_DIR, exist_ok=True)

detector = SignDetector()

# Shared playback/detection state for the single active stream. This app is a
# one-viewer local tool (see README), so a couple of module-level globals are
# enough; a multi-user deployment would need this keyed per session instead.
_state_lock = threading.Lock()
_latest_detections = []
_stream_active = False
playback_paused = threading.Event()


def _set_latest_detections(detections):
    global _latest_detections
    with _state_lock:
        _latest_detections = detections


def _clear_latest_detections():
    _set_latest_detections([])


def _set_stream_active(active):
    global _stream_active
    with _state_lock:
        _stream_active = active


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_VIDEO_EXTENSIONS


def parse_requested_classes():
    """Read the ?classes=a,b,c query param. Absent -> detector default (the
    configured road-class scope). Present but empty -> show nothing."""
    if "classes" not in request.args:
        return _UNSET
    raw = request.args.get("classes", "")
    return {c.strip() for c in raw.split(",") if c.strip()}


def generate_frames(capture, target_classes):
    _set_stream_active(True)
    try:
        while True:
            if playback_paused.is_set():
                time.sleep(0.1)
                continue
            success, frame = capture.read()
            if not success:
                break
            frame, detections = detector.detect_and_annotate(frame, target_classes=target_classes)
            _set_latest_detections(detections)
            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
    finally:
        capture.release()
        _clear_latest_detections()
        _set_stream_active(False)


@app.route("/")
def index():
    return render_template("index.html", uploaded_filename=request.args.get("filename"))


@app.route("/api/classes")
def api_classes():
    return jsonify(detector.class_names())


@app.route("/api/latest_detections")
def api_latest_detections():
    with _state_lock:
        detections = list(_latest_detections)
    return jsonify(build_alerts(detections))


@app.route("/api/stream_status")
def api_stream_status():
    """Whether a frame generator is currently running. The frontend polls
    this to notice when an uploaded video reaches its end on its own (as
    opposed to the user clicking Stop), so it can offer Replay."""
    with _state_lock:
        active = _stream_active
    return jsonify({"active": active})


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
    playback_paused.clear()
    capture = cv2.VideoCapture(filepath)
    return Response(generate_frames(capture, classes), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/webcam_feed")
def webcam_feed():
    classes = parse_requested_classes()
    playback_paused.clear()
    capture = cv2.VideoCapture(0)
    return Response(generate_frames(capture, classes), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
