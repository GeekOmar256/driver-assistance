import os
import uuid

import cv2
from flask import Flask, Response, abort, redirect, render_template, request, url_for

import config
from detector import SignDetector

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = config.UPLOAD_DIR
os.makedirs(config.UPLOAD_DIR, exist_ok=True)

detector = SignDetector()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_VIDEO_EXTENSIONS


def generate_frames(capture):
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            frame = detector.annotate(frame)
            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
    finally:
        capture.release()


@app.route("/")
def index():
    return render_template("index.html", uploaded_filename=request.args.get("filename"))


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
    capture = cv2.VideoCapture(filepath)
    return Response(generate_frames(capture), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/webcam_feed")
def webcam_feed():
    capture = cv2.VideoCapture(0)
    return Response(generate_frames(capture), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
