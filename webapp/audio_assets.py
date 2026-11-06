"""Playback filenames and descriptions for the bundled alert recordings."""
from pathlib import Path

PHRASES = {
    "person": "Person detected.",
    "sound enabled": "Sound alerts enabled.",
    "bicycle": "Bicycle detected.",
    "car": "Car detected.",
    "motorcycle": "Motorcycle detected.",
    "bus": "Bus detected.",
    "truck": "Truck detected.",
    "traffic light": "Traffic light detected. Check the signal.",
    "stop sign": "Stop sign detected.",
    "yield sign": "Yield sign detected.",
    "no entry": "No entry sign detected.",
    "speed limit 40": "Speed limit forty kilometers per hour.",
    "speed limit 60": "Speed limit sixty kilometers per hour.",
    "speed limit 80": "Speed limit eighty kilometers per hour.",
    "red traffic light": "Red traffic light detected.",
    "yellow traffic light": "Yellow traffic light detected.",
    "green traffic light": "Green traffic light detected. Check before proceeding.",
}
AUDIO_DIR = Path(__file__).parent / "static" / "audio"
DIRECTION_PHRASES = {
    "person": "Person detected.","left": "on your left", "right": "on your right", "ahead": "ahead"}


def filename(name, direction=None):
    suffix = "_" + direction if direction in DIRECTION_PHRASES and name != "sound enabled" else ""
    return name.replace(" ", "_") + suffix + (".wav" if name == "person" else ".mp3")
