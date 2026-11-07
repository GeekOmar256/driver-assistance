import queue
import time
import unittest
from unittest.mock import patch
import numpy as np
import desktop

class FakeDetector:
    def class_names(self):
        return ["person", "stop sign"]
    def detect_and_annotate(self, frame, **settings):
        items = [{"class_name":"person","direction":"ahead","confidence":.9}] if "person" in settings["target_classes"] else []
        return frame, items

class Capture:
    def __init__(self):
        self.released = False
    def read(self):
        return True, np.zeros((64,64,3),dtype=np.uint8)
    def get(self, prop):
        return 25
    def release(self):
        self.released = True

class DesktopWorkerTests(unittest.TestCase):
    def wait_event(self, worker, kind, value=None):
        deadline = time.monotonic()+3
        while time.monotonic()<deadline:
            try:
                event,data = worker.events.get(timeout=.1)
            except queue.Empty:
                continue
            if event==kind and (value is None or data==value):
                return data
        self.fail(f"Missing {kind}: {value}")

    def test_video_person_alert_filter_stop_and_release(self):
        capture = Capture()
        with patch.object(desktop,"make_detector",return_value=FakeDetector()), patch.object(desktop,"open_source",return_value=capture):
            worker = desktop.DetectionWorker()
            worker.start()
            try:
                worker.commands.put(("model","hybrid"))
                self.assertIn("person", self.wait_event(worker,"classes"))
                worker.commands.put(("settings",(["person"],.4,640,1)))
                worker.commands.put(("start",("video","sample.mp4")))
                self.wait_event(worker,"status","Running")
                deadline = time.monotonic()+3
                alerts=[]
                while time.monotonic()<deadline and not alerts:
                    frame, alerts = worker.frames.get(timeout=1)
                self.assertEqual(alerts[0]["class_name"],"person")
                self.assertTrue(alerts[0]["audio_url"].endswith("person_ahead.wav"))
                worker.commands.put(("stop",None))
                self.wait_event(worker,"status","Stopped")
                self.assertTrue(capture.released)
                frame, alerts = worker.frames.get(timeout=1)
                self.assertIsNone(frame)
                self.assertEqual(alerts,[])
                worker.commands.put(("settings",([],.4,640,1)))
                worker.commands.put(("start",("video","sample.mp4")))
                self.wait_event(worker,"status","Running")
                frame,alerts=worker.frames.get(timeout=1)
                self.assertEqual(alerts,[])
            finally:
                worker.quit.set()
                worker.join(timeout=3)
                self.assertFalse(worker.is_alive())

    def test_model_failure_is_reported(self):
        with patch.object(desktop,"make_detector",side_effect=FileNotFoundError("missing weights")):
            worker=desktop.DetectionWorker()
            worker.start()
            try:
                worker.commands.put(("model","hybrid"))
                self.assertIn("missing weights",self.wait_event(worker,"error"))
            finally:
                worker.quit.set()
                worker.join(timeout=3)
