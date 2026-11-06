import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import app
from alerts import build_alerts
import test_hybrid_detector as fixtures


class PersonReleaseTests(unittest.TestCase):
    def test_person_is_selectable_and_bypasses_crop_in_both_modes(self):
        for mode in (False, True):
            model = fixtures.HybridTests().detector(color=mode)
            model.names.append("person")
            model.localizer.predict.return_value = [
                SimpleNamespace(names={0: "person"}, boxes=[fixtures.box([55, 10, 95, 140], cls=0)])
            ]
            frame = np.zeros((150, 150, 3), dtype=np.uint8)
            _, detections = model.detect_and_annotate(frame.copy(), {"person"})
            self.assertEqual(detections[0]["recognition_method"], "pretrained")
            self.assertEqual(build_alerts(detections)[0]["text"], "Person ahead")
            self.assertTrue(build_alerts(detections)[0]["audio_url"].endswith("person_ahead.wav"))
            _, excluded = model.detect_and_annotate(frame, {"red traffic light"})
            self.assertEqual(excluded, [])
            model.recognizer.predict.assert_not_called()

    def test_default_selection_in_browser_and_api(self):
        with app.app.test_client() as client, patch.object(app, "HybridDetector") as factory:
            factory.return_value.class_names.return_value = ["person"]
            self.assertIn(b'let selectedModel = "hybrid";', client.get("/").data)
            self.assertEqual(client.get("/api/classes").json, ["person"])
            factory.assert_called_once()
            for direction in ('ahead', 'left', 'right'):
                with client.get(f'/static/audio/person_{direction}.wav') as response:
                    self.assertEqual(response.status_code, 200)
