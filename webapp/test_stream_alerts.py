"""Regression tests for detections appearing between browser polls."""
import unittest
from unittest.mock import Mock

import numpy as np

import app
from alerts import RecentDetections


class StreamAlertTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.previous_recent = app._recent_detections
        self.previous_state = (app._stream_active, app._stream_version, app._stream_client_id)
        app._recent_detections = RecentDetections(clock=lambda: self.now)
        self.client = app.app.test_client()
        self.frame = np.zeros((32, 32, 3), dtype=np.uint8)
        self.sign = {"class_name": "stop sign", "direction": "ahead", "confidence": .8}

    def tearDown(self):
        app._recent_detections = self.previous_recent
        app._stream_active, app._stream_version, app._stream_client_id = self.previous_state

    def frames(self, detections, stream_id):
        cap = Mock()
        cap.read.side_effect = [(True, self.frame) for _ in detections] + [(False, None)]
        detector = Mock()
        detector.detect_and_annotate.side_effect = [(self.frame, items) for items in detections]
        generator = app.generate_frames(cap, {"stop sign"}, detector, .4, 640, stream_id)
        return generator, cap

    def test_brief_detection_reaches_api_after_empty_frames_and_end(self):
        frames, cap = self.frames([[self.sign], [], []], "brief")
        list(frames)
        self.now = .7  # First browser poll happens after the detection vanished.
        alerts = self.client.get("/api/latest_detections?stream_id=brief").get_json()
        self.assertEqual(alerts[0]["class_name"], "stop sign")
        self.assertTrue(alerts[0]["recent"])
        self.assertEqual(self.client.get("/api/stream_status?stream_id=brief").get_json(),
                         {"active": False, "started": True})
        cap.release.assert_called_once()
        self.now = 3.1
        self.assertEqual(self.client.get("/api/latest_detections?stream_id=brief").get_json(), [])

    def test_new_stream_hides_previous_alerts_and_ignores_old_cleanup(self):
        old, _ = self.frames([[self.sign]], "old")
        next(old)
        new, _ = self.frames([[]], "new")
        next(new)
        old.close()
        self.assertEqual(self.client.get("/api/latest_detections?stream_id=new").get_json(), [])
        self.assertTrue(self.client.get("/api/stream_status?stream_id=new").get_json()["active"])
        self.assertFalse(self.client.get("/api/stream_status?stream_id=old").get_json()["started"])
        new.close()

    def test_not_started_is_distinct_from_ended(self):
        status = self.client.get("/api/stream_status?stream_id=not-started").get_json()
        self.assertFalse(status["started"])
        self.assertEqual(self.client.get("/api/latest_detections?stream_id=not-started").get_json(), [])


if __name__ == "__main__":
    unittest.main()
