import unittest
import json
import config
from alerts import RecentDetections, build_alerts
from audio_assets import AUDIO_DIR, PHRASES, filename


class AlertTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.recent = RecentDetections(clock=lambda: self.now)
        self.stop = {"class_name": "stop sign", "direction": "ahead"}

    def test_single_frame_detection_survives_empty_frames_until_poll(self):
        self.recent.update([self.stop])
        for _ in range(20):
            self.now += .03
            self.recent.update([])
        alerts = build_alerts(self.recent.snapshot())
        self.assertEqual(alerts[0]["class_name"], "stop sign")
        self.assertTrue(alerts[0]["recent"])
        self.now = 3.0
        self.assertEqual(self.recent.snapshot(), [])

    def test_repeated_frames_do_not_inflate_counts(self):
        for _ in range(10):
            self.recent.update([self.stop, self.stop])
        self.assertEqual(build_alerts(self.recent.snapshot())[0]["text"], "2 Stop signs ahead")

    def test_new_stream_clears_recent_detections(self):
        self.recent.update([self.stop])
        self.recent.clear()
        self.assertEqual(self.recent.snapshot(), [])

    def test_last_frame_remains_available_after_end(self):
        self.recent.update([self.stop])
        self.recent.finish()
        self.assertTrue(build_alerts(self.recent.snapshot())[0]["recent"])

    def test_changed_light_replaces_held_color(self):
        self.recent.update([{"class_name": "green traffic light", "direction": "ahead"}])
        self.recent.update([{"class_name": "red traffic light", "direction": "ahead"}])
        self.assertEqual([a["class_name"] for a in self.recent.snapshot()], ["red traffic light"])

    def test_every_supported_class_has_saved_audio(self):
        for name in set(config.ROAD_CLASSES) | set(PHRASES):
            for direction in ["left", "right", "ahead"]:
                with self.subTest(name=name, direction=direction):
                    alerts = build_alerts([{"class_name": name, "direction": direction}])
                    self.assertEqual(alerts[0]["audio_url"], "/static/audio/" + filename(name, direction))
                    self.assertGreater((AUDIO_DIR / filename(name, direction)).stat().st_size, 100)

    def test_direction_is_spoken_before_additional_instruction(self):
        manifest = json.loads((AUDIO_DIR / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["traffic light:left"]["text"],
                         "Traffic light detected on your left. Check the signal.")
        self.assertEqual(manifest["stop sign:ahead"]["text"], "Stop sign detected ahead.")

    def test_left_and_right_have_separate_tracks_for_same_class(self):
        alerts = build_alerts([dict(self.stop, direction="left"), dict(self.stop, direction="right")])
        self.assertEqual(len(alerts), 2)
        self.assertNotEqual(alerts[0]["audio_url"], alerts[1]["audio_url"])

    def test_grouping_and_critical_priority(self):
        alerts = build_alerts([
            {"class_name": "car", "direction": "left"},
            {"class_name": "car", "direction": "left"},
            {"class_name": "red traffic light", "direction": "ahead"},
        ])
        self.assertEqual(len(alerts), 2)
        self.assertEqual(alerts[0]["class_name"], "red traffic light")
        self.assertEqual(alerts[0]["kind"], "signal")
        self.assertEqual(alerts[1]["text"], "2 Cars on your left")

    def test_unknown_class_does_not_request_missing_audio(self):
        self.assertIsNone(build_alerts([{"class_name": "unknown", "direction": "ahead"}])[0]["audio_url"])


if __name__ == "__main__":
    unittest.main()
