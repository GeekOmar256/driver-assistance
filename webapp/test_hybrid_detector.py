import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from hybrid_detector import HybridDetector, LIGHT_TYPES, SIGN_TYPES, light_color, red_sign_candidates, stop_has_red_face


def box(coords, cls=9, confidence=.85):
    return SimpleNamespace(xyxy=np.array([coords], dtype=float),
                           cls=np.array([cls]), conf=np.array([confidence]))


class HybridTests(unittest.TestCase):
    def light(self, color):
        image = np.zeros((90, 30, 3), dtype=np.uint8)
        cv2.circle(image, (15, 45), 8, color, -1)
        return image

    def test_lamp_colors_and_unknown(self):
        for color, bgr in [("red", (0, 0, 255)), ("yellow", (0, 255, 255)), ("green", (0, 255, 0))]:
            self.assertEqual(light_color(self.light(bgr)), color + " traffic light")
        self.assertIsNone(light_color(np.zeros((90, 30, 3), dtype=np.uint8)))
        self.assertIsNone(light_color(np.full((90, 30, 3), 255, dtype=np.uint8)))
        self.assertIsNone(light_color(np.zeros((5, 5, 3), dtype=np.uint8)))

    def test_ambiguous_lamps_remain_unknown(self):
        image = self.light((0, 0, 255))
        cv2.circle(image, (15, 15), 8, (0, 255, 0), -1)
        self.assertIsNone(light_color(image))

    def test_white_speed_sign_is_not_accepted_as_red_stop_face(self):
        speed = np.full((100,100,3),255,dtype=np.uint8)
        cv2.circle(speed,(50,50),45,(0,0,255),6)
        self.assertFalse(stop_has_red_face(speed,[0,0,100,100]))
        stop = np.zeros((100,100,3),dtype=np.uint8)
        stop[:]=(0,0,200)
        self.assertTrue(stop_has_red_face(stop,[0,0,100,100]))

    def detector(self, color=False):
        model = HybridDetector.__new__(HybridDetector)
        model.color_fallback = color
        model.names = sorted(SIGN_TYPES | LIGHT_TYPES)
        model.localizer_ids = [9, 11]
        model.localizer = Mock()
        model.localizer.predict.return_value = [SimpleNamespace(names={9: "traffic light"}, boxes=[box([30, 20, 60, 110])])]
        model.recognizer = Mock()
        model.recognizer.names = {0: "red traffic light", 1: "stop sign"}
        model.recognizer.predict.return_value = [SimpleNamespace(names=model.recognizer.names, boxes=[])]
        return model

    @patch("hybrid_detector.red_sign_candidates", return_value=[])
    def test_unrecognized_light_is_omitted_without_color(self, _):
        model = self.detector()
        frame = np.zeros((150, 150, 3), dtype=np.uint8)
        _, detections = model.detect_and_annotate(frame)
        self.assertEqual(detections, [])
        self.assertEqual(model.recognizer.predict.call_args.kwargs["classes"], [0])

    @patch("hybrid_detector.red_sign_candidates", return_value=[])
    def test_color_fallback_and_final_class_filter(self, _):
        model = self.detector(True)
        frame = np.zeros((150, 150, 3), dtype=np.uint8)
        frame[20:110, 30:60] = self.light((0, 0, 255))
        _, detections = model.detect_and_annotate(frame.copy(), {"red traffic light"})
        self.assertEqual(detections[0]["class_name"], "red traffic light")
        self.assertEqual(detections[0]["recognition_method"], "HSV-color")
        _, detections = model.detect_and_annotate(frame, {"green traffic light"})
        self.assertEqual(detections, [])

    def test_empty_selection_skips_both_models(self):
        model = self.detector()
        _, detections = model.detect_and_annotate(np.zeros((150,150,3), dtype=np.uint8), set())
        self.assertEqual(detections, [])
        model.localizer.predict.assert_not_called()
        model.recognizer.predict.assert_not_called()

    def test_red_circle_proposed_but_not_emitted_without_recognition(self):
        frame = np.zeros((200,200,3), dtype=np.uint8)
        cv2.circle(frame,(100,100),35,(0,0,255),6)
        self.assertTrue(red_sign_candidates(frame))
        model = self.detector()
        model.localizer.predict.return_value = [SimpleNamespace(names={}, boxes=[])]
        _, detections = model.detect_and_annotate(frame)
        self.assertEqual(detections, [])
        self.assertEqual(model.recognizer.predict.call_args.kwargs["conf"], .65)

    @patch("hybrid_detector.red_sign_candidates", return_value=[])
    def test_crop_coordinates_match_original_proposal_and_preserve_confidence(self, _):
        model = self.detector()
        model.recognizer.predict.return_value = [SimpleNamespace(names={0:"red traffic light"},
            boxes=[box([23,20,53,110],0,.92)])]
        _, detections = model.detect_and_annotate(np.zeros((150,150,3),dtype=np.uint8))
        self.assertEqual(detections[0]["class_name"], "red traffic light")
        self.assertEqual(detections[0]["recognition_method"], "crop-model")
        self.assertAlmostEqual(detections[0]["confidence"], .85)
        self.assertEqual(detections[0]["bbox"], [30,20,60,110])

    def test_recognition_from_nearby_object_outside_proposal_is_rejected(self):
        model = self.detector()
        model.recognizer.predict.return_value = [SimpleNamespace(names={0:"red traffic light"}, boxes=[box([0,0,5,5],0)])]
        _, detections = model.detect_and_annotate(np.zeros((150,150,3), dtype=np.uint8))
        self.assertEqual(detections, [])


if __name__ == "__main__":
    unittest.main()
