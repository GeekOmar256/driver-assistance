import unittest

from roboflow_detect import normalize_predictions


class PredictionConversionTests(unittest.TestCase):
    def test_converts_centre_boxes_and_filters_unselected_classes(self):
        result = {"image": {"width": 900}, "predictions": [
            {"class": "Traffic signal_red", "confidence": 0.91, "x": 100, "y": 200,
             "width": 20, "height": 60},
            {"class": "speed limit 60km/h", "confidence": 0.85, "x": 700, "y": 100,
             "width": 50, "height": 50},
            {"class": "Pedestrians", "confidence": 0.8, "x": 450, "y": 200,
             "width": 50, "height": 80},
        ]}
        detections = normalize_predictions(result)
        self.assertEqual(len(detections), 2)
        self.assertEqual(detections[0]["class_name"], "red traffic light")
        self.assertEqual(detections[0]["bbox"], [90, 170, 110, 230])
        self.assertEqual(detections[0]["direction"], "left")
        self.assertEqual(detections[1]["class_name"], "speed limit 60")
        self.assertEqual(detections[1]["direction"], "right")


if __name__ == "__main__":
    unittest.main()
