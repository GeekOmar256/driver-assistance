"""Run the hosted road-sign model without changing the local web detector."""
import argparse
import json
import os
from pathlib import Path

MODEL_ID = "road-and-traffic-signs-recognition/2"
API_URL = "https://serverless.roboflow.com"
SELECTED_CLASSES = {
    "stop": "stop sign",
    "yield": "yield sign",
    "no entry": "no entry",
    "speed limit 40km/h": "speed limit 40",
    "speed limit 60km/h": "speed limit 60",
    "speed limit 80km/h": "speed limit 80",
    "traffic signal red": "red traffic light",
    "traffic signal yellow": "yellow traffic light",
    "traffic signal green": "green traffic light",
}


def normalize_predictions(result):
    """Translate supported model labels into the project's nine-class schema."""
    width = result.get("image", {}).get("width", 0)
    detections = []
    for prediction in result.get("predictions", []):
        name = " ".join(prediction["class"].lower().replace("km-h", "km/h").replace("_", " ").replace("-", " ").split())
        canonical = SELECTED_CLASSES.get(name)
        if canonical is None:
            continue
        x, y = float(prediction["x"]), float(prediction["y"])
        box_width, box_height = float(prediction["width"]), float(prediction["height"])
        direction = "ahead"
        if width:
            direction = "left" if x < width / 3 else "right" if x > 2 * width / 3 else "ahead"
        detections.append({"class_name": canonical, "source_class": prediction["class"],
                           "confidence": float(prediction["confidence"]), "direction": direction,
                           "bbox": [x-box_width/2, y-box_height/2, x+box_width/2, y+box_height/2]})
    return detections


def infer_image(image, confidence=0.4):
    from inference_sdk import InferenceHTTPClient, InferenceConfiguration
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError("Set the ROBOFLOW_API_KEY environment variable before running inference.")
    client = InferenceHTTPClient(api_url=API_URL, api_key=api_key).configure(
        InferenceConfiguration(api_key_transport="header", confidence_threshold=confidence)
    )
    result = client.infer(str(image), model_id=MODEL_ID)
    return {"model_id": MODEL_ID, "raw_result": result, "detections": normalize_predictions(result)}


def annotate(image, detections, output):
    from PIL import Image, ImageDraw
    with Image.open(image) as source:
        canvas = source.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for detection in detections:
        draw.rectangle(detection["bbox"], outline="#00dd66", width=3)
        x, y = detection["bbox"][:2]
        draw.text((max(0, x), max(0, y-14)),
                  f"{detection['class_name']} {detection['confidence']:.2f}", fill="#00dd66")
    canvas.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Local image to send to Roboflow")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--confidence", type=float, default=0.4)
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"Image does not exist: {args.image}")
    if not 0 <= args.confidence <= 1:
        parser.error("Confidence must be between 0 and 1.")
    try:
        payload = infer_image(args.image, args.confidence)
    except Exception as error:
        key = os.environ.get("ROBOFLOW_API_KEY", "")
        message = str(error).replace(key, "[REDACTED]") if key else str(error)
        parser.exit(1, f"Inference failed: {message}\n")
    args.output.mkdir(parents=True, exist_ok=True)
    json_path = args.output / f"{args.image.stem}.json"
    annotated_path = args.output / f"{args.image.stem}_annotated.png"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    annotate(args.image, payload["detections"], annotated_path)
    print(f"Model: {MODEL_ID}")
    print(f"Raw detections: {len(payload['raw_result'].get('predictions', []))}")
    print(f"Selected-class detections: {len(payload['detections'])}")
    print(f"JSON: {json_path.resolve()}")
    print(f"Annotated image: {annotated_path.resolve()}")


if __name__ == "__main__":
    main()
