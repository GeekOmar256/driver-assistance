"""Validate a compact YOLO detection dataset and fine-tune pretrained YOLOv8."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import yaml

CLASSES = [
    "stop sign", "yield sign", "no entry", "speed limit 40", "speed limit 60",
    "speed limit 80", "red traffic light", "yellow traffic light", "green traffic light",
]
ALIASES = {"stop": "stop sign", "yield": "yield sign", "give way": "yield sign",
           "red": "red traffic light", "yellow": "yellow traffic light",
           "amber": "yellow traffic light", "green": "green traffic light"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def canonical_name(name):
    name = " ".join(str(name).lower().replace("_", " ").replace("-", " ").split())
    return ALIASES.get(name, name)


def prepare_data(source, output):
    source = Path(source).resolve()
    output = Path(output).resolve()
    root = source.parent
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw_names = config.get("names", [])
    if isinstance(raw_names, dict):
        raw_names = [raw_names.get(i, raw_names.get(str(i))) for i in range(len(raw_names))]
    names = [canonical_name(name) for name in raw_names]
    if len(names) != len(CLASSES) or set(names) != set(CLASSES):
        raise ValueError(f"Dataset must contain exactly these nine classes: {CLASSES}. "
                         f"Found {names}. Export a detection dataset with these classes first.")
    # Keep class IDs in the export's original order, while normalizing names.
    prepared = {"names": names, "nc": len(names)}
    counts = {}
    seen = {}
    for split in ("train", "val", "test"):
        value = config.get(split)
        if not isinstance(value, str):
            raise ValueError(f"Provide a {split} image directory in data.yaml.")
        split_alias = "valid" if split == "val" else split
        candidates = [root / value, root / str(config.get("path", ".")) / value,
                      root / split_alias / "images", root / split / "images"]
        folder = next((p.resolve() for p in candidates if p.is_dir()
                       and p.resolve().is_relative_to(root)), None)
        if folder is None:
            raise ValueError(f"Cannot find {split} images inside {root}.")
        if folder.name != "images":
            raise ValueError("Use train/images, valid/images and test/images with sibling labels folders.")
        images = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            raise ValueError(f"Empty {split} split.")
        split_counts = [0] * len(names)
        for image in images:
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            previous = seen.get(digest)
            if previous is not None and previous != split:
                raise ValueError(f"Identical image appears in {previous} and {split}: {image.name}")
            seen[digest] = split
            label = folder.parent / "labels" / image.relative_to(folder).with_suffix(".txt")
            if not label.is_file():
                raise ValueError(f"Missing label {label}; negative images need an empty label file.")
            for line in label.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) != 5:
                    raise ValueError(f"Expected detection label: class x y width height in {label}")
                class_id = int(parts[0])
                box = [float(v) for v in parts[1:]]
                if class_id not in range(len(names)) or not all(math.isfinite(v) and 0 <= v <= 1 for v in box):
                    raise ValueError(f"Invalid class ID or normalized coordinates in {label}")
                x, y, width, height = box
                if width <= 0 or height <= 0 or x-width/2 < -0.001 or x+width/2 > 1.001 or y-height/2 < -0.001 or y+height/2 > 1.001:
                    raise ValueError(f"Bounding box extends outside the image in {label}")
                split_counts[class_id] += 1
        missing = [names[i] for i, count in enumerate(split_counts) if count == 0]
        if missing:
            raise ValueError(f"{split} has no examples for: {missing}. Add examples before training.")
        prepared[split] = str(folder)
        counts[split] = {"images": len(images), "objects": dict(zip(names, split_counts))}
    output.mkdir(parents=True, exist_ok=True)
    data = output / "data.yaml"
    # No dataset download directive is copied from the source export.
    data.write_text(yaml.safe_dump(prepared, sort_keys=False), encoding="utf-8")
    (output / "dataset_counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print(json.dumps(counts, indent=2))
    return data


def train(data, output, epochs=50, imgsz=960):
    from ultralytics import YOLO
    existing_weights = Path(__file__).resolve().parents[1] / "webapp" / "yolov8n.pt"
    model = YOLO(str(existing_weights) if existing_weights.is_file() else "yolov8n.pt")
    model.train(data=str(data), epochs=epochs, imgsz=imgsz, batch=8, patience=12,
                device=0, workers=2, seed=42, deterministic=True, cache=False,
                fliplr=0.0, flipud=0.0, hsv_h=0.0, hsv_s=0.0,
                project=str(output), name="road_signs_lights", exist_ok=False)
    best = Path(model.trainer.best)
    evaluation = YOLO(str(best)).val(data=str(data), split="test", imgsz=imgsz,
                                   device=0, project=str(output), name="held_out_test")
    metrics = {"mAP50": float(evaluation.box.map50), "mAP50_95": float(evaluation.box.map),
               "precision": float(evaluation.box.mp), "recall": float(evaluation.box.mr)}
    (best.parent.parent / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Trained checkpoint: {best}")
    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Compact nine-class YOLO export's data.yaml")
    parser.add_argument("--output", default="training_runs")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    data = prepare_data(args.data, Path(args.output) / "prepared")
    if not args.validate_only:
        train(data, args.output, args.epochs, args.imgsz)
