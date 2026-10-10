"""Convert the selected Roboflow version's 20 labels to the nine-class scope."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import yaml

from train_limited import CLASSES

SOURCE_TO_TARGET = {
    "Stop": "stop sign", "Yield": "yield sign", "No entry": "no entry",
    "speed limit 40km-h": "speed limit 40", "speed limit 60km-h": "speed limit 60",
    "speed limit 80km-h": "speed limit 80", "Traffic signal_red": "red traffic light",
    "Traffic signal_yellow": "yellow traffic light", "Traffic signal_green": "green traffic light",
}


def prepare(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError(f"Destination already exists: {destination}")
    metadata = yaml.safe_load((source / "data.yaml").read_text())
    names = metadata["names"]
    mapping = {i: CLASSES.index(SOURCE_TO_TARGET[name]) for i, name in enumerate(names)
               if name in SOURCE_TO_TARGET}
    if set(mapping.values()) != set(range(len(CLASSES))):
        raise ValueError("Source export does not include all nine selected classes.")
    counts = {}
    seen_hashes, seen_origins = {}, {}
    removed = {}
    # Reserve held-out images first so leakage is removed from training.
    for split in ("test", "valid", "train"):
        images = source / split / "images"
        labels = source / split / "labels"
        if not images.is_dir() or not labels.is_dir():
            raise ValueError(f"Missing {split} images or labels")
        (destination / split / "images").mkdir(parents=True)
        (destination / split / "labels").mkdir()
        count = [0] * len(CLASSES)
        removed[split] = 0
        for image in sorted(images.iterdir()):
            if not image.is_file():
                continue
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            origin = image.stem.split(".rf.")[0]
            if ((digest in seen_hashes and seen_hashes[digest] != split)
                    or (origin in seen_origins and seen_origins[origin] != split)):
                removed[split] += 1
                continue
            seen_hashes[digest], seen_origins[origin] = split, split
            label = labels / (image.stem + ".txt")
            if not label.is_file():
                raise ValueError(f"Missing source annotation: {label}")
            converted = []
            for row in label.read_text().splitlines():
                if not row.strip():
                    continue
                parts = row.split()
                is_polygon = len(parts) >= 7 and len(parts) % 2 == 1
                if (len(parts) != 5 and not is_polygon) or int(parts[0]) not in range(len(names)):
                    raise ValueError(f"Invalid detection label in {label}")
                original_id = int(parts[0])
                if original_id in mapping:
                    target_id = mapping[original_id]
                    coordinates = parts[1:]
                    if is_polygon:
                        points = [float(value) for value in coordinates]
                        xs, ys = points[::2], points[1::2]
                        xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
                        coordinates = [str((xmin+xmax)/2), str((ymin+ymax)/2),
                                       str(xmax-xmin), str(ymax-ymin)]
                    converted.append(" ".join([str(target_id), *coordinates]))
                    count[target_id] += 1
            # Retain empty-label images as useful background examples.
            shutil.copy2(image, destination / split / "images" / image.name)
            (destination / split / "labels" / label.name).write_text("\n".join(converted))
        counts[split] = dict(zip(CLASSES, count))
    config = {"train": "train/images", "val": "valid/images", "test": "test/images",
              "names": CLASSES, "nc": len(CLASSES)}
    (destination / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    for name in ("README.roboflow.txt", "README.dataset.txt"):
        if (source / name).is_file():
            shutil.copy2(source / name, destination / name)
    (destination / "source.json").write_text(json.dumps({
        "workspace": "ibtissammouaad", "project": "road-and-traffic-signs-recognition",
        "version": 2, "license": metadata.get("roboflow", {}).get("license"),
        "class_counts": counts, "cross_split_duplicates_removed": removed}, indent=2))
    print(json.dumps(counts, indent=2))
    return destination / "data.yaml"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    prepare(args.source, args.destination)
