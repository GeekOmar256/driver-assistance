"""Build the self-contained Colab notebook from the training helper."""
import json
from pathlib import Path

root = Path(__file__).parent


def cell(kind, source):
    value = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    return value


cells = [
    cell("markdown", """# VisionGuard: nine-class YOLO fine-tuning

Start from pretrained **YOLOv8n** and fine-tune for stop, yield, no entry, speed limits 40/60/80 km/h, and red/yellow/green traffic lights.

**You need a compact YOLO detection dataset ZIP before running this notebook.** Aim initially for roughly 2,000–5,000 varied images covering all nine classes. This notebook does not fetch the large Mapillary, Bosch, TT100K or BDD100K archives.

Export exactly the selected classes, with `data.yaml` and `train/images`, `train/labels`, `valid/images`, `valid/labels`, `test/images`, `test/labels`. Every image needs a label file, which is empty for negative images. Use bounding boxes in full road scenes, rather than classification-only sign crops. Label all selected signs and light states visible in every image. A generic traffic-light label is insufficient for learning its state. Speed-limit labels must be in km/h.

Keep different recording sequences/journeys in different splits and remove near-duplicates. Exact duplicate images across splits are checked automatically. Include each class in every split. Do not merge sign and light datasets without labelling any missing selected objects in those images.

In Colab choose **Runtime → Change runtime type → GPU**, then run cells in order. Free GPU availability varies. Training has not been run yet; metrics appear only after actual training.

References: [Ultralytics training](https://docs.ultralytics.com/modes/train/), [YOLO dataset format](https://docs.ultralytics.com/datasets/detect/), [Colab](https://docs.ultralytics.com/integrations/google-colab/).
"""),
    cell("code", """%pip install -q "ultralytics>=8.3,<9" pyyaml
import torch
assert torch.cuda.is_available(), "Enable a GPU runtime before training."
print(torch.cuda.get_device_name(0))
"""),
    cell("markdown", """## Get only the compact dataset
Paste a direct HTTPS ZIP download link when prompted, or leave it blank to upload your compact ZIP to Colab. URLs are entered privately and are not stored in the notebook. The download limit is 1 GiB, and extraction is limited to 3 GiB. If an export is larger, create a smaller export first. The ZIP is downloaded to the cloud runtime, not to your computer.
"""),
    cell("code", """import getpass
import shutil
import stat
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlparse
from google.colab import files

workspace = Path('/content/visionguard')
workspace.mkdir(exist_ok=True)
archive = workspace / 'compact_dataset.zip'
MAX_DOWNLOAD = 1024 ** 3
url = getpass.getpass('Direct dataset ZIP URL (blank to upload): ').strip()
if url:
    assert urlparse(url).scheme == 'https', 'Use an HTTPS download link.'
    request = urllib.request.Request(url, headers={'User-Agent': 'VisionGuard-training'})
    with urllib.request.urlopen(request, timeout=60) as response:
        assert urlparse(response.url).scheme == 'https', 'Redirect must also use HTTPS.'
        length = int(response.headers.get('Content-Length', 0))
        assert length <= MAX_DOWNLOAD, 'Dataset exceeds the 1 GiB limit.'
        received = 0
        with archive.open('wb') as output:
            while chunk := response.read(1024 ** 2):
                received += len(chunk)
                assert received <= MAX_DOWNLOAD, 'Dataset exceeds the 1 GiB limit.'
                output.write(chunk)
else:
    uploaded = files.upload()
    assert len(uploaded) == 1, 'Upload exactly one dataset ZIP.'
    filename, content = next(iter(uploaded.items()))
    assert len(content) <= MAX_DOWNLOAD, 'Dataset exceeds the 1 GiB limit.'
    archive.write_bytes(content)
    del uploaded, content

dataset_dir = workspace / 'dataset'
assert not dataset_dir.exists(), 'Use a fresh runtime before importing a different dataset.'
with zipfile.ZipFile(archive) as bundle:
    assert sum(info.file_size for info in bundle.infolist()) <= 3 * 1024 ** 3, 'Export exceeds the extraction limit.'
    for info in bundle.infolist():
        destination = (dataset_dir / info.filename).resolve()
        assert destination.is_relative_to(dataset_dir.resolve()), 'Unsafe ZIP member path.'
        assert not stat.S_ISLNK(info.external_attr >> 16), 'ZIP symlinks are not supported.'
    bundle.extractall(dataset_dir)
yaml_files = list(dataset_dir.rglob('data.yaml'))
assert len(yaml_files) == 1, 'Dataset must contain exactly one data.yaml.'
source_yaml = yaml_files[0]
print('Dataset imported:', source_yaml)
"""),
    cell("code", "%%writefile /content/visionguard/train_limited.py\n" + (root / "train_limited.py").read_text(encoding="utf-8")),
    cell("markdown", """## Validate class names and annotations
Accepted aliases include `stop`, `yield`, `give way`, `red`, `yellow`, `amber`, `green`, and underscore/hyphen variants of the canonical names. Class IDs retain the export's order. Other class names must be renamed explicitly in the export; do not assign a generic light label to a colour.
"""),
    cell("code", """import sys
sys.path.insert(0, str(workspace))
from train_limited import prepare_data, train
prepared_yaml = prepare_data(source_yaml, workspace / 'prepared')
"""),
    cell("markdown", """## Fine-tune and evaluate
Initial settings: 50 epochs, 960-pixel inputs, batch size 8, early stopping, and no flips or hue/saturation augmentation. These are starting settings, not an accuracy guarantee. If GPU memory is insufficient, reduce `batch=8` in the helper cell above, rerun it, and restart the Python runtime before importing it again. The held-out test split is evaluated once after training; use validation, not test results, to choose settings.
"""),
    cell("code", """from google.colab import drive
drive.mount('/content/drive')
from datetime import datetime
run_dir = Path('/content/drive/MyDrive/VisionGuardTraining') / datetime.now().strftime('%Y%m%d_%H%M%S')
run_dir.mkdir(parents=True, exist_ok=False)
shutil.copy(workspace / 'prepared/dataset_counts.json', run_dir / 'dataset_counts.json')
best = train(prepared_yaml, run_dir, epochs=50, imgsz=960)
"""),
    cell("markdown", """## Download the checkpoint
The trained checkpoint and training results are saved in your Google Drive. Download the checkpoint below and put it at `webapp/models/road_signs_lights.pt` in your local VisionGuard project, then restart the app. Until a checkpoint is installed, the app uses its pretrained fallback for stop signs and generic traffic lights only. For interrupted runs, retain `last.pt`; the dataset needs to be loaded again into a new runtime before resuming.
"""),
    cell("code", """checkpoint = workspace / 'road_signs_lights.pt'
shutil.copy(best, checkpoint)
files.download(str(checkpoint))
print('Install at: webapp/models/road_signs_lights.pt')
print('Test metrics:', best.parent.parent / 'test_metrics.json')
"""),
]
notebook = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"name": "VisionGuard_9_class_training.ipynb"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
(root / "VisionGuard_9_class_training.ipynb").write_text(json.dumps(notebook, indent=2), encoding="utf-8")
