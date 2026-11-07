#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
sudo apt-get update
sudo apt-get install -y python3-tk ffmpeg
bash scripts/install-pi.sh
.venv/bin/python -m pip install --only-binary=:all: -r requirements-tkinter.txt
.venv/bin/python -m pip check
.venv/bin/python -c 'import tkinter; from PIL import Image, ImageTk; print("Tkinter and Pillow imports passed")'
echo "Run bash scripts/run-tkinter-pi.sh from a terminal on the Pi graphical desktop."
