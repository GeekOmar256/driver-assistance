#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$(uname -m)" != aarch64 || "$(dpkg --print-architecture)" != arm64 ]]; then
  echo "This release needs a 64-bit ARM OS (aarch64/arm64)." >&2
  exit 1
fi
sudo apt-get update
glib=libglib2.0-0
if apt-cache show libglib2.0-0t64 2>/dev/null | grep -q '^Package:'; then
  glib=libglib2.0-0t64
fi
sudo apt-get install -y python3-venv libgl1 "$glib" libgomp1 v4l-utils curl
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install --only-binary=:all: --index-url https://download.pytorch.org/whl/cpu 'torch>=2.6,<3' 'torchvision>=0.21,<1'
constraints=()
if grep -q '^VERSION_CODENAME=bookworm$' /etc/os-release; then
  # Keep NumPy compatible with Bookworm's optional system Picamera2 bindings.
  constraints=(-c constraints-bookworm.txt)
fi
.venv/bin/python -m pip install --only-binary=:all: "${constraints[@]}" -r requirements-pi.txt
.venv/bin/python -m pip check
sha256sum -c MODELS.sha256
(cd webapp && ../.venv/bin/python -m unittest discover -p 'test_*.py')
.venv/bin/python scripts/smoke.py
