#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../webapp"
export VISIONGUARD_AUTO_IMGSZ="${VISIONGUARD_AUTO_IMGSZ:-640}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
exec ../.venv/bin/waitress-serve --listen=127.0.0.1:5000 --threads=4 --channel-timeout=120 app:app
