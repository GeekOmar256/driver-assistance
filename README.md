# VisionGuard for Raspberry Pi

This deployment branch targets Raspberry Pi 5 with 64-bit Raspberry Pi OS Bookworm or Trixie. Debian 13 Trixie ARM64 with Python 3.13 is also permitted for installation and runtime validation; full operation on that OS must be verified on the device. It bundles the pretrained YOLOv8n checkpoint, the nine-class Roboflow-trained checkpoint, and local alert audio. No API key is required. Use one browser viewer at a time.

The standard two-stage mode is the default. Both two-stage modes expose `person` as a selectable class. The pretrained model detects people directly and sends their image direction to the visual and audio alerts; the crop recognizer handles signs and light colors. Generic unclassified traffic lights are suppressed in both two-stage modes. Direction describes image position, not distance or a collision prediction.

## Install and start

```bash
sudo apt-get update
sudo apt-get install -y git
git clone --single-branch --branch codex/raspberry-pi https://github.com/GeekOmar256/driver-assistance.git ~/visionguard-pi
cd ~/visionguard-pi
bash scripts/install-pi.sh
bash scripts/run-pi.sh
```

Open **http://127.0.0.1:5000** on the Pi. For a browser on another computer, use an SSH tunnel from that computer, then open the same URL there:

```bash
ssh -N -L 5000:127.0.0.1:5000 <pi-user>@<pi-host>
```

The service binds to loopback. Start with an uploaded MP4 and enable sound in the browser. Automatic inference size is 640; explicitly selecting 960 or 1280 costs more CPU time. Uploads are limited to 256 MiB and stored in `webapp/uploads/`. Stop the server with Ctrl+C.

### Camera selection

The webcam button opens a camera attached to the **Pi**, not the remote browser. For a USB webcam, use `v4l2-ctl --list-devices`; the default is camera index 0. For example:

```bash
VISIONGUARD_CAMERA=/dev/video2 bash scripts/run-pi.sh
```

On plain Debian, first check that the Raspberry Pi camera tools and `python3-picamera2` are available from the configured OS repositories. If unavailable, use USB or video testing and report CSI as pending; do not add external apt repositories. For a Raspberry Pi Camera Module connected by CSI with the camera stack available, check that `rpicam-hello --list-cameras` detects it, then install the OS bindings and select the CSI backend:

```bash
sudo apt-get install -y python3-picamera2
VISIONGUARD_CAMERA=picamera2 bash scripts/run-pi.sh
```

The installer uses a virtual environment with access to system packages so the CSI backend can load the OS camera bindings. It verifies checkpoint checksums and runs software and CPU inference tests. If a binary wheel is unavailable, report the OS, architecture, Python version, and pip error rather than attempting an unattended source build. Actual camera and performance validation must be done on the Pi.

## Sources

- [Raspberry Pi OS and Python virtual environments](https://www.raspberrypi.com/documentation/computers/os.html) and [Camera Module software](https://www.raspberrypi.com/documentation/accessories/camera-software.html).
- [Ultralytics Raspberry Pi guide](https://docs.ultralytics.com/guides/raspberry-pi/) and [PyTorch CPU wheels](https://download.pytorch.org/whl/cpu/). The bundled YOLOv8n weights and Ultralytics code are subject to upstream licensing; see `licenses/ultralytics-AGPL-3.0.txt`.
- The custom checkpoint was fine-tuned using a selected nine-class subset of [Road and Traffic Signs Recognition, version 2](https://universe.roboflow.com/ibtissammouaad/road-and-traffic-signs-recognition/dataset/2), by ibtissammouaad, listed as [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The dataset is not bundled.
