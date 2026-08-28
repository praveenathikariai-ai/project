# Low-Cost Smart Glasses for Real-Time Obstacle Detection 

Dissertation code submission. University of Stirling.

This archive backs up the dissertation with the actual source code used in the
project. It is not assessed directly — it's evidence for the claims made in
the dissertation.

## What's in this archive

| `smart_glasses.py` | Main prototype. Live webcam feed → YOLOv8 object detection → direction (left/right/ahead) → monocular distance estimate → spoken alert via Windows text-to-speech. Also logs a session and saves a 4-panel report (FPS, objects/frame, avg. distance by class, detection frequency by class) as a PNG when the run ends. |
| `colab_report_from_video.py` | Same detection, direction, and distance logic as `smart_glasses.py`, but reads from an uploaded video file instead of a live webcam, so it can run in Google Colab (no webcam/display access there). Produces the same 4-panel report PNG. Used to generate result plots for the dissertation without needing a Windows machine on hand. |


## Provenance — third-party code and libraries

This project is built entirely on top of published open-source libraries. No
code was supplied by a supervisor or third party; nothing here was copied
from another student's or public project's source. Libraries used:
from gettext import install
import queue
import subprocess
import threading
import time
from collections import defaultdict
 
import cv2
import matplotlib
import pip
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
from ultralytics import YOLO

[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)** (AGPL-3.0) — object detection model and inference API. The prototype loads the pretrained `yolov8n.pt` checkpoint (COCO-trained, 80 classes) via `YOLO("yolov8n.pt")`; the library downloads it automatically on first run. Not redistributed in this archive.

[OpenCV](https://opencv.org/) (`opencv-python`)** — camera capture, frame annotation, on-screen HUD text.
[Matplotlib](https://matplotlib.org/)** — session report plots.
Windows `System.Speech` (via PowerShell)** — built into Windows; used for text-to-speech. No external package.


Distance estimation uses the standard pinhole-camera relation
(`distance = real_width × focal_length_px / box_width_px`), a well-known
technique in monocular distance estimation, not sourced from a specific
codebase — implemented directly for this project.

## Build and run instructions

### `smart_glasses.py` (Windows only — needs webcam + PowerShell TTS)

```
pip install ultralytics opencv-python matplotlib
python smart_glasses.py
```

Point the camera at objects; alerts are spoken and shown on-screen with
direction and distance. Press `q` in the video window to quit — this saves
`session_report_<timestamp>.png` in the working directory.

Before trusting the distance readings: `FOCAL_LENGTH_PX` (line 32) is a
placeholder value (700). 

```
pip install ultralytics opencv-python-headless matplotlib
python colab_report_from_video.py path/to/video.mp4
```

Produces the same session report PNG, built from real detections in the
supplied video (no live camera or synthetic data involved).

## Dependencies not included in this archive

- **`yolov8n.pt`** — pretrained YOLOv8 nano weights (~6 MB), downloaded
  automatically by the `ultralytics` package on first run from
  [Ultralytics' releases](https://github.com/ultralytics/assets/releases).
  Not a project output, so not included per the "no binaries" rule.
- **Video/image test data** used to produce result plots — not included per
  the "no large datasets" rule. Any footage used for evaluation was captured
  by the student with a personal camera/phone; there is no fixed dataset to
  redistribute or link.

## Use of AI tools

Generative AI (Claude) was used during the final development of this
prototype as i was getting ideas to add on. Mainly for -- 

- Reviewing `smart_glasses.py` for correctness and flagging platform
  assumptions  and calibration.
- Writing `colab_report_from_video.py`, adapting the detection/direction/
  distance logic already present in `smart_glasses.py`.


