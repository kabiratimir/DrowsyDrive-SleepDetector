# Student Sleep Detection & Alert System

A Python computer-vision project for detecting prolonged eye closure/drowsiness while a student studies using a webcam.

## Current version

This ZIP is the **starter implementation** following the requested incremental roadmap.

Implemented:
- Webcam capture
- Face landmark detection with MediaPipe
- Eye landmark extraction
- Eye Aspect Ratio (EAR)
- Basic blink detection
- Drowsy / sleeping state based on eye-closure duration
- On-screen status and timers

Not yet implemented:
- Alarm playback integration
- Full sleep episode database
- Study-session reports
- GUI
- Head-pose and yawn detection
- Advanced smoothing/calibration

Those should be added only after this foundation is tested successfully.

## Requirements

- Windows/macOS/Linux
- Python 3.10 or 3.11 recommended
- Working webcam
- Internet access for installing packages and downloading the MediaPipe model

## Setup

Open a terminal in this folder.

### 1. Create virtual environment

Windows:

    python -m venv .venv
    .venv\Scripts\activate

macOS/Linux:

    python3 -m venv .venv
    source .venv/bin/activate

### 2. Install packages

    python -m pip install --upgrade pip
    pip install -r requirements.txt

## 3. Download the MediaPipe Face Landmarker model

Create:

    models/

Download the official `face_landmarker.task` model from the MediaPipe documentation/model-hosting source and place it here:

    models/face_landmarker.task

The code intentionally does not silently download a model so the project remains explicit about its runtime dependency.

## 4. Run

    python main.py

Press `Q` to quit.

## Important calibration note

The default EAR threshold (`0.21`) and timing values are starting values only. Camera position, lighting, face shape, glasses, and frame rate can affect detection.

Do not treat this prototype as a medical sleep diagnostic system.

## Next development phases

1. Validate webcam + landmark detection.
2. Calibrate EAR threshold.
3. Improve blink/drowsiness state machine.
4. Add sleep episode tracking.
5. Add alarm.
6. Add SQLite.
7. Add GUI.
8. Add reports and testing.

## Privacy

Process camera frames locally. Do not store or upload video unless explicitly added as a future feature.
