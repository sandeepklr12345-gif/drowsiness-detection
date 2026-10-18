# Drowsiness Detection

Developed a computer-vision based system to detect driver fatigue and drowsiness in real time by analysing facial features and eye states. The system identifies signs of reduced alertness and generates timely alerts to improve driver safety and reduce the risk of fatigue-related accidents.

## Features
- Real-time webcam-based detection
- Face & eye detection (OpenCV / dlib / mediapipe)
- Drowsiness scoring and threshold-based alerts (sound / visual)
- Lightweight model suitable for edge deployment

## System architecture
This repository did not have a root README.md, so I added this System Architecture section so it's visible on the repository main page. Below is a short high-level overview and a simple flow diagram you can expand with an image (architecture.png) if you prefer.

High-level components:

- Camera / Video Source: captures live frames from a webcam or video stream.
- Frame pre-processing: resize, convert to grayscale, normalize, and optionally blur/denoise.
- Face & eye detection: detect face and eye regions (Haar cascades, DNN, or MediaPipe).
- Feature extraction: compute eye aspect ratio (EAR), blink rate, or extract face landmarks / facial embeddings used by the model.
- Drowsiness model / decision logic: a ML model (CNN / classifier) or heuristic (EAR threshold + time window) that outputs a fatigue_score.
- Alerting subsystem: triggers audio alarms, visual warnings on-screen, or sends notifications (webhook / REST API) when drowsiness is detected.
- Persistence & logging (optional): save events, timestamps, and example frames for offline analysis.

Simple flow diagram (text):

Camera/Video -> Frame Capture -> Preprocessing -> Face & Eye Detection -> Feature Extraction -> Drowsiness Model -> Decision & Alerts

To add a visual diagram, put an image at `docs/architecture.png` or `assets/architecture.png` and reference it like:

```
![System Architecture](assets/architecture.png)
```

### Where to show architecture in the README
GitHub displays the README.md located at the repository root (default branch). If your architecture is in another file (for example `docs/architecture.md` or `pages/architecture.py`), it won't appear on the repo front page unless the root README links to it. I recommend one of the following:

- Keep a concise System Architecture section inside root `README.md` (what I added).
- Add a diagram image in `assets/` or `docs/` and reference it from README.
- If you prefer a detailed architecture doc, add `docs/ARCHITECTURE.md` and link to it from README.

## Quick setup
1. Create a virtual env: `python -m venv .venv`
2. Activate and install: `pip install -r requirements.txt`
3. Run app: `python app.py` (or run the appropriate script in `pages/`)

## Notes
- I noticed you have `pages/` Python files that render UI or pages (drowsiness.py, emotion.py). If those contain diagrams or documentation, link them from README.
- If you want, I can:
  - add an `assets/architecture.png` placeholder and link it from README,
  - or create a `docs/ARCHITECTURE.md` with more detail and sequence diagrams,
  - or update the README text further to match the exact architecture used by your code (I can inspect the detection / model files to write precise descriptions).
