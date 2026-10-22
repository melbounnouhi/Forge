# Forge

**Video analysis for real-world workspaces** — person detection and tracking on fixed-camera footage, built with YOLO and OpenCV.

> The physical world is the next data frontier. Machines will need to understand how humans actually work — in garages, workshops, and small businesses, not just in labs. Forge is my training ground for that problem.

## What it does

Give it a video from a fixed camera, it gives you back:
- **Person detection & tracking** (persistent IDs across frames)
- **Zone occupancy analysis** — time spent and entries per custom zone
- **Movement patterns** — transitions between zones per person
- **A one-page report** (console + JSON export)

## Usage

```bash
pip install ultralytics opencv-python
python forge_v01.py your_video.mp4
```

Zones are defined as relative coordinates (x1, y1, x2, y2) at the top of the script — adapt them to your camera setup.

## Privacy principles

- Designed for **consent-based** analysis only
- Tracks **positions, not identities** — no face recognition, ever
- Data stays local: the tool outputs statistics, not footage

## Roadmap

- [ ] Smarter re-identification (one person = one ID)
- [ ] Per-site zone calibration
- [ ] Structured JSON export → LeRobot-compatible formats
- [ ] Enrichment pipeline: pose estimation, depth

## Benchmark (MOT16-02, 54 pedestrians, CC BY-NC-SA — research use only)

| Config | Unique IDs | Max simultaneous | Score |
|---|---|---|---|
| yolov8n, imgsz 640, every 10th frame | ~9 | 4 | ~17% |
| + imgsz 1280 | 18 | 7 | 33% |
| + every 2nd frame | 33 | 11 | 61% |
| + yolov8s | 42 | 11 | 78% |

12 of the 42 IDs live under 1 second (likely fragments) → ~30 solid tracks (56%).
Next: tracker memory (`track_buffer`) and proper MOT evaluation with `motmetrics`.

`frames_to_video.py` converts a MOT image sequence into a video for testing.

## Why

I'm a 16-year-old student in France, learning computer vision by building tools for the messy real world. This is project #1.
