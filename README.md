# People Counter AI (Line Crossing, Web UI)

Production-minded demo for counting people crossing a virtual line using a webcam or video file.
Built with Flask + OpenCV + YOLO (Ultralytics). Includes iPhone-friendly web UI.

## Features
- Line-cross counting with direction (IN/OUT)
- Works with webcam, RTSP, or local video (sample included as placeholder)
- Minimal Flask UI that streams annotated frames (MJPEG)
- Dockerfile-ready (optional)

## Quick Start
```bash
git clone <your-repo-url> people-counter-ai
cd people-counter-ai
python -m venv .venv && . .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
Open: http://127.0.0.1:5000

## Configure Source
Set env `VIDEO_SOURCE` to:
- `0` for default webcam
- RTSP URL
- Path to a video file, e.g., `sample_data/sample_video.mp4`

## Security & Privacy
- Do **not** commit real camera URLs or faces without consent.
- Use `.env` (not provided) and keep it out of git.

## Folder Structure
```text
people-counter-ai/
├─ app.py
├─ requirements.txt
├─ static/
├─ templates/
├─ sample_data/
│  └─ README.txt
├─ models/  # put your YOLO weights if needed
├─ .gitignore
└─ LICENSE
```

## Notes
- YOLO weights auto-download on first run (by Ultralytics).
- If `lap` install fails on Windows, either skip trackers or install `lap==0.4.0`.