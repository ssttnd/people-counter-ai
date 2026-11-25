"""ARIA automated people counter - Flask server.

Pipeline for every frame (live view and fast count use exactly the same code):
    YOLO person detection -> PeopleTracker (BoT-SORT + ghosts) -> LineCounter
See counting.py for the counting rules and README.md for the measured accuracy.
"""
import json
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

from counting import IN, LineCounter, PeopleTracker

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - shown as an error in the UI
    YOLO = None


app = Flask(__name__)

ROOT = Path(__file__).resolve().parent
CONFIG_FILE = ROOT / "line_config.json"
SETTINGS_FILE = ROOT / "people_counter_settings.json"
TRACKER_FILE = ROOT / "people_tracker.yaml"

DEFAULT_LINE = [[0.15, 0.5], [0.85, 0.5]]
VIDEO_EXTENSIONS = ("*.mp4 *.avi *.mov *.mkv *.wmv *.webm *.m4v")

# Detection settings shared by live view and fast count. They were tuned on a
# hand-labelled 10-minute entrance video (see README.md, "Accuracy").
MODEL_NAME = os.getenv("PEOPLE_COUNTER_MODEL", "yolov8n.pt")
DETECTION_CONF = 0.25
DETECTION_IMGSZ = 640

state_lock = threading.Lock()
line_lock = threading.Lock()
inference_lock = threading.Lock()


# --------------------------------------------------------------------- settings
def load_line():
    try:
        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        points = [[float(v) for v in p] for p in data]
        if len(points) == 2 and all(len(p) == 2 and all(0 <= v <= 1 for v in p) for p in points):
            return points
    except (OSError, ValueError, TypeError):
        pass
    return [p[:] for p in DEFAULT_LINE]


SETTING_LIMITS = {
    # name: (default, min, max)
    "size_tolerance_cm": (30, 10, 100),
    "reference_line_cm": (180, 50, 300),
    "ghost_timeout_seconds": (3, 1, 20),
}
settings = {
    "size_filter_enabled": True,
    "show_invalid_size_boxes": True,
    "show_ghosts": True,
    **{k: v[0] for k, v in SETTING_LIMITS.items()},
}


def _clamp_setting(name, value):
    _, lo, hi = SETTING_LIMITS[name]
    return max(lo, min(hi, int(value)))


def load_settings():
    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if not isinstance(data, dict):
        return
    for key in ("size_filter_enabled", "show_invalid_size_boxes", "show_ghosts"):
        if isinstance(data.get(key), bool):
            settings[key] = data[key]
    for key in SETTING_LIMITS:
        try:
            if key in data:
                settings[key] = _clamp_setting(key, data[key])
        except (TypeError, ValueError):
            pass


def save_settings():
    SETTINGS_FILE.write_text(json.dumps(settings, indent=2), encoding="utf-8")


LINE_NORM = load_line()
load_settings()


def _initial_source():
    value = os.getenv("VIDEO_SOURCE")
    if value is None:
        return None
    return int(value) if value.isdigit() else value


# ------------------------------------------------------------------- app state
source = _initial_source()          # file path, webcam index (int) or stream URL
stream_generation = 0               # bumped whenever the live view must restart
is_paused = False
count_in = 0
count_out = 0
video_state = {"duration": 0.0, "current_time": 0.0, "seek_to": None,
               "is_file": False, "is_live": False, "error": None, "fps": 0.0}

process_cancel_event = threading.Event()
process_state = {"running": False, "done": False, "cancelled": False, "progress": 0.0,
                 "frame": 0, "total_frames": 0, "status": "idle", "error": None,
                 "count_in": 0, "count_out": 0, "fps": 0.0, "eta_seconds": None,
                 "elapsed_seconds": 0.0, "video_seconds": 0.0, "events": []}


def source_kind(src):
    if src is None:
        return "none"
    if isinstance(src, int):
        return "webcam"
    if os.path.isfile(src):
        return "file"
    return "stream"


def source_label(src):
    kind = source_kind(src)
    if kind == "file":
        return os.path.basename(src)
    if kind == "webcam":
        return f"Webcam {src}"
    if kind == "stream":
        # never show credentials embedded in camera URLs
        scheme, _, rest = str(src).partition("://")
        host = rest.split("@")[-1]
        return f"{scheme}://{host}" if rest else str(src)
    return "No source"


# -------------------------------------------------------------------- pipeline
_model = None


def get_model():
    global _model
    if YOLO is None:
        raise RuntimeError("Ultralytics/YOLO is not installed")
    if _model is None:
        _model = YOLO(MODEL_NAME)
    return _model


def detect_people(frame, imgsz=DETECTION_IMGSZ):
    """(N,5) array [x1, y1, x2, y2, conf] of people in the frame."""
    with inference_lock:
        result = get_model().predict(frame, classes=[0], conf=DETECTION_CONF,
                                     imgsz=imgsz, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return np.zeros((0, 5), np.float32)
    return np.hstack([result.boxes.xyxy.cpu().numpy(),
                      result.boxes.conf.cpu().numpy()[:, None]]).astype(np.float32)


def size_is_plausible(box, p1, p2, frame_width, frame_height, tolerance_cm, reference_cm):
    """Height filter: compare the box height with the real length of the line.

    The pixel length of the counting line is the scale reference: a person is
    countable when their apparent height is reference_cm +/- tolerance_cm.
    Very long lines (> 45 % of the frame diagonal) disable the check.
    """
    box_height = max(0.0, float(box[3]) - float(box[1]))
    line_pixels = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
    if line_pixels < 8.0 or box_height < 1.0:
        return True
    if line_pixels > 0.45 * float(np.hypot(frame_width, frame_height)):
        return True
    apparent_cm = box_height / line_pixels * reference_cm
    return max(1.0, reference_cm - tolerance_cm) <= apparent_cm <= reference_cm + tolerance_cm


# Backwards-compatible name used by tools/eval_cache.py
def _size_is_plausible(box, p1, p2, frame_width=None, frame_height=None,
                       enabled=True, tolerance_cm=30, reference_cm=180):
    if not enabled:
        return True
    return size_is_plausible(box, p1, p2, frame_width or 1e9, frame_height or 1e9,
                             tolerance_cm, reference_cm)


class Pipeline:
    """Detector + tracker + counter for one video source."""

    def __init__(self, fps, width, height, ghost_timeout_s):
        scale = max(width, height) / 1280.0
        self.width, self.height = width, height
        self.people = PeopleTracker(str(TRACKER_FILE), frame_rate=fps or 30.0,
                                    ghost_timeout_s=ghost_timeout_s)
        self.counter = LineCounter(
            (0, 0), (1, 1), min_offset_px=15.0 * scale, lost_near_px=60.0 * scale,
            cancel_window_s=1.5, lost_predict_s=0.0, ghost_count_s=20.0,
        )

    def step(self, frame, t, p1, p2, cfg, dets=None):
        """Process one frame (or ready-made detections). Returns (tracked, ghosts, fired).

        tracked: [(id, box, countable)]  ghosts: [(id, box, age_s)]
        """
        self.people.set_ghost_timeout(cfg["ghost_timeout_seconds"])
        self.counter.set_line(p1, p2)
        if dets is None:
            dets = detect_people(frame)
        raw, ghosts = self.people.update(dets, frame.shape[:2])
        tracked = []
        for tid, box, _conf in raw:
            countable = (not cfg["size_filter_enabled"]) or size_is_plausible(
                box, p1, p2, self.width, self.height,
                cfg["size_tolerance_cm"], cfg["reference_line_cm"])
            tracked.append((tid, box, countable))
        fired = self.counter.update(t, tracked, ghosts)
        if cfg["size_filter_enabled"]:
            # no ghosts for filtered (e.g. far away) people: they would only clutter the view
            ghosts = [g for g in ghosts if size_is_plausible(
                g[1], p1, p2, self.width, self.height,
                cfg["size_tolerance_cm"], cfg["reference_line_cm"])]
        return tracked, ghosts, fired

    def entered_ids(self):
        """People currently counted IN (hidden from the overlay)."""
        return {tid for tid, tr in self.counter.tracks.items() if IN in tr.counted}

    def finish(self, t, fps):
        """Let people who vanished in the last frames complete their crossing."""
        for k in range(1, int(2 * fps)):
            self.counter.update(t + k / fps, [], [])

    def reset(self):
        self.people.reset()
        self.counter.reset()


def line_pixels(width, height):
    with line_lock:
        norm = [p[:] for p in LINE_NORM]
    return ((int(norm[0][0] * width), int(norm[0][1] * height)),
            (int(norm[1][0] * width), int(norm[1][1] * height)))


def settings_snapshot():
    with state_lock:
        return dict(settings)


# --------------------------------------------------------------------- drawing
def draw_overlay(frame, tracked, ghosts, hidden_ids, cfg):
    for tid, box, countable in tracked:
        if tid in hidden_ids:
            continue
        x1, y1, x2, y2 = map(int, box)
        if countable or not cfg["size_filter_enabled"]:
            color, label = (255, 160, 40), f"ID {tid}"
        elif cfg["show_invalid_size_boxes"]:
            color, label = (90, 200, 90), f"IGNORED ID {tid}"
        else:
            continue
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(frame, (cx, cy), 4, (60, 60, 255), -1)
        cv2.putText(frame, label, (x1, max(18, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 2, cv2.LINE_AA)
    if cfg["show_ghosts"]:
        for tid, box, _age in ghosts:
            if tid in hidden_ids:
                continue
            x1, y1, x2, y2 = map(int, box)
            color = (200, 200, 200)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 40, 40), 5)       # dark outline
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            label = f"GHOST ID {tid}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            ty = max(th + 8, y1 - 6)
            cv2.rectangle(frame, (x1, ty - th - 6), (x1 + tw + 8, ty + 4), (30, 30, 30), -1)
            cv2.putText(frame, label, (x1 + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def message_frame(text, width=960, height=540):
    frame = np.full((height, width, 3), (32, 43, 10), np.uint8)
    for i, line in enumerate(text.split("\n")):
        cv2.putText(frame, line, (40, height // 2 - 20 + 36 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (220, 236, 243), 2, cv2.LINE_AA)
    return frame


def mjpeg(frame):
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return b""
    return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"


class LatestFrameReader:
    """Reads a live camera in the background and keeps only the newest frame,
    so processing never falls behind real time (old frames are dropped)."""

    def __init__(self, cap):
        self.cap = cap
        self.frame = None
        self.ok = True
        self.cond = threading.Condition()
        self.stopped = False
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            while not self.stopped:
                try:
                    ok, frame = self.cap.read()
                except cv2.error:
                    ok, frame = False, None
                with self.cond:
                    self.ok, self.frame = ok, frame if ok else None
                    self.cond.notify_all()
                if not ok:
                    break
        finally:
            self.cap.release()          # the reader owns the capture

    def read(self, timeout=5.0):
        with self.cond:
            if self.frame is None and self.ok:
                self.cond.wait(timeout)
            frame, self.frame = self.frame, None
            return (frame is not None), frame

    def stop(self):
        self.stopped = True


def open_capture(src):
    if isinstance(src, str) and source_kind(src) == "stream":
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
    else:
        cap = cv2.VideoCapture(src)
    return cap


# ---------------------------------------------------------------- live stream
def gen():
    global count_in, count_out, is_paused

    with state_lock:
        src = source
        my_generation = stream_generation

    if src is None:
        yield mjpeg(message_frame("Open a video file or connect a camera\nusing the buttons below."))
        return

    cap = open_capture(src)
    if not cap.isOpened():
        with state_lock:
            video_state.update({"error": f"Cannot open {source_label(src)}", "is_file": False,
                                "is_live": False})
        yield mjpeg(message_frame(f"Cannot open\n{source_label(src)}"))
        cap.release()
        return

    kind = source_kind(src)
    is_file = kind == "file"
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not 1.0 <= fps <= 120.0:
        fps = 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if is_file else 0
    duration = frame_count / fps if is_file else 0.0
    reader = None if is_file else LatestFrameReader(cap)

    with state_lock:
        video_state.update({"duration": duration, "current_time": 0.0, "seek_to": None,
                            "is_file": is_file, "is_live": not is_file, "error": None, "fps": 0.0})
        is_paused = False
        count_in = count_out = 0

    pipeline = None
    frame_index = 0
    started = time.monotonic()
    rate_window = []
    try:
        while True:
            with state_lock:
                if my_generation != stream_generation:
                    break
                paused = is_paused and is_file
                seek_to = video_state["seek_to"] if is_file else None
                video_state["seek_to"] = None

            if paused and seek_to is None:
                time.sleep(0.05)
                continue          # (a seek while paused still shows that frame)

            if seek_to is not None:
                frame_index = int(max(0.0, min(duration, seek_to)) * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                if pipeline is not None:
                    pipeline.reset()
                with state_lock:
                    count_in = count_out = 0
                    video_state["current_time"] = frame_index / fps

            if is_file:
                ok, frame = cap.read()
            else:
                ok, frame = reader.read()
            if not ok:
                with state_lock:
                    if is_file:
                        video_state["current_time"] = duration
                        is_paused = True
                    else:
                        video_state["error"] = f"Lost connection to {source_label(src)}"
                break

            height, width = frame.shape[:2]
            t = frame_index / fps if is_file else time.monotonic() - started
            frame_index += 1
            cfg = settings_snapshot()
            p1, p2 = line_pixels(width, height)
            if pipeline is None:
                pipeline = Pipeline(fps, width, height, cfg["ghost_timeout_seconds"])
            tracked, ghosts, _ = pipeline.step(frame, t, p1, p2, cfg)
            draw_overlay(frame, tracked, ghosts, pipeline.entered_ids(), cfg)

            now = time.monotonic()
            rate_window = [x for x in rate_window if now - x < 2.0] + [now]
            with state_lock:
                count_in = pipeline.counter.count_in
                count_out = pipeline.counter.count_out
                video_state["current_time"] = t if is_file else 0.0
                video_state["fps"] = len(rate_window) / 2.0
            yield mjpeg(frame)
    finally:
        if reader is not None:
            reader.stop()               # releases the capture when its read returns
        else:
            cap.release()


# ------------------------------------------------------------------ fast count
# Video count modes. Both detect only around the counting line (ROI).
#   fast:     every 2nd frame - ~4x real time on a laptop CPU, F1 0.90 on the test video
#   accurate: every frame     - ~2.5x real time, same accuracy as the live view
COUNT_MODES = {"fast": 2, "accurate": 1}
FAST_ROI_MARGIN = 1.0      # analysed area: line bounding box + 1 line length on each side


def counting_roi(p1, p2, width, height, margin=FAST_ROI_MARGIN):
    """Region around the counting line that people must pass through."""
    length = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
    pad = max(margin * length, 0.1 * max(width, height))
    x0 = int(max(0, min(p1[0], p2[0]) - pad))
    x1 = int(min(width, max(p1[0], p2[0]) + pad))
    y0 = int(max(0, min(p1[1], p2[1]) - pad))
    y1 = int(min(height, max(p1[1], p2[1]) + pad))
    return x0, y0, x1, y1


def detect_people_in_roi(frame, roi):
    """Detect only inside the ROI, at the same pixel scale as a full-frame pass."""
    x0, y0, x1, y1 = roi
    crop = frame[y0:y1, x0:x1]
    scale = DETECTION_IMGSZ / max(frame.shape[:2])
    imgsz = max(160, int(np.ceil(max(crop.shape[:2]) * scale / 32.0)) * 32)
    dets = detect_people(crop, imgsz=min(imgsz, DETECTION_IMGSZ))
    dets[:, [0, 2]] += x0
    dets[:, [1, 3]] += y0
    return dets


def _process_video_worker(video_path, cfg, line_norm, stride=2):
    """Count a whole video as fast as possible: no drawing, no streaming, only the
    area around the counting line, and every ``stride``-th frame."""
    global count_in, count_out
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise RuntimeError("Could not open the selected video")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            raise RuntimeError("The video length could not be read")
        get_model()
        with state_lock:
            process_state.update({"total_frames": total, "status": "Counting",
                                  "video_seconds": total / fps, "video_fps": fps})

        pipeline = None
        started = time.monotonic()
        frame_no = 0
        roi = None
        while not process_cancel_event.is_set():
            if frame_no % stride:
                ok = cap.grab()             # skipped frame: no pixel conversion
                frame_no += 1
                if not ok:
                    break
                continue
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            p1 = (int(line_norm[0][0] * w), int(line_norm[0][1] * h))
            p2 = (int(line_norm[1][0] * w), int(line_norm[1][1] * h))
            if pipeline is None:
                pipeline = Pipeline(fps / stride, w, h, cfg["ghost_timeout_seconds"])
                roi = counting_roi(p1, p2, w, h)
            pipeline.step(frame, frame_no / fps, p1, p2, cfg, dets=detect_people_in_roi(frame, roi))
            frame_no += 1
            if frame_no % 10 < stride or frame_no >= total:
                elapsed = time.monotonic() - started
                rate = frame_no / max(elapsed, 1e-6)
                with state_lock:
                    process_state.update({
                        "frame": frame_no, "progress": min(1.0, frame_no / total),
                        "fps": round(rate, 1), "elapsed_seconds": round(elapsed, 1),
                        "eta_seconds": round(max(0, total - frame_no) / rate, 1) if frame_no >= 30 else None,
                        "count_in": pipeline.counter.count_in,
                        "count_out": pipeline.counter.count_out,
                    })

        if pipeline is not None:
            pipeline.finish(frame_no / fps, fps / stride)
        cancelled = process_cancel_event.is_set()
        events = [] if pipeline is None else [
            {"t": round(e["t"], 2), "dir": e["dir"], "id": e["id"]}
            for e in _net_events(pipeline.counter.events)
        ]
        with state_lock:
            process_state.update({
                "running": False, "done": not cancelled, "cancelled": cancelled,
                "status": "Cancelled" if cancelled else "Completed",
                "progress": 1.0 if not cancelled else process_state["progress"],
                "eta_seconds": 0 if not cancelled else None,
                "elapsed_seconds": round(time.monotonic() - started, 1),
                "count_in": pipeline.counter.count_in if pipeline else 0,
                "count_out": pipeline.counter.count_out if pipeline else 0,
                "events": events,
            })
            count_in, count_out = process_state["count_in"], process_state["count_out"]
    except Exception as exc:
        with state_lock:
            process_state.update({"running": False, "done": False, "status": "Error", "error": str(exc)})
    finally:
        cap.release()


def _net_events(events):
    """Counted events with turn-back cancellations removed."""
    result = [e for e in events if not e.get("cancel")]
    for c in (e for e in events if e.get("cancel")):
        for i in range(len(result) - 1, -1, -1):
            if result[i]["id"] == c["id"] and result[i]["dir"] == c["dir"]:
                result.pop(i)
                break
    return result


# ---------------------------------------------------------------------- routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/stream")
def stream():
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame",
                    headers={"Cache-Control": "no-cache"})


def _set_source(new_source):
    global source, stream_generation, is_paused
    with state_lock:
        source = new_source
        stream_generation += 1
        is_paused = False
        video_state.update({"seek_to": None, "error": None, "current_time": 0.0})


@app.route("/choose-video", methods=["POST"])
def choose_video():
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        path = filedialog.askopenfilename(
            parent=root, title="Select video",
            filetypes=[("Video files", VIDEO_EXTENSIONS), ("All files", "*.*")])
    finally:
        root.destroy()
    if not path:
        return jsonify({"success": False, "cancelled": True})
    _set_source(path)
    return jsonify({"success": True, "label": source_label(path)})


@app.route("/source", methods=["POST"])
def set_source():
    """Switch to a webcam ({"webcam": 0}) or an IP camera ({"url": "rtsp://..."})."""
    data = request.get_json(silent=True) or {}
    if "webcam" in data:
        try:
            new = int(data["webcam"])
        except (TypeError, ValueError):
            return jsonify(success=False, error="Webcam index must be a number"), 400
    else:
        new = str(data.get("url", "")).strip()
        if new and "://" not in new:
            new = "rtsp://" + new            # "192.168.1.64:554/stream" -> RTSP
        if not new.lower().startswith(("rtsp://", "rtsps://", "http://", "https://", "rtmp://")):
            return jsonify(success=False,
                           error="Camera address must start with rtsp://, http://, https:// or rtmp://"), 400
    _set_source(new)
    return jsonify(success=True, label=source_label(new))


@app.route("/line", methods=["GET", "POST"])
def line_settings():
    global LINE_NORM
    if request.method == "GET":
        with line_lock:
            return jsonify(LINE_NORM)
    data = request.get_json(silent=True)
    try:
        points = [[float(p[0]), float(p[1])] for p in data]
        if len(points) != 2 or any(not 0 <= v <= 1 for p in points for v in p):
            raise ValueError
    except (TypeError, ValueError, IndexError):
        return jsonify(error="Invalid line coordinates"), 400
    with line_lock:
        LINE_NORM = points
        CONFIG_FILE.write_text(json.dumps(LINE_NORM), encoding="utf-8")
    return jsonify(ok=True, line=LINE_NORM)


@app.route("/settings", methods=["GET", "POST"])
def settings_route():
    if request.method == "POST":
        payload = request.get_json(silent=True) or {}
        with state_lock:
            try:
                for key in ("size_filter_enabled", "show_invalid_size_boxes", "show_ghosts"):
                    if key in payload:
                        if not isinstance(payload[key], bool):
                            raise ValueError(key)
                        settings[key] = payload[key]
                for key in SETTING_LIMITS:
                    if key in payload:
                        settings[key] = _clamp_setting(key, payload[key])
            except (TypeError, ValueError):
                return jsonify(success=False, error="Invalid settings"), 400
            save_settings()
    with state_lock:
        return jsonify(success=True, **settings)


@app.route("/status")
def status():
    with state_lock:
        return jsonify({**video_state, "is_paused": is_paused, "count_in": count_in,
                        "count_out": count_out, "source": source_label(source),
                        "source_kind": source_kind(source)})


@app.route("/playback", methods=["POST"])
def playback():
    global is_paused
    action = (request.get_json(silent=True) or {}).get("action")
    if action not in ("play", "stop"):
        return jsonify(error="Action must be 'play' or 'stop'"), 400
    with state_lock:
        if not video_state["is_file"]:
            return jsonify(error="Playback controls require a video file"), 400
        if action == "play" and video_state["current_time"] >= video_state["duration"] - 0.05:
            video_state["seek_to"] = 0.0      # replay from the start
        is_paused = action == "stop"
    return jsonify(ok=True, is_paused=is_paused)


@app.route("/seek", methods=["POST"])
def seek():
    try:
        seconds = float((request.get_json(silent=True) or {})["seconds"])
    except (KeyError, TypeError, ValueError):
        return jsonify(error="Invalid seek position"), 400
    with state_lock:
        if not video_state["is_file"]:
            return jsonify(error="Seeking requires a video file"), 400
        video_state["seek_to"] = max(0.0, min(video_state["duration"], seconds))
    return jsonify(ok=True)


@app.route("/process/start", methods=["POST"])
def start_process():
    global stream_generation
    mode = (request.get_json(silent=True) or {}).get("mode", "fast")
    if mode not in COUNT_MODES:
        return jsonify(success=False, error="mode must be 'fast' or 'accurate'"), 400
    with state_lock:
        if process_state["running"]:
            return jsonify(success=False, error="Counting is already running"), 409
        if source_kind(source) != "file":
            return jsonify(success=False, error="Fast count requires a video file"), 400
        cfg = dict(settings)
        with line_lock:
            line_norm = [p[:] for p in LINE_NORM]
        process_cancel_event.clear()
        process_state.update({"running": True, "done": False, "cancelled": False, "progress": 0.0,
                              "frame": 0, "total_frames": 0, "status": "Starting", "error": None,
                              "count_in": 0, "count_out": 0, "fps": 0.0, "eta_seconds": None,
                              "elapsed_seconds": 0.0, "events": [], "source": source_label(source),
                              "mode": mode})
        stream_generation += 1               # stop the live view: all CPU goes to counting
        path = source
    threading.Thread(target=_process_video_worker, args=(path, cfg, line_norm, COUNT_MODES[mode]),
                     daemon=True).start()
    return jsonify(success=True)


@app.route("/process/cancel", methods=["POST"])
def cancel_process():
    process_cancel_event.set()
    return jsonify(success=True)


@app.route("/process/status")
def process_status():
    with state_lock:
        return jsonify(dict(process_state))


@app.route("/process/return", methods=["POST"])
def return_from_process():
    """Leave the results screen and restart the live view of the same source."""
    with state_lock:
        if process_state["running"]:
            return jsonify(success=False, error="Counting is still running"), 409
        src = source
    _set_source(src)
    return jsonify(success=True)


# ----------------------------------------------------------------------- main
def _open_browser_when_ready(url, timeout=60.0):
    """Open the UI in the default browser as soon as the server answers."""
    import urllib.request
    import webbrowser

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(url + "line", timeout=1)
            break
        except OSError:
            time.sleep(0.3)
    webbrowser.open(url)


@app.route("/shutdown", methods=["POST"])
def shutdown():
    """Quit button: stop the server (it runs without a console window)."""
    process_cancel_event.set()
    threading.Timer(0.5, os._exit, args=(0,)).start()
    return jsonify(success=True)


def _already_running(url):
    import urllib.request
    try:
        urllib.request.urlopen(url + "line", timeout=1)
        return True
    except OSError:
        return False


if __name__ == "__main__":
    import sys
    import webbrowser

    URL = "http://127.0.0.1:5000/"
    if sys.stdout is None or sys.stderr is None:
        # started with pythonw (no console): keep the output in a log file
        log = open(ROOT / "people_counter.log", "a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stderr = log
    if _already_running(URL):
        # second start: don't launch another server, just show the open one
        if "--open-browser" in sys.argv:
            webbrowser.open(URL)
        sys.exit(0)
    if "--open-browser" in sys.argv:
        threading.Thread(target=_open_browser_when_ready, args=(URL,), daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True, use_reloader=False)
