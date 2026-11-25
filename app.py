
import os
import json
import threading
import time
from collections import deque
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np
from flask import Flask, Response, render_template, request, jsonify

from counting import IN, LineCounter

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


app = Flask(__name__)

CONFIG_FILE = Path(__file__).with_name("line_config.json")
SETTINGS_FILE = Path(__file__).with_name("people_counter_settings.json")
TRACKER_FILE = Path(__file__).with_name("people_tracker.yaml")

DEFAULT_LINE = [[0.15, 0.5], [0.85, 0.5]]

state_lock = threading.Lock()
line_lock = threading.Lock()
inference_lock = threading.Lock()
process_cancel_event = threading.Event()
process_thread = None
process_state = {
    "running": False, "done": False, "cancelled": False, "progress": 0.0,
    "frame": 0, "total_frames": 0, "status": "idle", "error": None,
    "count_in": 0, "count_out": 0,
}

LINE_NORM = DEFAULT_LINE
count_in = 0
count_out = 0
selected_video = None
stream_generation = 0
is_paused = False
show_ghost_ids = True
ghost_timeout_seconds = 5
size_filter_enabled = True
show_invalid_size_boxes = True
size_tolerance_cm = 75
reference_line_cm = 170

video_state = {
    "duration": 0.0,
    "current_time": 0.0,
    "seek_to": None,
    "is_file": False,
    "error": None,
}


def load_line():
    try:
        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list) and len(data) == 2:
            points = [[float(v) for v in p] for p in data]
            if all(
                len(p) == 2 and all(0 <= v <= 1 for v in p)
                for p in points
            ):
                return points
    except (OSError, ValueError, TypeError):
        pass

    return [p[:] for p in DEFAULT_LINE]


LINE_NORM = load_line()


def load_counter_settings():
    global size_filter_enabled, show_invalid_size_boxes
    global size_tolerance_cm, reference_line_cm, ghost_timeout_seconds
    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            size_filter_enabled = bool(data.get("size_filter_enabled", True))
            show_invalid_size_boxes = bool(data.get("show_invalid_size_boxes", True))
            size_tolerance_cm = max(10, min(100, int(data.get("size_tolerance_cm", 75))))
            reference_line_cm = max(50, min(300, int(data.get("reference_line_cm", 170))))
            ghost_timeout_seconds = max(1, min(20, int(data.get("ghost_timeout_seconds", 5))))
    except (OSError, ValueError, TypeError):
        pass


def save_counter_settings():
    payload = {
        "size_filter_enabled": size_filter_enabled,
        "show_invalid_size_boxes": show_invalid_size_boxes,
        "size_tolerance_cm": size_tolerance_cm,
        "reference_line_cm": reference_line_cm,
        "ghost_timeout_seconds": ghost_timeout_seconds,
    }
    SETTINGS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


load_counter_settings()


def get_source():
    source = os.getenv("VIDEO_SOURCE", "0")
    try:
        return int(source)
    except ValueError:
        return source


def load_model():
    if YOLO is None:
        return None

    # Override with e.g. PEOPLE_COUNTER_MODEL=yolo11s.pt (more accurate, ~2x slower on CPU).
    return YOLO(os.getenv("PEOPLE_COUNTER_MODEL", MODEL_NAME))


# Detection settings shared by the live stream and offline processing. They were
# tuned on a hand-labelled 10-minute entrance video (see README.md, "Accuracy").
MODEL_NAME = "yolov8n.pt"
DETECTION_CONF = 0.25


def make_line_counter(width, height):
    """Line-crossing counter with thresholds scaled to the frame size."""
    scale = max(width, height) / 1280.0
    return LineCounter(
        (0, 0), (1, 1),
        min_offset_px=15.0 * scale,
        lost_near_px=60.0 * scale,
        cancel_window_s=1.5,
        lost_predict_s=0.7,
    )


def counter_feed(detections, p1, p2, width, height, settings):
    """(track_id, box, countable) tuples for LineCounter.update.

    Only real tracker IDs are passed; detections without an ID (negative
    synthetic IDs) are never counted. The height filter, when enabled, decides
    whether a crossing may be counted, but the person is still tracked.
    """
    feed = []
    for item in detections:
        if item["raw_id"] < 0:
            continue
        countable = _size_is_plausible(
            item["box"], p1, p2, width, height,
            enabled=settings["size_filter_enabled"],
            tolerance_cm=settings["size_tolerance_cm"],
            reference_cm=settings["reference_line_cm"],
        )
        feed.append((item["raw_id"], item["box"], countable))
    return feed


def line_side(point, p1, p2):
    return (
        (p2[0] - p1[0]) * (point[1] - p1[1])
        - (p2[1] - p1[1]) * (point[0] - p1[0])
    )


def draw_line_and_labels(frame, p1, p2):
    height, width = frame.shape[:2]

    cv2.line(frame, p1, p2, (0, 255, 0), 3)

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = float(np.hypot(dx, dy))

    if length < 1:
        return

    midpoint = (
        (p1[0] + p2[0]) // 2,
        (p1[1] + p2[1]) // 2,
    )

    nx = -dy / length
    ny = dx / length
    offset = 48

    positive = (
        int(midpoint[0] + nx * offset),
        int(midpoint[1] + ny * offset),
    )
    negative = (
        int(midpoint[0] - nx * offset),
        int(midpoint[1] - ny * offset),
    )

    cv2.arrowedLine(
        frame, midpoint, positive, (0, 255, 0), 2,
        tipLength=0.25
    )
    cv2.arrowedLine(
        frame, midpoint, negative, (0, 165, 255), 2,
        tipLength=0.25
    )

    def draw_label(text, point, color):
        x = max(5, min(width - 65, point[0] + 6))
        y = max(25, min(height - 8, point[1] - 6))

        cv2.putText(
            frame, text, (x, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
            cv2.LINE_AA
        )

    draw_label("IN", positive, (0, 255, 0))
    draw_label("OUT", negative, (0, 165, 255))



def _center(box):
    x1, y1, x2, y2 = map(float, box)
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=float)


def _box_diagonal(box):
    x1, y1, x2, y2 = map(float, box)
    return float(np.hypot(max(1.0, x2 - x1), max(1.0, y2 - y1)))


def _distance_to_segment(point, p1, p2):
    """Shortest pixel distance from a point to the finite counting-line segment."""
    point = np.asarray(point, dtype=float)
    start = np.asarray(p1, dtype=float)
    end = np.asarray(p2, dtype=float)
    segment = end - start
    length_sq = float(np.dot(segment, segment))
    if length_sq <= 1e-9:
        return float(np.linalg.norm(point - start))
    fraction = float(np.dot(point - start, segment) / length_sq)
    fraction = max(0.0, min(1.0, fraction))
    closest = start + fraction * segment
    return float(np.linalg.norm(point - closest))


def _distance_from_counting_line_is_plausible(box, p1, p2, frame_width, frame_height):
    """Keep detections within half the larger frame dimension of the line."""
    center = _center(box)
    max_distance = 0.5 * max(float(frame_width), float(frame_height))
    return _distance_to_segment(center, p1, p2) <= max_distance


def _outside_fraction(box, frame_width, frame_height, reference_box=None):
    """Return the fraction of an estimated full hitbox outside the frame.

    Detector boxes are clipped at image boundaries. When a previous box is
    available, use its dimensions to estimate the clipped-away part; otherwise
    use the supplied box dimensions. The returned value is always in [0, 1].
    """
    x1, y1, x2, y2 = map(float, box)
    frame_width = max(1.0, float(frame_width))
    frame_height = max(1.0, float(frame_height))

    visible_x1, visible_y1 = max(0.0, x1), max(0.0, y1)
    visible_x2, visible_y2 = min(frame_width, x2), min(frame_height, y2)
    visible_w = max(0.0, visible_x2 - visible_x1)
    visible_h = max(0.0, visible_y2 - visible_y1)

    full_x1, full_y1, full_x2, full_y2 = x1, y1, x2, y2
    if reference_box is not None:
        rx1, ry1, rx2, ry2 = map(float, reference_box)
        ref_w = max(1.0, rx2 - rx1)
        ref_h = max(1.0, ry2 - ry1)
        # If the observed box touches an edge, estimate its missing part using
        # the previous hitbox size, while preserving the visible edge location.
        if x1 <= 1.0:
            full_x1, full_x2 = x1, x1 + ref_w
        elif x2 >= frame_width - 1.0:
            full_x1, full_x2 = x2 - ref_w, x2
        if y1 <= 1.0:
            full_y1, full_y2 = y1, y1 + ref_h
        elif y2 >= frame_height - 1.0:
            full_y1, full_y2 = y2 - ref_h, y2

    full_w = max(1.0, full_x2 - full_x1)
    full_h = max(1.0, full_y2 - full_y1)
    full_area = full_w * full_h
    visible_area = min(full_area, visible_w * visible_h)
    return max(0.0, min(1.0, 1.0 - visible_area / full_area))


def _size_is_plausible(
    box, p1, p2, frame_width=None, frame_height=None,
    enabled=True, tolerance_cm=75, reference_cm=170,
):
    """Check box height against the configured real-world length of the line.

    The pixel length of the counting line is the scale reference. A detection is
    accepted when its height corresponds to reference_cm +/- tolerance_cm.
    If filtering is disabled, all sizes are accepted.
    """
    if not enabled:
        return True

    x1, y1, x2, y2 = map(float, box)
    box_height = max(0.0, y2 - y1)
    line_pixels = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
    reference_cm = max(1.0, float(reference_cm))
    tolerance_cm = max(0.0, min(100.0, float(tolerance_cm)))

    # Do not apply a meaningless calibration before the line is configured.
    if line_pixels < 8.0 or box_height < 1.0:
        return True
    if frame_width is not None and frame_height is not None:
        frame_diagonal = float(np.hypot(frame_width, frame_height))
        if line_pixels > 0.45 * frame_diagonal:
            return True

    apparent_height_cm = box_height / line_pixels * reference_cm
    min_height_cm = max(1.0, reference_cm - tolerance_cm)
    max_height_cm = reference_cm + tolerance_cm
    return min_height_cm <= apparent_height_cm <= max_height_cm


def _kalman_predict(track, motion_time):
    """Constant-velocity Kalman prediction for the center point [x, y, vx, vy]."""
    state = np.asarray(track.get("kf_state", [*track["center"], *track.get("velocity", [0.0, 0.0])]), dtype=float).reshape(4)
    covariance = np.asarray(track.get("kf_cov", np.diag([36.0, 36.0, 900.0, 900.0])), dtype=float).reshape(4, 4)
    elapsed = max(0.0, min(20.0, motion_time - track.get("last_motion_time", motion_time)))

    f = np.array([
        [1.0, 0.0, elapsed, 0.0],
        [0.0, 1.0, 0.0, elapsed],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=float)
    # White-noise acceleration model: uncertainty grows with the prediction horizon.
    q = 220.0
    dt2, dt3, dt4 = elapsed**2, elapsed**3, elapsed**4
    q_block = np.array([
        [dt4 / 4.0, dt3 / 2.0],
        [dt3 / 2.0, dt2],
    ], dtype=float) * (q**2)
    process_noise = np.zeros((4, 4), dtype=float)
    process_noise[np.ix_([0, 2], [0, 2])] = q_block
    process_noise[np.ix_([1, 3], [1, 3])] = q_block

    predicted_state = f @ state
    predicted_cov = f @ covariance @ f.T + process_noise
    predicted_cov = (predicted_cov + predicted_cov.T) * 0.5
    return predicted_state, predicted_cov, elapsed


def _kalman_update(old_track, observed_center, motion_time, box):
    """Update the motion estimate while adapting measurement noise to detector jitter."""
    state, covariance, _ = _kalman_predict(old_track, motion_time)
    z = np.asarray(observed_center, dtype=float).reshape(2)
    h = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=float)

    # Bounding-box centers jitter by a few pixels. Larger boxes generally have
    # noisier centers, so measurement covariance scales mildly with box size.
    diagonal = _box_diagonal(box)
    sigma = max(3.0, min(11.0, diagonal * 0.035))
    r = np.eye(2, dtype=float) * (sigma**2)
    innovation = z - h @ state
    s = h @ covariance @ h.T + r
    try:
        mahalanobis = float(innovation.T @ np.linalg.solve(s, innovation))
    except np.linalg.LinAlgError:
        mahalanobis = 0.0

    # A large one-frame center jump is more likely detector jitter or an ID switch.
    # Down-weight that measurement rather than letting it instantly rotate velocity.
    if mahalanobis > 9.0:
        r *= min(10.0, mahalanobis / 9.0)
        s = h @ covariance @ h.T + r
    try:
        gain = covariance @ h.T @ np.linalg.inv(s)
    except np.linalg.LinAlgError:
        gain = covariance @ h.T @ np.linalg.pinv(s)
    updated_state = state + gain @ innovation
    identity = np.eye(4, dtype=float)
    updated_cov = (identity - gain @ h) @ covariance @ (identity - gain @ h).T + gain @ r @ gain.T
    updated_cov = (updated_cov + updated_cov.T) * 0.5
    return updated_state, updated_cov


def _robust_velocity(history, fallback=None, max_speed=500.0):
    """Estimate image-plane velocity from robust pairwise slopes in seconds."""
    points = list(history or [])
    slopes = []
    for i in range(len(points)):
        ti, pi = points[i]
        for j in range(i + 1, len(points)):
            tj, pj = points[j]
            dt = float(tj) - float(ti)
            if 0.035 <= dt <= 0.85:
                delta = (np.asarray(pj, dtype=float) - np.asarray(pi, dtype=float)) / dt
                if np.all(np.isfinite(delta)):
                    slopes.append(delta)
    if slopes:
        velocity = np.median(np.stack(slopes), axis=0)
    elif fallback is not None:
        velocity = np.asarray(fallback, dtype=float).reshape(2).copy()
    else:
        velocity = np.zeros(2, dtype=float)
    speed = float(np.linalg.norm(velocity))
    if not np.isfinite(speed):
        return np.zeros(2, dtype=float)
    if speed > max_speed:
        velocity *= max_speed / max(speed, 1e-6)
    return velocity


def _predict_track(track, motion_time):
    """Predict a lost hitbox using seconds, not processed-frame count.

    The robust velocity is used for the first 1.2 seconds. Longer gaps gradually
    damp displacement so a stale vector cannot send a ghost racing across the frame.
    """
    last_time = float(track.get("last_motion_time", motion_time))
    elapsed = max(0.0, float(motion_time) - last_time)
    velocity = np.asarray(
        track.get("velocity", np.asarray(track.get("kf_state", [0, 0, 0, 0]), dtype=float)[2:]),
        dtype=float,
    ).reshape(2)
    # Preserve near-term linear motion; damp only after a sustained detector gap.
    effective_elapsed = min(elapsed, 1.2) + 2.0 * (1.0 - np.exp(-max(0.0, elapsed - 1.2) / 2.0))
    base_center = np.asarray(track.get("center", _center(track["box"])), dtype=float).reshape(2)
    center = base_center + velocity * effective_elapsed
    box = np.asarray(track["box"], dtype=float).copy()
    old_center = _center(box)
    shift = center - old_center
    box[[0, 2]] += shift[0]
    box[[1, 3]] += shift[1]
    return center, box, elapsed


def _box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = iw * ih
    area_a = max(1.0, ax2 - ax1) * max(1.0, ay2 - ay1)
    area_b = max(1.0, bx2 - bx1) * max(1.0, by2 - by1)
    return intersection / max(1.0, area_a + area_b - intersection)


def _association_score(track, detection_box, motion_time):
    """Fixed-cost re-identification using Kalman prediction, overlap, size and soft direction."""
    predicted_state, predicted_cov, elapsed = _kalman_predict(track, motion_time)
    predicted_center = predicted_state[:2]
    detection_center = _center(detection_box)
    error_vector = detection_center - predicted_center
    distance = float(np.linalg.norm(error_vector))
    velocity = predicted_state[2:]
    speed = float(np.linalg.norm(velocity))
    box_scale = (_box_diagonal(track["box"]) + _box_diagonal(detection_box)) / 2.0
    pos_cov = predicted_cov[:2, :2]
    sigma = float(np.sqrt(max(1.0, np.max(np.linalg.eigvalsh(pos_cov)))))

    # A hard spatial gate prevents a nearby-but-unrelated person from inheriting a ghost.
    gate = min(220.0, max(48.0, 0.62 * box_scale + 0.35 * speed * min(elapsed, 1.5) + 2.2 * sigma))
    if distance > gate:
        return None

    predicted_box = np.asarray(track["box"], dtype=float).copy()
    old_box_center = _center(predicted_box)
    predicted_box[[0, 2]] += predicted_center[0] - old_box_center[0]
    predicted_box[[1, 3]] += predicted_center[1] - old_box_center[1]
    overlap_penalty = 1.0 - _box_iou(predicted_box, detection_box)

    direction_penalty = 0.0
    observed_displacement = detection_center - np.asarray(track["center"], dtype=float)
    displacement_length = float(np.linalg.norm(observed_displacement))
    if speed >= 12.0 and displacement_length >= 12.0:
        cosine = float(np.dot(velocity, observed_displacement) / max(1e-6, speed * displacement_length))
        cosine = max(-1.0, min(1.0, cosine))
        # Direction is a soft cue, not a veto: people can turn during occlusion.
        direction_penalty = (1.0 - cosine) * 0.5

    old_w = max(1.0, float(track["box"][2] - track["box"][0]))
    old_h = max(1.0, float(track["box"][3] - track["box"][1]))
    new_w = max(1.0, float(detection_box[2] - detection_box[0]))
    new_h = max(1.0, float(detection_box[3] - detection_box[1]))
    size_penalty = min(1.0, abs(float(np.log(max(1e-6, (new_w * new_h) / (old_w * old_h))))) / 2.2)

    return (distance / gate) + 0.22 * overlap_penalty + 0.18 * direction_penalty + 0.08 * size_penalty


def _missing_age(track, now, motion_time, is_file):
    """Use video timeline for files and wall-clock time for live cameras."""
    if is_file:
        return max(0.0, motion_time - track.get("last_motion_time", motion_time))
    return max(0.0, now - track.get("last_seen", now))


def gen():
    global count_in, count_out, is_paused

    with state_lock:
        source = selected_video if selected_video else get_source()
        my_generation = stream_generation

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        with state_lock:
            video_state["error"] = "Cannot open video source"
        cap.release()
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    is_file = isinstance(source, str) and frame_count > 0 and fps > 0
    duration = frame_count / fps if is_file else 0.0

    with state_lock:
        video_state.update({
            "duration": duration,
            "current_time": 0.0,
            "seek_to": None,
            "is_file": is_file,
            "error": None,
            "is_paused": False,
        })
        is_paused = False

    model = load_model()
    line_counter = None
    # Stable IDs are independent of Ultralytics' tracker IDs.
    raw_to_stable = {}
    tracks = {}
    next_stable_id = 1
    tracker_config = (
        str(TRACKER_FILE)
        if TRACKER_FILE.exists()
        else "bytetrack.yaml"
    )
    paused_since = None
    next_synthetic_raw_id = -1
    terminal_raw_ids = set()

    try:
        while True:
            with state_lock:
                if my_generation != stream_generation:
                    break
                paused = is_paused

            if paused:
                if paused_since is None:
                    paused_since = time.monotonic()
                time.sleep(0.05)
                continue
            elif paused_since is not None:
                # Pausing the video must not age lost tracks in the background.
                paused_duration = time.monotonic() - paused_since
                if not is_file:
                    for track in tracks.values():
                        track["last_seen"] += paused_duration
                        track["last_motion_time"] += paused_duration
                # File motion_time comes from the media timeline, which does not advance while paused.
                paused_since = None

            seek_to = None
            with state_lock:
                if video_state["seek_to"] is not None:
                    seek_to = video_state["seek_to"]
                    video_state["seek_to"] = None

            if seek_to is not None and is_file:
                seek_to = max(0.0, min(duration, seek_to))
                cap.set(cv2.CAP_PROP_POS_MSEC, seek_to * 1000)
                raw_to_stable.clear()
                tracks.clear()
                next_stable_id = 1
                terminal_raw_ids.clear()
                if line_counter is not None:
                    line_counter.reset()

                with state_lock:
                    count_in = 0
                    count_out = 0
                    video_state["current_time"] = seek_to

                if model is not None:
                    model = load_model()

            ok, frame = cap.read()
            if not ok:
                if is_file:
                    with state_lock:
                        video_state["current_time"] = duration
                        video_state["is_paused"] = True
                        is_paused = True
                break

            height, width = frame.shape[:2]
            now = time.monotonic()

            if is_file:
                current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                with state_lock:
                    video_state["current_time"] = current_time
                motion_time = current_time
            else:
                motion_time = now

            with line_lock:
                norm = [p[:] for p in LINE_NORM]

            p1 = (int(norm[0][0] * width), int(norm[0][1] * height))
            p2 = (int(norm[1][0] * width), int(norm[1][1] * height))
            # Keep an untouched copy for the detector: the overlay (line, arrows,
            # labels) is drawn exactly where people cross and would hide them.
            clean_frame = frame.copy()
            draw_line_and_labels(frame, p1, p2)

            with state_lock:
                frame_size_filter_enabled = size_filter_enabled
                frame_show_invalid_boxes = show_invalid_size_boxes
                frame_size_tolerance_cm = size_tolerance_cm
                frame_reference_line_cm = reference_line_cm
                frame_ghost_timeout_seconds = ghost_timeout_seconds

            detections = []
            invalid_size_detections = []
            if model is not None:
                with inference_lock:
                    results = model.track(
                        clean_frame,
                        persist=True,
                        tracker=tracker_config,
                        classes=[0],
                        conf=DETECTION_CONF,
                        imgsz=640,
                        verbose=False,
                    )
                for result in results:
                    if result.boxes is None or len(result.boxes) == 0:
                        continue
                    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
                    if result.boxes.id is None:
                        raw_ids = [None] * len(boxes)
                    else:
                        raw_ids = result.boxes.id.int().cpu().tolist()
                    for box, raw_id in zip(boxes, raw_ids):
                        # Some detections can exist before/without a tracker ID.
                        # Give them a temporary unique ID so our own association
                        # logic can reconnect them to a stable track or ghost.
                        if raw_id is None:
                            raw_id = next_synthetic_raw_id
                            next_synthetic_raw_id -= 1
                        detections.append({
                            "raw_id": int(raw_id),
                            "box": box.copy(),
                        })

            # Counting: one LineCounter step per frame on the tracker's own IDs.
            # It sees every detection (including people already counted IN, so a
            # person who immediately walks back out cancels the IN).
            if line_counter is None:
                line_counter = make_line_counter(width, height)
            line_counter.set_line(p1, p2)
            fired = line_counter.update(motion_time, counter_feed(
                detections, p1, p2, width, height, {
                    "size_filter_enabled": frame_size_filter_enabled,
                    "size_tolerance_cm": frame_size_tolerance_cm,
                    "reference_line_cm": frame_reference_line_cm,
                },
            ))
            for raw_id, direction in fired:
                if direction == IN:
                    # Entered people disappear from the live overlay and cannot
                    # spawn a ghost or reappear under a newly allocated stable ID.
                    terminal_raw_ids.add(raw_id)
                    entered_track = tracks.get(raw_to_stable.get(raw_id))
                    if entered_track is not None:
                        entered_track["crossed_line"] = True
            with state_lock:
                count_in = line_counter.count_in
                count_out = line_counter.count_out

            # Hide detections whose identity has already crossed IN. Keep the raw ID
            # reserved so the same person does not reappear with a new visible ID.
            detections = [d for d in detections if d["raw_id"] not in terminal_raw_ids]

            # Only discard detections that are substantially outside the image.
            # Height/distance checks mark detections as out of range, but keep them
            # in tracking so temporary perspective changes do not break IDs/ghosts.
            kept_detections = []
            for item in detections:
                mapped_id = raw_to_stable.get(item["raw_id"])
                reference_track = tracks.get(mapped_id) if mapped_id is not None else None
                reference_box = reference_track.get("box") if reference_track else None
                outside = _outside_fraction(item["box"], width, height, reference_box) >= 0.30
                if outside:
                    if mapped_id is not None:
                        tracks.pop(mapped_id, None)
                    raw_to_stable = {
                        raw_id: stable_id for raw_id, stable_id in raw_to_stable.items()
                        if stable_id != mapped_id and raw_id != item["raw_id"]
                    }
                    continue

                size_ok = _size_is_plausible(
                    item["box"], p1, p2, width, height,
                    enabled=frame_size_filter_enabled,
                    tolerance_cm=frame_size_tolerance_cm,
                    reference_cm=frame_reference_line_cm,
                )
                # The height-filter checkbox controls all range-based filtering,
                # including the distance-from-line gate.
                distance_ok = (
                    not frame_size_filter_enabled
                    or _distance_from_counting_line_is_plausible(
                        item["box"], p1, p2, width, height
                    )
                )
                in_range = (not frame_size_filter_enabled) or (size_ok and distance_ok)
                item["in_range"] = in_range
                if frame_size_filter_enabled and not in_range:
                    invalid_size_detections.append({
                        "box": item["box"].copy(),
                        "id": item["raw_id"],
                    })
                kept_detections.append(item)
            detections = kept_detections

            # A raw tracker ID that returns after a gap must be re-validated.
            for item in detections:
                raw_id = item["raw_id"]
                mapped_id = raw_to_stable.get(raw_id)
                mapped_track = tracks.get(mapped_id) if mapped_id is not None else None
                if mapped_track is not None and not mapped_track.get("crossed_line", False) and (
                    _missing_age(mapped_track, now, motion_time, is_file) > 0.50
                ):
                    raw_to_stable.pop(raw_id, None)

            active_raw_ids = {item["raw_id"] for item in detections}

            # Validate raw-ID mappings against recent motion. A stable identity can be
            # claimed by at most one detection in a frame; duplicate raw mappings are
            # released and resolved through the one-to-one association below.
            unsuppressed = []
            claimed_mapped_ids = set()
            active_window = max(0.12, 1.5 / fps) if is_file and fps > 0 else 0.18
            for item in detections:
                center = _center(item["box"])
                raw_id = item["raw_id"]
                mapped_id = raw_to_stable.get(raw_id)
                mapped_track = tracks.get(mapped_id) if mapped_id is not None else None
                if mapped_track is not None and mapped_id not in claimed_mapped_ids:
                    mapped_age = _missing_age(mapped_track, now, motion_time, is_file)
                    mapped_prediction, _, _ = _predict_track(mapped_track, motion_time)
                    mapped_distance = float(np.linalg.norm(center - mapped_prediction))
                    mapped_gate = max(44.0, 0.65 * _box_diagonal(item["box"]))
                    if mapped_age <= active_window and mapped_distance <= mapped_gate:
                        unsuppressed.append(item)
                        claimed_mapped_ids.add(mapped_id)
                        continue
                raw_to_stable.pop(raw_id, None)
                unsuppressed.append(item)
            detections = unsuppressed
            active_raw_ids = {item["raw_id"] for item in detections}

            expired = [
                stable_id for stable_id, track in tracks.items()
                if _missing_age(track, now, motion_time, is_file) > frame_ghost_timeout_seconds
                and stable_id not in {
                    raw_to_stable.get(raw_id) for raw_id in active_raw_ids
                }
            ]
            for stable_id in expired:
                tracks.pop(stable_id, None)
                raw_to_stable = {
                    raw_id: mapped_id
                    for raw_id, mapped_id in raw_to_stable.items()
                    if mapped_id != stable_id
                }

            # Associate only tracker IDs that have not already been assigned a stable ID.
            new_detections = [
                item for item in detections
                if item["raw_id"] not in raw_to_stable
            ]
            lost_tracks = {
                stable_id: track for stable_id, track in tracks.items()
                if not track.get("crossed_line", False)
                and _missing_age(track, now, motion_time, is_file) <= frame_ghost_timeout_seconds
                and stable_id not in {
                    raw_to_stable.get(raw_id) for raw_id in active_raw_ids
                }
            }

            candidate_pairs = []
            for index, item in enumerate(new_detections):
                for stable_id, track in lost_tracks.items():
                    score = _association_score(track, item["box"], motion_time)
                    if score is not None and score <= 1.22:
                        candidate_pairs.append((score, index, stable_id))

            # Resolve candidates by global sorted cost with one-to-one constraints.
            used_detections = set()
            used_tracks = set()
            for score, index, stable_id in sorted(candidate_pairs):
                if index in used_detections or stable_id in used_tracks:
                    continue
                raw_id = new_detections[index]["raw_id"]
                raw_to_stable = {
                    old_raw: old_stable for old_raw, old_stable in raw_to_stable.items()
                    if old_stable != stable_id and old_raw != raw_id
                }
                raw_to_stable[raw_id] = stable_id
                used_detections.add(index)
                used_tracks.add(stable_id)

            # Give unmatched tracker IDs new stable IDs.
            for item in detections:
                raw_id = item["raw_id"]
                if raw_id not in raw_to_stable:
                    raw_to_stable[raw_id] = next_stable_id
                    next_stable_id += 1

            # Keep only current raw-ID mappings and guarantee a one-to-one map for this frame.
            frame_mapping = {}
            used_stable_in_frame = set()
            for item in detections:
                raw_id = item["raw_id"]
                stable_id = raw_to_stable[raw_id]
                if stable_id in used_stable_in_frame:
                    # This should be rare after association; allocate a fresh identity rather
                    # than drawing duplicate IDs on two simultaneous people.
                    stable_id = next_stable_id
                    next_stable_id += 1
                    raw_to_stable[raw_id] = stable_id
                used_stable_in_frame.add(stable_id)
                frame_mapping[raw_id] = stable_id
            raw_to_stable = frame_mapping

            active_ids = set()
            for item in detections:
                raw_id = item["raw_id"]
                stable_id = raw_to_stable[raw_id]
                active_ids.add(stable_id)
                box = item["box"]
                center = _center(box)

                old_track = tracks.get(stable_id)
                if old_track is not None:
                    kf_state, kf_cov = _kalman_update(old_track, center, motion_time, box)
                    filtered_center = kf_state[:2].copy()
                    velocity = kf_state[2:].copy()
                    velocity_history = old_track.get("velocity_history", deque(maxlen=7))
                    motion_history = old_track.get("motion_history", deque(maxlen=9))
                    first_seen = old_track.get("first_seen", now)
                else:
                    filtered_center = center.copy()
                    velocity = np.zeros(2, dtype=float)
                    kf_state = np.array([center[0], center[1], 0.0, 0.0], dtype=float)
                    scale = max(3.0, min(11.0, _box_diagonal(box) * 0.035))
                    kf_cov = np.diag([scale**2, scale**2, 900.0, 900.0]).astype(float)
                    velocity_history = deque(maxlen=7)
                    motion_history = deque(maxlen=9)
                    first_seen = now

                # Estimate motion from observed positions in seconds. Median pairwise
                # slopes suppress box jitter and occasional one-frame detector jumps.
                if not motion_history or motion_time > motion_history[-1][0] + 1e-6:
                    motion_history.append((float(motion_time), center.copy()))
                robust_velocity = _robust_velocity(
                    motion_history,
                    fallback=kf_state[2:],
                    max_speed=max(140.0, min(600.0, 3.5 * _box_diagonal(box))),
                )
                if len(motion_history) >= 3:
                    velocity = 0.62 * robust_velocity + 0.38 * np.asarray(kf_state[2:], dtype=float)
                    speed = float(np.linalg.norm(velocity))
                    speed_cap = max(140.0, min(600.0, 3.5 * _box_diagonal(box)))
                    if speed > speed_cap:
                        velocity *= speed_cap / max(speed, 1e-6)
                    kf_state[2:] = velocity
                else:
                    velocity = np.asarray(kf_state[2:], dtype=float).copy()
                velocity_history.append(velocity.copy())

                tracks[stable_id] = {
                    "box": box.astype(float).copy(),
                    "center": filtered_center,
                    "observed_center": center.copy(),
                    "velocity": velocity,
                    "kf_state": kf_state,
                    "kf_cov": kf_cov,
                    "velocity_history": velocity_history,
                    "motion_history": motion_history,
                    "position_uncertainty": float(np.sqrt(max(1.0, np.max(np.linalg.eigvalsh(kf_cov[:2, :2]))))),
                    "first_seen": first_seen,
                    "crossed_line": bool(old_track.get("crossed_line", False)) if old_track else False,
                    "last_seen": now,
                    "last_motion_time": motion_time,
                }

                x1, y1, x2, y2 = map(int, box)
                cx, cy = map(int, center)
                if tracks.get(stable_id, {}).get("crossed_line", False):
                    continue

                # A green out-of-range box replaces the blue box, but keeps the same
                # stable ID. Never draw two labels/boxes for one detection.
                show_as_out_of_range = (
                    frame_size_filter_enabled and frame_show_invalid_boxes
                    and not item.get("in_range", True)
                )
                if show_as_out_of_range:
                    color = (0, 255, 0)
                    label = f"OUT OF RANGE: ID {stable_id}"
                    thickness = 2
                    text_scale = 0.5
                else:
                    color = (255, 0, 0)
                    label = f"ID {stable_id}"
                    thickness = 2
                    text_scale = 0.6
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(
                    frame, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, text_scale, color, 2,
                    cv2.LINE_AA,
                )

            # Optional ghost boxes: predicted positions of recently lost tracks.
            # They are visual only and do not create new tracking identities.
            with state_lock:
                ghosts_visible = show_ghost_ids
            if ghosts_visible:
                for ghost_id, ghost_track in list(tracks.items()):
                    if ghost_id in active_ids or ghost_track.get("crossed_line", False):
                        continue
                    ghost_age = _missing_age(ghost_track, now, motion_time, is_file)
                    if ghost_age <= 0 or ghost_age > frame_ghost_timeout_seconds:
                        continue
                    _, ghost_box, _ = _predict_track(ghost_track, motion_time)
                    gx1, gy1, gx2, gy2 = map(int, ghost_box)
                    # Skip only if the predicted box is entirely off-screen; retain the
                    # track internally so an OUT trajectory can reconnect to the same ID.
                    if gx2 < 0 or gy2 < 0 or gx1 >= width or gy1 >= height:
                        continue
                    gcx, gcy = map(int, _center(ghost_box))
                    cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (180, 180, 180), 2)
                    cv2.circle(frame, (gcx, gcy), 4, (180, 180, 180), -1)
                    cv2.putText(
                        frame, f"GHOST ID {ghost_id}",
                        (gx1, max(20, gy1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 2,
                        cv2.LINE_AA,
                    )

            with state_lock:
                current_in = count_in
                current_out = count_out

            cv2.putText(
                frame, f"IN: {current_in}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
            )
            cv2.putText(
                frame, f"OUT: {current_out}", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2,
            )


            ok, buffer = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
            )
            if not ok:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes() + b"\r\n"
            )
    finally:
        cap.release()


@app.route("/choose-video", methods=["POST"])
def choose_video():
    global selected_video, stream_generation, is_paused

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    try:
        path = filedialog.askopenfilename(
            parent=root,
            title="Select video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.webm"),
                ("All files", "*.*"),
            ],
        )
    finally:
        root.destroy()

    if not path:
        return jsonify({"success": False, "cancelled": True})

    with state_lock:
        selected_video = path
        stream_generation += 1
        video_state["seek_to"] = None
        video_state["is_paused"] = False
        is_paused = False

    return jsonify({"success": True})

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/stream")
def stream():
    return Response(
        gen(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache"},
    )


@app.route("/line", methods=["GET", "POST"])
def line_settings():
    global LINE_NORM

    if request.method == "GET":
        with line_lock:
            return jsonify(LINE_NORM)

    data = request.get_json(silent=True)

    try:
        if not isinstance(data, list) or len(data) != 2:
            raise ValueError

        points = [[float(p[0]), float(p[1])] for p in data]

        if any(
            not 0 <= value <= 1
            for point in points for value in point
        ):
            raise ValueError
    except (TypeError, ValueError, IndexError):
        return jsonify(error="Invalid line coordinates"), 400

    with line_lock:
        LINE_NORM = points
        CONFIG_FILE.write_text(
            json.dumps(LINE_NORM), encoding="utf-8"
        )

    return jsonify(ok=True, line=LINE_NORM)


@app.route("/ghost-visibility", methods=["GET", "POST"])
def ghost_visibility():
    global show_ghost_ids, ghost_timeout_seconds
    if request.method == "POST":
        payload = request.get_json(silent=True) or {}
        value = payload.get("visible")
        timeout_value = payload.get("timeout_seconds")
        if value is not None and not isinstance(value, bool):
            return jsonify({"success": False, "error": "visible must be a boolean"}), 400
        if timeout_value is not None:
            try:
                timeout_value = max(1, min(20, int(timeout_value)))
            except (ValueError, TypeError):
                return jsonify({"success": False, "error": "timeout_seconds must be from 1 to 20"}), 400
        with state_lock:
            if value is not None:
                show_ghost_ids = value
            if timeout_value is not None:
                ghost_timeout_seconds = timeout_value
                save_counter_settings()
    with state_lock:
        value = show_ghost_ids
        timeout_value = ghost_timeout_seconds
    return jsonify({"success": True, "visible": value, "timeout_seconds": timeout_value})


@app.route("/size-filter-settings", methods=["GET", "POST"])
def size_filter_settings():
    global size_filter_enabled, show_invalid_size_boxes
    global size_tolerance_cm, reference_line_cm

    if request.method == "POST":
        payload = request.get_json(silent=True) or {}
        with state_lock:
            try:
                if "size_filter_enabled" in payload:
                    if not isinstance(payload["size_filter_enabled"], bool):
                        raise ValueError
                    size_filter_enabled = payload["size_filter_enabled"]
                if "show_invalid_size_boxes" in payload:
                    if not isinstance(payload["show_invalid_size_boxes"], bool):
                        raise ValueError
                    show_invalid_size_boxes = payload["show_invalid_size_boxes"]
                if "size_tolerance_cm" in payload:
                    size_tolerance_cm = max(10, min(100, int(payload["size_tolerance_cm"])))
                if "reference_line_cm" in payload:
                    reference_line_cm = max(50, min(300, int(payload["reference_line_cm"])))
                if not size_filter_enabled:
                    show_invalid_size_boxes = False
                save_counter_settings()
            except (ValueError, TypeError):
                return jsonify({"success": False, "error": "Invalid size filter settings"}), 400

    with state_lock:
        return jsonify({
            "success": True,
            "size_filter_enabled": size_filter_enabled,
            "show_invalid_size_boxes": show_invalid_size_boxes,
            "size_tolerance_cm": size_tolerance_cm,
            "reference_line_cm": reference_line_cm,
        })


@app.route("/status")
def status():
    with state_lock:
        return jsonify({
            "duration": video_state["duration"],
            "current_time": video_state["current_time"],
            "is_file": video_state["is_file"],
            "error": video_state["error"],
            "is_paused": is_paused,
            "count_in": count_in,
            "count_out": count_out,
        })


@app.route("/playback", methods=["POST"])
def playback():
    global is_paused

    data = request.get_json(silent=True) or {}
    action = data.get("action")

    if action not in ("play", "stop"):
        return jsonify(error="Action must be 'play' or 'stop'"), 400

    with state_lock:
        if not video_state["is_file"]:
            return jsonify(error="Playback controls require a selected video file"), 400

        is_paused = (action == "stop")
        video_state["is_paused"] = is_paused

    return jsonify(ok=True, is_paused=is_paused)


@app.route("/seek", methods=["POST"])
def seek():
    data = request.get_json(silent=True) or {}

    try:
        seconds = float(data["seconds"])
    except (KeyError, TypeError, ValueError):
        return jsonify(error="Invalid seek position"), 400

    with state_lock:
        if not video_state["is_file"]:
            return jsonify(error="Seeking requires a video file"), 400

        seconds = max(0.0, min(video_state["duration"], seconds))
        video_state["seek_to"] = seconds

    return jsonify(ok=True, seconds=seconds)



def _process_video_worker(video_path, settings, line_norm):
    """Analyze a complete video as fast as possible, without rendering frames.

    Uses exactly the same detector, tracker and LineCounter as the live stream.
    """
    global count_in, count_out, is_paused
    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Could not open the selected video")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0:
            fps = 25.0
        if total_frames <= 0:
            raise RuntimeError("The video duration could not be read")

        model = load_model()
        if model is None:
            raise RuntimeError("Ultralytics/YOLO is not available")
        tracker_config = str(TRACKER_FILE) if TRACKER_FILE.exists() else "bytetrack.yaml"

        counter = None
        frame_no = 0
        with state_lock:
            process_state.update({"total_frames": total_frames, "frame": 0, "progress": 0.0, "status": "Processing video", "error": None})
            count_in = 0
            count_out = 0

        while not process_cancel_event.is_set():
            ok, frame = cap.read()
            if not ok:
                break
            frame_no += 1
            h, w = frame.shape[:2]
            p1 = (int(line_norm[0][0] * w), int(line_norm[0][1] * h))
            p2 = (int(line_norm[1][0] * w), int(line_norm[1][1] * h))
            if counter is None:
                counter = make_line_counter(w, h)
                counter.set_line(p1, p2)

            with inference_lock:
                results = model.track(
                    frame, persist=True, tracker=tracker_config, classes=[0],
                    conf=DETECTION_CONF, imgsz=640, verbose=False,
                )
            detections = []
            for result in results:
                if result.boxes is None or result.boxes.id is None:
                    continue
                boxes = result.boxes.xyxy.cpu().numpy().astype(float)
                for box, raw_id in zip(boxes, result.boxes.id.int().cpu().tolist()):
                    detections.append({"raw_id": int(raw_id), "box": box})
            counter.update((frame_no - 1) / fps, counter_feed(detections, p1, p2, w, h, settings))

            with state_lock:
                process_state["frame"] = frame_no
                process_state["progress"] = min(1.0, frame_no / max(1, total_frames))
                process_state["count_in"] = counter.count_in
                process_state["count_out"] = counter.count_out
                count_in = counter.count_in
                count_out = counter.count_out
                video_state["current_time"] = min(frame_no / fps, total_frames / fps)
                video_state["duration"] = total_frames / fps
                video_state["is_file"] = True

        if counter is not None:
            # Let tracks that vanished in the last frames finish (door occlusion).
            for k in range(1, int(2 * fps)):
                counter.update((frame_no + k) / fps, [])
        local_in = counter.count_in if counter else 0
        local_out = counter.count_out if counter else 0
        was_cancelled = process_cancel_event.is_set()
        with state_lock:
            process_state.update({
                "running": False,
                "done": not was_cancelled,
                "cancelled": was_cancelled,
                "progress": process_state.get("progress", 0.0) if was_cancelled else 1.0,
                "status": "Cancelled" if was_cancelled else "Completed",
                "count_in": local_in,
                "count_out": local_out,
                "events": [] if counter is None else [
                    {"t": round(e["t"], 2), "dir": e["dir"], "cancel": bool(e.get("cancel"))}
                    for e in counter.events
                ],
            })
            count_in = local_in
            count_out = local_out
            is_paused = True
            video_state["is_paused"] = True
            if not was_cancelled:
                video_state["current_time"] = video_state["duration"]
    except Exception as exc:
        with state_lock:
            process_state.update({"running": False, "done": False, "cancelled": False, "status": "Error", "error": str(exc)})
    finally:
        if cap is not None:
            cap.release()


@app.route("/process/start", methods=["POST"])
def start_process():
    global process_thread, stream_generation, is_paused
    with state_lock:
        if process_state.get("running"):
            return jsonify({"success": False, "error": "Processing is already running"}), 409
        video_path = selected_video
        if not video_path:
            return jsonify({"success": False, "error": "Select a local video with Open Video first"}), 400
        if not os.path.isfile(video_path):
            return jsonify({"success": False, "error": "The selected video file no longer exists"}), 400
        settings = {
            "size_filter_enabled": size_filter_enabled,
            "size_tolerance_cm": size_tolerance_cm,
            "reference_line_cm": reference_line_cm,
            "ghost_timeout_seconds": ghost_timeout_seconds,
        }
        with line_lock:
            line_norm = [p[:] for p in LINE_NORM]
        process_cancel_event.clear()
        process_state.update({"running": True, "done": False, "cancelled": False, "progress": 0.0, "frame": 0, "total_frames": 0, "status": "Starting", "error": None, "count_in": 0, "count_out": 0})
        stream_generation += 1
        is_paused = True
        video_state["is_paused"] = True
        process_thread = threading.Thread(target=_process_video_worker, args=(video_path, settings, line_norm), daemon=True)
        process_thread.start()
    return jsonify({"success": True})


@app.route("/process/cancel", methods=["POST"])
def cancel_process():
    process_cancel_event.set()
    return jsonify({"success": True})


@app.route("/process/status", methods=["GET"])
def process_status():
    with state_lock:
        return jsonify(dict(process_state))


@app.route("/process/return", methods=["POST"])
def return_from_process():
    global count_in, count_out, is_paused
    with state_lock:
        if process_state.get("running"):
            return jsonify({"success": False, "error": "Processing is still running"}), 409
        count_in = 0
        count_out = 0
        is_paused = False
        video_state["current_time"] = 0.0
        video_state["seek_to"] = None
        video_state["is_paused"] = False
        process_state.update({"running": False, "done": False, "cancelled": False, "progress": 0.0, "frame": 0, "total_frames": 0, "status": "idle", "error": None, "count_in": 0, "count_out": 0})
    return jsonify({"success": True})


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False,
    )
