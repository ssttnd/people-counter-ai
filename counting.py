"""Tracker-agnostic line-crossing counter.

The counter receives, for every frame, the tracked boxes of people and reports
IN / OUT events. It is deliberately independent of YOLO and Flask so it can be
evaluated offline on cached detections (see tools/).

Counting rules (each one fixes a failure mode seen with a naive
"centre changed side" test):

* Finite segment - a crossing only counts if the path of the person intersects
  the drawn segment (plus a small margin), not its infinite extension. People
  walking past the ends of the line are ignored.
* Hysteresis - a person has to be at least ``min_offset_px`` away from the line
  on each side. Box jitter while standing on the line cannot flip the side.
* Confirmed tracks - a track must have been seen in ``min_hits`` frames before
  its first side is recorded, so one-frame false detections never count.
* Re-crossing - a track counts at most once per direction, and if it returns
  across the line within ``cancel_window_s`` the first event is cancelled
  (someone who steps through the door and immediately turns back).
* Disappearing at the line - when a track that was close to the line on one
  side vanishes while moving towards the line (typical when a person walks
  through a door and is hidden by the frame), the crossing is completed if its
  constant-velocity prediction crosses the segment within ``lost_predict_s``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

IN, OUT = "in", "out"


def box_anchor(box, anchor="center"):
    x1, y1, x2, y2 = map(float, box[:4])
    if anchor == "bottom":
        return np.array([(x1 + x2) / 2.0, y2])
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])


@dataclass
class _Track:
    hits: int = 0
    side: int = 0                 # last confirmed side (+1 IN side, -1 OUT side)
    side_point: np.ndarray | None = None
    side_time: float = 0.0
    last_point: np.ndarray | None = None
    last_time: float = 0.0
    history: list = field(default_factory=list)   # (t, point)
    counted: dict = field(default_factory=dict)   # direction -> time
    in_range: bool = True
    finished: bool = False


class LineCounter:
    def __init__(
        self, p1, p2, *, min_offset_px=8.0, segment_margin=0.15, min_hits=3,
        cancel_window_s=1.5, lost_predict_s=0.7, lost_near_px=60.0,
        lost_after_s=0.35, forget_after_s=5.0, anchor="center",
    ):
        self.p1 = np.asarray(p1, float)
        self.p2 = np.asarray(p2, float)
        self.min_offset_px = min_offset_px
        self.segment_margin = segment_margin
        self.min_hits = min_hits
        self.cancel_window_s = cancel_window_s
        self.lost_predict_s = lost_predict_s
        self.lost_near_px = lost_near_px
        self.lost_after_s = lost_after_s
        self.forget_after_s = forget_after_s
        self.anchor = anchor
        self.tracks: dict[int, _Track] = {}
        self.count_in = 0
        self.count_out = 0
        self.events: list[dict] = []

    # ------------------------------------------------------------------ geometry
    def set_line(self, p1, p2):
        p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
        if np.allclose(p1, self.p1) and np.allclose(p2, self.p2):
            return
        self.p1, self.p2 = p1, p2
        for tr in self.tracks.values():     # sides are meaningless for a new line
            tr.side, tr.side_point = 0, None

    @property
    def _length(self):
        return max(1e-6, float(np.linalg.norm(self.p2 - self.p1)))

    def signed_distance(self, point):
        d = self.p2 - self.p1
        v = np.asarray(point, float) - self.p1
        return float(d[0] * v[1] - d[1] * v[0]) / self._length

    def _crosses_segment(self, a, b):
        """Does the path a->b cross the (margin-extended) segment?"""
        da, db = self.signed_distance(a), self.signed_distance(b)
        if da == db:
            return False
        frac = da / (da - db)
        crossing = np.asarray(a, float) + frac * (np.asarray(b, float) - np.asarray(a, float))
        seg = self.p2 - self.p1
        u = float((crossing - self.p1) @ seg) / float(seg @ seg)
        return -self.segment_margin <= u <= 1.0 + self.segment_margin

    def distance_to_segment(self, point):
        seg = self.p2 - self.p1
        u = float(np.clip((np.asarray(point, float) - self.p1) @ seg / float(seg @ seg), 0.0, 1.0))
        return float(np.linalg.norm(np.asarray(point, float) - (self.p1 + u * seg)))

    # ------------------------------------------------------------------ counting
    def _emit(self, tid, tr, new_side, t):
        direction = IN if new_side > 0 else OUT
        opposite = OUT if direction == IN else IN
        if opposite in tr.counted and t - tr.counted[opposite] <= self.cancel_window_s:
            # Came straight back: undo the earlier event instead of adding a new one.
            del tr.counted[opposite]
            if opposite == IN:
                self.count_in -= 1
            else:
                self.count_out -= 1
            self.events.append({"t": t, "id": tid, "dir": opposite, "cancel": True})
            return None
        if direction in tr.counted or not tr.in_range:
            return None
        tr.counted[direction] = t
        if direction == IN:
            self.count_in += 1
        else:
            self.count_out += 1
        self.events.append({"t": t, "id": tid, "dir": direction})
        return direction

    def update(self, t, tracked):
        """tracked: iterable of (track_id, box, in_range). Returns list of (id, direction)."""
        fired = []
        seen = set()
        for tid, box, in_range in tracked:
            seen.add(tid)
            point = box_anchor(box, self.anchor)
            tr = self.tracks.setdefault(tid, _Track())
            tr.hits += 1
            tr.in_range = bool(in_range)
            tr.last_point, tr.last_time = point, t
            tr.history.append((t, point))
            if len(tr.history) > 15:
                tr.history.pop(0)
            if tr.hits < self.min_hits:
                continue
            dist = self.signed_distance(point)
            if abs(dist) < self.min_offset_px:
                continue                      # inside the hysteresis band
            new_side = 1 if dist > 0 else -1
            if tr.side == 0:
                tr.side, tr.side_point, tr.side_time = new_side, point, t
                continue
            if new_side != tr.side:
                if self._crosses_segment(tr.side_point, point):
                    direction = self._emit(tid, tr, new_side, t)
                    if direction:
                        fired.append((tid, direction))
                tr.side = new_side
            tr.side_point, tr.side_time = point, t

        for tid, tr in list(self.tracks.items()):
            if tid in seen:
                continue
            gap = t - tr.last_time
            if gap > self.forget_after_s:
                del self.tracks[tid]
                continue
            if gap >= self.lost_after_s and not tr.finished:
                tr.finished = True
                direction = self._complete_lost_track(tid, tr, t)
                if direction:
                    fired.append((tid, direction))
        return fired

    def _velocity(self, tr):
        hist = tr.history[-8:]
        if len(hist) < 3:
            return None
        ts = np.array([h[0] for h in hist])
        ps = np.array([h[1] for h in hist])
        if ts[-1] - ts[0] < 0.1:
            return None
        return np.array([np.polyfit(ts, ps[:, k], 1)[0] for k in range(2)])

    def _complete_lost_track(self, tid, tr, t):
        if tr.side == 0 or tr.last_point is None:
            return None
        if self.distance_to_segment(tr.last_point) > self.lost_near_px:
            return None
        vel = self._velocity(tr)
        if vel is None:
            return None
        predicted = tr.last_point + vel * self.lost_predict_s
        new_dist = self.signed_distance(predicted)
        if abs(new_dist) < self.min_offset_px or (1 if new_dist > 0 else -1) == tr.side:
            return None
        if not self._crosses_segment(tr.last_point, predicted):
            return None
        new_side = -tr.side
        direction = self._emit(tid, tr, new_side, tr.last_time)
        tr.side = new_side
        return direction

    def reset(self):
        self.tracks.clear()
        self.count_in = self.count_out = 0
        self.events.clear()


# ---------------------------------------------------------------------- offline
def make_tracker(cfg_path=None, frame_rate=30):
    """Create an Ultralytics BYTETracker/BOTSORT instance from a yaml file."""
    from ultralytics.trackers.track import TRACKER_MAP
    from ultralytics.utils import IterableSimpleNamespace, yaml_load
    from ultralytics.utils.checks import check_yaml

    cfg = IterableSimpleNamespace(**yaml_load(check_yaml(cfg_path or "bytetrack.yaml")))
    return TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=frame_rate)


def track_detections(tracker, dets, shape, frame=None):
    """Run one tracker step on an (N,5) array [x1,y1,x2,y2,conf]. Returns [(id, box, conf)]."""
    from ultralytics.engine.results import Boxes

    if len(dets) == 0:
        dets = np.zeros((0, 6), np.float32)
    else:
        dets = np.hstack([dets[:, :5], np.zeros((len(dets), 1), np.float32)])
    out = tracker.update(Boxes(dets, shape), frame)
    return [(int(r[4]), r[:4].astype(float), float(r[5])) for r in out]


def run_tracker_on_cache(cache_path, tracker_cfg=None, conf_floor=0.1):
    data = np.load(cache_path)
    frames, det = data["frame"], data["det"]
    n = int(data["frames"])
    shape = (int(data["height"]), int(data["width"]))
    tracker = make_tracker(tracker_cfg, frame_rate=round(float(data["fps"])))
    order = np.argsort(frames, kind="stable")
    frames, det = frames[order], det[order]
    bounds = np.searchsorted(frames, np.arange(n + 1))
    for i in range(n):
        d = det[bounds[i]:bounds[i + 1]]
        d = d[d[:, 4] >= conf_floor]
        for tid, box, conf in track_detections(tracker, d, shape):
            yield i, tid, box, conf
