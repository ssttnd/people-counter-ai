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
        lost_after_s=0.35, forget_after_s=5.0, anchor="center", ghost_count_s=1.0,
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
        self.ghost_count_s = ghost_count_s
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

    def update(self, t, tracked, ghosts=()):
        """Advance the counter by one frame.

        tracked: iterable of (track_id, box, countable) for people detected now.
        ghosts:  iterable of (track_id, predicted_box, age_s) for people the tracker
                 has temporarily lost (the grey "ghost" boxes). A ghost that crosses
                 the line within ``ghost_count_s`` of being lost is counted, because
                 people walking through a door are often hidden by the frame before
                 their detected centre crosses. If the person then reappears on the
                 original side, the event is cancelled like any turn-back.
        Returns a list of (track_id, direction) for events counted in this frame.
        """
        fired = []
        seen = set()
        for tid, box, age in ghosts:
            tr = self.tracks.get(tid)
            if tr is None or tr.hits < self.min_hits or age > self.ghost_count_s:
                continue
            seen.add(tid)
            direction = self._observe(tid, tr, box_anchor(box, self.anchor), t)
            if direction:
                fired.append((tid, direction))
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
            tr.finished = False
            if tr.hits < self.min_hits:
                continue
            direction = self._observe(tid, tr, point, t)
            if direction:
                fired.append((tid, direction))

        for tid, tr in list(self.tracks.items()):
            if tid in seen and t == tr.last_time:
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

    def _observe(self, tid, tr, point, t):
        """Apply hysteresis + finite-segment crossing test to one position."""
        dist = self.signed_distance(point)
        if abs(dist) < self.min_offset_px:
            return None                       # inside the hysteresis band
        new_side = 1 if dist > 0 else -1
        direction = None
        if tr.side == 0:
            tr.side = new_side
        elif new_side != tr.side:
            if self._crosses_segment(tr.side_point, point):
                direction = self._emit(tid, tr, new_side, t)
            tr.side = new_side
        tr.side_point, tr.side_time = point, t
        return direction

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


# --------------------------------------------------------------------- tracking
def make_tracker(cfg_path=None, frame_rate=30):
    """Create an Ultralytics BYTETracker/BOTSORT instance from a yaml file."""
    from ultralytics.trackers.track import TRACKER_MAP
    from ultralytics.utils import IterableSimpleNamespace, yaml_load
    from ultralytics.utils.checks import check_yaml

    cfg = IterableSimpleNamespace(**yaml_load(check_yaml(cfg_path or "bytetrack.yaml")))
    return TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=frame_rate)


def _iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class PeopleTracker:
    """BoT-SORT/ByteTrack driven frame by frame, plus "ghosts".

    A ghost is a person the tracker has temporarily lost (behind a pillar, a door
    frame or another person). It moves on from the person's last *observed*
    position with the person's *measured* walking speed (least-squares over the
    last ``velocity_window_s`` of real detections) until the timeout.

    The tracker's own prediction for a lost person is overwritten with the ghost
    every frame, so the tracker looks for the person exactly where the ghost is
    drawn. If the tracker still starts a new ID for someone who reappears near a
    ghost of similar size, the new ID is mapped back to the ghost's ID.

    Ghosts are only kept for real people (confirmed for ``ghost_min_frames``
    frames), inside the image and not on top of someone currently detected.
    """

    def __init__(self, cfg_path=None, frame_rate=30, ghost_timeout_s=3.0, ghost_min_frames=8,
                 ghost_glide_s=None, velocity_window_s=0.5):
        self.frame_rate = float(frame_rate) or 30.0
        self.tracker = make_tracker(cfg_path, frame_rate=int(round(self.frame_rate)))
        self.ghost_min_frames = ghost_min_frames
        self.glide_frames = float("inf") if ghost_glide_s is None else ghost_glide_s * self.frame_rate
        self.window_frames = max(3, int(round(velocity_window_s * self.frame_rate)))
        self._history = {}             # id -> [(frame_id, box xyxy)] of real observations
        self._alias = {}               # tracker id -> id shown/counted (re-identified ghosts)
        self._motion = {}              # id -> [first_frame, first_centre, max_displacement]
        self.static_after_frames = 2.0 * self.frame_rate
        self.set_ghost_timeout(ghost_timeout_s)

    def set_ghost_timeout(self, seconds):
        """How long a lost person keeps their ID (and ghost) before being dropped."""
        self.ghost_timeout_s = float(seconds)
        self.tracker.max_time_lost = max(1, int(round(self.ghost_timeout_s * self.frame_rate)))

    def reset(self):
        self.tracker.reset()
        self._history.clear()
        self._alias.clear()
        self._motion.clear()

    def _velocity(self, hist):
        """Pixels per frame of the box centre, fitted on the last observations."""
        if len(hist) < 3:
            return np.zeros(2)
        f = np.array([h[0] for h in hist], float)
        if f[-1] - f[0] < 2:
            return np.zeros(2)
        c = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for _, b in hist])
        v = np.array([np.polyfit(f, c[:, k], 1)[0] for k in range(2)])
        height = hist[-1][1][3] - hist[-1][1][1]
        max_speed = 1.5 * height / self.frame_rate     # ~1.5 body heights per second
        speed = float(np.linalg.norm(v))
        return v * (max_speed / speed) if speed > max_speed > 0 else v

    def _ghost_box(self, tid, frames_lost):
        hist = self._history[tid]
        box = hist[-1][1]
        shift = self._velocity(hist) * min(frames_lost, self.glide_frames)
        return box + np.array([shift[0], shift[1], shift[0], shift[1]])

    def _reidentify(self, box, frame_id):
        """A brand-new tracker ID: return the ID of a nearby ghost of similar size, if any."""
        cx, cy, h = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[3] - box[1]
        best, best_d = None, None
        for st in self.tracker.lost_stracks:
            tid = self._display_id(st.track_id)
            if tid not in self._history or st.tracklet_len < self.ghost_min_frames:
                continue
            g = self._ghost_box(tid, frame_id - st.end_frame)
            gh = g[3] - g[1]
            if not 0.7 <= h / max(gh, 1.0) <= 1.4:
                continue
            d = float(np.hypot(cx - (g[0] + g[2]) / 2, cy - (g[1] + g[3]) / 2))
            if d <= 0.6 * max(h, gh) and (best_d is None or d < best_d):
                best, best_d = st, d
        if best is None:
            return None
        best.mark_removed()                 # the ghost is now this detection
        self.tracker.lost_stracks.remove(best)
        return self._display_id(best.track_id)

    def is_static(self, tid, box):
        """True for objects that have not moved for 2 s (parked scooters, posters...).

        A person who stands still and then walks off becomes non-static again.
        """
        m = self._motion.get(tid)
        if m is None or self.tracker.frame_id - m[0] < self.static_after_frames:
            return False
        return m[2] < 0.25 * (box[3] - box[1])

    def _display_id(self, raw):
        return self._alias.get(int(raw)) or int(raw)

    def update(self, dets, shape):
        """dets: (N,5) [x1,y1,x2,y2,conf]. Returns (tracked, ghosts).

        tracked: [(id, box, conf)]   ghosts: [(id, predicted_box, age_s)]
        """
        from ultralytics.engine.results import Boxes

        dets = np.asarray(dets, np.float32).reshape(-1, 5)
        data = np.hstack([dets, np.zeros((len(dets), 1), np.float32)])
        out = self.tracker.update(Boxes(data, shape), None)
        frame_id = self.tracker.frame_id
        tracked = []
        for r in out:
            raw, box = int(r[4]), dets[int(r[-1])][:4].astype(float)   # the real detection box
            if raw not in self._alias and raw not in self._history:
                self._alias[raw] = self._reidentify(box, frame_id)
            tid = self._display_id(raw)
            tracked.append((tid, box, float(r[5])))
            centre = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
            m = self._motion.setdefault(tid, [frame_id, centre, 0.0])
            m[2] = max(m[2], float(np.linalg.norm(centre - m[1])))
            hist = self._history.setdefault(tid, [])
            hist.append((frame_id, box))
            while hist and frame_id - hist[0][0] > self.window_frames:
                hist.pop(0)

        alive = {self._display_id(st.track_id) for st in self.tracker.tracked_stracks + self.tracker.lost_stracks}
        if len(self._history) > 2 * len(alive) + 50:
            self._history = {k: v for k, v in self._history.items() if k in alive}
            self._motion = {k: v for k, v in self._motion.items() if k in alive}

        height, width = shape[:2]
        live_boxes = [box for _, box, _ in tracked]
        ghosts = []
        for st in self.tracker.lost_stracks:
            tid = self._display_id(st.track_id)
            if tid not in self._history or st.mean is None:
                continue
            lost = frame_id - st.end_frame
            box = self._ghost_box(tid, lost)
            # Make the tracker search for this person where the ghost is, and move
            # it with the same speed during the glide, then keep it still.
            st.mean[:4] = st.convert_coords(np.array([box[0], box[1], box[2] - box[0], box[3] - box[1]]))
            st.mean[4:8] = 0.0
            if lost + 1 <= self.glide_frames:
                st.mean[4:6] = self._velocity(self._history[tid])
            age = lost / self.frame_rate
            if st.tracklet_len < self.ghost_min_frames or age <= 0 or age > self.ghost_timeout_s:
                continue
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            if not (0 <= cx < width and 0 <= cy < height):
                continue                      # walked out of the image
            if any(_iou(box, live) > 0.6 for live in live_boxes):
                continue                      # duplicate of someone visible
            ghosts.append((tid, box, float(age)))
        return tracked, ghosts


# ---------------------------------------------------------------------- offline
def run_tracker_on_cache(cache_path, tracker_cfg=None, conf_floor=0.1, ghost_timeout_s=3.0):
    """Replay cached detections through PeopleTracker. Yields (frame, tracked, ghosts)."""
    data = np.load(cache_path)
    frames, det = data["frame"], data["det"]
    n = int(data["frames"])
    shape = (int(data["height"]), int(data["width"]))
    people = PeopleTracker(tracker_cfg, frame_rate=float(data["fps"]), ghost_timeout_s=ghost_timeout_s)
    order = np.argsort(frames, kind="stable")
    frames, det = frames[order], det[order]
    bounds = np.searchsorted(frames, np.arange(n + 1))
    for i in range(n):
        d = det[bounds[i]:bounds[i + 1]]
        tracked, ghosts = people.update(d[d[:, 4] >= conf_floor], shape)
        yield i, tracked, ghosts
