"""Evaluate the counting pipeline on cached detections against ground truth.

Usage:
    python tools/eval_cache.py dets.npz ground_truth.json [--tracker people_tracker.yaml]
        [--conf 0.25] [--size-filter] [--verbose]

ground_truth.json: [{"t": 12.3, "dir": "in"}, ...]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from counting import LineCounter, run_tracker_on_cache  # noqa: E402
from tools.score import score  # noqa: E402


def run(cache, tracker_cfg, conf, counter_kwargs, line=None, size_filter=None):
    """size_filter: None or dict(tolerance_cm=.., reference_cm=..) as in the app."""
    from app import _size_is_plausible
    data = np.load(cache)
    w, h, fps = float(data["width"]), float(data["height"]), float(data["fps"])
    line = line or json.loads((ROOT / "line_config.json").read_text())
    p1 = (line[0][0] * w, line[0][1] * h)
    p2 = (line[1][0] * w, line[1][1] * h)
    counter = LineCounter(p1, p2, **counter_kwargs)
    current, batch = 0, []
    for frame_no, tid, box, _c in run_tracker_on_cache(cache, tracker_cfg, conf_floor=conf):
        if frame_no != current:
            counter.update(current / fps, batch)
            for k in range(current + 1, frame_no):     # frames without tracks
                counter.update(k / fps, [])
            current, batch = frame_no, []
        ok = True
        if size_filter:
            ok = _size_is_plausible(box, p1, p2, w, h, enabled=True, **size_filter)
        batch.append((tid, box, ok))
    counter.update(current / fps, batch)
    for k in range(1, int(5 * fps)):
        counter.update((current + k) / fps, [])
    events = [e for e in counter.events if not e.get("cancel")]
    cancelled = [e for e in counter.events if e.get("cancel")]
    for c in cancelled:   # remove the event that was undone
        for i in range(len(events) - 1, -1, -1):
            if events[i]["id"] == c["id"] and events[i]["dir"] == c["dir"]:
                events.pop(i)
                break
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("gt")
    ap.add_argument("--tracker", default=str(ROOT / "people_tracker.yaml"))
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--params", default="{}", help="JSON dict of LineCounter kwargs")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--size-filter", default=None, help='JSON e.g. {"tolerance_cm":30,"reference_cm":180}')
    args = ap.parse_args()
    events = run(args.cache, args.tracker, args.conf, json.loads(args.params),
                 size_filter=json.loads(args.size_filter) if args.size_filter else None)
    gt = json.loads(Path(args.gt).read_text())
    result = score(events, gt, verbose=args.verbose)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
