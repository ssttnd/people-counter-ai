"""Run the live-stream counting pipeline (app.gen) headlessly on a video file.

Prints every IN/OUT count change with its video timestamp so results can be
compared against a hand-labelled ground truth (see tools/score.py).

Usage:
    python tools/eval_live.py path/to/video.mp4 [--out events.json] [--max-seconds N]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-seconds", type=float, default=None)
    args = parser.parse_args()

    app.source = str(Path(args.video).resolve())
    events = []
    last_in = last_out = 0
    started = time.time()
    frames = 0
    for _ in app.gen():
        frames += 1
        with app.state_lock:
            t = app.video_state["current_time"]
            cin, cout = app.count_in, app.count_out
        for _ in range(cin - last_in):
            events.append({"t": round(t, 2), "dir": "in"})
            print(f"{t:8.2f}s IN   (total in={cin} out={cout})", flush=True)
        for _ in range(cout - last_out):
            events.append({"t": round(t, 2), "dir": "out"})
            print(f"{t:8.2f}s OUT  (total in={cin} out={cout})", flush=True)
        last_in, last_out = cin, cout
        if args.max_seconds is not None and t >= args.max_seconds:
            break

    elapsed = time.time() - started
    print(f"frames={frames} wall={elapsed:.1f}s fps={frames / max(elapsed, 1e-6):.1f}")
    print(f"TOTAL in={last_in} out={last_out}")
    if args.out:
        Path(args.out).write_text(json.dumps(events, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
