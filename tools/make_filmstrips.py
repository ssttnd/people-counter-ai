"""Build filmstrips of people near the counting line, for manual ground-truth labelling.

Tracks cached detections with Ultralytics' ByteTrack, keeps tracks that come
near the counting segment, and writes one image strip per candidate so a human
can label it as in / out / none.

Usage:
    python tools/make_filmstrips.py video.mp4 dets.npz outdir
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from counting import run_tracker_on_cache  # noqa: E402


def main():
    video, cache, outdir = sys.argv[1:4]
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    line = json.loads((Path(__file__).resolve().parents[1] / "line_config.json").read_text())
    data = np.load(cache)
    w, h = float(data["width"]), float(data["height"])
    p1 = np.array([line[0][0] * w, line[0][1] * h])
    p2 = np.array([line[1][0] * w, line[1][1] * h])

    tracks = {}
    for frame_no, tracked, _ghosts in run_tracker_on_cache(cache, conf_floor=0.15):
        for tid, box, _conf in tracked:
            tracks.setdefault(tid, []).append((frame_no, box))

    seg = p2 - p1
    seg_len2 = float(seg @ seg)
    candidates = []
    for tid, obs in tracks.items():
        best = 1e9
        for frame_no, box in obs:
            c = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
            f = np.clip((c - p1) @ seg / seg_len2, 0, 1)
            best = min(best, float(np.linalg.norm(c - (p1 + f * seg))))
        if best < 160 and len(obs) >= 3:
            candidates.append((obs[0][0], tid, obs))
    candidates.sort()

    cap = cv2.VideoCapture(video)
    fps = float(data["fps"])
    index = []
    for k, (_start, tid, obs) in enumerate(candidates):
        pick = [obs[int(i)] for i in np.linspace(0, len(obs) - 1, 6)]
        tiles = []
        for frame_no, box in pick:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ok, frame = cap.read()
            if not ok:
                continue
            cv2.line(frame, tuple(map(int, p1)), tuple(map(int, p2)), (0, 255, 0), 2)
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            crop = cv2.resize(frame[100:720, 400:1280], (352, 248))
            cv2.putText(crop, f"{frame_no / fps:.1f}s", (4, 20), 0, 0.6, (0, 0, 255), 2)
            tiles.append(crop)
        strip = np.hstack(tiles)
        cv2.putText(strip, f"#{k} track {tid}", (4, 240), 0, 0.7, (255, 255, 0), 2)
        cv2.imwrite(str(outdir / f"cand_{k:03d}.jpg"), strip, [cv2.IMWRITE_JPEG_QUALITY, 75])
        index.append({"k": k, "track": tid, "t0": round(obs[0][0] / fps, 2), "t1": round(obs[-1][0] / fps, 2)})
    (outdir / "index.json").write_text(json.dumps(index, indent=1))
    print(len(candidates), "candidates")


if __name__ == "__main__":
    main()
