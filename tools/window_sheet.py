"""Render dense frame strips around given timestamps for manual verification.

Usage: python tools/window_sheet.py video.mp4 outdir t1 t2 ...
Each row: 8 frames from t-2.0s to t+1.5s (0.5s apart), door region, line drawn.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    video, outdir, *times = sys.argv[1:]
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    line = json.loads((ROOT / "line_config.json").read_text())
    cap = cv2.VideoCapture(video)
    w, h = cap.get(3), cap.get(4)
    p1 = (int(line[0][0] * w), int(line[0][1] * h))
    p2 = (int(line[1][0] * w), int(line[1][1] * h))
    rows = []
    for t in map(float, times):
        tiles = []
        for dt in np.arange(-2.0, 1.51, 0.5):
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t + dt) * 1000)
            ok, frame = cap.read()
            if not ok:
                frame = np.zeros((int(h), int(w), 3), np.uint8)
            cv2.line(frame, p1, p2, (0, 255, 0), 2)
            crop = cv2.resize(frame[120:720, 560:1280], (264, 220))
            cv2.putText(crop, f"{t + dt:.1f}", (3, 16), 0, 0.5, (0, 0, 255), 2)
            tiles.append(crop)
        row = np.hstack(tiles)
        cv2.putText(row, f"t={t:.1f}", (3, 212), 0, 0.6, (255, 255, 0), 2)
        rows.append(np.pad(row, ((0, 4), (0, 0), (0, 0))))
    for k in range(0, len(rows), 3):
        name = outdir / f"win_{times[k]}.jpg"
        cv2.imwrite(str(name), np.vstack(rows[k:k + 3]), [cv2.IMWRITE_JPEG_QUALITY, 80])
        print(name)


if __name__ == "__main__":
    main()
