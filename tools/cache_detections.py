"""Run a YOLO person detector over a whole video once and cache the raw boxes.

The cache (.npz) lets the tracking/counting logic be re-run and tuned in
seconds instead of re-running the neural network for every experiment.

Usage:
    python tools/cache_detections.py video.mp4 --model yolo11s.pt --out dets.npz
"""
import argparse
import time

import cv2
import numpy as np
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx, rows = [], []
    started = time.time()
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = model.predict(frame, classes=[0], conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        boxes = result.boxes
        if boxes is not None and len(boxes):
            data = np.hstack([boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()[:, None]])
            rows.append(data.astype(np.float32))
            frame_idx.extend([i] * len(data))
        i += 1
        if i % 500 == 0:
            rate = i / (time.time() - started)
            print(f"{i}/{total} frames, {rate:.1f} fps", flush=True)
    det = np.vstack(rows) if rows else np.zeros((0, 5), np.float32)
    np.savez_compressed(
        args.out, frame=np.asarray(frame_idx, np.int32), det=det,
        fps=fps, frames=i, width=cap.get(3), height=cap.get(4),
    )
    print(f"done: {i} frames, {len(det)} boxes, {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
