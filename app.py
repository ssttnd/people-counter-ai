import os
from flask import Flask, Response, render_template
import cv2
import numpy as np

# Lazy import for Ultralytics to speed startup if not used
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

app = Flask(__name__)

def get_source():
    src = os.getenv("VIDEO_SOURCE", "0")
    try:
        return int(src)
    except ValueError:
        return src

# Basic line (x1,y1) -> (x2,y2)
LINE_PTS = ((100, 300), (540, 300))  # adjust in UI as needed
crossed_ids = set()
count_in, count_out = 0, 0

def load_model():
    if YOLO is None:
        return None
    # Use a lightweight model; customize if needed
    return YOLO("yolov8n.pt")

def gen():
    global count_in, count_out, crossed_ids
    cap = cv2.VideoCapture(get_source())
    model = load_model()

    if not cap.isOpened():
        raise RuntimeError("Cannot open VIDEO_SOURCE")

    line_p1, line_p2 = LINE_PTS
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        # Draw line
        cv2.line(frame, line_p1, line_p2, (0, 255, 0), 2)

        # Run detection (very naive demo)
        if model is not None:
            results = model.predict(frame, imgsz=640, conf=0.4, verbose=False)
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    if cls != 0:  # person class
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2)//2, (y1 + y2)//2
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                    # Fake line-cross (center over/under line), no tracking for brevity
                    line_y = line_p1[1]
                    above = cy < line_y
                    key = (x1, y1, x2, y2)  # placeholder ID
                    if key not in crossed_ids:
                        if above:
                            count_in += 1
                        else:
                            count_out += 1
                        crossed_ids.add(key)

        # Overlay counts
        cv2.putText(frame, f"IN: {count_in}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        cv2.putText(frame, f"OUT: {count_out}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stream')
def stream():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)