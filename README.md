# ARIA automated people counter

Counts people walking **in** and **out** through a door by detecting when they cross a
virtual line. Runs locally in the browser: Flask + OpenCV + YOLO (Ultralytics) + BoT-SORT
tracking. Built for a university Operations Management project (entrance traffic at
KIMEP University).

Based on the open-source project
[nhatbo286/people-counter-ai](https://github.com/nhatbo286/people-counter-ai) (MIT), extended
with a new UI, a hand-labelled evaluation, and a rewritten counting algorithm.

## Features

- Drag the counting line on top of the video, save it; arrows show the IN / OUT sides
- Open any local video (or webcam / RTSP via `VIDEO_SOURCE`), play/pause and seek
- Stable person IDs with "ghost" boxes that predict where a temporarily hidden person is
- Height filter: the line's real length (cm) is used as a scale, so people that are much
  too small or too large for the doorway (people far away, someone right in front of the
  camera) are tracked but never counted
- Offline "process whole video" endpoint (`POST /process/start`) that uses the same logic

## Quick start (Windows)

Double-click **`start.bat`**. On the first run it creates a Python environment and
installs the packages (a few minutes, needs Python 3.10+). It then starts the server
and opens http://127.0.0.1:5000 in your browser. Close the console window to stop.

Manual alternative:

```powershell
python -m venv .venv
.venv\Scriptsctivate
pip install -r requirements.txt
python app.py --open-browser
```

The YOLO weights (`yolov8n.pt`, 6 MB) are downloaded automatically on first run.

Environment variables:

| variable | default | meaning |
|---|---|---|
| `VIDEO_SOURCE` | `0` | webcam index, RTSP URL or video path used before a file is opened |
| `PEOPLE_COUNTER_MODEL` | `yolov8n.pt` | any Ultralytics detection model, e.g. `yolo11s.pt` |

## How counting works

Detection and tracking come from YOLO + BoT-SORT (`people_tracker.yaml`). The counting
decision lives in [`counting.py`](counting.py) (`LineCounter`). It has no dependency on YOLO
or Flask, so it can be tested on its own:

1. **Finite line.** A crossing counts only if the person's path intersects the drawn
   segment (±15 %), not its infinite extension. People walking past the end of the line
   are ignored.
2. **Hysteresis.** A person must be at least 15 px (at 1280 px width) away from the line
   on both sides. Box jitter while someone stands in the doorway can't flip the side.
3. **Confirmed tracks only.** A track has to exist for 3 frames before it can count.
4. **Turning back.** If the same person crosses back within 1.5 s, the first event is
   cancelled.
5. **Vanishing in the door.** When a person close to the line disappears, which is
   typical when they walk through the door, their last motion is extrapolated for
   0.7 s. If that crosses the segment, the crossing is completed.
6. **Height filter** (optional, see above) decides whether a crossing may be counted.

## Accuracy

Evaluated on a 10-minute 1280×720 recording of the KIMEP entrance (busy, groups, people
standing in front of the door, a hand-held camera). The video is not published because it
shows identifiable students. Ground truth: 32 IN and 20 OUT crossings, labelled by hand
([`eval/make_gt.py`](eval/make_gt.py)). A predicted event is correct if it matches a true
event of the same direction within ±2.5 s.

Full app pipeline (`tools/eval_live.py`, what you see in the browser):

| version | precision | recall | F1 | IN (true 32) | OUT (true 20) | speed (CPU) |
|---|---|---|---|---|---|---|
| previous app logic (stable IDs + ghost crossings) | 0.72 | 0.79 | 0.75 | 37 | 20 | ~17 fps |
| **this version** | **0.92** | **0.90** | **0.91** | 32 | 17 | ~33 fps (1.1× real time) |

The old version's totals looked close only because its errors cancelled out: it had
16 wrong events and 11 missed ones. Besides the new counting rules, two fixes contributed:

- the line and labels were drawn onto the frame *before* detection, so they covered
  exactly the people standing in the doorway. Detection now runs on a clean copy
- BoT-SORT's camera-motion compensation (`gmc_method`) is now off. The camera is
  static, and it cost about 40 % of the CPU time

Counter-only ablation on cached detections (`tools/eval_cache.py`):

| setting | precision | recall | F1 |
|---|---|---|---|
| height filter off | 0.83 | 0.94 | 0.88 |
| height filter on (180 cm ± 30) | 0.96 | 0.90 | 0.93 |

Other things that were tested and did not help on this video: `yolo11s` (F1 0.91, 2× slower),
detection confidence 0.35, and plain ByteTrack instead of BoT-SORT.

The ground truth was built by checking every moment at which any tested version reported
a crossing, plus a full manual pass over the first 40 s. A crossing that *every* version
missed could therefore be absent from the ground truth, so recall may be slightly
optimistic for all rows.

### Reproduce

```powershell
# 1. cache detections once (≈10 min on CPU)
python tools/cache_detections.py path\to\video.mp4 --out dets.npz
# 2. evaluate the counter in seconds (try other parameters via --params / --size-filter)
python tools/eval_cache.py dets.npz eval/kimep_entrance_gt.json --conf 0.25 `
    --params '{\"min_offset_px\":15}' --size-filter '{\"tolerance_cm\":30,\"reference_cm\":180}' --verbose
# 3. or run the real app pipeline headless and score it
python tools/eval_live.py path\to\video.mp4 --out events.json
python tools/score.py events.json eval/kimep_entrance_gt.json
```

`tools/make_filmstrips.py` and `tools/window_sheet.py` produce the image strips that were used
for labelling.

## Related open-source projects

- [Ultralytics `ObjectCounter`](https://docs.ultralytics.com/guides/object-counting/):
  built-in line/region in-out counting (centroid crosses the line).
- [Roboflow Supervision `LineZone`](https://supervision.roboflow.com/latest/detection/tools/line_zone/):
  line counting with multi-anchor triggering and ByteTrack.
- [bee1997/People-Counting-In-Out](https://github.com/bee1997/People-Counting-In-Out): a
  small MIT project on top of the Ultralytics counter.
- [ailia APPS people counter](https://medium.com/axinc-ai/ailia-apps-people-counter-an-ai-application-that-can-perform-human-flow-analysis-fcc9f55e301a):
  a GUI app for line-crossing counting.

These give a working baseline quickly. This project adds a counting rule set that is
tuned and measured on a real entrance, the height filter, and the browser UI.

## Privacy

Don't commit real camera footage or RTSP URLs. People in an entrance video are
identifiable, so get consent before recording, and keep recordings out of git
(`*.mp4` is in `.gitignore`).

## License

MIT (see [LICENSE](LICENSE)). Note that the Ultralytics package this project depends on is
licensed under AGPL-3.0.
