# ARIA automated people counter

Counts people walking **in** and **out** through a door by detecting when they cross a
virtual line. Runs locally in the browser: Flask + OpenCV + YOLO (Ultralytics) + BoT-SORT
tracking. Built for a university Operations Management project (entrance traffic at
KIMEP University).

Based on the open-source project
[nhatbo286/people-counter-ai](https://github.com/nhatbo286/people-counter-ai) (MIT), extended
with a new UI, a hand-labelled evaluation, and a rewritten counting algorithm.

## Features

- **Sources**: local video file, webcam, or an **IP camera** (RTSP / HTTP, e.g. Hikvision,
  Dahua, or a phone running the IP Webcam app)
- Drag the counting line on top of the video and save it; arrows show the IN / OUT sides
  (*Swap IN/OUT* flips them)
- **Count video**: analyses a whole video without displaying it, only in the area around the
  counting line. *Fast* mode uses every 2nd frame: ~4× real time on a laptop CPU (10-min video
  in 2.5 min, F1 0.90 on the test video). *Accurate* uses every frame (~2.5× real time).
  Progress bar, time remaining, speed, live totals and cancel. At the end you get a per-minute in/out chart
  and a CSV of every crossing
- **Ghost IDs**: a person the detector briefly loses (hidden by the door frame or another
  person) is shown as a grey ghost at their predicted position and gets the same ID back
  when they reappear. A ghost that crosses the line is counted
- Height filter: the line's real length (cm) is used as a scale, so people that are much
  too small or too large for the doorway (people far away, someone right in front of the
  camera) are tracked but never counted

## Quick start (Windows)

Double-click **`start.bat`**. On the first run it creates a Python environment and
installs the packages (a few minutes, needs Python 3.10+). It then starts the server
and opens http://127.0.0.1:5000 in your browser. The server runs in the background without a
console window (output goes to `people_counter.log`); stop it with **Quit** in the app.
Running `start.bat` again while the server is up just opens the browser.

Manual alternative:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py --open-browser
```

The YOLO weights (`yolov8n.pt`, 6 MB) are downloaded automatically on first run.

Environment variables:

| variable | default | meaning |
|---|---|---|
| `VIDEO_SOURCE` | none | webcam index, camera URL or video path to open on start |
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
4. **Once per direction.** A track counts at most one IN and one OUT. Counts never go
   down: cancelling a "turn-back" was dropped because in a doorway it was mostly two
   different people swapping IDs (one entering, one leaving).
5. **Ghosts.** When the tracker loses a person (typically while walking through the door),
   their ghost keeps moving at the person's measured walking speed (fitted on the last
   0.5 s of real detections) for 1 s, then waits in place until the ghost timeout. The
   tracker searches for the person exactly where the ghost is, so an ID only comes back
   to a detection overlapping the ghost. If the ghost crosses the segment, the crossing counts; if the person reappears on
   the original side, rule 4 cancels it. The ghosts are the tracker's own lost tracks
   (`PeopleTracker` in `counting.py`), so the ghost you see, the ID that comes back and the
   ID that is counted are always the same. When a lost person is picked up again, the ID
   is confirmed only if they keep walking in the ghost's direction (checked over 0.3 s);
   someone walking the opposite way gets a new ID, and nothing is counted for the ID
   until it is confirmed.
6. **Height filter** (optional, see above) decides whether a crossing may be counted.

## Accuracy

Evaluated on a 10-minute 1280×720 recording of the KIMEP entrance (busy, groups, people
standing in front of the door, a hand-held camera). The video is not published because it
shows identifiable students. Ground truth: 32 IN and 20 OUT crossings, labelled by hand
([`eval/make_gt.py`](eval/make_gt.py)). A predicted event is correct if it matches a true
event of the same direction within ±2.5 s.

Full pipeline (detector → tracker with ghosts → counter; live view, Fast count and the evaluation run the same code):

| version | precision | recall | F1 | IN (true 32) | OUT (true 20) | speed (CPU) |
|---|---|---|---|---|---|---|
| previous app logic (stable IDs + ghost crossings) | 0.72 | 0.79 | 0.75 | 37 | 20 | ~17 fps |
| **this version** | **0.91** | **0.94** | **0.93** | 35 | 19 | ~33 fps live, 1.3× real time in Fast count |

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
