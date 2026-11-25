"""Hand-labelled ground truth for the KIMEP entrance test video (20261027_132153.mp4).

Each timestamp (seconds) is when a person passed the counting line drawn across
the door (line_config.json). Labelled by reviewing dense frame strips around every
moment at which any tested algorithm reported a crossing.
Not included: one ambiguous movement at ~585 s (someone hovering in the doorway).
"""
import json
from pathlib import Path

IN = [10.4, 59.0, 63.0, 82.3, 85.8, 108.6, 121.2, 123.4, 133.6, 134.8, 136.2, 151.2,
      156.5, 180.0, 212.0, 253.0, 329.7, 407.2, 437.2, 478.2, 481.3, 481.9, 482.6,
      483.2, 485.7, 531.6, 532.4, 534.3, 548.8, 567.2, 567.8, 581.8]
OUT = [13.2, 24.6, 57.6, 58.4, 59.3, 112.5, 114.9, 158.4, 347.3, 348.5, 401.2, 450.5,
       461.4, 486.8, 490.2, 502.7, 511.4, 541.4, 551.0, 592.8]

gt = sorted([{"t": t, "dir": "in"} for t in IN] + [{"t": t, "dir": "out"} for t in OUT],
            key=lambda e: e["t"])
Path(__file__).with_name("kimep_entrance_gt.json").write_text(json.dumps(gt, indent=1))
print(len(IN), "in,", len(OUT), "out")
