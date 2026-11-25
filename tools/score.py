"""Match predicted IN/OUT events to ground truth events by time (one-to-one)."""
import json
import sys
from pathlib import Path


def score(pred, gt, tol=2.5, verbose=False):
    result = {}
    for direction in ("in", "out"):
        p = sorted(e["t"] for e in pred if e["dir"] == direction)
        g = sorted(e["t"] for e in gt if e["dir"] == direction)
        used = set()
        tp = 0
        missed = []
        for gt_t in g:
            best = None
            for i, pt in enumerate(p):
                if i in used or abs(pt - gt_t) > tol:
                    continue
                if best is None or abs(pt - gt_t) < abs(p[best] - gt_t):
                    best = i
            if best is None:
                missed.append(gt_t)
            else:
                used.add(best)
                tp += 1
        false = [p[i] for i in range(len(p)) if i not in used]
        result[direction] = {"gt": len(g), "pred": len(p), "tp": tp, "fn": len(missed), "fp": len(false)}
        if verbose:
            print(direction, "missed:", missed, "false:", false, file=sys.stderr)
    tp = sum(r["tp"] for r in result.values())
    n_gt = sum(r["gt"] for r in result.values())
    n_pred = sum(r["pred"] for r in result.values())
    result["precision"] = round(tp / max(1, n_pred), 3)
    result["recall"] = round(tp / max(1, n_gt), 3)
    result["f1"] = round(2 * tp / max(1, n_pred + n_gt), 3)
    return result


if __name__ == "__main__":
    pred = json.loads(Path(sys.argv[1]).read_text())
    gt = json.loads(Path(sys.argv[2]).read_text())
    print(json.dumps(score(pred, gt, verbose=True), indent=1))
