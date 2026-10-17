"""T12.1 — mean±sd, paired-by-seed Δ (bootstrap CI95, B=2000) and the worst-class confusion row.

    PYTHONPATH=src .venv/bin/python scripts/t121_concat_stats.py

Reads only ``artifacts/<run_id>/metrics.json`` and ``predictions_dev.parquet`` (dev).
Writes ``artifacts/t121/summary.json`` and ``artifacts/dl-cat-concat-cpu-s42/confusion_worst_row.csv``.
Seeds are paired: Δ_s = concat_s − reference_s, and the CI resamples the five Δ_s with replacement.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from vietjobs import config as C

SEEDS = [42, 43, 44, 45, 46]
B = 2000
RNG = np.random.default_rng(20260826)

# (label, run_id per seed); seed 42 has the "-0920" name for the PhoBERT runs
RUNS = {
    "category": {
        "concat": [f"dl-cat-concat-cpu-s{s}" for s in SEEDS],
        "phobert": ["dl-cat-ce-cpu-0920"] + [f"dl-cat-ce-cpu-s{s}" for s in SEEDS[1:]],
        "cafebert": ["dl-cat-cafe-dense-512"] + [f"dl-cat-cafe-dense-512-s{s}" for s in SEEDS[1:]],
        "metric": ("metrics", "f1_macro"),
    },
    "salary": {
        "concat": [f"dl-sal-concat-cpu-s{s}" for s in SEEDS],
        "phobert": ["dl-sal-cpu-0920"] + [f"dl-sal-cpu-s{s}" for s in SEEDS[1:]],
        "cafebert": ["dl-sal-cafe-dense-512"] + [f"dl-sal-cafe-dense-512-s{s}" for s in SEEDS[1:]],
        "metric": ("metrics", "mae_trieu"),
    },
}


def value(run: str, key: str) -> float:
    m = json.loads((C.ARTIFACT_DIR / run / "metrics.json").read_text(encoding="utf-8"))
    return float(m["metrics"][key])


def paired(a: list[float], b: list[float]) -> dict:
    d = np.array(a) - np.array(b)
    boots = RNG.choice(d, size=(B, len(d)), replace=True).mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean": float(d.mean()), "ci95": [float(lo), float(hi)],
            "deltas": d.round(5).tolist()}


def confusion_worst_row(run: str) -> None:
    pred = pd.read_parquet(C.ARTIFACT_DIR / run / "predictions_dev.parquet")
    m = json.loads((C.ARTIFACT_DIR / run / "metrics.json").read_text(encoding="utf-8"))
    per = {k: m["per_class"][k] for k in m["labels"] if m["per_class"][k]["support"] > 0}
    worst = min(per, key=lambda k: per[k]["f1-score"])
    labels = m["labels"]
    row = pred.loc[pred["y_true"] == worst, "y_pred"].value_counts().reindex(labels, fill_value=0)
    out = C.ARTIFACT_DIR / run / "confusion_worst_row.csv"
    pd.DataFrame({"true_class": worst, "predicted_as": row.index, "count": row.values,
                  "f1": per[worst]["f1-score"], "support": per[worst]["support"]}).to_csv(out, index=False)
    print(f"worst class {worst}: f1={per[worst]['f1-score']:.4f} support={per[worst]['support']} -> {out}")


def main() -> None:
    summary = {}
    for task, spec in RUNS.items():
        key = spec["metric"][1]
        vals = {name: [value(r, key) for r in spec[name]] for name in ("concat", "phobert", "cafebert")}
        summary[task] = {"metric": key, "seeds": SEEDS, "values": vals, "stats": {}}
        for name in ("concat", "phobert", "cafebert"):
            v = np.array(vals[name])
            summary[task]["stats"][name] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1))}
        for ref in ("phobert", "cafebert"):
            summary[task]["stats"][f"concat_minus_{ref}"] = paired(vals["concat"], vals[ref])
        for k in ("concat", "phobert", "cafebert"):
            print(task, k, [round(x, 4) for x in vals[k]])
    confusion_worst_row(RUNS["category"]["concat"][0])
    out = C.ARTIFACT_DIR / "t121" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
