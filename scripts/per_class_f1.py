"""F1 từng ngành (precision, recall, F1 = 2PR/(P+R)) của các run `category`, tính từ
``predictions_dev.parquet`` trên `dev` nhãn gốc.

Có vì các run trước 2026-09-24 không lưu ``per_class`` trong ``metrics.json``. Ghi
``artifacts/eda/per-class-f1.json`` và in bảng F1, kèm số dòng dự đoán khác nhau giữa
từng cặp run (0 = hai run cho dự đoán giống hệt).

    PYTHONPATH=src .venv/bin/python scripts/per_class_f1.py dl-cat-ce-cpu-0920 dl-cat-ce-16lop-0924
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vietjobs import config as C  # noqa: E402
from vietjobs import evaluate as E  # noqa: E402

OUT = C.ARTIFACT_DIR / "eda" / "per-class-f1.json"


def main(runs: list[str]) -> None:
    preds, y = {}, None
    for run in runs:
        df = pd.read_parquet(C.ARTIFACT_DIR / run / "predictions_dev.parquet")
        if y is None:
            y = df["y_true"].to_numpy(dtype=object)
        elif not np.array_equal(df["y_true"].to_numpy(dtype=object), y):
            raise SystemExit(f"{run}: y_true dev không khớp")
        preds[run] = df["y_pred"].to_numpy(dtype=object)
    labels = sorted(set(y.tolist()))
    rep = {"n": int(len(y)), "labels": labels, "runs": {}, "n_diff_predictions": {}}
    for run, p in preds.items():
        pc = E.per_class_report(y, p, labels)
        m = E.classification_metrics(y, p, labels)
        rep["runs"][run] = {
            "per_class": {lab: {k: round(pc[lab][k], 4) for k in ("precision", "recall", "f1-score")}
                          | {"support": int(pc[lab]["support"])} for lab in labels},
            "f1_macro": round(m["f1_macro"], 4), "f1_weighted": round(m["f1_weighted"], 4),
            "accuracy": round(m["accuracy"], 4)}
    for a, b in itertools.combinations(runs, 2):
        rep["n_diff_predictions"][f"{a} vs {b}"] = int((preds[a] != preds[b]).sum())
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

    print("ngành".ljust(24) + "".join(r[7:25].rjust(20) for r in runs))
    for lab in labels:
        print(lab[:24].ljust(24) + "".join(f"{rep['runs'][r]['per_class'][lab]['f1-score']:20.3f}" for r in runs))
    for k in ("f1_macro", "f1_weighted", "accuracy"):
        print(k.ljust(24) + "".join(f"{rep['runs'][r][k]:20.4f}" for r in runs))
    for k, v in rep["n_diff_predictions"].items():
        print(f"khác dự đoán: {k}: {v}")
    print(f"  → {OUT.relative_to(C.ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1:])
