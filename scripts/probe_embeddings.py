"""Dò tuyến tính trên vector PhoBERT — câu hỏi: đặc trưng có tín hiệu không?

Khi mạng dense cho điểm thấp, có đúng hai khả năng: **đặc trưng nghèo** hoặc
**phần đầu mạng hỏng**. Một mô hình tuyến tính có nghiệm lồi, không có tốc độ học
để chỉnh sai, tách được hai khả năng đó — nó đo cái *trần dưới* mà bất kỳ mạng nào
đặt trên cùng vector cũng phải vượt.

    PYTHONPATH=src .venv-dl/bin/python scripts/probe_embeddings.py --task category

Cần cache của ``dl/encode.py``. Ghi một dòng vào ``docs/04-results.md`` như mọi
lần chạy khác, để con số nằm cùng chỗ với phần còn lại.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vietjobs import config as C  # noqa: E402
from vietjobs import dataset as D  # noqa: E402
from vietjobs import evaluate as E  # noqa: E402
from vietjobs.dl import encode as ENC  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default=C.TASK_CATEGORY, choices=[C.TASK_CATEGORY, C.TASK_SALARY])
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--standardize", action="store_true")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--no-log", action="store_true", help="không ghi vào 04-results.md")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler

    Xtr = np.load(ENC.cache_path("train", args.task, args.max_len))
    Xva = np.load(ENC.cache_path("dev", args.task, args.max_len))
    tr = D.load_split("train")
    va = D.load_split("dev")

    if args.standardize:
        sc = StandardScaler().fit(Xtr)
        Xtr, Xva = sc.transform(Xtr), sc.transform(Xva)

    t0 = time.time()
    if args.task == C.TASK_CATEGORY:
        clf = LogisticRegression(max_iter=1000, C=args.C, n_jobs=-1).fit(Xtr, tr["category"])
        m = E.classification_metrics(va["category"].to_numpy(), clf.predict(Xva),
                                     list(clf.classes_))
        headline = (f"macroF1={m['f1_macro']:.4f} · F1={m['f1_weighted']:.4f} · "
                    f"acc={m['accuracy']:.4f}")
        model = f"logreg-probe(C={args.C})"
    else:
        ktr = ((tr["salary_disclosed"] == 1) & tr["salary_mid"].notna()).to_numpy()
        kva = ((va["salary_disclosed"] == 1) & va["salary_mid"].notna()).to_numpy()
        ytr = np.log1p(tr.loc[ktr, "salary_mid"].to_numpy())
        yva = np.log1p(va.loc[kva, "salary_mid"].to_numpy())
        reg = Ridge(alpha=1.0).fit(Xtr[ktr], ytr)
        m = E.regression_metrics(yva, reg.predict(Xva[kva]))
        headline = (f"MAE={m['mae_trieu']:.2f}tr · RMSE={m['rmse_trieu']:.2f}tr · "
                    f"R2log={m['r2_log']:.3f} · ±20%={m['within_20pct'] * 100:.1f}%")
        model = "ridge-probe(alpha=1.0)"

    seconds = time.time() - t0
    run_id = args.run_id or f"probe-{args.task[:3]}{'-std' if args.standardize else ''}"
    print(f"[{run_id}] {headline}  ({seconds:.0f}s)")

    if not args.no_log:
        with C.RESULTS_LOG.open("a", encoding="utf-8") as fh:
            fh.write(f"| {run_id} | {datetime.now(timezone.utc).strftime('%m-%d %H:%M')} "
                     f"| {args.task} | {model} on phobert-frozen | title+desc+req "
                     f"| segment{'+std' if args.standardize else ''} | dev | {m['n']} "
                     f"| {headline} | {seconds:.1f}s | probe |\n")


if __name__ == "__main__":
    main()
