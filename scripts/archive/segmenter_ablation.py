#!/usr/bin/env python3
"""So sánh hai bộ tách từ — underthesea và pyvi — trên bài toán 1.

    python scripts/segmenter_ablation.py

Câu hỏi: đổi bộ tách từ có cứu được bước 5 (tách từ) không? Bảng ablation trong
docs/02-vietnamese-nlp.md đo underthesea và thấy nó *kéo điểm xuống* dưới mức
không tách từ. Nhưng phép đo đó gắn với **một** bộ tách từ. Script này giữ
nguyên mọi thứ khác — cùng splits đã đóng băng, cùng LinearSVC C=0,02, cùng
chuẩn tỉnh — và chỉ đổi bộ tách từ.

Không đụng vào splits trên đĩa và không đụng vào tập test. Cột ``*_seg`` bản pyvi
được tính lại trong bộ nhớ từ cột văn bản đã sạch nằm sẵn trong parquet, rồi
cache xuống artifacts/segmenter-ablation/ để chạy lại cho nhanh.

Ba cấu hình, so bằng bootstrap **cặp** (cùng một ma trận resample cho cả ba):

    province             không tách từ          (dòng cat-T-svm-C0.02)
    segment+province     underthesea            (dòng cat-R-svm-C0.02-seg)
    segment+province     pyvi                   (mới)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from vietjobs import config as C  # noqa: E402
from vietjobs import evaluate as E  # noqa: E402
from vietjobs import features as F  # noqa: E402
from vietjobs import vitext as V  # noqa: E402
from vietjobs.dataset import load_split  # noqa: E402
from vietjobs.models import build_estimator  # noqa: E402

CACHE = ROOT / "artifacts" / "segmenter-ablation"

# Only the columns (task=category, scope=full) actually reads in segmented form.
# The remaining blocks (benefits, soft skills, languages) have no _seg mirror on
# the unmasked branch, so segmenting them could not reach the matrix anyway.
SEG_COLUMNS = {
    "job_title": "job_title_seg",
    "description": "description_seg",
    "requirements_text": "requirements_seg",
    "technical_skills_text": "technical_skills_seg",
    "qualifications_text": "qualifications_seg",
}


def resegment(split: str, df: pd.DataFrame, backend: str) -> tuple[pd.DataFrame, float | None]:
    """Trả về ``(df với cột *_seg tách lại bằng backend, số giây)``.

    Giây là ``None`` khi đọc từ cache — chỉ lần chạy đầu mới đo được thời gian.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{split}-{backend}.parquet"
    seconds: float | None = None
    if path.exists():
        seg = pd.read_parquet(path)
    else:
        V.use_segmenter(backend)
        if V.segmenter_name() != backend:
            raise SystemExit(f"không nạp được bộ tách từ {backend!r} — đã cài chưa?")
        out = {}
        t0 = time.perf_counter()
        for src, dst in SEG_COLUMNS.items():
            out[dst] = V.segment_many(df[src].fillna("").astype(str))
        seconds = time.perf_counter() - t0
        print(f"  {backend:12s} {split:5s} 5 cột · {len(df):,} dòng {seconds:7.1f}s", flush=True)
        seg = pd.DataFrame(out, index=df.index)
        seg.to_parquet(path)
    out_df = df.copy()
    for dst in SEG_COLUMNS.values():
        out_df[dst] = seg[dst].to_numpy()
    return out_df, seconds


def run(name: str, prep: F.PrepConfig, train_df, val_df, y_tr, y_val) -> dict:
    from sklearn.pipeline import Pipeline

    t0 = time.perf_counter()
    ct = F.build_features(C.TASK_CATEGORY, scope="full", prep=prep)
    est, _ = build_estimator(C.TASK_CATEGORY, "svm", C.RANDOM_SEED)
    est.set_params(C=0.02)
    pipe = Pipeline([("features", ct), ("model", est)])
    pipe.fit(train_df, y_tr)
    y_pred = pipe.predict(val_df)
    m = E.classification_metrics(y_val, y_pred, labels=sorted(set(y_tr)))
    m["name"], m["seconds"], m["y_pred"] = name, time.perf_counter() - t0, y_pred
    print(
        f"{name:34s} macroF1={m['f1_macro']:.4f}  acc={m['accuracy']:.4f}  "
        f"balAcc={m['balanced_accuracy']:.4f}  ({m['seconds']:.0f}s)",
        flush=True,
    )
    return m


def main() -> None:
    train_df, val_df = load_split("train"), load_split("val")
    y_tr = train_df["category"].to_numpy()
    y_val = val_df["category"].to_numpy()
    print(f"train {len(train_df):,} · val {len(val_df):,}\n")

    # Re-segment with BOTH backends over the same five columns. The underthesea
    # mirror already sits in the parquet; recomputing it is what makes the timing
    # comparison fair — a speed number measured on a different workload is noise.
    print("tách lại (chỉ 5 cột mà bài toán 1 thật sự đọc):")
    timing = {}
    tr_uts, t1 = resegment("train", train_df, "underthesea")
    va_uts, t2 = resegment("val", val_df, "underthesea")
    tr_pyvi, t3 = resegment("train", train_df, "pyvi")
    va_pyvi, t4 = resegment("val", val_df, "pyvi")
    # Tripwire: the two backends must produce DIFFERENT text. The first version of
    # this script wrote two byte-identical files — joblib had kept its old worker
    # pool alive, and those workers were still running the old segmenter. That
    # "comparison" looked perfectly valid on screen.
    same = (tr_uts["description_seg"] == tr_pyvi["description_seg"]).mean()
    if same > 0.99:
        raise SystemExit(
            f"{same:.1%} dòng giống hệt nhau — bộ tách từ không thật sự đổi"
        )
    print(f"  hai bộ khác nhau ở {(1 - same):.1%} số dòng mô tả")

    if None not in (t1, t2, t3, t4):
        timing = {"underthesea_seconds": t1 + t2, "pyvi_seconds": t3 + t4,
                  "speedup": (t1 + t2) / (t3 + t4)}
        print(f"  underthesea {timing['underthesea_seconds']:.1f}s · "
              f"pyvi {timing['pyvi_seconds']:.1f}s · nhanh gấp {timing['speedup']:.1f} lần")
    print()

    np.random.seed(C.RANDOM_SEED)
    results = [
        run("province (không tách từ)", F.PrepConfig(province=True), train_df, val_df, y_tr, y_val),
        run("segment+province · underthesea", F.PrepConfig(segment=True, province=True),
            tr_uts, va_uts, y_tr, y_val),
        run("segment+province · pyvi", F.PrepConfig(segment=True, province=True),
            tr_pyvi, va_pyvi, y_tr, y_val),
    ]

    idx = E.bootstrap_indices(len(y_val), n_boot=1000)
    scores = {r["name"]: E.bootstrap_scores(y_val, r["y_pred"], idx) for r in results}
    base = results[0]["name"]
    print("\nbootstrap cặp so với", base)
    for r in results[1:]:
        d = E.paired_delta(scores[r["name"]], scores[base])
        print(
            f"  {r['name']:32s} Δ={d['mean']:+.4f}  "
            f"CI95 [{d['ci95_lo']:+.4f}, {d['ci95_hi']:+.4f}]  P(>0)={d['p_gt_0']:.3f}"
        )
    head = E.paired_delta(scores[results[2]["name"]], scores[results[1]["name"]])
    print(
        f"\npyvi − underthesea               Δ={head['mean']:+.4f}  "
        f"CI95 [{head['ci95_lo']:+.4f}, {head['ci95_hi']:+.4f}]  P(>0)={head['p_gt_0']:.3f}"
    )

    # Written out so the table in docs/02-vietnamese-nlp.md has a checkable source.
    payload = {
        "timing": timing,
        "paired_pyvi_minus_underthesea": head,
        "runs": [
            {k: v for k, v in r.items() if k not in {"y_pred", "confusion_matrix"}}
            for r in results
        ],
        "paired_vs_no_segment": {
            r["name"]: E.paired_delta(scores[r["name"]], scores[base]) for r in results[1:]
        },
    }
    (CACHE / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    print(f"\nđã ghi {CACHE / 'metrics.json'}")


if __name__ == "__main__":
    main()
