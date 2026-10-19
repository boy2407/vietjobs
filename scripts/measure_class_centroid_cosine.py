"""Cosine giữa tâm lớp của ba cặp nhầm lẫn (T14.3): khối A6 của skill dataset-diagnosis.

Tâm lớp lấy trên train (vector CafeBERT raw-len512, đúng thứ run chẩn đoán đọc); số lần nhầm
đếm trên dev từ predictions_dev.parquet. Không đọc test (Rule 5).
    .venv/bin/python scripts/measure_class_centroid_cosine.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vietjobs import config as C  # noqa: E402

RUN = "dl-cat-cafe-rnn-ce-s42"
VECTORS = ROOT / "artifacts" / "embeddings" / "cafebert-train-raw-len512.npy"
OUT = ROOT / "artifacts" / "eda" / "class-centroid-cosine.json"
PAIRS = [
    ("du_lịch_nhà_hàng_khách_sạn_dịch_vụ", "kinh_doanh_bán_hàng_chăm_sóc_khách_hàng"),
    ("thiết_kế_nghệ_thuật_giải_trí_truyền_hình_báo_chí", "xây_dựng_kiến_trúc_bất_động_sản"),
    ("du_lịch_nhà_hàng_khách_sạn_dịch_vụ", "sản_xuất_lao_động_phổ_thông_cơ_khí"),
]
# Cặp đối chứng chọn trước khi đo: hai ngành không chung nghiệp vụ.
CONTROL = ("công_nghệ_thông_tin_kỹ_thuật_số", "nông_nghiệp_năng_lượng_môi_trường")


def verdict(cos: float, confused: bool) -> str:
    if cos > 0.80 and confused:
        return "label_finding"
    if cos >= 0.65:
        return "feature_candidate" if cos <= 0.80 else "high_cosine_not_in_confusion_list"
    if cos < 0.50:
        return "model_error"
    return "between_0.50_and_0.65_no_rule"


def main() -> None:
    tr = pd.read_csv(C.SPLIT_DIR / "train.csv", usecols=["category"])
    X = np.load(VECTORS, mmap_mode="r")
    assert len(X) == len(tr), (len(X), len(tr))
    labs = sorted(tr["category"].unique())
    raw_c = np.vstack([np.asarray(X[(tr["category"] == l).values]).mean(axis=0) for l in labs])
    S = normalize(raw_c) @ normalize(raw_c).T
    # Vector mean-pool dùng chung một hướng trung bình, nên cosine thô cao ở mọi cặp: trừ trung bình train.
    Sc = normalize(raw_c - np.asarray(X).mean(axis=0)) @ normalize(raw_c - np.asarray(X).mean(axis=0)).T
    ix = {l: i for i, l in enumerate(labs)}
    off = S[np.triu_indices(len(labs), 1)]
    off_c = Sc[np.triu_indices(len(labs), 1)]

    p = pd.read_parquet(ROOT / "artifacts" / RUN / "predictions_dev.parquet")
    err = p[p["y_true"] != p["y_pred"]]
    top = err.groupby(["y_true", "y_pred"]).size().sort_values(ascending=False).head(12)
    top_pairs = {frozenset(k) for k in top.index}

    def row(a: str, b: str) -> dict:
        n_ab = int(((err["y_true"] == a) & (err["y_pred"] == b)).sum())
        n_ba = int(((err["y_true"] == b) & (err["y_pred"] == a)).sum())
        cos = float(S[ix[a], ix[b]])
        conf = frozenset((a, b)) in top_pairs
        return {"a": a, "b": b, "cosine": cos, "cosine_centered": float(Sc[ix[a], ix[b]]),
                "confused_a_to_b": n_ab, "confused_b_to_a": n_ba,
                "in_top12_confusions": conf, "verdict": verdict(cos, conf)}

    res = {
        "scope": "centroids fitted on train, confusions counted on dev",
        "vectors": str(VECTORS.relative_to(ROOT)),
        "n_train": len(tr),
        "run": RUN,
        "pairs": [row(a, b) for a, b in PAIRS],
        "control": row(*CONTROL),
        "all_pairs_cosine": {"min": float(off.min()), "mean": float(off.mean()), "max": float(off.max())},
        "all_pairs_cosine_centered": {"min": float(off_c.min()), "mean": float(off_c.mean()),
                                      "max": float(off_c.max())},
    }
    if res["control"]["cosine"] > 0.80:
        for r in res["pairs"]:
            r["verdict_raw_rule"], r["verdict"] = r["verdict"], "not_informative_raw_cosine"
        res["note"] = ("control pair cosine > 0.80: the raw cosine is high for every pair, so the A6 "
                       "verdicts above are not informative; read cosine_centered instead")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
