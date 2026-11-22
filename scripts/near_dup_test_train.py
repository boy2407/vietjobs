"""Đo mức trùng gần giữa ``test`` và ``train`` mà ``group_key`` không bắt được.

Chỉ đọc, không biến đổi ``test`` (Rule 5). Mỗi tin test được so với tin train giống nhất
theo Jaccard trên tập từ (job_title + description + requirements_text, chữ thường).
Với mỗi ngưỡng: số tin test, tỷ lệ cùng nhãn ngành, và độ chênh lương ở các cặp cả hai
đều công bố lương. Với ``test``, đo thêm so với train + dev gộp lại (``vs_train_dev``):
dev không dùng để học nhưng dùng để chọn mô hình.

    .venv/bin/python scripts/near_dup_test_train.py [dev|test]   # -> artifacts/eda/near_dup_<split>.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vietjobs import config as C  # noqa: E402

COLS = ["job_title", "description", "requirements_text", "category",
        "salary_disclosed", "salary_mid"]
THRESHOLDS = (1.0, 0.95, 0.9, 0.8)
CHUNK = 400


def read(split: str) -> pd.DataFrame:
    return pd.read_csv(C.SPLIT_DIR / f"{split}.csv", dtype=str,
                       keep_default_na=False, usecols=COLS)


def text(d: pd.DataFrame) -> pd.Series:
    return (d.job_title + " " + d.description + " " + d.requirements_text).str.lower()


def best_match(ref: pd.DataFrame, te: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Jaccard cao nhất của mỗi tin ``te`` với một tin ``ref``, và vị trí tin đó."""
    vec = CountVectorizer(binary=True, token_pattern=r"(?u)\b\w+\b")
    a = vec.fit_transform(text(ref)).astype(np.float32).tocsr()
    b = vec.transform(text(te)).astype(np.float32).tocsr()
    na, nb = np.asarray(a.sum(1)).ravel(), np.asarray(b.sum(1)).ravel()
    best, arg = np.zeros(b.shape[0]), np.zeros(b.shape[0], dtype=int)
    for i in range(0, b.shape[0], CHUNK):
        inter = (b[i:i + CHUNK] @ a.T).toarray()
        jac = inter / (nb[i:i + CHUNK, None] + na[None, :] - inter + 1e-9)
        best[i:i + CHUNK], arg[i:i + CHUNK] = jac.max(1), jac.argmax(1)
    return best, arg


def main() -> None:
    target = sys.argv[1] if len(sys.argv) > 1 else "test"
    tr, te = read("train"), read(target)
    best, arg = best_match(tr, te)

    sal_te = pd.to_numeric(te.salary_mid, errors="coerce").to_numpy()
    sal_tr = pd.to_numeric(tr.salary_mid, errors="coerce").to_numpy()[arg]
    both = (te.salary_disclosed.to_numpy() == "1") & \
           (tr.salary_disclosed.to_numpy()[arg] == "1")
    rows = []
    for t in THRESHOLDS:
        m = best >= t - 1e-9
        s = m & both
        diff = np.abs(sal_te[s] - sal_tr[s])
        rows.append({
            "jaccard_min": t,
            "test_rows": int(m.sum()),
            "test_share": round(float(m.mean()), 4),
            "same_category": round(float((te.category.to_numpy()[m]
                                          == tr.category.to_numpy()[arg[m]]).mean()), 4),
            "pairs_both_salary": int(s.sum()),
            "salary_equal_share": round(float((diff < 1e-9).mean()), 4),
            "salary_abs_diff_median": round(float(np.median(diff)), 2),
        })
    out = {"split": target, "n_test": len(te), "n_train": len(tr), "rows": rows}
    if target == "test":
        best_td, _ = best_match(pd.concat([tr, read("dev")], ignore_index=True), te)
        out["vs_train_dev"] = [{"jaccard_min": t, "test_rows": int((best_td >= t - 1e-9).sum())}
                               for t in THRESHOLDS]
        np.save(C.ARTIFACT_DIR / "eda" / "near_dup_test_best_traindev.npy", best_td)
    path = C.ARTIFACT_DIR / "eda" / f"near_dup_{target}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    # Jaccard cao nhất của từng tin, theo thứ tự dòng của <split>.csv: để tách phần "sạch".
    np.save(C.ARTIFACT_DIR / "eda" / f"near_dup_{target}_best.npy", best)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
