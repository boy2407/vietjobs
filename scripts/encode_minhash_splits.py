"""Vector CafeBERT (raw, 512 token) cho bộ chia MinHash: dùng lại vector cũ, chỉ mã hoá các dòng còn thiếu.

Bộ mới (data/processed/splits-minhash-1label) là các dòng của khung đã làm sạch 47.707 dòng (cột raw_row).
Vector cũ chỉ có cho train và dev của bộ chia cũ; các dòng từng nằm trong test cũ được mã hoá mới.
Chạy với biến môi trường để cache vào artifacts/embeddings/splits-minhash-1label/:
    VIETJOBS_SPLITS=splits-minhash-1label PYTHONPATH=src .venv-dl/bin/python scripts/encode_minhash_splits.py [--dry]
    ... --task salary --encoder phobert --max-len 256     # lương: họ masked
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vietjobs import config as C  # noqa: E402
from vietjobs import dataset as D  # noqa: E402
from vietjobs.dl import encode as ENC  # noqa: E402
from vietjobs.dl import text as T  # noqa: E402

OLD_CACHE = C.ARTIFACT_DIR / "embeddings"


def base_rows(new: pd.DataFrame, raw_dd: pd.DataFrame) -> np.ndarray:
    """Chỉ số dòng trong khung 47.707 dòng: dùng cột raw_row nếu có, không thì ghép bằng 18 cột gốc raw_*."""
    if "raw_row" in new.columns:
        return new["raw_row"].astype(int).to_numpy()
    cols = list(raw_dd.columns)
    sep = "\x1f"
    base_key = raw_dd[cols].fillna("").astype(str).agg(sep.join, axis=1)
    assert base_key.is_unique, "18 cột gốc phải xác định duy nhất một dòng"
    idx = pd.Series(range(len(base_key)), index=base_key.values)
    new_key = new[["raw_" + c for c in cols]].fillna("").astype(str).agg(sep.join, axis=1)
    rows = new_key.map(idx)
    assert rows.notna().all(), f"{int(rows.isna().sum())} dòng không tìm thấy trong file gốc"
    return rows.astype(int).to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="chỉ đếm, không mã hoá")
    ap.add_argument("--task", default="category", choices=list(C.TASKS))
    ap.add_argument("--encoder", default="cafebert", choices=list(ENC.ENCODERS))
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    ENCODER, MAX_LEN, TASK = args.encoder, args.max_len, args.task
    seg = ENC.ENCODERS[ENCODER]["segmented"]
    prefix = "" if ENCODER == ENC.DEFAULT_ENCODER else f"{ENCODER}-"
    old_name = lambda split: f"{prefix}{split}-{ENC.family(TASK)}-len{MAX_LEN}.npy"  # noqa: E731
    assert C.SPLIT_TAG, "đặt VIETJOBS_SPLITS=<tên thư mục bộ chia>"

    # 1. Dựng lại phép chia cũ trên khung 47.707 dòng để biết dòng nào nằm ở đâu trong cache cũ.
    raw = D.load_raw(C.RAW_CSV)
    raw_dd = raw.drop_duplicates().reset_index(drop=True)
    base = D.clean(raw, segment=False)
    old = D.two_stage_split(base)
    counts = old.value_counts().to_dict()
    assert counts == {"train": 34354, "dev": 3812, "test": 9541}, counts
    pos = old.groupby(old).cumcount().to_numpy()
    old_split = old.to_numpy()
    for name in ("train", "dev"):  # thứ tự dòng khớp file cũ: so tiêu đề, không đọc test
        ref = pd.read_csv(C.PROCESSED_DIR / "splits" / f"{name}.csv", usecols=["job_title"], keep_default_na=False)
        assert (ref.job_title.to_numpy() == base.job_title[old == name].to_numpy()).all(), name
    old_vec = {n: np.load(OLD_CACHE / old_name(n), mmap_mode="r") for n in ("train", "dev")}

    out_dir = ENC.CACHE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = {}
    for split in C.SPLIT_NAMES:
        new = D.load_split(split)
        rows = base_rows(new, raw_dd)
        have = np.isin(old_split[rows], ["train", "dev"])
        plan[split] = (new, rows, have)
        print(f"[{split}] {len(new):,} dòng · có vector cũ {int(have.sum()):,} · cần mã hoá {int((~have).sum()):,}")
    if args.dry:
        return

    for split, (new, rows, have) in plan.items():
        X = np.zeros((len(new), ENC.hidden_of(ENCODER)), dtype=np.float32)
        for i in np.flatnonzero(have):
            X[i] = old_vec[old_split[rows[i]]][pos[rows[i]]]
        need = np.flatnonzero(~have)
        texts_all = T.build_text(new, task=TASK, segmented=seg)
        if len(need):
            X[need] = ENC.encode_texts([texts_all[i] for i in need], max_len=MAX_LEN, batch=args.batch,
                                       device=args.device, encoder=ENCODER)
        path = ENC.cache_path(split, TASK, MAX_LEN, encoder=ENCODER)
        np.save(path, X)
        ENC._write_sidecar(path, split=split, task=TASK, max_len=MAX_LEN, n=len(new), encoder=ENCODER, texts=texts_all)
        print(f"  -> {path}  {X.shape}")

    # 2. Kiểm tra ánh xạ: mã hoá lại 48 dòng có vector cũ, so cosine với vector đã lấy.
    new, rows, have = plan["train"]
    X = np.load(ENC.cache_path("train", TASK, MAX_LEN, encoder=ENCODER))
    pick = np.random.default_rng(0).choice(np.flatnonzero(have), 48, replace=False)
    texts_all = T.build_text(new, task=TASK, segmented=seg)
    Y = ENC.encode_texts([texts_all[i] for i in pick], max_len=MAX_LEN, batch=args.batch,
                         device=args.device, encoder=ENCODER, quiet=True)
    cos = (X[pick] * Y).sum(1) / (np.linalg.norm(X[pick], axis=1) * np.linalg.norm(Y, axis=1))
    print(f"kiểm tra ánh xạ: cosine nhỏ nhất {cos.min():.5f} trên {len(pick)} dòng")
    assert cos.min() > 0.999, "ánh xạ vector cũ sai"


if __name__ == "__main__":
    main()
