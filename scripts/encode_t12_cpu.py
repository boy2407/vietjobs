"""Mã hoá pooled cho ``test`` và ``ext37k-sal`` để chấm lại các best.pt (T12.1).

``encode.py`` từ chối ``test`` (docs/03-protocol.md điều 2). Việc này chỉ *đọc* test để
chấm các checkpoint đã có, theo quyền tác giả 2026-10-15 (AGENTS.md Rule 5 item 6); không
biến đổi test. Mặc định CPU (Rule 13), `--device cuda` trên máy thuê. Ghi cùng tên file mà
``encode.cache_path`` và ``eval_external_mtl.embed`` dùng, nên cache này khớp đường train.

    PYTHONPATH=src .venv-dl/bin/python scripts/encode_t12_cpu.py --jobs phobert:256 cafebert:512

Mỗi job ghi ``artifacts/embeddings/<tên>.npy`` một lần; có sẵn thì bỏ qua.
"""
from __future__ import annotations

import argparse
import ast
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vietjobs import config as C  # noqa: E402
from vietjobs import dataset as D  # noqa: E402
from vietjobs import external as X  # noqa: E402
from vietjobs.dl import encode as ENC  # noqa: E402
from vietjobs import vitext as V  # noqa: E402
from vietjobs.dl import text as T  # noqa: E402

SAL_FILE = X.EXTERNAL_DIR / "ext37k-sal.csv"
# ext37k-sal không có cột *_masked_seg (chưa tách từ). Tách từ chạy một lần trong
# .venv (có underthesea) và lưu ở đây; .venv-dl không có underthesea.
SEG_FILE = C.ARTIFACT_DIR / "t12" / "ext37k-sal-masked-seg.parquet"
SEG_SRC = ["job_title_masked", "description_masked", "requirements_masked"]


def segment_sal() -> None:
    """Chạy bằng .venv: tách từ 3 cột masked của ext37k-sal, ghi parquet cạnh artifacts."""
    df = pd.read_csv(SAL_FILE, keep_default_na=False, na_values=[""], usecols=["id"] + SEG_SRC)
    out = pd.DataFrame({"id": df["id"]})
    for c in SEG_SRC:
        out[c + "_seg"] = V.segment_many(df[c].tolist())
    SEG_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(SEG_FILE)
    print(f"segment: wrote {SEG_FILE} failures={V.segment_failures}", flush=True)


def cache_path_t12(split: str, family: str, encoder: str, max_len: int) -> Path:
    """Cùng quy ước tên với ``encode.cache_path``; ``split`` có thể là ``ext37k-sal``."""
    stem = f"{'' if encoder == ENC.DEFAULT_ENCODER else encoder + '-'}{split}-{family}-len{max_len}"
    return ENC.CACHE_DIR / f"{stem}.npy"


def ext_sal_frame(segmented: bool, family: str = "masked") -> pd.DataFrame:
    """``ext37k-sal.csv``. masked: thêm ``*_masked_seg`` từ SEG_FILE khi encoder cần.

    raw (cho phân lớp): cùng ``eval_external.prepare_frame`` đã dùng cho gold/test của 37K
    (preprocess như ``dataset.clean`` rồi tách từ) — cần underthesea, có trên máy thuê.
    """
    df = pd.read_csv(SAL_FILE, keep_default_na=False, na_values=[""])
    for col in ("labels", "label_ids"):
        df[col] = df[col].map(ast.literal_eval)
    if family == "raw":
        from eval_external import prepare_frame
        return prepare_frame(df)
    if segmented:
        seg = pd.read_parquet(SEG_FILE)
        assert (seg["id"].to_numpy() == df["id"].to_numpy()).all(), "SEG_FILE lệch thứ tự id"
        for c in SEG_SRC:
            df[c + "_seg"] = seg[c + "_seg"].to_numpy()
    return df


def job(split: str, family: str, encoder: str, max_len: int, device: str = "cpu") -> None:
    """family: raw (phân lớp) hoặc masked (lương)."""
    path = cache_path_t12(split, family, encoder, max_len)
    if path.exists():
        print(f"skip {path.name} (có sẵn)", flush=True)
        return
    if split == "test":
        df = D.load_split("test")
    else:
        df = ext_sal_frame(ENC.ENCODERS[encoder]["segmented"], family)
    task = C.TASK_SALARY if family == "masked" else C.TASK_CATEGORY
    texts = T.build_text(df, task=task, segmented=ENC.ENCODERS[encoder]["segmented"])
    t0 = time.time()
    emb = ENC.encode_texts(texts, max_len=max_len, batch=16, device=device, encoder=encoder)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    dt = time.time() - t0
    print(f"wrote {path.name} n={len(emb)} in {dt / 60:.1f} min ({len(emb) / dt:.2f} texts/s)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", nargs="+", help="encoder:max_len, theo thứ tự ưu tiên")
    ap.add_argument("--device", default="cpu", help="cpu (mặc định, Mac) | cuda (máy thuê)")
    ap.add_argument("--segment-only", action="store_true", help="chạy trong .venv: tách từ ext37k-sal")
    ap.add_argument("--ext-masked-only", action="store_true",
                    help="chỉ nhúng ext37k-sal họ masked (đủ cho bài lương); không nhúng test, không nhúng họ raw")
    args = ap.parse_args()
    if args.segment_only:
        segment_sal()
        return
    for spec in args.jobs:
        encoder, max_len = spec.split(":")
        max_len = int(max_len)
        job("ext37k-sal", "masked", encoder, max_len, args.device)
        if args.ext_masked_only:
            continue
        job("ext37k-sal", "raw", encoder, max_len, args.device)
        job("test", "masked", encoder, max_len, args.device)
        job("test", "raw", encoder, max_len, args.device)


if __name__ == "__main__":
    main()
