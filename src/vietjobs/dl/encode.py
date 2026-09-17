"""Nhúng tin tuyển dụng bằng PhoBERT — chạy một lần, cache ra đĩa.

Bậc 4a giữ PhoBERT **đóng băng**: nó chỉ là bộ trích đặc trưng. Vì trọng số không
đổi, vector của một tin cũng không đổi giữa các epoch — nên tính một lần rồi cache
là đúng, không phải mẹo tiết kiệm. Toàn bộ vòng huấn luyện sau đó chạy trên ma
trận ``float32 [n, 768]``, nhanh gấp hàng trăm lần so với đẩy lại qua encoder.

    python -m vietjobs.dl.encode --task category --splits train val
    python -m vietjobs.dl.encode --task salary   --splits train val

Cache tách theo *họ cột*, không theo tên bài toán: ``category`` đọc văn bản thô,
``salary``/``disclosed`` đọc bản đã che lương, nên hai họ cho hai file khác nhau.
Trộn nhầm hai file này là rò rỉ lương im lặng — xem ``dl/text.py``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from .. import dataset as D
from . import text as T

MODEL_NAME = "vinai/phobert-base-v2"
HIDDEN = 768
CACHE_DIR = C.ARTIFACT_DIR / "embeddings"


def family(task: str) -> str:
    """Hai họ cột: ``masked`` cho bài lương, ``raw`` cho phân lớp."""
    return "masked" if task in C.MASKED_TASKS else "raw"


def cache_path(split: str, task: str, max_len: int) -> Path:
    return CACHE_DIR / f"{split}-{family(task)}-len{max_len}.npy"


def pick_device(name: str = "auto"):
    import torch

    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def encode_texts(texts, *, max_len: int = 256, batch: int = 32,
                 device: str = "auto", quiet: bool = False) -> np.ndarray:
    """Trả về ``[n, 768]`` — trung bình có mặt nạ của lớp ẩn cuối.

    Dùng mean pooling thay cho vector ``<s>``: với PhoBERT chưa tinh chỉnh, vector
    câu ``<s>`` không được huấn luyện cho tác vụ nào, còn trung bình token giữ
    được nhiều tín hiệu từ vựng hơn — thứ mà nhãn ngành nghề phụ thuộc vào.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    # PhoBERT chỉ có tokenizer "chậm"; ghim use_fast=False để một bản
    # transformers sau này không đổi tokenizer âm thầm — cache .npy sẽ lệch
    # mà không có lỗi nào. Lớp tokenizer thật được ghi vào sidecar .json.
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = AutoModel.from_pretrained(MODEL_NAME).eval().to(pick_device(device))
    dev = next(model.parameters()).device

    out = np.zeros((len(texts), HIDDEN), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            enc = tok(chunk, padding=True, truncation=True,
                      max_length=max_len, return_tensors="pt").to(dev)
            hidden = model(**enc).last_hidden_state           # [b, t, 768]
            mask = enc["attention_mask"].unsqueeze(-1).float()  # [b, t, 1]
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            out[i:i + len(chunk)] = pooled.cpu().numpy()
            if not quiet and (i // batch) % 20 == 0:
                done = i + len(chunk)
                rate = done / max(time.time() - t0, 1e-9)
                print(f"  {done:>6,}/{len(texts):,}  {rate:6.1f} tin/s  "
                      f"còn ~{(len(texts) - done) / max(rate, 1e-9) / 60:5.1f} phút",
                      flush=True)
    return out


def embeddings_for(split: str, task: str, *, max_len: int = 256, batch: int = 32,
                   device: str = "auto", force: bool = False) -> np.ndarray:
    """Đọc cache nếu có, nếu chưa thì tính rồi ghi cache."""
    path = cache_path(split, task, max_len)
    if path.exists() and not force:
        return np.load(path)
    df = D.load_split(split)
    texts = T.build_text(df, task=task, segmented=True)
    print(f"[{split}] {len(texts):,} tin · cột {T.columns_for(task)} · max_len={max_len}")
    emb = encode_texts(texts, max_len=max_len, batch=batch, device=device)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    _write_sidecar(path, split=split, task=task, max_len=max_len, n=len(texts))
    print(f"  -> {path}  {emb.shape}")
    return emb


def _write_sidecar(path: Path, *, split: str, task: str, max_len: int, n: int) -> None:
    """Ghi ``<cache>.json`` cạnh file ``.npy``: đúng những gì đã tạo ra các vector.

    Một cache ``.npy`` không tự nói nó được nhúng bằng tokenizer nào, cắt ở đâu,
    đọc cột nào. Sidecar là bằng chứng để so sánh hai cache với nhau; cache cũ
    không có sidecar vẫn đọc được bình thường.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    info = {
        "model": MODEL_NAME,
        "tokenizer_class": type(tok).__name__,
        "add_special_tokens": True,
        "max_len": max_len,
        "pooling": "masked_mean",
        "split": split,
        "family": family(task),
        "columns": T.columns_for(task),
        "n": n,
    }
    path.with_suffix(".json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, choices=list(C.TASKS))
    ap.add_argument("--splits", nargs="+", default=["train", "dev"])
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if "test" in args.splits:
        raise SystemExit("test không được nhúng ở đây — xem docs/03-protocol.md, điều 2")
    for split in args.splits:
        embeddings_for(split, args.task, max_len=args.max_len, batch=args.batch,
                       device=args.device, force=args.force)


if __name__ == "__main__":
    main()
