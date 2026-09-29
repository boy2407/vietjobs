"""Nhúng tin tuyển dụng bằng PhoBERT — chạy một lần, cache ra đĩa.

Bậc 4a giữ PhoBERT **đóng băng**: nó chỉ là bộ trích đặc trưng. Vì trọng số không
đổi, vector của một tin cũng không đổi giữa các epoch — nên tính một lần rồi cache
là đúng, không phải mẹo tiết kiệm. Toàn bộ vòng huấn luyện sau đó chạy trên ma
trận ``float32 [n, 768]``, nhanh gấp hàng trăm lần so với đẩy lại qua encoder.

    python -m vietjobs.dl.encode --task category --splits train dev
    python -m vietjobs.dl.encode --task salary   --splits train dev
    python -m vietjobs.dl.encode --task category --splits train dev --pooling none

Cache tách theo *họ cột*, không theo tên bài toán: ``category`` đọc văn bản thô,
``salary``/``disclosed`` đọc bản đã che lương, nên hai họ cho hai file khác nhau.
Trộn nhầm hai file này là rò rỉ lương im lặng — xem ``dl/text.py``.

Hai mức cache cho cùng một họ cột (T8.1):

    {split}-{family}-len256.npy           float32 [n, 768]       trung bình có mặt nạ
    {split}-{family}-len256-tok.f16.npy   float16 [n, 256, 768]  từng token, chưa gộp
    {split}-{family}-len256-mask.npy      uint8   [n, 256]       1 = token thật, 0 = pad

Mức token giữ nguyên thứ mà phép trung bình vứt đi — thứ tự, phủ định, token
mạnh — để một head đọc chuỗi (Bi-GRU/Bi-LSTM) tự học cách gộp. 13,5 GB cho
``train`` nên nó được ghi tăng dần qua memmap, không bao giờ gom trong RAM.
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

# T8.6: encoder là một tham số. ``phobert`` giữ nguyên tên cache cũ (không phải
# nhúng lại 30 GB); encoder khác mang tiền tố riêng trong tên file. CafeBERT là
# XLM-R-large (SentencePiece), học trên văn bản **chưa** tách từ — đưa nó chuỗi
# có dấu ``_`` của underthesea là đưa nó thứ nó chưa từng thấy lúc tiền huấn luyện.
ENCODERS = {
    "phobert": {"name": "vinai/phobert-base-v2", "hidden": 768,
                "segmented": True, "use_fast": False},
    "cafebert": {"name": "uitnlp/CafeBERT", "hidden": 1024,
                 "segmented": False, "use_fast": True},
}
DEFAULT_ENCODER = "phobert"
MODEL_NAME = ENCODERS[DEFAULT_ENCODER]["name"]   # alias cho script cũ
HIDDEN = ENCODERS[DEFAULT_ENCODER]["hidden"]
CACHE_DIR = C.ARTIFACT_DIR / "embeddings"


def family(task: str) -> str:
    """Hai họ cột: ``masked`` cho bài lương, ``raw`` cho phân lớp."""
    return "masked" if task in C.MASKED_TASKS else "raw"


def hidden_of(encoder: str) -> int:
    return ENCODERS[encoder]["hidden"]


def cache_path(split: str, task: str, max_len: int, kind: str = "pooled",
               encoder: str = DEFAULT_ENCODER) -> Path:
    """``kind``: ``pooled`` (tên cũ, giữ nguyên để cache đã có vẫn hợp lệ), ``tok``, ``mask``."""
    prefix = "" if encoder == DEFAULT_ENCODER else f"{encoder}-"
    stem = f"{prefix}{split}-{family(task)}-len{max_len}"
    if kind == "pooled":
        return CACHE_DIR / f"{stem}.npy"
    if kind == "tok":
        return CACHE_DIR / f"{stem}-tok.f16.npy"
    if kind == "mask":
        return CACHE_DIR / f"{stem}-mask.npy"
    raise ValueError(f"kind lạ: {kind}")


def _load_model(device: str, encoder: str = DEFAULT_ENCODER):
    """Tokenizer + encoder đóng băng, dùng chung cho cả hai mức cache."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    # PhoBERT chỉ có tokenizer "chậm"; ghim use_fast=False để một bản
    # transformers sau này không đổi tokenizer âm thầm — cache .npy sẽ lệch
    # mà không có lỗi nào. Lớp tokenizer thật được ghi vào sidecar .json.
    spec = ENCODERS[encoder]
    tok = AutoTokenizer.from_pretrained(spec["name"], use_fast=spec["use_fast"])
    model = AutoModel.from_pretrained(spec["name"]).eval().to(pick_device(device))
    return tok, model, next(model.parameters()).device


def _progress(i: int, n: int, batch: int, t0: float, quiet: bool) -> None:
    if quiet or (i // batch) % 20:
        return
    done = min(i + batch, n)
    rate = done / max(time.time() - t0, 1e-9)
    print(f"  {done:>6,}/{n:,}  {rate:6.1f} tin/s  "
          f"còn ~{(n - done) / max(rate, 1e-9) / 60:5.1f} phút", flush=True)


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
                 device: str = "auto", quiet: bool = False,
                 encoder: str = DEFAULT_ENCODER) -> np.ndarray:
    """Trả về ``[n, 768]`` — trung bình có mặt nạ của lớp ẩn cuối.

    Dùng mean pooling thay cho vector ``<s>``: với PhoBERT chưa tinh chỉnh, vector
    câu ``<s>`` không được huấn luyện cho tác vụ nào, còn trung bình token giữ
    được nhiều tín hiệu từ vựng hơn — thứ mà nhãn ngành nghề phụ thuộc vào.
    """
    import torch

    tok, model, dev = _load_model(device, encoder)
    out = np.zeros((len(texts), hidden_of(encoder)), dtype=np.float32)
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
            _progress(i, len(texts), batch, t0, quiet)
    return out


def encode_tokens(texts, tok_out: np.memmap, mask_out: np.ndarray, *, max_len: int = 256,
                  batch: int = 32, device: str = "auto", quiet: bool = False,
                  encoder: str = DEFAULT_ENCODER) -> None:
    """Ghi lớp ẩn cuối **từng token** vào ``tok_out [n, max_len, 768]`` float16 và
    mặt nạ vào ``mask_out [n, max_len]`` uint8 — không gộp gì cả.

    Tokenizer pad theo batch (``t`` ≤ ``max_len``), nên mỗi batch được đặt vào
    ``t`` cột đầu; phần còn lại giữ 0 và mask = 0. Chính trung bình có mặt nạ của
    mảng này phải bằng ``encode_texts`` — ``tests/test_dl_encode.py`` canh điều đó.
    """
    import torch

    tok, model, dev = _load_model(device, encoder)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            enc = tok(chunk, padding=True, truncation=True,
                      max_length=max_len, return_tensors="pt").to(dev)
            hidden = model(**enc).last_hidden_state              # [b, t, 768]
            m = enc["attention_mask"]
            # Ghi 0 ở vị trí pad: PhoBERT vẫn trả một vector cho token <pad>, và
            # một head quên mặt nạ sẽ đọc phải nó. Mảng trên đĩa vì thế tự bảo vệ.
            hidden = (hidden * m.unsqueeze(-1)).to(torch.float16).cpu().numpy()
            mask = m.to(torch.uint8).cpu().numpy()
            t = hidden.shape[1]
            tok_out[i:i + len(chunk), :t] = hidden
            mask_out[i:i + len(chunk), :t] = mask
            _progress(i, len(texts), batch, t0, quiet)
    tok_out.flush()


def embeddings_for(split: str, task: str, *, max_len: int = 256, batch: int = 32,
                   device: str = "auto", force: bool = False,
                   encoder: str = DEFAULT_ENCODER) -> np.ndarray:
    """Đọc cache nếu có, nếu chưa thì tính rồi ghi cache."""
    path = cache_path(split, task, max_len, encoder=encoder)
    if path.exists() and not force:
        return np.load(path)
    df = D.load_split(split)
    seg = ENCODERS[encoder]["segmented"]
    texts = T.build_text(df, task=task, segmented=seg)
    print(f"[{split}] {encoder} · {len(texts):,} tin · cột {T.columns_for(task, segmented=seg)} · max_len={max_len}")
    emb = encode_texts(texts, max_len=max_len, batch=batch, device=device, encoder=encoder)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    _write_sidecar(path, split=split, task=task, max_len=max_len, n=len(texts),
                   encoder=encoder, texts=texts)
    print(f"  -> {path}  {emb.shape}")
    return emb


def token_cache_for(split: str, task: str, *, max_len: int = 256, batch: int = 32,
                    device: str = "auto", force: bool = False,
                    encoder: str = DEFAULT_ENCODER):
    """Trả về ``(tok, mask)`` — ``tok`` là memmap chỉ-đọc float16 ``[n, max_len, 768]``,
    ``mask`` là mảng uint8 ``[n, max_len]`` nạp hẳn vào RAM (vài MB).

    Tính rồi ghi cache nếu chưa có. Memmap được mở ở chế độ ghi bằng
    ``open_memmap`` và điền từng batch, nên ``train`` 13,5 GB không bao giờ nằm
    trọn trong RAM — đọc lại cũng qua memmap, head lấy batch nào thì đĩa trả batch đó.
    """
    tok_path = cache_path(split, task, max_len, "tok", encoder)
    mask_path = cache_path(split, task, max_len, "mask", encoder)
    hidden = hidden_of(encoder)
    if tok_path.exists() and mask_path.exists() and not force:
        return np.load(tok_path, mmap_mode="r"), np.load(mask_path)
    df = D.load_split(split)
    seg = ENCODERS[encoder]["segmented"]
    texts = T.build_text(df, task=task, segmented=seg)
    print(f"[{split}] {encoder} · {len(texts):,} tin · cột {T.columns_for(task, segmented=seg)} · max_len={max_len} · token-level")
    tok_path.parent.mkdir(parents=True, exist_ok=True)
    tok_mm = np.lib.format.open_memmap(tok_path, mode="w+", dtype=np.float16,
                                       shape=(len(texts), max_len, hidden))
    mask = np.zeros((len(texts), max_len), dtype=np.uint8)
    encode_tokens(texts, tok_mm, mask, max_len=max_len, batch=batch, device=device,
                  encoder=encoder)
    del tok_mm  # đóng handle ghi trước khi mở lại chỉ-đọc
    np.save(mask_path, mask)
    _write_sidecar(tok_path, split=split, task=task, max_len=max_len, n=len(texts),
                   pooling="none", dtype="float16", shape=[len(texts), max_len, hidden],
                   encoder=encoder, texts=texts)
    print(f"  -> {tok_path}  ({len(texts):,}, {max_len}, {hidden}) float16 · mask {mask_path.name}")
    return np.load(tok_path, mmap_mode="r"), mask


def _write_sidecar(path: Path, *, split: str, task: str, max_len: int, n: int,
                   pooling: str = "masked_mean", dtype: str = "float32",
                   shape: list | None = None, encoder: str = DEFAULT_ENCODER,
                   texts: list | None = None) -> None:
    """Ghi ``<cache>.json`` cạnh file ``.npy``: đúng những gì đã tạo ra các vector.

    Một cache ``.npy`` không tự nói nó được nhúng bằng tokenizer nào, cắt ở đâu,
    đọc cột nào. Sidecar là bằng chứng để so sánh hai cache với nhau; cache cũ
    không có sidecar vẫn đọc được bình thường.
    """
    from transformers import AutoTokenizer

    spec = ENCODERS[encoder]
    tok = AutoTokenizer.from_pretrained(spec["name"], use_fast=spec["use_fast"])
    # Tỷ lệ tin bị cắt đo lại cho từng tokenizer: con số 19,8 % là của BPE PhoBERT,
    # SentencePiece của XLM-R chia chuỗi khác nên cắt ở chỗ khác.
    truncated = None
    if texts is not None:
        lens = [len(tok(t, add_special_tokens=True)["input_ids"]) for t in texts]
        truncated = float(np.mean([n_ > max_len for n_ in lens]))
    info = {
        "model": spec["name"],
        "encoder": encoder,
        "segmented": spec["segmented"],
        "truncated_frac": truncated,
        "tokenizer_class": type(tok).__name__,
        "add_special_tokens": True,
        "max_len": max_len,
        "pooling": pooling,
        "dtype": dtype,
        "shape": shape or [n, spec["hidden"]],
        "split": split,
        "family": family(task),
        "columns": T.columns_for(task, segmented=spec["segmented"]),
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
    ap.add_argument("--encoder", default=DEFAULT_ENCODER, choices=list(ENCODERS),
                    help="phobert (mặc định, cache cũ) | cafebert (T8.6, 1024 chiều, không tách từ)")
    ap.add_argument("--pooling", default="mean", choices=["mean", "none"],
                    help="mean: một vector 768/tin (mặc định, cache cũ); "
                         "none: giữ từng token [n, max_len, 768] float16 cho head đọc chuỗi (T8)")
    args = ap.parse_args()

    if "test" in args.splits:
        raise SystemExit("test không được nhúng ở đây — xem docs/03-protocol.md, điều 2")
    for split in args.splits:
        if args.pooling == "none":
            token_cache_for(split, args.task, max_len=args.max_len, batch=args.batch,
                            device=args.device, force=args.force, encoder=args.encoder)
        else:
            embeddings_for(split, args.task, max_len=args.max_len, batch=args.batch,
                           device=args.device, force=args.force, encoder=args.encoder)


if __name__ == "__main__":
    main()
