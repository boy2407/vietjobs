"""T22 — cột ngoài ba trường văn bản, lọc tin xung đột nhãn, lặp tin lớp nhỏ.

Ba cột văn bản chính vẫn đi qua ``dl/text.py``. Module này thêm hai khối:

* **khối văn bản ngắn**: ``technical_skills_text`` và ``benefits_masked``, mỗi cột
  một vector CafeBERT riêng (len 128) — nhúng riêng thì không cột nào chiếm chỗ
  của mô tả trong 512 token.
* **khối có cấu trúc**: kinh nghiệm, hợp đồng, tỉnh, bậc học, giờ làm, ngôn ngữ.

Phúc lợi dùng bản ``_masked`` cho **cả hai** bài: 10,7 % tin ``train`` còn thẻ
``<SALARY>`` sau khi che, tức bản thô chứa con số lương (quy tắc số 3).

    PYTHONPATH=src .venv-dl/bin/python -m vietjobs.dl.extra --splits train dev
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np
import pandas as pd

from .. import dataset as D
from .. import features as F
from . import encode as ENC

TEXT_COLUMNS = ("technical_skills_text", "benefits_masked")
TEXT_MAX_LEN = 128
INPUT_FIELDS = ("job_title", "description", "requirements_text")

CONTRACTS = ("toàn thời gian", "thực tập", "bán thời gian", "khác", "thời vụ", "cộng tác viên")
LANGUAGES = ("tiếng anh", "tiếng trung", "tiếng nhật", "tiếng hàn")
# Bậc thấp nhất được nêu: tin thường ghi "cao đẳng trở lên", nên bậc thấp là ngưỡng.
EDUCATION = (("thpt", 0), ("phổ thông", 0), ("cấp 3", 0), ("trung cấp", 1),
             ("cao đẳng", 2), ("đại học", 3), ("thạc sĩ", 4), ("tiến sĩ", 5))
PROVINCE_MIN_ROWS = 100


def text_cache_path(split: str, column: str, encoder: str = "cafebert",
                    max_len: int = TEXT_MAX_LEN):
    return ENC.CACHE_DIR / f"{encoder}-{split}-xf-{column}-len{max_len}.npy"


def text_block(split: str, n_rows: int, encoder: str = "cafebert") -> np.ndarray:
    """Nối vector của hai cột văn bản ngắn; cache thiếu hoặc lệch số dòng thì dừng."""
    parts = []
    for col in TEXT_COLUMNS:
        path = text_cache_path(split, col, encoder)
        if not path.exists():
            raise SystemExit(f"thiếu {path.name} — chạy `python -m vietjobs.dl.extra` trước")
        arr = np.load(path)
        if len(arr) != n_rows:
            raise SystemExit(f"{path.name} có {len(arr)} dòng, split có {n_rows}")
        parts.append(arr)
    return np.concatenate(parts, axis=1)


def _days(s: str) -> set[int]:
    """Thứ 2..7 → 2..7, chủ nhật → 8; đọc cả khoảng "thứ 2 - thứ 7" lẫn ngày lẻ."""
    s = s.replace("chủ nhật", "thứ 8").replace("cn", "thứ 8")
    days: set[int] = set()
    for a, b in re.findall(r"thứ\s*([2-8])\s*[-–đến]+\s*thứ\s*([2-8])", s):
        days |= set(range(int(a), int(b) + 1))
    days |= {int(d) for d in re.findall(r"thứ\s*([2-8])", s)}
    return days


def _span(s: str) -> float:
    t = re.findall(r"(\d{1,2})[:h](\d{2})", s)
    if len(t) < 2:
        return np.nan
    (h1, m1), (h2, m2) = t[0], t[1]
    d = (int(h2) + int(m2) / 60) - (int(h1) + int(m1) / 60)
    return d + 24 if d < 0 else d


def _education(s: str) -> float:
    found = [lv for key, lv in EDUCATION if key in s]
    return float(min(found)) if found else np.nan


def fit_structured(train: pd.DataFrame) -> dict:
    """Những gì phải khớp trên ``train``: danh sách tỉnh và trung vị để điền chỗ trống."""
    counts = train["province"].astype(str).str.strip().value_counts()
    raw = _raw_structured(train)
    return {"provinces": sorted(counts[counts >= PROVINCE_MIN_ROWS].index.tolist()),
            "medians": {k: float(np.nanmedian(v)) for k, v in raw.items()}}


def _raw_structured(df: pd.DataFrame) -> dict[str, np.ndarray]:
    wh = df["working_hours"].astype(str).str.lower()
    days = [_days(s) for s in wh]
    return {
        "days_per_week": np.array([len(d) if d else np.nan for d in days], dtype=float),
        "daily_span_h": np.array([_span(s) for s in wh], dtype=float),
        "education": np.array([_education(s) for s in
                               df["qualifications_text"].astype(str).str.lower()], dtype=float),
    }


def structured(df: pd.DataFrame, stats: dict) -> tuple[np.ndarray, list[str]]:
    """Ma trận ``[n, k]`` float32 và tên ``k`` cột — thứ tự cố định cho mọi split."""
    cols: dict[str, np.ndarray] = {}
    exp = pd.to_numeric(df["experience_months"], errors="coerce").fillna(0).to_numpy(float)
    cols["log1p_experience_months"] = np.log1p(np.clip(exp, 0, None))
    contract = df["contract_type"].astype(str).str.lower()
    for c in CONTRACTS:
        cols[f"contract={c}"] = contract.str.split(",").apply(
            lambda xs, c=c: any(x.strip() == c for x in xs)).to_numpy(float)
    prov = df["province"].astype(str).str.strip()
    for p in stats["provinces"]:
        cols[f"province={p}"] = (prov == p).to_numpy(float)
    cols["province=other"] = (~prov.isin(stats["provinces"])).to_numpy(float)
    raw = _raw_structured(df)
    for k, v in raw.items():
        cols[f"{k}_missing"] = np.isnan(v).astype(float)
        cols[k] = np.where(np.isnan(v), stats["medians"][k], v)
    wh = df["working_hours"].astype(str).str.lower()
    days = [_days(s) for s in wh]
    cols["works_saturday"] = np.array([7 in d for d in days], dtype=float)
    cols["works_sunday"] = np.array([8 in d for d in days], dtype=float)
    cols["shift_work"] = wh.str.contains(r"\bca\b", regex=True).to_numpy(float)
    lang = df["languages_text"].astype(str).str.lower()
    for lg in LANGUAGES:
        cols[f"lang={lg}"] = lang.str.contains(lg, regex=False).to_numpy(float)
    rest = lang
    for lg in LANGUAGES + ("tiếng việt",):
        rest = rest.str.replace(lg, "", regex=False)
    cols["lang=other"] = rest.str.contains("tiếng", regex=False).to_numpy(float)
    names = list(cols)
    return np.stack([cols[n] for n in names], axis=1).astype(np.float32), names


def conflict_free(df: pd.DataFrame) -> np.ndarray:
    """``True`` cho dòng KHÔNG thuộc nhóm cùng ba cột đầu vào mà khác ``category``."""
    n_labels = df.groupby(list(INPUT_FIELDS), dropna=False)["category"].transform("nunique")
    return (n_labels <= 1).to_numpy()


def oversample(y: np.ndarray, n_min: int, seed: int) -> np.ndarray:
    """Chỉ số dòng sau khi lặp tin của lớp có ít hơn ``n_min`` dòng lên đúng ``n_min``.

    Lặp nguyên vòng trước, phần dư rút ngẫu nhiên không lặp lại — mỗi tin của lớp
    nhỏ xuất hiện số lần chênh nhau tối đa 1.
    """
    rng = np.random.default_rng(seed)
    idx = [np.arange(len(y))]
    for lab in np.unique(y):
        rows = np.flatnonzero(y == lab)
        need = n_min - len(rows)
        if need <= 0:
            continue
        reps, rem = divmod(need, len(rows))
        idx += [rows] * reps + [rng.choice(rows, rem, replace=False)]
    return np.concatenate(idx)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", nargs="+", default=["train", "dev"])
    ap.add_argument("--encoder", default="cafebert", choices=list(ENC.ENCODERS))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    if "test" in args.splits:
        raise SystemExit("test không được nhúng ở đây — xem docs/03-protocol.md, điều 2")
    for col in TEXT_COLUMNS:
        # Cùng một cache cho cả hai bài, nên cột phải an toàn với bài lương.
        assert col not in F.UNMASKED_COLUMNS, col
    for split in args.splits:
        df = D.load_split(split)
        for col in TEXT_COLUMNS:
            path = text_cache_path(split, col, args.encoder)
            if path.exists():
                print(f"  có sẵn {path.name}")
                continue
            texts = df[col].astype(str).str.strip().tolist()
            print(f"[{split}] {args.encoder} · {col} · {len(texts):,} tin · len{TEXT_MAX_LEN}")
            emb = ENC.encode_texts(texts, max_len=TEXT_MAX_LEN, batch=args.batch,
                                   device=args.device, encoder=args.encoder)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, emb)
            path.with_suffix(".json").write_text(json.dumps({
                "encoder": args.encoder, "model": ENC.ENCODERS[args.encoder]["name"],
                "split": split, "column": col, "max_len": TEXT_MAX_LEN,
                "pooling": "masked_mean", "n": len(texts), "shape": list(emb.shape),
            }, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"  -> {path.name} {emb.shape}")


if __name__ == "__main__":
    main()
