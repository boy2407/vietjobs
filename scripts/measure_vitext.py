#!/usr/bin/env python3
"""Đo bằng chứng cho từng bước tiếng Việt trong docs/00-tong-quan.md §2.

Mỗi con số trong bảng "9 bước" của tài liệu tổng quan sinh ra từ file này.
Chạy lại bất cứ lúc nào để kiểm chứng:

    python scripts/measure_vitext.py

Đọc data/raw/VietJobs.csv, khử trùng lặp y hệt dataset.py (47.707 dòng),
rồi in từng dòng bằng chứng theo đúng thứ tự các bước.
"""
from __future__ import annotations

import collections
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from vietjobs import dataset as D  # noqa: E402
from vietjobs import vitext as V  # noqa: E402

TEXT_COLS = ["job_title", "description", "requirements_text", "benefits"]


def main() -> None:
    df = pd.read_csv(ROOT / "data/raw/VietJobs.csv", low_memory=False).drop_duplicates()
    n = len(df)
    col = {c: df[c].fillna("").astype(str) for c in TEXT_COLS}
    norm = {c: col[c].map(V.normalize_unicode) for c in TEXT_COLS}
    print(f"n = {n} dòng sau khử trùng lặp\n")

    # 1 — Unicode NFC
    def not_nfc(s: str) -> bool:
        return s != unicodedata.normalize("NFC", s)

    rows = sum(1 for i in range(n) if any(not_nfc(col[c].iloc[i]) for c in TEXT_COLS))
    print(f"1 NFC          {rows} dòng ({rows/n:.2%}) chứa ký tự tổ hợp chưa gộp")

    # 2 — tone placement
    style_a = re.compile("|".join(V._TONE_PAIRS))          # hòa, thúy
    style_b = re.compile("|".join(V._TONE_PAIRS.values()))  # hoà, thuý
    a = sum(1 for s in norm["description"] if style_a.search(s))
    b = sum(1 for s in norm["description"] if style_b.search(s))
    tok = collections.Counter(
        t for s in norm["job_title"] for t in s.lower().split()
    )
    merged = [(t, tok[t], V.normalize_tone(t), tok[V.normalize_tone(t)])
              for t in tok if V.normalize_tone(t) != t and V.normalize_tone(t) in tok]
    merged.sort(key=lambda x: -(x[1] + x[3]))
    print(f"2 dấu thanh    mô tả kiểu 'hòa' {a} ({a/n:.1%}) · kiểu 'hoà' {b} ({b/n:.1%})")
    print(f"               {len(merged)} cặp token tiêu đề gộp lại, lớn nhất: "
          + " · ".join(f"{x[0]}({x[1]})+{x[2]}({x[3]})" for x in merged[:3]))

    # 3 — abbreviations
    simple, contextual = V._abbreviations()
    hits: collections.Counter = collections.Counter()
    rows = 0
    for s in norm["description"]:
        found = {m.group(0).lower().rstrip(".") for m in V._TOKEN_RE.finditer(s)} & set(simple)
        if found:
            rows += 1
            hits.update(found)
    tp = sum(1 for s in norm["description"] if re.search(r"\bTP\b", s, re.I))
    print(f"3 viết tắt     {rows} mô tả ({rows/n:.1%}) chứa 1 trong {len(simple)} viết tắt · "
          + " ".join(f"{k}={v}" for k, v in hits.most_common(4)))
    print(f"               {len(contextual)} mục nhập nhằng theo ngữ cảnh · 'TP' xuất hiện ở {tp} mô tả")

    # 4 — salary masking
    for c in TEXT_COLS:
        masked = norm[c].map(V.mask_salary)
        k = int((masked != norm[c]).sum())
        print(f"4 che lương    {c:<18} {k} dòng ({k/n:.2%}) nhắc lại con số lương")

    # 5 — segmentation
    prev: collections.Counter = collections.Counter()
    for s in norm["job_title"]:
        w = s.lower().split()
        for i, t in enumerate(w):
            if t == "viên" and i:
                prev[w[i - 1]] += 1
    print(f"5 tách từ      âm tiết 'viên' xuất hiện {sum(prev.values())} lần sau "
          f"{len(prev)} âm tiết khác nhau · " + " ".join(f"{k}={v}" for k, v in prev.most_common(3)))

    # 6 — accent folding
    k = sum(1 for s in norm["job_title"] if s and V.fold_accents(s) == s)
    print(f"6 gấp dấu      {k} tiêu đề ({k/n:.2%}) viết không dấu")

    # 7 — provinces
    loc = df["location"].fillna("").astype(str)
    mapped = loc.map(V.normalize_province)
    raw_distinct = loc.str.strip().str.lower().replace("", pd.NA).nunique()
    changed = int((loc.str.strip().str.lower() != mapped.str.strip().str.lower()).sum())
    top = mapped.value_counts().index[0]
    variants = collections.Counter(
        x.strip().lower() for x, m in zip(loc, mapped) if m == top
    )
    print(f"7 chuẩn tỉnh   {raw_distinct} chuỗi địa điểm khác nhau -> {mapped.nunique()} giá trị · "
          f"{changed} dòng ({changed/n:.1%}) đổi giá trị")
    print(f"               '{top}' gom {len(variants)} biến thể: "
          + " · ".join(f"{k}={v}" for k, v in variants.most_common(3)))

    # 8 — stopwords
    raw_sw = sum(1 for l in (ROOT / "resources/stopwords_vi.txt").read_text(encoding="utf-8").splitlines()
              if l.strip() and not l.startswith("#"))
    print(f"8 từ dừng      {raw_sw} mục trong stopwords_vi.txt -> {len(V.stopwords())} dạng "
          "(kèm dạng gạch dưới cho văn bản đã tách từ)")

    # 9 — grouping key
    groups = pd.concat(
        [D.load_split(s)["group_id"] for s in ("train", "dev", "test")]
    )
    vc = groups.value_counts()
    dup = len(groups) - vc.size
    print(f"9 khoá nhóm    {len(groups)} dòng -> {vc.size} nhóm · {dup} dòng ({dup/len(groups):.1%}) "
          f"là tin đăng lại · nhóm lớn nhất {vc.max()} dòng")


if __name__ == "__main__":
    main()
