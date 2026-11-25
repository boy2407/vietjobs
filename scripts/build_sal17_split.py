"""T23 — dựng bộ split `splits-sal17`: chỉ tin có lương, bỏ tin trùng 17 cột khác nhãn.

Tác giả quyết định 2026-11-24 (ngoại lệ Rule 2/5): bộ này chia lại cả `test`.
Bộ `splits` gốc giữ nguyên; bộ mới chọn bằng ``VIETJOBS_SPLITS=splits-sal17``.

Lọc trên file gốc, sau ``drop_duplicates()``:
  1. bỏ mọi dòng thuộc nhóm trùng đủ 17 cột (khác `category`) — 5.220 dòng;
  2. giữ dòng ``salary_disclosed == 1`` (``derive_salary``).
Rồi ``clean`` + ``two_stage_split`` như ``dataset.build`` (8:2 rồi 9:1, seed 20260826).
Không gọi ``dataset.build`` vì hàm đó ghi đè ``data/processed/manifest.json``.

    PYTHONPATH=src .venv/bin/python scripts/build_sal17_split.py
"""
from __future__ import annotations

import json

from vietjobs import config as C
from vietjobs import dataset as D

OUT = C.PROCESSED_DIR / "splits-sal17"


def main() -> None:
    raw = D.load_raw().drop_duplicates().reset_index(drop=True)
    keys = [c for c in raw.columns if c != "category"]
    n_conf = raw.groupby(keys, dropna=False)["category"].transform("size")
    disc = D.derive_salary(raw)["salary_disclosed"].to_numpy()
    keep = raw[(n_conf.to_numpy() == 1) & (disc == 1)].reset_index(drop=True)
    print(f"raw dedup {len(raw):,} -> kept {len(keep):,}", flush=True)

    df = D.clean(keep, segment=True)
    df["split"] = D.two_stage_split(df, seed=C.SPLIT_SEED)
    assert (df.groupby("group_id")["split"].nunique() > 1).sum() == 0

    OUT.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in C.SPLIT_NAMES:
        part = df[df["split"] == split].drop(columns=["split"]).reset_index(drop=True)
        part.to_csv(OUT / f"{split}{C.SPLIT_SUFFIX}", index=False)
        counts[split] = {"rows": len(part), "groups": int(part["group_id"].nunique()),
                         "categories": int(part["category"].nunique())}
    meta = {"raw_dedup_rows": len(raw), "dropped_17col_conflict": int((n_conf > 1).sum()),
            "kept_rows": len(keep), "rows_after_clean": len(df), "split_seed": C.SPLIT_SEED,
            "split_scheme": "two-stage: train:test = 8:2, then train:dev = 9:1",
            "splits": counts}
    (OUT / "build_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
