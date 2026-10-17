"""Cấu trúc nhóm của train và dev (T2.6): nhóm lớn nhất, nhóm đa nhãn, trùng nhóm giữa hai tập.

Chỉ đọc train.csv và dev.csv; không đọc test (Rule 5). In ra stdout.
    .venv/bin/python scripts/measure_group_structure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vietjobs import config as C  # noqa: E402


def main() -> None:
    sp = {n: pd.read_csv(C.SPLIT_DIR / f"{n}.csv") for n in ("train", "dev")}
    for n, d in sp.items():
        size = d.groupby("group_id").size()
        ncat = d.groupby("group_id")["category"].nunique()
        ml = d["group_id"].map(ncat) > 1
        print(f"{n}: {len(d)} dòng, {len(size)} nhóm, {len(d) / len(size):.3f} dòng/nhóm, "
              f"nhóm lớn nhất {size.max()}, nhóm >=2 dòng {(size >= 2).sum()}, "
              f"nhóm >=3 dòng {(size >= 3).sum()}, nhóm >=2 nhãn {(ncat > 1).sum()}, "
              f"dòng trong nhóm >=2 dòng {size[size >= 2].sum() / len(d):.4f}, "
              f"dòng trong nhóm đa nhãn {ml.mean():.4f}")
        print(d.assign(ml=ml).groupby("category")["ml"].mean().round(3).sort_values().to_string())
    both = set(sp["dev"]["group_id"]) & set(sp["train"]["group_id"])
    print(f"nhóm dev nằm trong train: {len(both) / sp['dev']['group_id'].nunique():.4f}")


if __name__ == "__main__":
    main()
