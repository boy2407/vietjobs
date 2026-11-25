"""T22 R7 — dựng bộ split `splits-r7text` (chỉ train, dev) cho bài lương, một text đầu vào.

Năm trường do tác giả chọn (2026-11-24) được viết thành chữ, nối vào cuối ``requirements_masked``:

    ... yêu cầu . Địa điểm: … . Bằng cấp: … . Kỹ năng: … . Ngôn ngữ: … . Kinh nghiệm: …

Trường trống thì bỏ. Đoạn thêm đi qua ``mask_salary`` (quy tắc số 3). ``test`` không dựng ở đây (quy tắc số 5).

    PYTHONPATH=src python scripts/build_r7text_split.py
"""
from __future__ import annotations

import pandas as pd

from vietjobs import config as C
from vietjobs import dataset as D
from vietjobs import vitext as V

FIELDS = (("Địa điểm", "location_raw"), ("Bằng cấp", "qualifications_text"),
          ("Kỹ năng", "technical_skills_text"), ("Ngôn ngữ", "languages_text"),
          ("Kinh nghiệm", "experience_raw"))
OUT = C.PROCESSED_DIR / "splits-r7text"


def extra_text(df: pd.DataFrame) -> pd.Series:
    def one(row) -> str:
        parts = [f"{name}: {str(row[col]).strip()}" for name, col in FIELDS
                 if str(row[col]).strip() and str(row[col]).strip().lower() != "nan"]
        return V.mask_salary(" . ".join(parts))
    return df.apply(one, axis=1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev"):
        df = pd.read_csv(D.split_path(split), na_filter=False, dtype=str)
        add = extra_text(df)
        df["requirements_masked"] = [f"{r} . {a}" if a else r
                                     for r, a in zip(df["requirements_masked"], add)]
        # PhoBERT đọc cột đã tách từ, nên đoạn thêm cũng phải được tách từ.
        add_seg = V.segment_many(add.tolist())
        df["requirements_masked_seg"] = [f"{r} . {a}" if a else r
                                         for r, a in zip(df["requirements_masked_seg"], add_seg)]
        df.to_csv(OUT / f"{split}.csv", index=False)
        print(f"{split}: {len(df):,} dòng · {(add != '').mean() * 100:.1f}% có đoạn thêm")


if __name__ == "__main__":
    main()
