"""T22 R6-L4 — dựng bộ split `splits-r6text` (chỉ train, dev) cho bài lương, một text đầu vào.

Sáu trường được viết thành chữ và nối vào cuối ``requirements_masked`` (tác giả 2026-11-23):

    ... yêu cầu . Bằng cấp: … . Kỹ năng: … . Ngôn ngữ: … . Kinh nghiệm: … . Hình thức: … . Giờ làm: …

Trường trống thì bỏ. Đoạn thêm đi qua ``mask_salary`` như mọi cột ``*_masked`` (quy tắc số 3).
Đặt ở cuối để khi vượt 512 token thì phần bị cắt là đoạn thêm, không phải mô tả.
``test`` không được dựng ở đây (quy tắc số 5).

    PYTHONPATH=src python scripts/build_r6text_split.py
    VIETJOBS_SPLITS=splits-r6text PYTHONPATH=src python -m vietjobs.dl.encode \
        --task salary --encoder cafebert --max-len 512 --layers 4 --splits train dev
"""
from __future__ import annotations

import pandas as pd

from vietjobs import config as C
from vietjobs import dataset as D
from vietjobs import vitext as V

FIELDS = (("Bằng cấp", "qualifications_text"), ("Kỹ năng", "technical_skills_text"),
          ("Ngôn ngữ", "languages_text"), ("Kinh nghiệm", "experience_raw"),
          ("Hình thức", "contract_type"), ("Giờ làm", "working_hours"))
OUT = C.PROCESSED_DIR / "splits-r6text"


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
