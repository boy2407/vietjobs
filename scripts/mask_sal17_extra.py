"""T23 — che lương ở 7 trường phụ của `splits-sal17` (tác giả 2026-11-24).

Bài lương sẽ đọc 11 cột. Ba cột chính và phúc lợi đã có bản ``*_masked`` từ
``dataset.clean``; bảy trường còn lại chưa có. Mỗi trường được đưa qua
``vitext.mask_salary`` và ghi thành cột mới ``<cột>_masked`` cạnh cột gốc.
Đo trước khi chạy: chỉ ``working_hours`` có số tiền (2 dòng train), nhưng che cả
bảy để mọi trường bài lương đọc đều đi cùng một đường (quy tắc số 3, số 4).

Áp dụng giống nhau cho train, dev, test: đây là bước đầu vào, không đổi dòng
hay nhãn nào. Ghi đè CSV trong ``data/processed/splits-sal17/``; bộ ``splits``
gốc không bị đụng tới.

    PYTHONPATH=src .venv/bin/python scripts/mask_sal17_extra.py
"""
from __future__ import annotations

import pandas as pd

from vietjobs import config as C
from vietjobs import vitext as V

DIR = C.PROCESSED_DIR / "splits-sal17"
EXTRA = ("location_raw", "qualifications_text", "technical_skills_text", "languages_text",
         "experience_raw", "working_hours", "contract_type")


def main() -> None:
    for split in C.SPLIT_NAMES:
        path = DIR / f"{split}{C.SPLIT_SUFFIX}"
        df = pd.read_csv(path, na_filter=False, dtype=str)
        for col in EXTRA:
            masked = df[col].map(V.mask_salary)
            df[f"{col}_masked"] = masked
            print(f"{split:5s} {col:22s} changed {int((masked != df[col]).sum())}")
        df.to_csv(path, index=False)


if __name__ == "__main__":
    main()
