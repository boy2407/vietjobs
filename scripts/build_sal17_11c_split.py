"""T23 — dựng `splits-sal17-11c`: cả hai bài đọc 11 cột trong một chuỗi (tác giả 2026-11-24).

Cùng cách với `splits-r6text`: tám trường phụ, đã che lương, được viết thành chữ
có nhãn và nối vào cuối ``requirements_masked`` (và bản tách từ của nó):

    ... yêu cầu . Địa điểm: … . Bằng cấp: … . Kỹ năng: … . Ngôn ngữ: … .
    Kinh nghiệm: … . Giờ làm: … . Hình thức: … . Phúc lợi: …

Vì vậy ``dl/text.py`` và ``features.resolve_column`` không đổi: chuỗi vẫn đi qua
cột ``*_masked`` (quy tắc số 3). Trường trống thì bỏ. Phúc lợi dài nhất nên đứng
cuối — khi vượt ``--max-len`` thì phần bị cắt là phần phụ.
Đọc `splits-sal17` (đã chạy ``mask_sal17_extra.py``), ghi sang thư mục mới để chạy
lại không nối hai lần. Chỉ họ ``masked`` đổi; cả ``category`` và ``salary`` đều đọc họ này (T23-train-lai).

    PYTHONPATH=src .venv/bin/python scripts/build_sal17_11c_split.py
    VIETJOBS_SPLITS=splits-sal17-11c PYTHONPATH=src .venv-dl/bin/python -m vietjobs.dl.encode \
        --task salary --splits train dev test
"""
from __future__ import annotations

import json
import sys

import pandas as pd

from vietjobs import config as C
from vietjobs import vitext as V

SRC = C.PROCESSED_DIR / "splits-sal17"
OUT = C.PROCESSED_DIR / "splits-sal17-11c"
FIELDS = (("Địa điểm", "location_raw_masked"), ("Bằng cấp", "qualifications_text_masked"),
          ("Kỹ năng", "technical_skills_text_masked"), ("Ngôn ngữ", "languages_text_masked"),
          ("Kinh nghiệm", "experience_raw_masked"), ("Giờ làm", "working_hours_masked"),
          ("Hình thức", "contract_type_masked"), ("Phúc lợi", "benefits_masked"))


def extra_text(df: pd.DataFrame) -> pd.Series:
    def one(row) -> str:
        parts = [f"{name}: {str(row[col]).strip()}" for name, col in FIELDS
                 if str(row[col]).strip() and str(row[col]).strip().lower() != "nan"]
        return " . ".join(parts)
    return df.apply(one, axis=1)


def write_manifest() -> None:
    """Manifest riêng cạnh CSV: manifest gốc + 7 cột ``*_masked`` mới kiểu ``str``.

    ``dataset.load_split`` từ chối cột lạ; manifest gốc của ``splits/`` giữ nguyên.
    ``--manifest-only`` chỉ ghi tệp này, không dựng lại CSV.
    """
    meta = json.loads(C.MANIFEST.read_text(encoding="utf-8"))
    cols = pd.read_csv(OUT / f"dev{C.SPLIT_SUFFIX}", nrows=0).columns
    types = meta["column_dtypes"]
    new = [c for c in cols if c not in types and not c.startswith("raw_")]
    types.update({c: "str" for c in new})
    meta["derived_from"] = "splits-sal17 + mask_sal17_extra.py + build_sal17_11c_split.py"
    (OUT / C.MANIFEST.name).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"manifest: +{len(new)} cột {new}")


def main() -> None:
    if "--manifest-only" in sys.argv:
        return write_manifest()
    OUT.mkdir(parents=True, exist_ok=True)
    for split in C.SPLIT_NAMES:
        df = pd.read_csv(SRC / f"{split}{C.SPLIT_SUFFIX}", na_filter=False, dtype=str)
        add = extra_text(df)
        df["requirements_masked"] = [f"{r} . {a}" if a else r
                                     for r, a in zip(df["requirements_masked"], add)]
        # PhoBERT đọc cột đã tách từ, nên đoạn thêm cũng phải được tách từ.
        add_seg = V.segment_many(add.tolist())
        df["requirements_masked_seg"] = [f"{r} . {a}" if a else r
                                         for r, a in zip(df["requirements_masked_seg"], add_seg)]
        df.to_csv(OUT / f"{split}{C.SPLIT_SUFFIX}", index=False)
        print(f"{split}: {len(df):,} dòng · {(add != '').mean() * 100:.1f}% có đoạn thêm", flush=True)
    write_manifest()


if __name__ == "__main__":
    main()
