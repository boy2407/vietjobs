"""Dựng chuỗi đầu vào cho PhoBERT — nơi duy nhất quyết định model đọc cột nào.

Hai điều bắt buộc, cả hai đều đo được bằng ``tests/test_dl_text.py``:

1. **Che lương.** Bài ``salary`` và ``disclosed`` chỉ được đọc bản ``*_masked``.
   Quyết định đó không nằm ở đây mà ở ``features.resolve_column`` — đường DL đi
   qua đúng cái cổng mà đường TF-IDF đã đi (quy tắc số 3).
2. **Tách từ.** PhoBERT được huấn luyện trên văn bản đã tách từ
   ("nhân_viên kinh_doanh"). Đưa văn bản chưa tách vào là đưa sai phân bố so với
   lúc tiền huấn luyện, nên ``segmented=True`` là mặc định ở đây — ngược với kết
   luận của đường TF-IDF, và đó là điều nên nói rõ trong tài liệu.
"""
from __future__ import annotations

import pandas as pd

from .. import features as F

# Ba trường văn bản, ghép theo thứ tự này. Tiêu đề đứng trước vì khi cắt còn 256
# token thì phần bị cắt phải là phần đuôi mô tả, không phải phần nhận dạng nghề.
FIELDS = ("job_title", "description", "requirements_text")
SEP = " . "


def columns_for(task: str, *, segmented: bool = True) -> list[str]:
    """Tên cột thật mà bài toán này được phép đọc."""
    return [F.resolve_column(f, task=task, segmented=segmented) for f in FIELDS]


def build_text(df: pd.DataFrame, *, task: str, segmented: bool = True) -> list[str]:
    """Ghép ba trường thành một chuỗi cho mỗi dòng."""
    cols = columns_for(task, segmented=segmented)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"thiếu cột {missing} — chạy lại `python -m vietjobs.dataset build`")
    parts = [df[c].fillna("").astype(str).str.strip() for c in cols]
    joined = parts[0]
    for p in parts[1:]:
        joined = joined.str.cat(p, sep=SEP)
    return joined.str.replace(r"\s+", " ", regex=True).str.strip().tolist()
