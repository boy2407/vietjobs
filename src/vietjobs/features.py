"""Cổng quyết định mỗi bài toán được đọc cột nào — và chỉ có thế.

Bài phân loại đọc tin nguyên văn. Hai bài lương chỉ được đọc bản ``*_masked``,
nơi con số lương đã bị thay bằng ``<SALARY>``. Sai chỗ này **không báo lỗi** —
nó lặng lẽ tạo ra một mô hình lương đọc chính đáp án của mình rồi báo về những
con số đẹp mà vô nghĩa. ``tests/test_no_leak.py`` canh đúng điểm đó.

Đường học sâu đi qua chính cổng này: ``dl/text.py`` gọi :func:`resolve_column`.
"""
from __future__ import annotations

from . import config as C

# ---------------------------------------------------------------------------
# Which column a task is allowed to read
# ---------------------------------------------------------------------------
# Four mirrors exist for each free-text field, built once in dataset.py:
#
#   description              raw
#   description_seg          raw, word-segmented
#   description_masked       pay figures removed
#   description_masked_seg   pay figures removed, word-segmented
#
# The salary tasks are restricted to the masked pair. `resolve_column` is the
# only place that decision is made.

_MASKABLE = {
    "job_title": "job_title_masked",
    "description": "description_masked",
    "requirements_text": "requirements_masked",
    "benefits_text": "benefits_masked",
}

# Not every column has a segmented mirror — short list-ish fields like
# soft_skills are left alone because segmenting them buys nothing.
_SEG_NAME = {
    "job_title": "job_title_seg",
    "description": "description_seg",
    "requirements_text": "requirements_seg",
    "job_title_masked": "job_title_masked_seg",
    "description_masked": "description_masked_seg",
    "requirements_masked": "requirements_masked_seg",
    "benefits_masked": "benefits_masked_seg",
    "technical_skills_text": "technical_skills_seg",
    "qualifications_text": "qualifications_seg",
}


#: Columns a salary/disclosed model must never see. Used by the leak test.
#
# The segmented names come from ``_SEG_NAME``, not from ``f"{c}_seg"``. Building
# them by string concatenation produced ``requirements_text_seg`` while the real
# column is ``requirements_seg`` — a shield guarding a name that does not exist,
# which is exactly the silent failure this set was written to prevent.
# ``tests/test_no_leak.py`` now asserts every one of these names is real.
UNMASKED_COLUMNS = (
    frozenset(_MASKABLE)
    | {_SEG_NAME[c] for c in _MASKABLE if c in _SEG_NAME}
    | set(C.TARGET_LEAK_COLUMNS)
    | {"salary_mid", "salary_mid_log", "salary_is_range",
       "salary_disclosed", "salary_extreme"}
)


def resolve_column(name: str, *, task: str, segmented: bool) -> str:
    """Map a logical field to the actual dataframe column for this task.

    ``("description", task="salary", segmented=True)`` -> ``description_masked_seg``
    """
    col = _MASKABLE[name] if (task in C.MASKED_TASKS and name in _MASKABLE) else name
    if segmented:
        seg = _SEG_NAME.get(col)
        if seg is not None:
            return seg
    return col
