"""Đường dữ liệu của nhánh học sâu — chủ yếu là canh rò rỉ lương."""
from __future__ import annotations

import pandas as pd
import pytest

from vietjobs import config as C
from vietjobs.dl import text as T


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "job_title": ["Nhân viên bán hàng lương 20 triệu"],
        "description": ["Mức lương 20 triệu/tháng"],
        "requirements_text": ["Tối thiểu 1 năm"],
        "job_title_masked": ["Nhân viên bán hàng lương <MONEY>"],
        "description_masked": ["Mức lương <MONEY>/tháng"],
        "requirements_masked": ["Tối thiểu 1 năm"],
        "job_title_seg": ["Nhân_viên bán_hàng lương 20 triệu"],
        "description_seg": ["Mức_lương 20 triệu/tháng"],
        "requirements_seg": ["Tối_thiểu 1 năm"],
        "job_title_masked_seg": ["Nhân_viên bán_hàng lương <MONEY>"],
        "description_masked_seg": ["Mức_lương <MONEY>/tháng"],
        "requirements_masked_seg": ["Tối_thiểu 1 năm"],
    })


@pytest.mark.parametrize("task", sorted(C.MASKED_TASKS))
def test_masked_tasks_never_see_a_number(task):
    """Bài lương đọc trúng số lương là đọc chính đáp án của nó."""
    for col in T.columns_for(task, segmented=True):
        assert "masked" in col, f"{task} đọc cột chưa che: {col}"
    blob = T.build_text(_frame(), task=task)[0]
    assert "20 triệu" not in blob
    assert "<MONEY>" in blob


def test_category_reads_raw_text():
    cols = T.columns_for(C.TASK_CATEGORY, segmented=True)
    assert not any("masked" in c for c in cols)
    assert all(c.endswith("_seg") for c in cols)


def test_segmented_is_the_default_because_phobert_was_trained_on_it():
    assert T.columns_for(C.TASK_CATEGORY) == T.columns_for(C.TASK_CATEGORY, segmented=True)
    assert T.build_text(_frame(), task=C.TASK_CATEGORY)[0].startswith("Nhân_viên bán_hàng")


def test_title_comes_first_so_truncation_eats_the_description_tail():
    blob = T.build_text(_frame(), task=C.TASK_CATEGORY)[0]
    assert blob.index("Nhân_viên") < blob.index("Mức_lương") < blob.index("Tối_thiểu")


def test_missing_column_fails_loudly():
    with pytest.raises(KeyError):
        T.build_text(_frame().drop(columns=["description_seg"]), task=C.TASK_CATEGORY)
