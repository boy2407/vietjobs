"""Đường dữ liệu của nhánh học sâu — chủ yếu là canh rò rỉ lương."""
from __future__ import annotations

import pandas as pd
import pytest

from vietjobs import config as C
from vietjobs import vitext as V
from vietjobs.dl import text as T


def _frame() -> pd.DataFrame:
    # Cột requirements cố tình chứa hư từ ("và", "của", "không") và một từ viết
    # tắt toàn chữ hoa ("IT") — hai thứ mà đường DL phải giữ nguyên.
    return pd.DataFrame({
        "job_title": ["Nhân viên bán hàng lương 20 triệu"],
        "description": ["Mức lương 20 triệu/tháng"],
        "requirements_text": ["Tối thiểu 1 năm và không yêu cầu bằng của ngành IT"],
        "job_title_masked": ["Nhân viên bán hàng lương <SALARY>"],
        "description_masked": ["Mức lương <SALARY>/tháng"],
        "requirements_masked": ["Tối thiểu 1 năm và không yêu cầu bằng của ngành IT"],
        "job_title_seg": ["Nhân_viên bán_hàng lương 20 triệu"],
        "description_seg": ["Mức_lương 20 triệu/tháng"],
        "requirements_seg": ["Tối_thiểu 1 năm và không yêu_cầu bằng của ngành IT"],
        "job_title_masked_seg": ["Nhân_viên bán_hàng lương <SALARY>"],
        "description_masked_seg": ["Mức_lương <SALARY>/tháng"],
        "requirements_masked_seg": ["Tối_thiểu 1 năm và không yêu_cầu bằng của ngành IT"],
    })


@pytest.mark.parametrize("task", sorted(C.MASKED_TASKS))
def test_masked_tasks_never_see_a_number(task):
    """Bài lương đọc trúng số lương là đọc chính đáp án của nó."""
    for col in T.columns_for(task, segmented=True):
        assert "masked" in col, f"{task} đọc cột chưa che: {col}"
    blob = T.build_text(_frame(), task=task)[0]
    assert "20 triệu" not in blob
    assert "<SALARY>" in blob


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


def test_dl_input_keeps_stopwords():
    """PhoBERT cần hư từ để dựng quan hệ giữa các từ — không được lọc stopword.

    `vitext.preprocess` biết bước này (`drop_stopwords`), nhưng `dataset.clean`
    không bao giờ bật nó. Test chốt cả hai tầng: mặc định của `preprocess` và
    chuỗi cuối cùng mà `build_text` trả về.
    """
    blob = T.build_text(_frame(), task=C.TASK_CATEGORY)[0]
    for word in ("và", "không", "của"):
        assert f" {word} " in blob, f"hư từ '{word}' bị lọc khỏi đầu vào DL"
    raw = "Tối thiểu 1 năm và không yêu cầu bằng của ngành"
    assert V.preprocess(raw) == raw
    assert V.remove_stopwords(raw) != raw, "danh sách stopword không còn tác dụng — test vô nghĩa"


def test_dl_input_is_not_lowercased():
    """Tokenizer của PhoBERT phân biệt hoa thường; chữ hoa đi thẳng vào mô hình."""
    blob = T.build_text(_frame(), task=C.TASK_CATEGORY)[0]
    assert "Nhân_viên" in blob
    assert "IT" in blob and " it" not in blob
    assert V.preprocess("Nhân viên IT") == "Nhân viên IT"


def test_missing_column_fails_loudly():
    with pytest.raises(KeyError):
        T.build_text(_frame().drop(columns=["description_seg"]), task=C.TASK_CATEGORY)
