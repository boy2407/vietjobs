"""T22 — khối cột thêm, lọc xung đột nhãn, lặp tin lớp nhỏ."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vietjobs import features as F
from vietjobs.dl import extra as XF


def _frame(n=4, province="hà nội"):
    return pd.DataFrame({
        "job_title": ["a"] * n, "description": ["d"] * n, "requirements_text": ["r"] * n,
        "category": ["x"] * n,
        "experience_months": ["12"] * n,
        "contract_type": ["Toàn thời gian, Bán thời gian"] * n,
        "province": [province] * n,
        "qualifications_text": ["cao đẳng ; đại học"] * n,
        "working_hours": ["Thứ 2 - Thứ 7 (từ 08:00 đến 17:30)"] * n,
        "languages_text": ["tiếng anh tiếng đức"] * n,
    })


def test_short_text_columns_are_safe_for_salary():
    # một cache dùng cho cả hai bài: cột nào chứa lương thô thì bài lương đọc được đáp án
    for col in XF.TEXT_COLUMNS:
        assert col not in F.UNMASKED_COLUMNS


def test_structured_parses_one_row():
    tr = _frame(XF.PROVINCE_MIN_ROWS)
    S, names = XF.structured(tr, XF.fit_structured(tr))
    row = dict(zip(names, S[0]))
    assert row["contract=toàn thời gian"] == row["contract=bán thời gian"] == 1
    assert row["province=hà nội"] == 1 and row["province=other"] == 0
    assert row["education"] == 2 and row["education_missing"] == 0     # bậc thấp nhất
    assert row["days_per_week"] == 6 and row["works_saturday"] == 1
    assert row["daily_span_h"] == pytest.approx(9.5)
    assert row["lang=tiếng anh"] == 1 and row["lang=other"] == 1


def test_province_list_is_fitted_on_train_only():
    stats = XF.fit_structured(_frame(XF.PROVINCE_MIN_ROWS))
    S, names = XF.structured(_frame(2, province="cần thơ"), stats)
    assert "province=cần thơ" not in names
    assert S[:, names.index("province=other")].tolist() == [1, 1]


def test_same_layout_for_every_split():
    stats = XF.fit_structured(_frame(XF.PROVINCE_MIN_ROWS))
    _, a = XF.structured(_frame(3), stats)
    _, b = XF.structured(_frame(5, province="huế"), stats)
    assert a == b


def test_exp_hours_skills_set_keeps_only_its_columns():
    tr = _frame(XF.PROVINCE_MIN_ROWS)
    _, names = XF.structured(tr, XF.fit_structured(tr))
    text_cols, prefixes = XF.EXTRA_SETS["exp_hours_skills"]
    kept = [names[i] for i in XF.select(names, prefixes)]
    assert text_cols == ("technical_skills_text",)
    assert "log1p_experience_months" in kept and "works_saturday" in kept
    assert not any(n.startswith(("province", "contract", "lang", "education")) for n in kept)
    assert XF.select(names, None) == list(range(len(names)))


def test_conflict_free_drops_whole_conflicting_group():
    df = _frame(5)
    df.loc[1, "category"] = "y"                       # nhóm a/d/r mang 2 nhãn → bỏ cả 5
    df.loc[4, ["job_title", "category"]] = ["b", "z"]  # nhóm khác, 1 nhãn → giữ
    assert XF.conflict_free(df).tolist() == [False, False, False, False, True]


def test_oversample_tops_small_classes_up_exactly():
    y = np.array(["a"] * 10 + ["b"] * 3)
    idx = XF.oversample(y, 8, seed=0)
    counts = pd.Series(y[idx]).value_counts()
    assert counts["a"] == 10 and counts["b"] == 8
    per_row = np.bincount(idx[y[idx] == "b"])[10:]
    assert per_row.max() - per_row.min() <= 1          # mỗi tin lặp đều nhau
