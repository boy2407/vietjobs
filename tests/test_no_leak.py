"""The salary model must not be able to read its own answer.

This is the single highest-consequence bug available in this project. If a
salary model reads an unmasked text column, ~5% of postings hand it the target
directly. Nothing raises, nothing looks wrong, and the reported MAE is simply
not a generalisation estimate.

These tests are cheap and run without the built dataset.
"""
from __future__ import annotations

import pytest

from vietjobs import config as C
from vietjobs import features as F
from vietjobs import vitext as V

SCOPES = ["title", "full"]


@pytest.mark.parametrize("scope", SCOPES)
@pytest.mark.parametrize("segment", [False, True])
@pytest.mark.parametrize("task", [C.TASK_SALARY, C.TASK_DISCLOSED])
def test_salary_tasks_never_read_unmasked_text(task, scope, segment):
    prep = F.PrepConfig(segment=segment, charfold=True, province=True)
    ct = F.build_features(task, scope, prep)
    used = set(F.source_columns(ct))
    leaked = used & F.UNMASKED_COLUMNS
    assert not leaked, (
        f"task={task} scope={scope} segment={segment} reads unmasked column(s): "
        f"{sorted(leaked)}"
    )


@pytest.mark.parametrize("scope", SCOPES)
@pytest.mark.parametrize("segment", [False, True])
def test_salary_tasks_actually_read_the_masked_mirrors(scope, segment):
    """The mirror must be wired up, not merely 'not the raw column'."""
    prep = F.PrepConfig(segment=segment, charfold=True, province=True)
    used = set(F.source_columns(F.build_features(C.TASK_SALARY, scope, prep)))
    expected = "job_title_masked_seg" if segment else "job_title_masked"
    assert expected in used, f"expected {expected} among {sorted(used)}"


@pytest.mark.parametrize("scope", SCOPES)
def test_classification_does_read_the_raw_text(scope):
    """The mirror-swap must apply ONLY to the salary tasks."""
    used = set(F.source_columns(F.build_features(C.TASK_CATEGORY, scope, F.PREP_RAW)))
    assert "job_title" in used
    assert not (used & {"job_title_masked", "description_masked"})


def test_no_task_reads_a_raw_salary_column():
    for task in C.TASKS:
        for scope in ["structured", "title", "full"]:
            used = set(F.source_columns(F.build_features(task, scope, F.PREP_FULL)))
            assert not (used & set(C.TARGET_LEAK_COLUMNS)), f"{task}/{scope}"


def test_resolve_column_swaps_only_for_salary_tasks():
    assert F.resolve_column("description", task=C.TASK_CATEGORY, segmented=False) == "description"
    assert F.resolve_column("description", task=C.TASK_SALARY, segmented=False) == "description_masked"
    assert F.resolve_column("description", task=C.TASK_SALARY, segmented=True) == "description_masked_seg"
    assert F.resolve_column("description", task=C.TASK_CATEGORY, segmented=True) == "description_seg"


def test_masking_actually_removes_the_figure():
    """End to end: a posting that states its pay loses the number."""
    posting = "Mức lương 18 triệu/tháng, thưởng theo doanh số"
    masked = V.preprocess(posting, mask=True)
    assert "18" not in masked
    assert "<SALARY>" in masked
    # ... but the classification branch keeps it, because pay is a useful
    # signal for guessing the job category.
    assert "18" in V.preprocess(posting, mask=False)


@pytest.mark.parametrize("segment", [False, True])
def test_char_channel_reads_unsegmented_text(segment):
    """Underscores from the segmenter would pollute character n-grams."""
    prep = F.PrepConfig(segment=segment, charfold=True)
    ct = F.build_features(C.TASK_CATEGORY, "title", prep)
    char_cols = [cols for name, _est, cols in ct.transformers if name == "title_char"]
    assert char_cols == ["job_title"]


# ---------------------------------------------------------------------------
# The shield must guard names that exist
# ---------------------------------------------------------------------------
#
# UNMASKED_COLUMNS once contained "requirements_text_seg" while the real column
# is "requirements_seg". A misspelled entry protects nothing and fails silently:
# every test above still passed, because a name that never appears in the data
# also never appears in the intersection. These three tests close that class of
# bug, and pin down the measurement that cleared docs/09-lo-trinh.md Ưu tiên 0b.

_SPLIT = C.PROCESSED_DIR / "splits" / "train.parquet"
_needs_data = pytest.mark.skipif(
    not _SPLIT.exists(), reason="cần data/processed/splits/train.parquet"
)


def _schema() -> set[str]:
    import pyarrow.parquet as pq

    return set(pq.read_schema(_SPLIT).names)


@_needs_data
def test_every_segmented_name_is_a_real_column():
    """_SEG_NAME is what resolve_column actually returns — a typo here routes
    the model to a column that does not exist, or worse, silently does not."""
    missing = sorted(v for v in F._SEG_NAME.values() if v not in _schema())
    assert missing == [], f"_SEG_NAME points at columns that do not exist: {missing}"


@_needs_data
def test_every_guarded_name_is_real_or_deliberately_dropped():
    schema = _schema()
    # The four raw salary columns are dropped by dataset.clean, so they are
    # absent on purpose. Everything else in the shield must be a live column.
    allowed_absent = set(C.TARGET_LEAK_COLUMNS)
    missing = sorted(c for c in F.UNMASKED_COLUMNS
                     if c not in schema and c not in allowed_absent)
    assert missing == [], f"UNMASKED_COLUMNS guards non-existent names: {missing}"


@_needs_data
def test_skill_columns_carry_no_pay_figures():
    """soft_skills_text, qualifications_text and technical_skills_text reach the
    salary model unmasked — features.py passes them without col().

    That is safe only because they contain no pay figures at all, which is a
    property of this corpus rather than of the code. This test pins the
    measurement so a future dataset that breaks it is caught here instead of
    quietly training a salary model on its own answer.

    Use re.search, not Series.str.contains: pandas does not preserve the \\b
    semantics of these compiled patterns and reports ~111 false hits on
    qualifications_text ("sinh viên năm 2 trở lên").
    """
    import pandas as pd

    cols = ["soft_skills_text", "qualifications_text", "technical_skills_text"]
    df = pd.read_parquet(_SPLIT, columns=cols)
    pats = (V._PAT_MILLIONS, V._PAT_DONG, V._PAT_USD)
    for c in cols:
        hits = sum(bool(p.search(t)) for t in df[c].fillna("") for p in pats)
        assert hits == 0, f"{c} now contains {hits} pay figures — mask it before training"
