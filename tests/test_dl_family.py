"""T23.0 — ``--family masked``: category đọc *_masked và dùng chung cache với salary."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vietjobs import config as C
from vietjobs.dl import encode as ENC
from vietjobs.dl import text as T


def test_default_keeps_old_family():
    assert ENC.cache_task(C.TASK_CATEGORY) == C.TASK_CATEGORY
    assert ENC.family(ENC.cache_task(C.TASK_CATEGORY)) == "raw"
    assert ENC.cache_task(C.TASK_SALARY) == C.TASK_SALARY


def test_masked_category_shares_salary_cache_and_columns():
    t = ENC.cache_task(C.TASK_CATEGORY, "masked")
    assert ENC.cache_path("train", t, 256) == ENC.cache_path("train", C.TASK_SALARY, 256)
    cols = T.columns_for(t)
    assert all("masked" in c for c in cols[1:]), cols


@pytest.mark.parametrize("fam,stem", [(None, "train-raw-len256"), ("masked", "train-masked-len256")])
def test_load_task_reads_family_cache(tmp_path, monkeypatch, fam, stem):
    pytest.importorskip("torch")
    from vietjobs.dl import train_dl as TD

    monkeypatch.setattr(ENC, "CACHE_DIR", tmp_path)
    np.save(tmp_path / f"{stem}.npy", np.full((3, 4), 1 if fam else 0, dtype=np.float32))
    df = pd.DataFrame({"category": ["a", "b", "a"]})
    monkeypatch.setattr(TD.D, "load_split", lambda split: df)
    X, _, y, _ = TD.load_task("train", C.TASK_CATEGORY, 256, family=fam)
    assert X.shape == (3, 4) and X[0, 0] == (1 if fam else 0)
    assert list(y) == ["a", "b", "a"]
