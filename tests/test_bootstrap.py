"""The bootstrap machinery behind every "is this difference real?" claim.

docs/ quoted sigma ~= 0.009 in five places long before any code computed it.
These tests pin down the code that now does, including the one property the
whole comparison rests on: two models resampled through the SAME index matrix
are compared pairwise, and identical models must come out as a tie rather than
as a win for whichever was passed first.
"""
from __future__ import annotations

import numpy as np
import pytest

from vietjobs import evaluate as E


LABELS = np.array(["a", "b", "c"])


def _fake(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return rng.choice(LABELS, size=n)


def test_same_seed_gives_identical_indices():
    a = E.bootstrap_indices(100, n_boot=20, seed=7)
    b = E.bootstrap_indices(100, n_boot=20, seed=7)
    assert np.array_equal(a, b)


def test_indices_shape_and_range():
    idx = E.bootstrap_indices(50, n_boot=13, seed=1)
    assert idx.shape == (13, 50)
    assert idx.min() >= 0 and idx.max() < 50


def test_perfect_model_scores_one_with_zero_spread():
    y = _fake()
    idx = E.bootstrap_indices(len(y), n_boot=50, seed=3)
    scores = E.bootstrap_scores(y, y, idx)
    assert np.allclose(scores, 1.0)
    assert E.bootstrap_summary(scores)["std"] == 0.0


def test_summary_reports_a_positive_spread_for_an_imperfect_model():
    y = _fake()
    y_pred = y.copy()
    y_pred[::4] = "a"  # break a quarter of the rows
    idx = E.bootstrap_indices(len(y), n_boot=200, seed=4)
    s = E.bootstrap_summary(E.bootstrap_scores(y, y_pred, idx))
    assert 0.0 < s["std"] < 0.5
    assert s["ci95_lo"] < s["mean"] < s["ci95_hi"]


def test_identical_models_tie_at_half():
    y = _fake()
    y_pred = y.copy()
    y_pred[::3] = "b"
    idx = E.bootstrap_indices(len(y), n_boot=100, seed=5)
    s = E.bootstrap_scores(y, y_pred, idx)
    d = E.paired_delta(s, s)
    assert d["mean"] == 0.0
    assert d["std"] == 0.0
    assert d["p_gt_0"] == 0.5


def test_clearly_better_model_wins_every_resample():
    y = _fake()
    good = y.copy()
    good[::20] = "a"  # 5% wrong
    bad = y.copy()
    bad[::2] = "a"  # 50% wrong
    idx = E.bootstrap_indices(len(y), n_boot=100, seed=6)
    d = E.paired_delta(
        E.bootstrap_scores(y, good, idx), E.bootstrap_scores(y, bad, idx)
    )
    assert d["p_gt_0"] == 1.0
    assert d["mean"] > 0


def test_pairing_is_more_sensitive_than_separate_error_bars():
    """The reason a shared index matrix exists at all.

    Two models that differ on a handful of rows share nearly all their errors.
    Resampling both through the same rows cancels that shared variance, so
    std(delta) lands far below either model's own sigma — which is exactly the
    difference between calling a small gap "noise" and calling it real.
    """
    y = _fake(n=600, seed=11)
    a = y.copy()
    a[::7] = "a"
    b = a.copy()
    b[:12] = "c"  # b is slightly worse, on a few rows only
    idx = E.bootstrap_indices(len(y), n_boot=300, seed=12)
    sa = E.bootstrap_scores(y, a, idx)
    sb = E.bootstrap_scores(y, b, idx)
    sigma_a = E.bootstrap_summary(sa)["std"]
    delta = E.paired_delta(sa, sb)
    assert delta["std"] < sigma_a


def test_mismatched_lengths_raise():
    y = _fake(n=10)
    idx = E.bootstrap_indices(10, n_boot=5, seed=8)
    with pytest.raises(ValueError):
        E.bootstrap_scores(y, y[:5], idx)
    with pytest.raises(ValueError):
        E.paired_delta(np.zeros(5), np.zeros(6))
