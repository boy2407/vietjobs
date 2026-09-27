"""The label-audit sampler (T2.3) and the hand-reading summary.

The sample is what a human reads and what ch.3 quotes as "label noise", so it
must be the same file every time it is regenerated, take only rows it was given
(the script feeds it `dev` errors only — `test` is never opened), and hit the
requested size even when a stratum is short.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from vietjobs import config as C

_spec = importlib.util.spec_from_file_location(
    "analyze_errors", Path(__file__).resolve().parents[1] / "scripts" / "analyze_errors.py")
AE = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(AE)

PAIRS = [("a", "b"), ("b", "a"), ("c", "d")]


def _errors(n=400, seed=0):
    rng = np.random.default_rng(seed)
    true = rng.choice(["a", "b", "c", "e", C.JUNK_CATEGORY], size=n)
    pred = rng.choice(["a", "b", "d", "e"], size=n)
    keep = true != pred
    return pd.DataFrame({"nhan_goc": true[keep], "du_doan": pred[keep],
                         "p_du_doan": rng.uniform(0.6, 1, keep.sum())},
                        index=np.flatnonzero(keep) + 1000)


def test_sample_has_requested_size_and_only_input_rows():
    err = _errors()
    s = AE.stratified_sample(err, PAIRS, n=100)
    assert len(s) == 100
    assert s.index.is_unique
    assert set(s.index) <= set(err.index)


def test_sample_is_deterministic_for_a_seed():
    err = _errors()
    a = AE.stratified_sample(err, PAIRS, n=100, seed=7)
    b = AE.stratified_sample(err, PAIRS, n=100, seed=7)
    assert a.index.tolist() == b.index.tolist()
    assert AE.stratified_sample(err, PAIRS, n=100, seed=8).index.tolist() != a.index.tolist()


def test_short_strata_are_topped_up_from_the_rest():
    err = _errors()
    s = AE.stratified_sample(err, [("a", "b"), ("zzz", "yyy")], n=100, n_pairs=70)
    assert len(s) == 100
    assert not s["nhom_mau"].str.contains("zzz").any()
    assert (s["nhom_mau"] == "rac").sum() <= 15


def test_sample_never_exceeds_available_rows():
    err = _errors(n=60)
    assert len(AE.stratified_sample(err, PAIRS, n=100)) == len(err)


def test_audit_summary_rates_and_kappa():
    df = pd.DataFrame({
        "nhom_mau": ["x", "x", "y", "y", "y"],
        "nhan_dung": ["goc", "du_doan", "ca_hai", "goc", ""],
        "nhan_dung_2": ["goc", "du_doan", "ca_hai", "goc", "goc"],
    })
    out = AE.audit_summary(df)
    assert out["total"]["n"] == 4           # the blank row is ignored
    assert out["total"]["goc"] == 0.5
    assert out["by_group"]["x"]["du_doan"] == 0.5
    assert out["kappa"]["n"] == 4 and out["kappa"]["value"] == 1.0
