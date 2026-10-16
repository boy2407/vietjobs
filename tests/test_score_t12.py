"""Phần thuần của ``scripts/score_t12.py`` (T12.1): chọn dòng, chặn hàng chờ tác giả, đường ghi."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import encode_t12_cpu as ENC_T12  # noqa: E402
import score_t12 as S  # noqa: E402


def test_blocked_rows_are_the_four_pending_the_author():
    for r in ("dl-mtl-h256-a1.0-1001-0735", "dl-mtl-h256-a1.0-1001-0746", "dl-cat-focal-0920",
              "dl-cat-focal-s2", "dl-sal-0920", "dl-sal-s2", "dl-cat-cafe-dense-512-p20-s43"):
        assert S.is_blocked(r), r
    for r in ("dl-cat-cafe-dense-512", "dl-cat-cafe-focal-512-s42", "dl-sal-cpu-0920", "dl-cat-ce-0920"):
        assert not S.is_blocked(r), r


def test_family_follows_rule_3():
    assert S.family_of("category") == "raw"
    assert S.family_of("salary") == "masked"
    assert S.family_of("mtl") == "masked"


def test_run_kind_defaults_for_old_and_mtl_runs(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(
        {"task": "category", "hidden": 100, "max_len": 256, "head": "rnn"}))
    k = S.run_kind(tmp_path)
    assert (k["task"], k["head"], k["encoder"]) == ("category", "rnn", "phobert")
    (tmp_path / "config.json").write_text(json.dumps({"alpha_sal": 5.0, "max_len": 512, "encoder": "cafebert"}))
    assert S.run_kind(tmp_path)["task"] == "mtl"


def test_score_frame_uses_only_labelled_and_disclosed_rows():
    df = pd.DataFrame({"category": ["a", "b", None, "a"], "salary_disclosed": [1, 0, 1, 1],
                       "salary_mid": [10.0, np.nan, 20.0, 30.0]})
    cat, sal = S.row_masks(df)
    assert cat.tolist() == [True, True, False, True]
    assert sal.tolist() == [True, False, True, True]
    proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.5, 0.5], [0.6, 0.4]])
    res = S.score_frame(df, proba, np.log1p(np.array([10.0, 99.0, 20.0, 30.0])), ["a", "b"])
    assert res["category"]["n"] == 3 and res["category"]["accuracy"] == 1.0
    assert res["salary"]["n"] == 3 and res["salary"]["mae_trieu"] == pytest.approx(0.0, abs=1e-4)


def test_cache_names_match_encode_py_convention():
    assert ENC_T12.cache_path_t12("test", "raw", "phobert", 256).name == "test-raw-len256.npy"
    assert ENC_T12.cache_path_t12("ext37k-sal", "masked", "cafebert", 512).name == "cafebert-ext37k-sal-masked-len512.npy"


def test_save_writes_under_artifacts_only(tmp_path, monkeypatch):
    monkeypatch.setattr(S.C, "ARTIFACT_DIR", tmp_path)
    S.save("run-x", "test", "cpu", pd.DataFrame({"a": [1, 2]}), {"category": {"f1_macro": 0.5}})
    out = json.loads((tmp_path / "run-x" / "t12_score" / "test.json").read_text())
    assert out["tag"] == "test: author override 2026-10-15" and out["n"] == 2
