"""T23 — a split set with extra columns reads the manifest beside its CSVs."""
from __future__ import annotations

import json

import pytest

from vietjobs import dataset as D


def _write(dir_, cols):
    (dir_ / "manifest.json").write_text(json.dumps({"column_dtypes": cols}), encoding="utf-8")


def test_own_manifest_wins_over_shared(tmp_path):
    split = tmp_path / "splits-x"
    split.mkdir()
    _write(tmp_path, {"a": "str"})
    (split / "dev.csv").write_text("a,b_masked\n1,x\n", encoding="utf-8")
    with pytest.raises(KeyError, match="b_masked"):
        D.load_split("dev", split)
    _write(split, {"a": "str", "b_masked": "str"})
    assert list(D.load_split("dev", split).columns) == ["a", "b_masked"]
