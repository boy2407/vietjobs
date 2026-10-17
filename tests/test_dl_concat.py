"""T12.1 — ghép hai cache vector đông lạnh chỉ khi cùng thứ tự hàng (sidecar)."""
from __future__ import annotations

import json

import pytest

from vietjobs.dl import encode as ENC


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(ENC, "CACHE_DIR", tmp_path)
    return tmp_path


def _sidecar(cache_dir, name, *, split, family, n):
    (cache_dir / f"{name}.json").write_text(
        json.dumps({"split": split, "family": family, "n": n}), encoding="utf-8")


def test_matching_sidecars_pass(cache_dir):
    _sidecar(cache_dir, "train-masked-len256", split="train", family="masked", n=10)
    _sidecar(cache_dir, "cafebert-train-masked-len512", split="train", family="masked", n=10)
    ENC.check_same_rows("train", "salary", 10, ("phobert", 256), ("cafebert", 512))


def test_pooled_phobert_falls_back_to_token_sidecar(cache_dir):
    # cache PhoBERT pooled cũ không có sidecar; sidecar của cache token cùng họ được dùng
    _sidecar(cache_dir, "train-raw-len256-tok.f16", split="train", family="raw", n=10)
    assert ENC.sidecar_of("train", "category", 256, "phobert")["n"] == 10


def test_row_count_mismatch_refused(cache_dir):
    _sidecar(cache_dir, "train-masked-len256", split="train", family="masked", n=10)
    _sidecar(cache_dir, "cafebert-train-masked-len512", split="train", family="masked", n=9)
    with pytest.raises(SystemExit, match="không ghép được"):
        ENC.check_same_rows("train", "salary", 10, ("phobert", 256), ("cafebert", 512))


def test_wrong_family_refused(cache_dir):
    # raw ghép masked là rò rỉ lương (Rule 3): phải bị từ chối
    _sidecar(cache_dir, "train-masked-len256", split="train", family="masked", n=10)
    _sidecar(cache_dir, "cafebert-train-raw-len512", split="train", family="raw", n=10)
    with pytest.raises(SystemExit):
        ENC.check_same_rows("train", "salary", 10, ("phobert", 256), ("cafebert", 512))


def test_missing_sidecar_refused(cache_dir):
    with pytest.raises(SystemExit, match="không có sidecar"):
        ENC.sidecar_of("train", "category", 512, "cafebert")
