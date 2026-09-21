"""Cache token-level (T8.1) phải chứa đúng thứ cache pooled đã nén.

Không cần torch: chỉ đọc cache trên đĩa. Chưa có cache thì skip. Chỉ đọc
``train``/``dev``; ``test`` không bao giờ được nhúng (Rule 5).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from vietjobs import config as C
from vietjobs.dl import encode as ENC

MAX_LEN = 256
COMBOS = [("train", C.TASK_CATEGORY), ("dev", C.TASK_CATEGORY),
          ("train", C.TASK_SALARY), ("dev", C.TASK_SALARY)]
N = 64  # số dòng đầu đọc để so mean(token) với pooled — đủ, không cần cả split


def _paths(split, task):
    return (ENC.cache_path(split, task, MAX_LEN, "tok"),
            ENC.cache_path(split, task, MAX_LEN, "mask"),
            ENC.cache_path(split, task, MAX_LEN, "pooled"))


def _has_cache(split, task):
    return all(p.exists() for p in _paths(split, task))


_needs_cache = pytest.mark.skipif(
    not all(_has_cache(s, t) for s, t in COMBOS),
    reason="cần chạy `encode.py --pooling none` cho cả train/dev × raw/masked")


def test_cache_paths_are_distinct_and_keep_the_old_name():
    """Tên cache pooled không đổi (cache cũ vẫn hợp lệ); tok/mask không thể trùng nó."""
    tok, mask, pooled = _paths("dev", C.TASK_CATEGORY)
    assert pooled.name == f"dev-raw-len{MAX_LEN}.npy"
    assert len({tok, mask, pooled}) == 3
    assert ENC.cache_path("dev", C.TASK_SALARY, MAX_LEN, "tok").name.startswith("dev-masked-")


@_needs_cache
@pytest.mark.parametrize("split,task", COMBOS)
def test_masked_mean_of_token_cache_equals_pooled_cache(split, task):
    """Tiêu chí T8.1 trên bảng: mean(token cache) == cache pooled, tol 1e-2 (fp16)."""
    tok_p, mask_p, pooled_p = _paths(split, task)
    tok = np.load(tok_p, mmap_mode="r")[:N].astype(np.float32)   # [N, 256, 768]
    mask = np.load(mask_p)[:N].astype(np.float32)[..., None]     # [N, 256, 1]
    pooled = np.load(pooled_p, mmap_mode="r")[:N]
    mean = (tok * mask).sum(1) / mask.sum(1)
    assert np.allclose(mean, pooled, atol=1e-2), float(np.abs(mean - pooled).max())


@_needs_cache
@pytest.mark.parametrize("split,task", COMBOS)
def test_token_cache_shape_matches_manifest_and_mask_is_sane(split, task):
    manifest = json.loads(C.MANIFEST.read_text())
    n_rows = manifest["splits"][split]["rows"]
    tok_p, mask_p, _ = _paths(split, task)
    tok = np.load(tok_p, mmap_mode="r")
    mask = np.load(mask_p)
    assert tok.shape == (n_rows, MAX_LEN, ENC.HIDDEN) and tok.dtype == np.float16
    assert mask.shape == (n_rows, MAX_LEN) and mask.dtype == np.uint8
    lengths = mask.sum(1)
    assert lengths.min() >= 2 and lengths.max() <= MAX_LEN   # ít nhất <s> và </s>
    # pad phải thật sự là 0 — head dựa vào mask, nhưng đây là bằng chứng ghi đúng chỗ
    assert not tok[0, lengths[0]:].any()


@_needs_cache
@pytest.mark.parametrize("split,task", COMBOS)
def test_token_sidecar_says_no_pooling(split, task):
    tok_p, _, _ = _paths(split, task)
    info = json.loads(tok_p.with_suffix(".json").read_text())
    assert info["pooling"] == "none" and info["dtype"] == "float16"
    expected = (["job_title_seg", "description_seg", "requirements_seg"] if task == C.TASK_CATEGORY
                else ["job_title_masked_seg", "description_masked_seg", "requirements_masked_seg"])
    assert info["columns"] == expected
