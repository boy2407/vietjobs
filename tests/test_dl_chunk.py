"""T23 — ``--chunked``: tin dài chia đoạn thay vì cắt bỏ; tin ngắn cho đúng vector cũ."""
from __future__ import annotations

import numpy as np
import pytest

from vietjobs import config as C
from vietjobs.dl import encode as ENC


def test_split_ids_covers_every_token_once():
    assert ENC.split_ids([], 3) == [[]]
    parts = ENC.split_ids(list(range(7)), 3)
    assert parts == [[0, 1, 2], [3, 4, 5], [6]]
    assert sum(parts, []) == list(range(7))


def test_chunked_cache_has_its_own_name():
    base = ENC.cache_path("train", C.TASK_SALARY, 256, last4=True)
    chunk = ENC.cache_path("train", C.TASK_SALARY, 256, last4=True, chunked=True)
    assert base.name == "train-masked-len256-last4.npy"
    assert chunk.name == "train-masked-len256-last4-chunk.npy"


class _Tok:
    """Tokenizer giả: mỗi ký tự là một id; <s>=1, </s>=2, pad=0."""

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def build_inputs_with_special_tokens(self, ids):
        return [1] + list(ids) + [2]

    def __call__(self, texts, add_special_tokens=True, padding=False, truncation=False,
                 max_length=None, return_tensors=None):
        import torch

        if isinstance(texts, str):
            return {"input_ids": [ord(c) for c in texts]}
        rows = [[ord(c) for c in t][: max_length - 2] for t in texts]
        return self.pad({"input_ids": [self.build_inputs_with_special_tokens(r) for r in rows]})

    def pad(self, enc, return_tensors="pt"):
        import torch

        n = max(len(r) for r in enc["input_ids"])
        ids = [r + [0] * (n - len(r)) for r in enc["input_ids"]]
        mask = [[1] * len(r) + [0] * (n - len(r)) for r in enc["input_ids"]]
        return _Batch(input_ids=torch.tensor(ids), attention_mask=torch.tensor(mask))


class _Batch(dict):
    def to(self, dev):
        return self


class _Model:
    """Vector của token = (id, 1) — trung bình có mặt nạ đọc được bằng tay."""

    def __call__(self, input_ids, attention_mask):
        import torch

        h = torch.stack([input_ids.float(), torch.ones_like(input_ids).float()], -1)
        return type("O", (), {"last_hidden_state": h})()


@pytest.fixture
def fake(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setattr(ENC, "_load_model", lambda device, encoder: (_Tok(), _Model(), "cpu"))
    monkeypatch.setattr(ENC, "hidden_of", lambda encoder: 2)


def test_short_text_matches_truncated_vector(fake):
    texts = ["abc", "hello"]
    old = ENC.encode_texts(texts, max_len=8, batch=2, quiet=True)
    new = ENC.encode_texts(texts, max_len=8, batch=1, quiet=True, chunked=True)
    np.testing.assert_allclose(new, old, rtol=1e-6)


def test_long_text_averages_every_token_of_every_chunk(fake):
    text = "abcdefghij"                       # 10 id, 6 id mỗi đoạn → 2 đoạn: 6 + 4
    got = ENC.encode_texts([text], max_len=8, batch=2, quiet=True, chunked=True)
    ids = [1, *map(ord, "abcdef"), 2, 1, *map(ord, "ghij"), 2]
    assert got[0, 0] == pytest.approx(np.mean(ids))
    cut = ENC.encode_texts([text], max_len=8, batch=2, quiet=True)
    assert cut[0, 0] != pytest.approx(got[0, 0])  # cách cũ bỏ mất "ghij"


def test_ragged_pad_rebuilds_each_posting():
    flat = np.arange(12, dtype=np.float16).reshape(6, 2)   # 6 token, 2 chiều
    rag = ENC.RaggedTokens(flat, np.array([0, 1, 4, 6]))   # tin dài 1, 3, 2 token
    assert len(rag) == 3 and rag.shape[-1] == 2
    x, m = rag.pad(np.array([2, 0]))
    assert x.shape == (2, 2, 2) and m.tolist() == [[1, 1], [1, 0]]
    np.testing.assert_array_equal(x[0], flat[4:6])
    np.testing.assert_array_equal(x[1, 0], flat[0])
    assert not x[1, 1].any()                                  # pad là 0


def test_chunked_token_cache_names():
    tok = ENC.cache_path("train", C.TASK_SALARY, 256, "tok", last4=True, chunked=True)
    off = ENC.cache_path("train", C.TASK_SALARY, 256, "off", last4=True, chunked=True)
    assert tok.name == "train-masked-len256-last4-chunk-tok.f16.npy"
    assert off.name == "train-masked-len256-last4-chunk-off.npy"


def test_chunked_tokens_keep_every_chunk_in_order(fake, tmp_path):
    flat_path, off_path = tmp_path / "t.npy", tmp_path / "o.npy"
    ENC.encode_tokens_chunked(["abcdefghij", "xy"], flat_path, off_path, max_len=8, batch=2,
                              quiet=True)
    rag = ENC.RaggedTokens(np.load(flat_path, mmap_mode="r"), np.load(off_path))
    x, m = rag.pad(np.array([0, 1]))
    first = [1, *map(ord, "abcdef"), 2, 1, *map(ord, "ghij"), 2]
    assert x[0, :, 0].tolist() == first and m[0].sum() == len(first)
    assert x[1, :4, 0].tolist() == [1, ord("x"), ord("y"), 2] and m[1].sum() == 4
    pooled = ENC.encode_texts(["abcdefghij"], max_len=8, batch=2, quiet=True, chunked=True)
    assert x[0, :, 0].astype(np.float32).mean() == pytest.approx(pooled[0, 0], rel=1e-3)


def test_token_stats_on_ragged_reads_only_real_tokens_of_the_rows():
    pytest.importorskip("torch")
    from vietjobs.dl.train_dl import token_stats

    flat = np.arange(12, dtype=np.float16).reshape(6, 2)
    rag = ENC.RaggedTokens(flat, np.array([0, 1, 4, 6]))
    mu, sigma = token_stats(rag, np.array([0, 2]), None)
    real = flat[[0, 4, 5]].astype(np.float32)               # tin 1 bị loại, pad không tính
    np.testing.assert_allclose(mu[0, 0], real.mean(0))
    np.testing.assert_allclose(sigma[0, 0], real.std(0), rtol=1e-5)
