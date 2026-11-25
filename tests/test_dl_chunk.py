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
