"""JEPA (T19): che khối, không rò qua attention, EMA, loss — chạy khi có torch."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from vietjobs.dl.jepa import (  # noqa: E402
    JepaEncoder, Predictor, block_mask, ema_update, jepa_step)


def _attn(lengths, T=40):
    a = torch.zeros(len(lengths), T, dtype=torch.long)
    for i, L in enumerate(lengths):
        a[i, :L] = 1
    return a


def _models(vocab=50, d=16, T=40):
    torch.manual_seed(0)
    table = torch.randn(vocab, 8)
    kw = dict(d=d, layers=1, heads=2, max_len=T, dropout=0.0)
    enc, tgt = JepaEncoder(table, **kw), JepaEncoder(table, **kw)
    tgt.load_state_dict(enc.state_dict())
    return enc, tgt, Predictor(d=d, dp=8, layers=1, heads=2, max_len=T, dropout=0.0)


def test_block_mask_only_inner_real_tokens():
    attn = _attn([40, 25, 3, 1])
    m = block_mask(attn, gen=torch.Generator().manual_seed(0))
    assert not (m & ~attn.bool()).any()                 # không che pad
    assert not m[:, 0].any()                            # không che <s>
    assert not m[0, 39] and not m[1, 24]                # không che </s>
    assert not m[2:].any()                              # câu quá ngắn: không che
    frac = m[0].sum().item() / 38
    assert 0.2 <= frac <= 0.55


def test_context_encoder_does_not_see_masked_tokens():
    enc, _, _ = _models()
    enc.eval()
    ids = torch.randint(0, 50, (1, 40))
    real = torch.ones(1, 40, dtype=torch.bool)
    masked = torch.zeros(1, 40, dtype=torch.bool)
    masked[0, 10:20] = True
    vis = real & ~masked
    ids2 = ids.clone()
    ids2[0, 10:20] = torch.randint(0, 50, (10,))         # đổi token bị che
    a, b = enc(ids, vis), enc(ids2, vis)
    assert torch.allclose(a[vis], b[vis], atol=1e-5)


def test_target_gets_no_gradient_and_ema_moves_it():
    enc, tgt, pred = _models()
    ids = torch.randint(0, 50, (2, 40))
    attn = _attn([40, 30])
    masked = block_mask(attn, gen=torch.Generator().manual_seed(1))
    loss = jepa_step(enc, tgt, pred, ids, attn, masked)
    loss.backward()
    assert all(p.grad is None for p in tgt.parameters())
    assert enc.proj.weight.grad is not None and pred.mask_token.grad is not None

    before = tgt.proj.weight.clone()
    with torch.no_grad():
        enc.proj.weight.add_(1.0)
    ema_update(tgt, enc, tau=0.75)
    expected = 0.75 * before + 0.25 * enc.proj.weight
    assert torch.allclose(tgt.proj.weight, expected, atol=1e-6)


def test_table_not_in_state_dict():
    enc, _, _ = _models()
    assert "table" not in enc.state_dict()
