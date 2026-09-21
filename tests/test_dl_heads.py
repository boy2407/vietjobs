"""FocalLoss (bản itakurah/focal-loss-pytorch, vendor tại dl/focal_loss.py) —
chạy khi có torch; no-op ở môi trường không có torch."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from vietjobs.dl.focal_loss import FocalLoss  # noqa: E402
from vietjobs.dl.heads import BiGruLstmCnn, DenseHead  # noqa: E402


def _fl(gamma, alpha=None):
    return FocalLoss(gamma=gamma, alpha=alpha, task_type="multi-class", num_classes=16)


def test_gamma_zero_equals_cross_entropy():
    torch.manual_seed(0)
    logits = torch.randn(8, 16)
    target = torch.randint(0, 16, (8,))
    focal = _fl(0.0)(logits, target)
    ce = nn.CrossEntropyLoss()(logits, target)
    assert torch.allclose(focal, ce, atol=1e-6)


def test_gamma_zero_weighted_divides_by_batch_not_sum_of_weights():
    """Khác torch: bản repo chia cho N, không chia cho tổng weight — ghi lại
    để không ai tưởng gamma=0 + alpha trùng CrossEntropyLoss(weight=...)."""
    torch.manual_seed(0)
    logits = torch.randn(8, 16)
    target = torch.randint(0, 16, (8,))
    w = torch.rand(16) + 0.1
    focal = _fl(0.0, w)(logits, target)
    ce_none = nn.functional.cross_entropy(logits, target, reduction="none")
    expected = (w.gather(0, target) * ce_none).mean()
    assert torch.allclose(focal, expected, atol=1e-6)
    assert not torch.allclose(focal, nn.CrossEntropyLoss(weight=w)(logits, target), atol=1e-4)


def test_gamma_down_weights_confident_correct_sample():
    """Mẫu đã đúng và tự tin (p_t cao) phải bị hạ trọng số nhiều hơn mẫu khó."""
    confident_correct = torch.tensor([[10.0] + [0.0] * 15])
    hard = torch.tensor([[0.1] + [0.0] * 15])
    target = torch.tensor([0])
    easy_ratio = _fl(2.0)(confident_correct, target) / _fl(0.0)(confident_correct, target)
    hard_ratio = _fl(2.0)(hard, target) / _fl(0.0)(hard, target)
    assert easy_ratio < hard_ratio


def test_uniform_alpha_scales_loss_by_that_constant():
    """Bản repo không chuẩn hoá theo weight, nên alpha đồng nhất = c nhân loss với c."""
    torch.manual_seed(0)
    logits = torch.randn(4, 16)
    target = torch.randint(0, 16, (4,))
    unweighted = _fl(2.0)(logits, target)
    weighted = _fl(2.0, torch.full((16,), 5.0))(logits, target)
    assert torch.allclose(weighted, 5.0 * unweighted, atol=1e-5)


# ---------------------------------------------------------------------------
# BiGruLstmCnn (T8.2) — head đọc từng token, không mean pooling trước RNN
# ---------------------------------------------------------------------------


def _tokens(b=3, T=40, d=768, lengths=(40, 25, 7)):
    torch.manual_seed(1)
    x = torch.randn(b, T, d)
    mask = torch.zeros(b, T, dtype=torch.uint8)
    for i, L in enumerate(lengths):
        mask[i, :L] = 1
    x = x * mask.unsqueeze(-1)          # cache trên đĩa cũng ghi 0 ở pad
    return x, mask


@pytest.mark.parametrize("out_dim", [16, 1])
@pytest.mark.parametrize("branches", ["both", "gru", "lstm"])
def test_bigru_lstm_cnn_forward_shape(out_dim, branches):
    x, mask = _tokens()
    head = BiGruLstmCnn(out_dim=out_dim, branches=branches).eval()
    y = head(x, mask)
    assert y.shape == (3, out_dim) and torch.isfinite(y).all()


def test_dense_head_accepts_and_ignores_mask():
    """Chung chữ ký với head chuỗi để vòng huấn luyện gọi model(x, mask) cho cả hai."""
    head = DenseHead(out_dim=16).eval()
    x = torch.randn(4, 768)
    assert torch.allclose(head(x), head(x, torch.ones(4, 256)))


def test_padding_invariance():
    """Cùng token thật, thêm pad → đầu ra không đổi (pack_padded + pool có mặt nạ)."""
    x, mask = _tokens(b=1, T=100, lengths=(100,))
    head = BiGruLstmCnn(out_dim=16).eval()
    y_short = head(x, mask)
    x_long = torch.cat([x, torch.zeros(1, 156, 768)], dim=1)          # pad lên 256
    mask_long = torch.cat([mask, torch.zeros(1, 156, dtype=torch.uint8)], dim=1)
    y_long = head(x_long, mask_long)
    assert torch.allclose(y_short, y_long, atol=1e-5), float((y_short - y_long).abs().max())


def test_max_pool_never_reads_pad():
    """Rác rất lớn ở vị trí pad không được lọt vào đầu ra."""
    x, mask = _tokens(b=1, T=30, lengths=(10,))
    head = BiGruLstmCnn(out_dim=4).eval()
    clean = head(x, mask)
    dirty = x.clone()
    dirty[0, 10:] = 1e4                 # cache hỏng: pad chứa số khổng lồ
    assert torch.allclose(clean, head(dirty, mask), atol=1e-4)


def test_branch_ablation_changes_parameter_count():
    both = sum(p.numel() for p in BiGruLstmCnn(branches="both").parameters())
    gru = sum(p.numel() for p in BiGruLstmCnn(branches="gru").parameters())
    lstm = sum(p.numel() for p in BiGruLstmCnn(branches="lstm").parameters())
    assert gru < both and lstm < both and lstm > gru      # LSTM có 4 cổng, GRU 3
