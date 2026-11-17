"""JEPA cho văn bản: encoder nhỏ học đoán **vector** của các khối token bị che (T19).

Vì sao không chạy JEPA trên token cache CafeBERT: CafeBERT đã đọc cả câu trước khi
ta che, nên vector của token lân cận mang sẵn đáp án — bài đoán thành tầm thường.
Ở đây đầu vào là **bảng embedding từ vựng** của CafeBERT (đóng băng, chưa qua
attention), nên token bị che thật sự vắng mặt với context encoder.

Ba thành phần (Assran et al. 2023, I-JEPA, chuyển sang chuỗi token):

* context encoder — chỉ thấy token không bị che (che bằng ``src_key_padding_mask``);
* target encoder — bản EMA của context encoder, đọc cả câu, không nhận gradient;
* predictor — từ đầu ra context + mask-token tại vị trí bị che, đoán vector đích.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def block_mask(attn: torch.Tensor, n_blocks: int = 4, ratio: tuple = (0.4, 0.5),
               gen: torch.Generator | None = None) -> torch.Tensor:
    """``attn [b, T]`` (1 = token thật) → ``bool [b, T]``, True = bị che.

    Che ``n_blocks`` khối liên tiếp, tổng ~``ratio`` số token thật (chồng lấn làm tỉ
    lệ thật thấp hơn một chút). Không bao giờ che token đầu ``<s>``, token cuối
    ``</s>`` hay pad, nên mỗi dòng luôn còn token nhìn thấy được.
    """
    attn = attn.bool()
    b, T = attn.shape
    out = torch.zeros(b, T, dtype=torch.bool)
    lengths = attn.sum(1).tolist()
    for i, L in enumerate(lengths):
        inner = L - 2                                   # bỏ <s> và </s>
        if inner < 2:
            continue
        r = ratio[0] + (ratio[1] - ratio[0]) * torch.rand(1, generator=gen).item()
        blen = max(1, int(round(r * inner / n_blocks)))
        for _ in range(n_blocks):
            s = 1 + int(torch.randint(0, max(1, inner - blen + 1), (1,), generator=gen))
            out[i, s:s + blen] = True
    return out


def _stack(d: int, layers: int, heads: int, dropout: float) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, activation="gelu",
                                       batch_first=True, norm_first=True)
    # enable_nested_tensor=False: đường nhanh của torch đổi đầu ra ở vị trí bị che.
    return nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(d),
                                 enable_nested_tensor=False)


class JepaEncoder(nn.Module):
    """``ids [b, T]`` → ``[b, T, d]``. Bảng từ vựng là buffer không lưu vào state_dict."""

    def __init__(self, table: torch.Tensor, d: int = 256, layers: int = 4, heads: int = 4,
                 max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.register_buffer("table", table, persistent=False)
        self.proj = nn.Linear(table.shape[1], d)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.body = _stack(d, layers, heads, dropout)

    def forward(self, ids: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
        """``visible [b, T]`` bool: token được phép nhìn (thật và không bị che)."""
        x = self.proj(self.table[ids].float()) + self.pos[:, :ids.shape[1]]
        return self.body(x, src_key_padding_mask=~visible)


class Predictor(nn.Module):
    """Đầu ra context (vị trí bị che thay bằng mask-token) → vector đoán ``[b, T, d]``."""

    def __init__(self, d: int = 256, dp: int = 128, layers: int = 2, heads: int = 4,
                 max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Linear(d, dp)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dp))
        self.pos = nn.Parameter(torch.zeros(1, max_len, dp))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.body = _stack(dp, layers, heads, dropout)
        self.out = nn.Linear(dp, d)

    def forward(self, ctx: torch.Tensor, real: torch.Tensor, masked: torch.Tensor) -> torch.Tensor:
        h = self.inp(ctx)
        # Đầu ra context tại vị trí bị che chứa chính token đó (residual) — thay hẳn.
        h = torch.where(masked.unsqueeze(-1), self.mask_token.expand_as(h), h)
        h = h + self.pos[:, :h.shape[1]]
        return self.out(self.body(h, src_key_padding_mask=~real))


@torch.no_grad()
def ema_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    for pt, po in zip(target.parameters(), online.parameters()):
        pt.mul_(tau).add_(po.detach(), alpha=1.0 - tau)


def jepa_loss(pred: torch.Tensor, target: torch.Tensor, masked: torch.Tensor) -> torch.Tensor:
    """Smooth-L1 giữa vector đoán và ``LayerNorm(vector đích)``, chỉ ở vị trí bị che."""
    tgt = F.layer_norm(target, target.shape[-1:])
    return F.smooth_l1_loss(pred[masked], tgt[masked])


def jepa_step(enc: JepaEncoder, tgt_enc: JepaEncoder, pred: Predictor,
              ids: torch.Tensor, attn: torch.Tensor, masked: torch.Tensor) -> torch.Tensor:
    """Một lượt xuôi của giai đoạn 1; trả loss (đích tính không gradient)."""
    real = attn.bool()
    with torch.no_grad():
        target = tgt_enc(ids, real)
    ctx = enc(ids, real & ~masked)
    return jepa_loss(pred(ctx, real, masked), target, masked)
