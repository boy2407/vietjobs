"""Phần dense đặt trên vector PhoBERT — chung khung cho cả hai bài toán.

Đây đúng là chỗ mà bản đa nhiệm sau này sẽ tách đôi: cùng ``thân`` (dense dùng
chung), hai ``đầu`` khác nhau. Ở bậc baseline mỗi bài dựng một mạng riêng, nên
thân và đầu nằm chung trong một khối — khi gộp thì thân được tách ra, đầu giữ
nguyên.
"""
from __future__ import annotations

import torch
from torch import nn


class DenseHead(nn.Module):
    """768 → hidden → (hidden//2) → out.

    Hai lớp ẩn là baseline có chủ ý: một lớp thì gần như hồi quy tuyến tính trên
    đặc trưng PhoBERT, ba lớp trở lên thì quá khớp nhanh trên 33k mẫu. Có số đo
    rồi mới đổi.
    """

    def __init__(self, in_dim: int = 768, hidden: int = 256, out_dim: int = 1,
                 dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, out_dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # ``mask`` chỉ để chung chữ ký với head đọc chuỗi; vector đã gộp không cần nó.
        return self.net(x)


class BiGruLstmCnn(nn.Module):
    """Head đọc **từng token** thay vì một vector trung bình (T8, theo Tran–Vo–Luu 2022).

    ``[b, T, 768]`` → SpatialDropout → Bi-GRU ‖ Bi-LSTM → Conv1d(k=3) → avg-pool ‖ max-pool
    có mặt nạ → ghép → LayerNorm → dense → out.

    Ba điều cố ý:

    * **Không mean pooling trước RNN** — RNN cần một dãy; trung bình trước là đưa
      cho nó chuỗi dài 1. Pooling nằm *sau* RNN/Conv, trên trạng thái đã học.
    * ``pack_padded_sequence`` để chiều ngược của RNN bắt đầu từ token thật cuối
      cùng chứ không từ pad; nhờ thế đầu ra không phụ thuộc số pad — có test canh.
    * Max-pool điền ``-inf`` ở pad, avg-pool chia cho số token thật: pad không bao
      giờ lọt vào vector cuối, kể cả khi cache có rác ở đó.

    ``branches``: ``both`` (mặc định) · ``gru`` · ``lstm`` — cho ablation T8.5.
    """

    def __init__(self, in_dim: int = 768, hidden: int = 128, out_dim: int = 1,
                 dropout: float = 0.3, spatial_dropout: float = 0.2,
                 conv_channels: int = 64, branches: str = "both"):
        super().__init__()
        if branches not in ("both", "gru", "lstm"):
            raise ValueError(f"branches lạ: {branches}")
        self.branches = branches
        # Dropout2d trên [b, C, T, 1] tắt cả một kênh trên mọi bước thời gian —
        # đúng nghĩa SpatialDropout1d của Keras mà bài gốc dùng.
        self.spatial = nn.Dropout2d(spatial_dropout)
        self.rnns = nn.ModuleDict()
        self.convs = nn.ModuleDict()
        if branches in ("both", "gru"):
            self.rnns["gru"] = nn.GRU(in_dim, hidden, batch_first=True, bidirectional=True)
            self.convs["gru"] = nn.Conv1d(2 * hidden, conv_channels, kernel_size=3, padding=1)
        if branches in ("both", "lstm"):
            self.rnns["lstm"] = nn.LSTM(in_dim, hidden, batch_first=True, bidirectional=True)
            self.convs["lstm"] = nn.Conv1d(2 * hidden, conv_channels, kernel_size=3, padding=1)
        feat = len(self.rnns) * 2 * conv_channels          # mỗi nhánh: avg ‖ max
        self.out = nn.Sequential(
            nn.LayerNorm(feat),
            nn.Linear(feat, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``x [b, T, 768]`` float32 · ``mask [b, T]`` (1 = token thật)."""
        mask = mask.bool()
        lengths = mask.sum(1)
        total = x.shape[1]
        x = self.spatial(x.transpose(1, 2).unsqueeze(-1)).squeeze(-1).transpose(1, 2)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        m = mask.unsqueeze(1)                                  # [b, 1, T]
        feats = []
        for name, rnn in self.rnns.items():
            h, _ = rnn(packed)
            h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=total)
            c = nn.functional.gelu(self.convs[name](h.transpose(1, 2)))   # [b, C, T]
            avg = (c * m).sum(2) / lengths.clamp(min=1).unsqueeze(1)
            mx = c.masked_fill(~m, float("-inf")).max(2).values
            feats += [avg, mx]
        return self.out(torch.cat(feats, dim=1))

