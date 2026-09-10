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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
