"""Phần dense đặt trên vector PhoBERT — chung khung cho cả hai bài toán.

Đây đúng là chỗ mà bản đa nhiệm sau này sẽ tách đôi: cùng ``thân`` (dense dùng
chung), hai ``đầu`` khác nhau. Ở bậc baseline mỗi bài dựng một mạng riêng, nên
thân và đầu nằm chung trong một khối — khi gộp thì thân được tách ra, đầu giữ
nguyên.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


def class_weights(counts: np.ndarray, mode: str = "inverse", beta: float = 0.9999) -> np.ndarray:
    """Trọng số lớp, chuẩn hoá để trung bình theo mẫu = 1 (thang loss ngang CE thường).

    ``inverse`` = 1/N_c · ``sqrt`` = 1/√N_c · ``effective`` = (1-β)/(1-β^N_c).
    """
    n = np.clip(np.asarray(counts, dtype=np.float64), 1, None)
    if mode == "inverse":
        raw = 1.0 / n
    elif mode == "sqrt":
        raw = 1.0 / np.sqrt(n)
    elif mode == "effective":
        raw = (1.0 - beta) / (1.0 - beta ** n)
    else:
        raise ValueError(f"cw-mode lạ: {mode}")
    return (raw * n.sum() / (raw * n).sum()).astype(np.float32)


class MultiTaskHead(nn.Module):
    """Trunk chung (chung dense layer), hai đầu ra: phân lớp và hồi quy lương."""
    def __init__(self, in_dim: int = 768, hidden: int = 256, n_classes: int = 16,
                 dropout: float = 0.3, pre_hidden: int = 0):
        super().__init__()
        pre = [nn.Linear(in_dim, pre_hidden), nn.GELU(), nn.Dropout(dropout)] if pre_hidden else []
        self.trunk = nn.Sequential(
            nn.LayerNorm(in_dim),
            *pre,
            nn.Linear(pre_hidden or in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head_cat = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes),
        )
        self.head_sal = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.head_cat(h), self.head_sal(h)


class DenseHead(nn.Module):
    """in_dim → [pre_hidden] → hidden → (hidden//2) → out.

    Hai lớp ẩn là baseline có chủ ý: một lớp thì gần như hồi quy tuyến tính trên
    đặc trưng encoder, ba lớp trở lên được cho là quá khớp nhanh trên 33k mẫu.
    ``pre_hidden > 0`` thêm lớp thứ ba ngay sau LayerNorm để đo lại chính câu đó
    (T8.10). ``pre_hidden = 0`` dựng đúng Sequential cũ — khoá state_dict không
    đổi, ``best.pt`` cũ vẫn nạp được.
    """

    def __init__(self, in_dim: int = 768, hidden: int = 256, out_dim: int = 1,
                 dropout: float = 0.3, pre_hidden: int = 0):
        super().__init__()
        pre = [nn.Linear(in_dim, pre_hidden), nn.GELU(), nn.Dropout(dropout)] if pre_hidden else []
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            *pre,
            nn.Linear(pre_hidden or in_dim, hidden),
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


class ResidualDenseHead(nn.Module):
    """in_dim → [Linear(in,2h) → GELU → Dropout → Linear(2h,h)] + Linear(in,h) → LayerNorm → out.

    Đường tắt tuyến tính in→h giữ lại phần thông tin tuyến tính của vector encoder;
    nhánh sâu chỉ học phần dư. Không có LayerNorm ở đầu vào như ``DenseHead``
    (đặc trưng đã được chuẩn hoá trong ``train_dl``).
    """

    def __init__(self, in_dim: int = 768, hidden: int = 256, out_dim: int = 1,
                 dropout: float = 0.3):
        super().__init__()
        self.deep = nn.Sequential(
            nn.Linear(in_dim, 2 * hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden, hidden),
        )
        self.skip = nn.Linear(in_dim, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.out(self.norm(self.deep(x) + self.skip(x)))


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
        return self.out(self.features(x, mask))

    def features(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Thân RNN → CNN → avg ‖ max có mặt nạ, trước lớp ra — trunk của T4.1."""
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
        return torch.cat(feats, dim=1)


class MultiTaskRnn(BiGruLstmCnn):
    """Trunk RNN chung (thân của ``BiGruLstmCnn``), hai đầu ra: phân lớp và hồi quy lương.

    Mỗi đầu đúng bằng ``out`` của head đơn nhiệm (LayerNorm → dense → out), nên
    khác biệt so với run đơn nhiệm chỉ còn là thân được chia sẻ.
    """

    def __init__(self, in_dim: int = 768, hidden: int = 128, n_classes: int = 16,
                 dropout: float = 0.3, spatial_dropout: float = 0.2,
                 conv_channels: int = 64, branches: str = "both"):
        super().__init__(in_dim, hidden, n_classes, dropout, spatial_dropout,
                         conv_channels, branches)
        self.head_cat = self.out
        del self.out
        feat = len(self.rnns) * 2 * conv_channels
        self.head_sal = nn.Sequential(
            nn.LayerNorm(feat),
            nn.Linear(feat, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.features(x, mask)
        return self.head_cat(h), self.head_sal(h)

