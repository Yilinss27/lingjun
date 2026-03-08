from __future__ import annotations

from typing import Protocol

import torch
import torch.nn as nn


class LSTMModelConfigLike(Protocol):
    emb_dim: int
    hidden: int
    layers: int
    drop_out: float


class FCResidual(nn.Module):
    """Two-layer MLP with residual + LayerNorm.

    Supports [batch, hidden_dim] and [batch, seq_len, hidden_dim].
    """

    def __init__(self, hidden_dim: int, fc_dim: int | None = None, dropout_rate: float = 0.15):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.fc_dim = int(fc_dim or hidden_dim * 4)
        self.dropout_rate = float(dropout_rate)

        self.fc1 = nn.Linear(self.hidden_dim, self.fc_dim)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Linear(self.fc_dim, self.hidden_dim)
        self.dropout2 = nn.Dropout(self.dropout_rate)
        self.ln = nn.LayerNorm(self.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dropout1(self.act1(self.fc1(x)))
        y = self.dropout2(self.fc2(y))
        return self.ln(x + y)
