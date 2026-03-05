from __future__ import annotations

from typing import Protocol

import torch
import torch.nn as nn


class GRUModelConfigLike(Protocol):
    emb_dim: int
    hidden: int
    layers: int
    drop_out: float


class FC_residual(nn.Module):
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


class GRUWithIndexEmbedding(nn.Module):
    def __init__(
        self,
        num_indices: int,
        num_features: int,
        emb_dim: int = 0,
        hidden: int = 64,
        layers: int = 2,
        drop_out: float = 0.3,
    ):
        super().__init__()

        self.num_indices = int(num_indices)
        self.num_features = int(num_features)
        self.emb_dim = int(emb_dim)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.drop_out = float(drop_out)

        self.use_emb = self.emb_dim > 0
        self.emb = nn.Embedding(self.num_indices, self.emb_dim) if self.use_emb else None

        self.fc_in = nn.Sequential(
            nn.Linear(self.num_features, self.hidden),
            nn.Tanh(),
            nn.LayerNorm(self.hidden),
        )

        self.gru = nn.GRU(
            input_size=self.hidden,
            hidden_size=self.hidden,
            num_layers=self.layers,
            batch_first=True,
            dropout=self.drop_out if self.layers > 1 else 0.0,
        )

        self.drop = nn.Dropout(self.drop_out)

        head_in = self.hidden + (self.emb_dim if self.use_emb else 0)
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            FC_residual(head_in, dropout_rate=self.drop_out),
            nn.Linear(head_in, 1),
        )

    def forward(self, X: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        h0 = self.fc_in(X)
        out, _ = self.gru(h0)
        h = out[:, -1, :]

        if self.use_emb:
            if self.emb is None:
                raise RuntimeError("Embedding layer is not initialized")
            e = self.emb(ids)
            z = torch.cat([h, e], dim=1)
        else:
            z = h

        z = self.drop(z)
        y = self.head(z)
        return y


def build_gru_model(cfg: GRUModelConfigLike, num_indices: int, num_features: int) -> nn.Module:
    for k in ("emb_dim", "hidden", "layers", "drop_out"):
        if not hasattr(cfg, k):
            raise AttributeError(f"cfg missing required attribute: {k}")

    return GRUWithIndexEmbedding(
        num_indices=num_indices,
        num_features=num_features,
        emb_dim=int(getattr(cfg, "emb_dim")),
        hidden=int(getattr(cfg, "hidden")),
        layers=int(getattr(cfg, "layers")),
        drop_out=float(getattr(cfg, "drop_out")),
    )


GRUModel = GRUWithIndexEmbedding


if __name__ == "__main__":
    class _Cfg:
        emb_dim = 8
        hidden = 32
        layers = 2
        drop_out = 0.2

    cfg = _Cfg()
    model = build_gru_model(cfg, num_indices=50, num_features=10)
    X = torch.randn(16, 60, 10)
    ids = torch.randint(0, 50, (16,))
    y = model(X, ids)
    print("y shape:", y.shape)
    assert y.shape == (16, 1)
