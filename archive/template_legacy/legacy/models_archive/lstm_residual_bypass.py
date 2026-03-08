from __future__ import annotations

import torch
import torch.nn as nn

from .common import LSTMModelConfigLike


class LSTMResidualBypass(nn.Module):
    """LSTM with residual bypass around the recurrent stack."""

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

        self.lstm = nn.LSTM(
            input_size=self.hidden,
            hidden_size=self.hidden,
            num_layers=self.layers,
            batch_first=True,
            dropout=self.drop_out if self.layers > 1 else 0.0,
        )

        self.lstm_res_drop = nn.Dropout(self.drop_out)
        self.lstm_res_ln = nn.LayerNorm(self.hidden)
        self.drop = nn.Dropout(self.drop_out)

        head_in = self.hidden + (self.emb_dim if self.use_emb else 0)
        self.head = nn.Linear(head_in, 1)

    def forward(self, X: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        h0 = self.fc_in(X)
        out, _ = self.lstm(h0)
        out = self.lstm_res_ln(h0 + self.lstm_res_drop(out))
        h = out[:, -1, :]

        if self.use_emb:
            if self.emb is None:
                raise RuntimeError("Embedding layer is not initialized")
            e = self.emb(ids)
            z = torch.cat([h, e], dim=1)
        else:
            z = h

        z = self.drop(z)
        return self.head(z)


def build_lstm_residual_bypass_model(
    cfg: LSTMModelConfigLike, num_indices: int, num_features: int
) -> nn.Module:
    for k in ("emb_dim", "hidden", "layers", "drop_out"):
        if not hasattr(cfg, k):
            raise AttributeError(f"cfg missing required attribute: {k}")

    return LSTMResidualBypass(
        num_indices=num_indices,
        num_features=num_features,
        emb_dim=int(getattr(cfg, "emb_dim")),
        hidden=int(getattr(cfg, "hidden")),
        layers=int(getattr(cfg, "layers")),
        drop_out=float(getattr(cfg, "drop_out")),
    )
