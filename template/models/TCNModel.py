from __future__ import annotations

from typing import Protocol, Sequence, cast

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


class TCNConfigLike(Protocol):
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


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self) -> None:
        conv1_weight = cast(torch.Tensor, self.conv1.weight)
        conv2_weight = cast(torch.Tensor, self.conv2.weight)
        nn.init.normal_(conv1_weight, 0.0, 0.01)
        nn.init.normal_(conv2_weight, 0.0, 0.01)
        if self.downsample is not None:
            down_weight = cast(torch.Tensor, self.downsample.weight)
            nn.init.normal_(down_weight, 0.0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TCNWithIndexEmbedding(nn.Module):
    def __init__(
        self,
        num_indices: int,
        num_features: int,
        emb_dim: int = 0,
        hidden: int = 64,
        layers: int = 3,
        drop_out: float = 0.2,
        kernel_size: int = 2,
        channels: Sequence[int] | None = None,
        clip_output: bool = True,
    ):
        super().__init__()

        self.num_indices = int(num_indices)
        self.num_features = int(num_features)
        self.emb_dim = int(emb_dim)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.drop_out = float(drop_out)
        self.kernel_size = int(kernel_size)
        self.clip_output = bool(clip_output)

        if channels is None:
            channels = [self.hidden] * max(1, self.layers)
        self.channels = [int(c) for c in channels]

        self.use_emb = self.emb_dim > 0
        self.emb = nn.Embedding(self.num_indices, self.emb_dim) if self.use_emb else None

        self.tcn = TemporalConvNet(self.num_features, self.channels, self.kernel_size, dropout=self.drop_out)

        self.drop = nn.Dropout(self.drop_out)

        head_in = self.channels[-1] + (self.emb_dim if self.use_emb else 0)
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            FC_residual(head_in, dropout_rate=self.drop_out),
            nn.Linear(head_in, 1),
        )

    def forward(self, X: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        x = X.permute(0, 2, 1)
        features = self.tcn(x)
        h = features[:, :, -1]

        if self.use_emb:
            if self.emb is None:
                raise RuntimeError("Embedding layer is not initialized")
            e = self.emb(ids)
            z = torch.cat([h, e], dim=1)
        else:
            z = h

        z = self.drop(z)
        y = self.head(z)
        if self.clip_output:
            y = torch.clamp(y, min=-5.0, max=5.0)
        return y


def build_tcn_model(cfg: TCNConfigLike, num_indices: int, num_features: int) -> nn.Module:
    for k in ("emb_dim", "hidden", "layers", "drop_out"):
        if not hasattr(cfg, k):
            raise AttributeError(f"cfg missing required attribute: {k}")

    drop_out = float(getattr(cfg, "tcn_drop_out", getattr(cfg, "drop_out")))
    channels = getattr(cfg, "tcn_channels", None)
    if channels is not None:
        channels = list(channels)

    return TCNWithIndexEmbedding(
        num_indices=num_indices,
        num_features=num_features,
        emb_dim=int(getattr(cfg, "emb_dim")),
        hidden=int(getattr(cfg, "hidden")),
        layers=int(getattr(cfg, "layers")),
        drop_out=drop_out,
        kernel_size=int(getattr(cfg, "tcn_kernel_size", 2)),
        channels=channels,
        clip_output=bool(getattr(cfg, "tcn_clip_output", True)),
    )


TCNModel = TCNWithIndexEmbedding


if __name__ == "__main__":
    class _Cfg:
        emb_dim = 4
        hidden = 32
        layers = 3
        drop_out = 0.2

    cfg = _Cfg()
    model = build_tcn_model(cfg, num_indices=50, num_features=7)
    X = torch.randn(32, 20, 7)
    ids = torch.randint(0, 50, (32,))
    y = model(X, ids)
    print("y shape:", y.shape)
    assert y.shape == (32, 1)
