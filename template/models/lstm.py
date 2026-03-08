"""
LSTM 模型定义（兼容现有框架）

目标：保持与你当前训练/评估框架的接口一致：
- 提供 `LSTMWithIndexEmbedding` 类
- 提供 `build_lstm_model(cfg, num_indices, num_features)` 工厂函数
- `forward(X, ids)`：X = [batch, seq_len, num_features], ids = [batch]

扩展：
- 可选 LSTM + Attention
- residual_position: "start" | "middle" | "end" | "none"
  - start: 输入投影后逐时间步残差 MLP（res_mlp_blocks）
  - middle: LSTM 残差旁路 LN(x + Dropout(LSTM(x)))
  - end: head 前残差 MLP（LayerNorm + FC_residual）
"""

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn as nn


class LSTMModelConfigLike(Protocol):
    """build_lstm_model() 所需的最小配置接口（用于类型检查）"""

    emb_dim: int
    hidden: int
    layers: int
    drop_out: float


class FC_residual(nn.Module):
    """两层 MLP + 残差 + LayerNorm

    既支持输入形状 [batch, hidden_dim]，也支持 [batch, seq_len, hidden_dim]。
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
        out = self.ln(x + y)
        return out


class LSTMWithIndexEmbedding(nn.Module):
    """带（可选）指数嵌入的 LSTM 模型"""

    def __init__(
        self,
        num_indices: int,
        num_features: int,
        emb_dim: int = 0,
        hidden: int = 64,
        layers: int = 2,
        drop_out: float = 0.3,
        use_attention: bool = False,
        residual_position: str = "middle",
        res_mlp_blocks: int = 2,
    ):
        super().__init__()

        self.num_indices = int(num_indices)
        self.num_features = int(num_features)
        self.emb_dim = int(emb_dim)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.drop_out = float(drop_out)
        self.use_attention = bool(use_attention)

        self.residual_position = str(residual_position).lower()
        if self.residual_position not in {"start", "middle", "end", "none"}:
            raise ValueError(
                "residual_position must be one of: start, middle, end, none; "
                f"got {self.residual_position}"
            )
        self.res_mlp_blocks = max(0, int(res_mlp_blocks))

        self.use_emb = self.emb_dim > 0
        self.emb = nn.Embedding(self.num_indices, self.emb_dim) if self.use_emb else None

        self.fc_in = nn.Sequential(
            nn.Linear(self.num_features, self.hidden),
            nn.Tanh(),
            nn.LayerNorm(self.hidden),
        )

        if self.residual_position == "start" and self.res_mlp_blocks > 0:
            self.pre_mlp = nn.Sequential(
                *[FC_residual(self.hidden, dropout_rate=self.drop_out) for _ in range(self.res_mlp_blocks)]
            )
        else:
            self.pre_mlp = nn.Identity()

        self.lstm = nn.LSTM(
            input_size=self.hidden,
            hidden_size=self.hidden,
            num_layers=self.layers,
            batch_first=True,
            dropout=self.drop_out if self.layers > 1 else 0.0,
        )

        self.lstm_res_drop = nn.Dropout(self.drop_out)
        self.lstm_res_ln = nn.LayerNorm(self.hidden)

        if self.use_attention:
            self.att_net = nn.Sequential(
                nn.Linear(self.hidden, self.hidden // 2),
                nn.Dropout(self.drop_out),
                nn.Tanh(),
                nn.Linear(self.hidden // 2, 1, bias=False),
                nn.Softmax(dim=1),
            )
        else:
            self.att_net = None

        self.drop = nn.Dropout(self.drop_out)

        feature_dim = self.hidden * 2 if self.use_attention else self.hidden
        head_in = feature_dim + (self.emb_dim if self.use_emb else 0)

        if self.residual_position == "end":
            self.head = nn.Sequential(
                nn.LayerNorm(head_in),
                FC_residual(head_in, dropout_rate=self.drop_out),
                nn.Linear(head_in, 1),
            )
        else:
            self.head = nn.Linear(head_in, 1)

    def forward(self, X: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        h_in = self.fc_in(X)
        h_in = self.pre_mlp(h_in)

        out, _ = self.lstm(h_in)

        if self.residual_position == "middle":
            out = self.lstm_res_ln(h_in + self.lstm_res_drop(out))

        if self.use_attention:
            if self.att_net is None:
                raise RuntimeError("Attention network is not initialized")
            attention_score = self.att_net(out)
            out_att = torch.sum(out * attention_score, dim=1)
            features = torch.cat((out[:, -1, :], out_att), dim=1)
        else:
            features = out[:, -1, :]

        if self.use_emb:
            if self.emb is None:
                raise RuntimeError("Embedding layer is not initialized")
            e = self.emb(ids)
            features = torch.cat([features, e], dim=1)

        features = self.drop(features)
        y = self.head(features)
        return y


def build_lstm_model(cfg: LSTMModelConfigLike, num_indices: int, num_features: int) -> nn.Module:
    """构建 LSTM 模型（与现有框架对齐）"""
    for k in ("emb_dim", "hidden", "layers", "drop_out"):
        if not hasattr(cfg, k):
            raise AttributeError(f"cfg missing required attribute: {k}")

    use_attention = bool(getattr(cfg, "use_attention", True))
    residual_position = str(getattr(cfg, "residual_position", "start"))
    res_mlp_blocks = int(getattr(cfg, "res_mlp_blocks", 2))

    return LSTMWithIndexEmbedding(
        num_indices=num_indices,
        num_features=num_features,
        emb_dim=int(getattr(cfg, "emb_dim")),
        hidden=int(getattr(cfg, "hidden")),
        layers=int(getattr(cfg, "layers")),
        drop_out=float(getattr(cfg, "drop_out")),
        use_attention=use_attention,
        residual_position=residual_position,
        res_mlp_blocks=res_mlp_blocks,
    )


LSTMModel = LSTMWithIndexEmbedding
