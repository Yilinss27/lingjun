from __future__ import annotations

from typing import Protocol

import torch
import torch.nn as nn

from .layers.Embed import DataEmbedding
from .layers.SelfAttention_Family import FullAttention, AttentionLayer
from .layers.Transformer_EncDec import Encoder, EncoderLayer, _build_norm


class TransformerEncConfigLike(Protocol):
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


class TransformerEnc(nn.Module):
    """Encoder-only Transformer backbone (no decoder)."""

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model

        pos_emb_type = getattr(configs, "pos_emb_type", "fixed")
        norm_type = getattr(configs, "norm_type", "ln")

        self.enc_embedding = DataEmbedding(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
            pos_emb_type=pos_emb_type,
        )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    norm_type=norm_type,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=_build_norm(norm_type, configs.d_model),
        )

    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        """
        x_enc: [B, L, F]
        returns: [B, L, d_model]
        """
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, _attns = self.encoder(enc_out, attn_mask=None)
        return enc_out


class TransformerEncModel(nn.Module):
    """
    Encoder-only wrapper that mirrors TransformerModel's external interface.

    Head input comes from flattened encoder output [B, seq_len * d_model]
    instead of the original decoder output [B, pred_len * c_out].
    """

    def __init__(
        self,
        configs: "Configs",
        num_indices: int,
        emb_dim: int = 0,
        drop_out: float | None = None,
    ):
        super().__init__()
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        self.transformer_enc = TransformerEnc(configs=configs)

        self.emb_dim = int(emb_dim)
        self.use_emb = self.emb_dim > 0
        self.emb = nn.Embedding(int(num_indices), self.emb_dim) if self.use_emb else None

        self.drop_out = float(drop_out if drop_out is not None else configs.dropout)
        self.drop = nn.Dropout(self.drop_out)

        head_in = configs.seq_len * configs.d_model + (self.emb_dim if self.use_emb else 0)
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            FC_residual(head_in, dropout_rate=self.drop_out),
            nn.Linear(head_in, 1),
        )

    def forward(self, x: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """
        x:   [B, L, F]
        ids: [B]
        returns: [B, 1]
        """
        enc_out = self.transformer_enc(x)       # [B, L, d_model]
        output = enc_out.reshape(enc_out.size(0), -1)  # [B, L * d_model]

        if self.use_emb:
            if self.emb is None:
                raise RuntimeError("Embedding layer is not initialized")
            e = self.emb(ids)
            output = torch.cat([output, e], dim=1)

        output = self.drop(output)
        output = self.head(output)
        return output


class Configs:
    def __init__(self, config_dict=None):
        self.seq_len = 96
        self.enc_in = 7
        self.d_model = 64
        self.embed = "fixed"
        self.freq = "d"
        self.dropout = 0.1
        self.n_heads = 8
        self.d_ff = None
        self.e_layers = 2
        self.activation = "gelu"
        self.factor = 1
        self.pos_emb_type = "fixed"
        self.norm_type = "ln"

        if config_dict:
            for key, value in config_dict.items():
                setattr(self, key, value)


def build_transformer_enc_model(cfg: TransformerEncConfigLike, num_indices: int, num_features: int) -> nn.Module:
    for k in ("emb_dim", "hidden", "layers", "drop_out"):
        if not hasattr(cfg, k):
            raise AttributeError(f"cfg missing required attribute: {k}")

    seq_len = int(getattr(cfg, "seq_len", getattr(cfg, "lookback", 96)))

    config_dict = {
        "seq_len": seq_len,
        "enc_in": int(num_features),
        "d_model": int(getattr(cfg, "transformer_d_model", getattr(cfg, "hidden"))),
        "embed": getattr(cfg, "transformer_embed", "fixed"),
        "freq": getattr(cfg, "transformer_freq", "d"),
        "dropout": float(getattr(cfg, "transformer_dropout", getattr(cfg, "drop_out"))),
        "n_heads": int(getattr(cfg, "transformer_n_heads", 8)),
        "d_ff": getattr(cfg, "transformer_d_ff", None),
        "e_layers": int(getattr(cfg, "transformer_e_layers", getattr(cfg, "layers"))),
        "activation": getattr(cfg, "transformer_activation", "gelu"),
        "factor": int(getattr(cfg, "transformer_factor", 1)),
        "pos_emb_type": getattr(cfg, "transformer_pos_emb_type", "fixed"),
        "norm_type": getattr(cfg, "transformer_norm_type", "ln"),
    }

    configs = Configs(config_dict=config_dict)
    return TransformerEncModel(
        configs=configs,
        num_indices=num_indices,
        emb_dim=int(getattr(cfg, "emb_dim")),
        drop_out=None,
    )
