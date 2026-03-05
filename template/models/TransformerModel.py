from __future__ import annotations

from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.Embed import DataEmbedding
from .layers.SelfAttention_Family import FullAttention, AttentionLayer
from .layers.Transformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, _build_norm


class TransformerConfigLike(Protocol):
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


class Transformer(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

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
        self.dec_embedding = DataEmbedding(
            configs.dec_in,
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
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model,
                        configs.n_heads,
                    ),
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
                for _ in range(configs.d_layers)
            ],
            norm_layer=_build_norm(norm_type, configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True),
        )

        if self.task_name == "imputation":
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == "anomaly_detection":
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == "classification":
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        return dec_out

    def imputation(self, x_enc, x_mark_enc):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out)
        return dec_out

    def anomaly_detection(self, x_enc):
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        output = self.act(enc_out)
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == "long_term_forecast" or self.task_name == "short_term_forecast":
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == "imputation":
            return self.imputation(x_enc, x_mark_enc)
        if self.task_name == "anomaly_detection":
            return self.anomaly_detection(x_enc)
        if self.task_name == "classification":
            return self.classification(x_enc, x_mark_enc)
        return None


class TransformerModel(nn.Module):
    def __init__(
        self,
        configs: "Configs",
        num_indices: int,
        emb_dim: int = 0,
        drop_out: float | None = None,
    ):
        super().__init__()
        self.out_len = configs.pred_len
        self.out_dim = configs.c_out
        self.transformer = Transformer(configs=configs)
        self.dec_input_mode = str(getattr(configs, "dec_input", "standard"))

        self.emb_dim = int(emb_dim)
        self.use_emb = self.emb_dim > 0
        self.emb = nn.Embedding(int(num_indices), self.emb_dim) if self.use_emb else None

        self.drop_out = float(drop_out if drop_out is not None else configs.dropout)
        self.drop = nn.Dropout(self.drop_out)

        head_in = configs.pred_len * configs.c_out + (self.emb_dim if self.use_emb else 0)
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            FC_residual(head_in, dropout_rate=self.drop_out),
            nn.Linear(head_in, 1),
        )

    def forward(self, x: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        x_enc = x
        x_mark_enc = None
        if self.dec_input_mode == "legacy":
            x_dec = x_enc
        else:
            label_len = int(getattr(self.transformer, "label_len", 0))
            pred_len = int(getattr(self.transformer, "pred_len", 0))
            if label_len > x_enc.size(1):
                label_len = x_enc.size(1)
            x_dec = torch.zeros(
                (x_enc.size(0), label_len + pred_len, x_enc.size(2)),
                device=x_enc.device,
                dtype=x_enc.dtype,
            )
            if label_len > 0:
                x_dec[:, :label_len, :] = x_enc[:, -label_len:, :]
        x_mark_dec = None

        output = self.transformer(x_enc, x_mark_enc, x_dec, x_mark_dec)
        output = output.reshape(output.shape[0], -1)

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
        self.task_name = "short_term_forecast"
        self.seq_len = 96
        self.label_len = 48
        self.pred_len = 2
        self.enc_in = 7
        self.dec_in = self.enc_in
        self.d_model = 64
        self.embed = "fixed"
        self.freq = "d"
        self.dropout = 0.1
        self.n_heads = 8
        self.d_ff = None
        self.e_layers = 2
        self.d_layers = 1
        self.c_out = self.enc_in
        self.activation = "gelu"
        self.factor = 1
        self.num_class = 2
        self.dec_input = "standard"
        # --- new: positional encoding & normalization ---
        self.pos_emb_type = "fixed"   # "fixed" (sinusoidal) | "learnable"
        self.norm_type = "ln"         # "ln" (LayerNorm) | "bn" (BatchNorm)

        if config_dict:
            for key, value in config_dict.items():
                setattr(self, key, value)
            self._sync_enc_dec_c_out()

    def _sync_enc_dec_c_out(self):
        self.dec_in = self.enc_in
        self.c_out = self.enc_in


def build_transformer_model(cfg: TransformerConfigLike, num_indices: int, num_features: int) -> nn.Module:
    for k in ("emb_dim", "hidden", "layers", "drop_out"):
        if not hasattr(cfg, k):
            raise AttributeError(f"cfg missing required attribute: {k}")

    seq_len = int(getattr(cfg, "seq_len", getattr(cfg, "lookback", 96)))
    label_len = int(getattr(cfg, "label_len", max(1, seq_len // 2)))
    pred_len = int(getattr(cfg, "pred_len", 1))

    config_dict = {
        "task_name": getattr(cfg, "transformer_task_name", "short_term_forecast"),
        "seq_len": seq_len,
        "label_len": label_len,
        "pred_len": pred_len,
        "enc_in": int(num_features),
        "d_model": int(getattr(cfg, "transformer_d_model", getattr(cfg, "hidden"))),
        "embed": getattr(cfg, "transformer_embed", "fixed"),
        "freq": getattr(cfg, "transformer_freq", "d"),
        "dropout": float(getattr(cfg, "transformer_dropout", getattr(cfg, "drop_out"))),
        "n_heads": int(getattr(cfg, "transformer_n_heads", 8)),
        "d_ff": getattr(cfg, "transformer_d_ff", None),
        "e_layers": int(getattr(cfg, "transformer_e_layers", getattr(cfg, "layers"))),
        "d_layers": int(getattr(cfg, "transformer_d_layers", 1)),
        "activation": getattr(cfg, "transformer_activation", "gelu"),
        "factor": int(getattr(cfg, "transformer_factor", 1)),
        "num_class": int(getattr(cfg, "transformer_num_class", 2)),
        "dec_input": getattr(cfg, "transformer_dec_input", "standard"),
        "pos_emb_type": getattr(cfg, "transformer_pos_emb_type", "fixed"),
        "norm_type": getattr(cfg, "transformer_norm_type", "ln"),
    }

    configs = Configs(config_dict=config_dict)
    return TransformerModel(
        configs=configs,
        num_indices=num_indices,
        emb_dim=int(getattr(cfg, "emb_dim")),
        drop_out=None,
    )
