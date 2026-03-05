from __future__ import annotations

import os
import sys
from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from .layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from .layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
import math
import numpy as np


class AutoformerConfigLike(Protocol):
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


class Autoformer(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    Paper link: https://openreview.net/pdf?id=I55UqU-M11y
    """

    def __init__(self, configs):
        super(Autoformer, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        # Decomp
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AutoCorrelationLayer(
                            AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=False),
                            configs.d_model, configs.n_heads),
                        AutoCorrelationLayer(
                            AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=False),
                            configs.d_model, configs.n_heads),
                        configs.c_out,
                        configs.d_model,
                        configs.d_ff,
                        moving_avg=configs.moving_avg,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for l in range(configs.d_layers)
                ],
                norm_layer=my_Layernorm(configs.d_model),
                projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
            )
            # self.season_trans = nn.Linear(configs.d_model,configs.d_model)
            
        if self.task_name == 'imputation':
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(
            1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len,
                             x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        
        trend_init = torch.cat(
            [trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat(
            [seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        # trend_init = self.dec_embedding(trend_init,x_mark_dec)2
        
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # final
        dec_out = self.projection(enc_out)
        return dec_out

    def anomaly_detection(self, x_enc):
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # final
        dec_out = self.projection(enc_out)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)   # [B, L+O,c_out]
            return dec_out[:, -self.pred_len:, :]  # [B, O, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None

class AutoformerModel(nn.Module):
    def __init__(
        self,
        configs: "Configs",
        num_indices: int,
        emb_dim: int = 0,
        drop_out: float | None = None,
    ):
        super(AutoformerModel, self).__init__()
        self.out_len = configs.pred_len
        self.out_dim = configs.c_out
        self.autoformer = Autoformer(configs=configs)

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
        x_dec = x_enc
        x_mark_dec = None

        output = self.autoformer(x_enc, x_mark_enc, x_dec, x_mark_dec)  # (B, O, D)
        output = output.reshape(output.shape[0], -1)  # (B, O*D)

        if self.use_emb:
            if self.emb is None:
                raise RuntimeError("Embedding layer is not initialized")
            e = self.emb(ids)
            output = torch.cat([output, e], dim=1)

        output = self.drop(output)
        output = self.head(output)  # (B, 1)
        return output
        
class Configs:
    def __init__(self, config_dict=None):
        self.task_name = 'short_term_forecast'
        self.seq_len = 96
        self.label_len = 50
        self.pred_len = 2
        self.moving_avg = 5
        self.enc_in = 7
        self.dec_in = self.enc_in
        self.d_model = 64
        self.embed = 'fixed'
        self.freq = 'd'
        self.dropout = 0.1
        self.n_heads = 8
        self.d_ff = None
        self.e_layers = 2
        self.d_layers = 1
        self.c_out = self.enc_in
        self.activation = 'gelu'
        self.factor = 1
        self.num_class = 2
        # 如果传入了字典，更新属性
        if config_dict:
            for key, value in config_dict.items():
                setattr(self, key, value)
            # 同步 enc_in, dec_in 和 c_out 的值
            self._sync_enc_dec_c_out()
    def _sync_enc_dec_c_out(self):
        # 以 enc_in 的值为准，同步 dec_in 和 c_out
        self.dec_in = self.enc_in
        self.c_out = self.enc_in
        

def build_autoformer_model(cfg: AutoformerConfigLike, num_indices: int, num_features: int) -> nn.Module:
    for k in ("emb_dim", "hidden", "layers", "drop_out"):
        if not hasattr(cfg, k):
            raise AttributeError(f"cfg missing required attribute: {k}")

    seq_len = int(getattr(cfg, "seq_len", getattr(cfg, "lookback", 96)))
    label_len = int(getattr(cfg, "label_len", max(1, seq_len // 2)))
    pred_len = int(getattr(cfg, "pred_len", 1))

    config_dict = {
        "task_name": getattr(cfg, "autoformer_task_name", "short_term_forecast"),
        "seq_len": seq_len,
        "label_len": label_len,
        "pred_len": pred_len,
        "moving_avg": int(getattr(cfg, "autoformer_moving_avg", 5)),
        "enc_in": int(num_features),
        "d_model": int(getattr(cfg, "autoformer_d_model", getattr(cfg, "hidden"))),
        "embed": getattr(cfg, "autoformer_embed", "fixed"),
        "freq": getattr(cfg, "autoformer_freq", "d"),
        "dropout": float(getattr(cfg, "autoformer_dropout", getattr(cfg, "drop_out"))),
        "n_heads": int(getattr(cfg, "autoformer_n_heads", 8)),
        "d_ff": getattr(cfg, "autoformer_d_ff", None),
        "e_layers": int(getattr(cfg, "autoformer_e_layers", getattr(cfg, "layers"))),
        "d_layers": int(getattr(cfg, "autoformer_d_layers", 1)),
        "activation": getattr(cfg, "autoformer_activation", "gelu"),
        "factor": int(getattr(cfg, "autoformer_factor", 1)),
        "num_class": int(getattr(cfg, "autoformer_num_class", 2)),
    }

    configs = Configs(config_dict=config_dict)
    return AutoformerModel(
        configs=configs,
        num_indices=num_indices,
        emb_dim=int(getattr(cfg, "emb_dim")),
        drop_out=None,
    )




if __name__ == "__main__":     
    

    
    # 定义一个字典更新参数
    config_dict = {
        'seq_len': 60,
        'pred_len': 3,
        'd_model': 128
    }
       
    configs = Configs(config_dict=config_dict)
    num_indices = 50
    emb_dim = 0
    model = AutoformerModel(configs=configs, num_indices=num_indices, emb_dim=emb_dim)
    
    x = torch.randn(32, configs.seq_len, configs.enc_in)
    ids = torch.randint(0, num_indices, (32,))
    #x_enc = torch.randn(32, configs.seq_len, configs.enc_in)
    #x_mark_enc = None
    #x_dec = x_enc
    #x_mark_dec = None
    output = model(x, ids)
    
    print(f"test finished ! out_put shape:{output.shape}")
