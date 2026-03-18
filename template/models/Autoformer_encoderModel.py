# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# pyright: reportConstantRedefinition=false
from __future__ import annotations

from typing import Protocol, cast

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AutoformerEncoderConfigLike(Protocol):
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

class AutoCorrelation(nn.Module):
    """
    AutoCorrelation Mechanism with the following two phases:
    (1) period-based dependencies discovery
    (2) time delay aggregation
    This block can replace the self-attention family mechanism seamlessly.
    """
    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        
        self.dropout = nn.Dropout(attention_dropout)

    def time_delay_agg_training(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the training phase.
        """
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def time_delay_agg_inference(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the inference phase.
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = (
            torch.arange(length, device=values.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(batch, head, channel, 1)
        )
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights = torch.topk(mean_value, top_k, dim=-1)[0]
        delay = torch.topk(mean_value, top_k, dim=-1)[1]
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def time_delay_agg_full(self, values, corr):
        """
        Standard version of Autocorrelation
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = (
            torch.arange(length, device=values.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(batch, head, channel, 1)
        )
        # find top k
        top_k = int(self.factor * math.log(length))
        weights = torch.topk(corr, top_k, dim=-1)[0]
        delay = torch.topk(corr, top_k, dim=-1)[1]
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[..., i].unsqueeze(-1)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (tmp_corr[..., i].unsqueeze(-1))
        return delays_agg

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        orig_dtype = queries.dtype
        # cuFFT in half precision only supports power-of-two sizes; run FFT in float32
        if orig_dtype in (torch.float16, torch.bfloat16):
            queries = queries.float()
            keys = keys.float()
            values = values.float()
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # period-based dependencies
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)

        # time delay agg
        if self.training:
            V = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
        else:
            V = self.time_delay_agg_inference(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)

        if orig_dtype in (torch.float16, torch.bfloat16):
            V = V.to(orig_dtype)
        return V.contiguous()


class AutoCorrelationLayer(nn.Module):
    def __init__(self, correlation, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AutoCorrelationLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_correlation(
            queries,
            keys,
            values
        )
        out = out.view(B, L, -1)

        return self.out_projection(out)


class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """
    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class EncoderLayer(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture
    """
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.layer_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])

    def forward(self, x):
        new_x= self.attention(
            x, x, x
        )
        x = x + self.dropout(new_x)
        
        x = self.layer_norm[0](x)
        
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x = x+y
        x = self.layer_norm[1](x)
        return x





class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, pe_freq=10000.0,max_len=1000):
        """
        Positional encoding module for adding positional information to input sequences.

        Args:
            hidden_dim (int): The dimensionality of the model's input and output embeddings.
            max_len (int, optional): The maximum length of the input sequence. Defaults to 1000.
        """
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(pe_freq) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Add positional encoding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape [T, N, F], where T is the sequence length,
                              N is the batch size, and F is the feature dimension.

        Returns:
            torch.Tensor: Tensor with positional encoding added, of the same shape as the input.
        """
        pe = cast(torch.Tensor, self.pe)
        return x + pe[: x.size(0), :]


class Autoformer_encoderModel(nn.Module):
    def __init__(
        self,
        num_indices: int,
        num_features: int,
        emb_dim: int = 0,
        hidden: int = 32,
        pe_freq: float = 10000.0,
        kernel_size: int = 5,
        n_heads: int = 4,
        num_layers: int = 2,
        drop_out: float = 0.25,
    ):
        super(Autoformer_encoderModel, self).__init__()

        self.num_indices = int(num_indices)
        self.num_features = int(num_features)
        self.emb_dim = int(emb_dim)
        self.hidden = int(hidden)
        self.kernel_size = int(kernel_size)
        self.drop_out = float(drop_out)

        self.use_emb = self.emb_dim > 0
        self.emb = nn.Embedding(self.num_indices, self.emb_dim) if self.use_emb else None

        self.conv1 = nn.Conv1d(
            in_channels=self.num_features,
            out_channels=self.num_features,
            kernel_size=self.kernel_size,
            padding=0,
            padding_mode="replicate",
        )
        self.conv1_act = nn.ReLU()
        self.conv1_dropout = nn.Dropout(p=self.drop_out)

        self.fc_in = nn.Sequential(
            nn.Linear(self.num_features, self.hidden),
            nn.LayerNorm(self.hidden),
        )

        self.pos_encoder = PositionalEncoding(self.hidden, pe_freq=pe_freq)
        self.encoder = nn.Sequential(
            *[
                EncoderLayer(
                    attention=AutoCorrelationLayer(
                        correlation=AutoCorrelation(mask_flag=False),
                        d_model=self.hidden,
                        n_heads=n_heads,
                    ),
                    d_model=self.hidden,
                    dropout=self.drop_out,
                )
                for _ in range(num_layers)
            ]
        )

        self.drop = nn.Dropout(self.drop_out)
        head_in = self.hidden + (self.emb_dim if self.use_emb else 0)
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            FC_residual(head_in, dropout_rate=self.drop_out),
            nn.Linear(head_in, 1),
        )

    def forward(self, src: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        # 调整输入维度，从 [N, T, F] 变为 [N, F, T] 以适应 Conv1d 的输入要求
        src = src.permute(0, 2, 1)

        # 手动计算填充量
        total_padding = self.kernel_size - 1
        left_padding = total_padding // 2
        right_padding = total_padding - left_padding

        # 手动进行填充
        padded_src = F.pad(src, (left_padding, right_padding), mode="replicate")

        # 进行卷积操作
        conv_output = self.conv1(padded_src)

        # 激活函数和 Dropout
        conv_output = self.conv1_dropout(self.conv1_act(conv_output))

        # 残差连接与ln
        src = src + conv_output

        # 调整维度回到 [N, T, F]
        src = src.permute(0, 2, 1)

        
        # src [N, T, F], e.g., [512, 60, 6]
        src = self.fc_in(src)  # [512, 60, 8]

        # src [N, T, F] --> [T, N, F], e.g., [60, 512, 8]
        src = src.transpose(1, 0)  # Using batch first, transpose shape for positional embedding.


        src = self.pos_encoder(src)
        
        # src [T, N, F] --> [N, T, F], e.g., [512, 60, 8]
        src = src.transpose(1,0)
        
        output = self.encoder(src)  # [N, T, H]
        h = output[:, -1, :]

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


def build_autoformer_encoder_model(
    cfg: AutoformerEncoderConfigLike, num_indices: int, num_features: int
) -> nn.Module:
    for k in ("emb_dim", "hidden", "layers", "drop_out"):
        if not hasattr(cfg, k):
            raise AttributeError(f"cfg missing required attribute: {k}")

    hidden = int(getattr(cfg, "encoder_hidden", getattr(cfg, "hidden")))
    layers = int(getattr(cfg, "encoder_layers", getattr(cfg, "layers")))
    drop_out = float(getattr(cfg, "encoder_drop_out", getattr(cfg, "drop_out")))

    return Autoformer_encoderModel(
        num_indices=num_indices,
        num_features=num_features,
        emb_dim=int(getattr(cfg, "emb_dim")),
        hidden=hidden,
        pe_freq=float(getattr(cfg, "pe_freq", 10000.0)),
        kernel_size=int(getattr(cfg, "encoder_kernel_size", 5)),
        n_heads=int(getattr(cfg, "encoder_n_heads", 4)),
        num_layers=layers,
        drop_out=drop_out,
    )


if __name__ == "__main__":
    # 检查是否有可用的 GPU，如果有则使用 GPU，否则使用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 初始化模型参数
    num_indices = 50
    num_features = 6
    emb_dim = 0
    hidden = 32
    pe_freq = 1000.0
    kernel_size = 5
    n_heads = 4
    num_layers = 2
    drop_out = 0.25
    # 创建模型实例并移动到指定设备
    model = Autoformer_encoderModel(
        num_indices,
        num_features,
        emb_dim,
        hidden,
        pe_freq,
        kernel_size,
        n_heads,
        num_layers,
        drop_out,
    ).to(device)
    # 生成随机输入数据并移动到指定设备
    batch_size = 32
    seq_length = 60
    input_tensor = torch.randn(batch_size, seq_length, num_features).to(device)
    ids = torch.randint(0, num_indices, (batch_size,), device=device)
    # 前向传播
    output = model(input_tensor, ids)
    print("Output shape:", output.shape)
