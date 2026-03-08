"""
GRU 模型板块
"""

# pyright: reportOptionalCall=false

from __future__ import annotations

import torch
import torch.nn as nn

from template.configs.pipeline_config import NNConfig


class GRUWithIndexEmbedding(nn.Module):
    """带指数嵌入的 GRU 模型
    
    - 可选的指数嵌入层（emb_dim > 0 时启用）
    - 多层 GRU
    - MLP head（拼接 GRU 输出 + 嵌入后映射到标量）
    
    Args:
        num_indices: 指数数量（用于嵌入层）
        num_features: 输入特征数量
        gru_hidden: GRU 隐藏层大小（对应 LSTM 的 hidden）
        gru_layers: GRU 层数（对应 LSTM 的 layers）
        emb_dim: 嵌入维度，0 表示不使用嵌入
        drop_out: Dropout 比例
    """
    
    def __init__(
        self,
        num_indices: int,
        num_features: int,
        gru_hidden: int = 64,
        gru_layers: int = 2,
        emb_dim: int = 0,
        drop_out: float = 0.3,
    ):
        super().__init__()
        
        self.use_emb = emb_dim > 0
        self.emb = nn.Embedding(num_indices, emb_dim) if self.use_emb else None
        

        self.gru = nn.GRU(
            input_size=num_features,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=drop_out if gru_layers > 1 else 0
        )
        
        self.drop = nn.Dropout(drop_out)
        
        head_in = gru_hidden + (emb_dim if self.use_emb else 0)
        self.head = nn.Sequential(
            nn.Linear(head_in, head_in//2),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(head_in//2, 1)
        )
    
    def forward(self, X: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            X: 输入序列 [batch, seq_len, num_features]
            ids: 指数 ID [batch]
            
        Returns:
            预测值 [batch, 1]
        """
        out, _ = self.gru(X)  # GRU 返回 (output, h_n)，LSTM 返回 (output, (h_n, c_n))
        h = out[:, -1, :]  # 取最后一个时间步
        
        if self.use_emb:
            e = self.emb(ids)  # type: ignore
            z = torch.cat([h, e], dim=1)
        else:
            z = h
        
        z = self.drop(z)
        y = self.head(z)
        return y


def build_gru_model(cfg: NNConfig, num_indices: int, num_features: int) -> nn.Module:
    """构建 GRU 模型。cfg 为 NNConfig，gru_hidden/gru_layers 可选，缺省复用 hidden/layers。"""
        num_indices: 指数数量
        num_features: 输入特征数量
        
    Returns:
        初始化好的模型实例
    """
    # 优先使用 gru_* 配置，否则复用 hidden/layers 配置
    gru_hidden = getattr(cfg, 'gru_hidden', cfg.hidden)
    gru_layers = getattr(cfg, 'gru_layers', cfg.layers)
    
    return GRUWithIndexEmbedding(
        num_indices=num_indices,
        num_features=num_features,
        gru_hidden=gru_hidden,
        gru_layers=gru_layers,
        emb_dim=cfg.emb_dim,
        drop_out=cfg.drop_out,
    )
