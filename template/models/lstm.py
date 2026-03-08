"""
LSTM 模型定义

只放模型结构，不放训练/评估/回测逻辑。
"""

# pyright: reportOptionalCall=false

from __future__ import annotations

import torch
import torch.nn as nn

from template.configs.pipeline_config import NNConfig


class LSTMWithIndexEmbedding(nn.Module):
    """带指数嵌入的 LSTM 模型
    
    结构：
    - 可选的指数嵌入层（emb_dim > 0 时启用）
    - 多层 LSTM
    - MLP head（拼接 LSTM 输出 + 嵌入后映射到标量）
    
    Args:
        num_indices: 指数数量（用于嵌入层）
        num_features: 输入特征数量
        emb_dim: 嵌入维度，0 表示不使用嵌入
        hidden: LSTM 隐藏层大小
        layers: LSTM 层数
        drop_out: Dropout 比例
    """
    
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
        
        self.use_emb = emb_dim > 0
        self.emb = nn.Embedding(num_indices, emb_dim) if self.use_emb else None
        
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=drop_out if layers > 1 else 0
        )
        
        self.drop = nn.Dropout(drop_out)
        
        head_in = hidden + (emb_dim if self.use_emb else 0)
        self.head = nn.Sequential(
            nn.Linear(head_in, head_in),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(head_in, 1)
        )
    
    def forward(self, X: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            X: 输入序列 [batch, seq_len, num_features]
            ids: 指数 ID [batch]
            
        Returns:
            预测值 [batch, 1]
        """
        out, _ = self.lstm(X)
        h = out[:, -1, :]  # 取最后一个时间步
        
        if self.use_emb:
            e = self.emb(ids)  # type: ignore
            z = torch.cat([h, e], dim=1)
        else:
            z = h
        
        z = self.drop(z)
        y = self.head(z)
        return y


def build_lstm_model(cfg: NNConfig, num_indices: int, num_features: int) -> nn.Module:
    """构建 LSTM 模型
    
    Args:
        cfg: 配置对象
        num_indices: 指数数量
        num_features: 输入特征数量
        
    Returns:
        初始化好的模型实例
    """
    return LSTMWithIndexEmbedding(
        num_indices=num_indices,
        num_features=num_features,
        emb_dim=cfg.emb_dim,
        hidden=cfg.hidden,
        layers=cfg.layers,
        drop_out=cfg.drop_out,
    )
