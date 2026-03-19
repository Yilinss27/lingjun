# pyright: reportAttributeAccessIssue=false
"""
train 模块：模型训练基础设施

包含：
    - nn_trainer: NN 训练器 (NNTrainer) + V2 评估 (evaluate_v2)
    - checkpoint: 检查点管理器
    - losses: 损失函数
    - torch_trainer: 设备管理 (get_torch_device)
    - gbdt_trainer: GBDT 训练器
"""

from .checkpoint import CheckpointManager

from .losses import (
    pearson_ic_loss,
    sigmoid_soft_rank,
    sigmoid_rankic_loss,
    neuralsort_permutation,
    neuralsort_soft_rank,
    neuralsort_rankic_loss,
    get_loss_fn,
)

from .torch_trainer import get_torch_device
from .nn_trainer import NNTrainer, evaluate_v2
from .gbdt_trainer import GBDTTrainer, GBDTTrainResult


__all__ = [
    'CheckpointManager',
    'pearson_ic_loss',
    'sigmoid_soft_rank',
    'sigmoid_rankic_loss',
    'neuralsort_permutation',
    'neuralsort_soft_rank',
    'neuralsort_rankic_loss',
    'get_loss_fn',
    'get_torch_device',
    'NNTrainer',
    'evaluate_v2',
    'GBDTTrainer',
    'GBDTTrainResult',
]
