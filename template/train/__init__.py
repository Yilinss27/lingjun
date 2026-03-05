# pyright: reportAttributeAccessIssue=false
"""
train 模块：PyTorch 模型训练基础设施

包含：
    - checkpoint: 检查点管理器
    - losses: 损失函数和评价指标
    - torch_trainer: 数据加载、训练循环、TorchTrainer 类
"""

from .checkpoint import CheckpointManager

from .losses import (
    # PyTorch 损失函数
    pearson_ic_loss,
    sigmoid_soft_rank,
    sigmoid_rankic_loss,
    neuralsort_permutation,
    neuralsort_soft_rank,
    neuralsort_rankic_loss,
    get_loss_fn,
)

from .torch_trainer import (
    # 随机种子
    seed_worker,
    # 设备管理
    get_torch_device,
    # 数据集类
    NpDatasetTyped,
    collate_fn,
    DailyBatchDataset,
    make_dataloader,
    # 训练/评估循环（函数式接口）
    train_one_epoch,
    train_one_epoch_daily,
    evaluate,
    evaluate_daily,
    # 推理辅助
    predict_to_df,
    # 训练器类（面向对象接口）
    TorchTrainer,
)
from .gbdt_trainer import GBDTTrainer, GBDTTrainResult


__all__ = [
    # checkpoint
    'CheckpointManager',
    # losses - PyTorch
    'pearson_ic_loss',
    'sigmoid_soft_rank',
    'sigmoid_rankic_loss',
    'neuralsort_permutation',
    'neuralsort_soft_rank',
    'neuralsort_rankic_loss',
    'get_loss_fn',
    # torch_trainer
    'seed_worker',
    'get_torch_device',
    'NpDatasetTyped',
    'collate_fn',
    'DailyBatchDataset',
    'make_dataloader',
    'train_one_epoch',
    'train_one_epoch_daily',
    'evaluate',
    'evaluate_daily',
    'predict_to_df',
    'TorchTrainer',
    'GBDTTrainer',
    'GBDTTrainResult',
]
