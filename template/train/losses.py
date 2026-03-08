"""
损失函数和评价指标模块

包含：
    - PyTorch 可微损失函数（用于训练））
    - IC/RankIC 相关函数

损失函数列表：
    - MSE, Huber, Quantile（常规回归）
    - pearson_ic_loss（IC损失）
    - sigmoid_rankic_loss（Sigmoid软排序RankIC）
    - neuralsort_rankic_loss（NeuralSort软排序RankIC）
"""
from __future__ import annotations
from typing import Callable, TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from template.configs.model_runtime_config import Config


# ============================================================
# PyTorch 可微损失函数（用于训练）
# ============================================================

def pearson_ic_loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Pearson IC 损失函数（负相关系数，越小越好）
    
    直接计算预测值与真实值的Pearson相关系数，取负值作为损失。
    完全可导，适合作为IC优化的baseline。
    """
    pred = pred.squeeze()
    y = y.squeeze()
    
    if len(pred) < 3:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    
    pred_mean = torch.mean(pred)
    y_mean = torch.mean(y)
    
    numerator = torch.sum((pred - pred_mean) * (y - y_mean))
    denominator = torch.sqrt(torch.sum((pred - pred_mean)**2) * torch.sum((y - y_mean)**2))
    
    corr = numerator / (denominator + 1e-8)
    return -corr  # 负相关作为损失


def sigmoid_soft_rank(x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """Sigmoid 软排序：用成对比较的sigmoid近似rank
    
    对于每个元素i，其软rank = sum_j(sigmoid((x_j - x_i) / tau))
    温度tau越小越接近硬排序，但梯度越不稳定
    
    Args:
        x: 输入张量 [N]
        tau: 温度参数，控制平滑程度
    
    Returns:
        soft_rank: 软排序结果 [N]，值域约为 [0, N-1]
    """
    x = x.squeeze()
    n = x.size(0)
    
    # 计算成对差异矩阵: diff[i,j] = x[j] - x[i]
    x_row = x.unsqueeze(0)  # [1, N]
    x_col = x.unsqueeze(1)  # [N, 1]
    diff = x_row - x_col     # [N, N], diff[i,j] = x[j] - x[i]
    
    # Sigmoid 近似：统计有多少元素比当前元素大
    soft_rank = torch.sum(torch.sigmoid(diff / tau), dim=1)  # [N]
    
    return soft_rank


def sigmoid_rankic_loss(pred: torch.Tensor, y: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """Sigmoid Soft-Rank IC 损失函数
    
    使用 Sigmoid 软排序近似 Spearman 相关（RankIC），取负值作为损失。
    参考：方正证券《基于可微RankIC损失函数的深度学习选股策略》
    """
    pred = pred.squeeze()
    y = y.squeeze()
    
    # 算相关系数时至少需要3个样本
    if len(pred) < 3:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    
    # 计算软rank
    pred_rank = sigmoid_soft_rank(pred, tau)
    y_rank = sigmoid_soft_rank(y, tau)
    
    # 在软rank上计算 Pearson 相关（等价于 Spearman）
    pred_mean = torch.mean(pred_rank)
    y_mean = torch.mean(y_rank)
    
    numerator = torch.sum((pred_rank - pred_mean) * (y_rank - y_mean))
    denominator = torch.sqrt(torch.sum((pred_rank - pred_mean)**2) * torch.sum((y_rank - y_mean)**2))
    
    corr = numerator / (denominator + 1e-8)
    return -corr


def neuralsort_permutation(x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """NeuralSort: 生成软置换矩阵
    
    基于论文 "Stochastic Optimization of Sorting Networks via Continuous Relaxations"
    
    Args:
        x: 输入张量 [N]
        tau: 温度参数
    
    Returns:
        P_soft: 软置换矩阵 [N, N]，P_soft[i,j] 表示元素i排在位置j的"概率"
    """
    x = x.squeeze()
    n = x.size(0)
    
    # 对每个元素，计算它与所有其他元素的差
    x_row = x.unsqueeze(0)  # [1, N]
    x_col = x.unsqueeze(1)  # [N, 1]
    
    # 计算 |x_i - x_j| 的矩阵
    abs_diff = torch.abs(x_row - x_col)  # [N, N]
    
    # 计算每个元素在每个位置的得分
    # 位置k的得分 = (n+1-2k) * x_i - sum_j(|x_i - x_j|)
    positions = torch.arange(1, n + 1, dtype=x.dtype, device=x.device)
    position_weights = (n + 1 - 2 * positions).unsqueeze(0)  # [1, N]
    
    scores = position_weights * x.unsqueeze(1) - torch.sum(abs_diff, dim=1, keepdim=True)  # [N, N]
    
    # Softmax 得到软置换矩阵
    P_soft = F.softmax(scores / tau, dim=1)  # [N, N]
    
    return P_soft


def neuralsort_soft_rank(x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """NeuralSort 软排序：通过软置换矩阵计算软rank
    
    Args:
        x: 输入张量 [N]
        tau: 温度参数
    
    Returns:
        soft_rank: 软排序结果 [N]
    """
    P_soft = neuralsort_permutation(x, tau)
    n = x.size(0)
    
    # 软rank = 置换矩阵 * 位置向量
    positions = torch.arange(n, dtype=x.dtype, device=x.device)  # [0, 1, ..., N-1]
    soft_rank = torch.matmul(P_soft, positions)  # [N]
    
    return soft_rank


def neuralsort_rankic_loss(pred: torch.Tensor, y: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """NeuralSort RankIC 损失函数
    
    使用 NeuralSort 软排序近似 Spearman 相关（RankIC），取负值作为损失。
    """
    pred = pred.squeeze()
    y = y.squeeze()
    
    if len(pred) < 3:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    
    # 计算软rank
    pred_rank = neuralsort_soft_rank(pred, tau)
    y_rank = neuralsort_soft_rank(y, tau)
    
    # 在软rank上计算 Pearson 相关
    pred_mean = torch.mean(pred_rank)
    y_mean = torch.mean(y_rank)
    
    numerator = torch.sum((pred_rank - pred_mean) * (y_rank - y_mean))
    denominator = torch.sqrt(torch.sum((pred_rank - pred_mean)**2) * torch.sum((y_rank - y_mean)**2))
    
    corr = numerator / (denominator + 1e-8)
    return -corr


def get_loss_fn(cfg: "Config") -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """根据配置返回损失函数
    
    Args:
        cfg: 配置对象，需包含 task_type, loss_name 等属性
    
    Returns:
        损失函数 (pred, y) -> loss
    
    支持的损失函数：
        回归任务：mse, huber, quantile_loss, pearson_ic, sigmoid_rankic, neuralsort_rankic
        分类任务：bce
    """
    def quantile_loss(pred: torch.Tensor, y: torch.Tensor, tau: float = 0.9) -> torch.Tensor:
        """分位数损失
        
        tau: 分位数,0-1之间,如0.1,0.5,0.9等
        理论上当tau取0.5时计算得到的结果与MAE作为损失函数结果一致
        """
        error = y - pred
        quantile_error = torch.where(error >= 0, 
                                    tau * error, 
                                    (tau - 1) * error)
        return torch.mean(quantile_error)
    
    if cfg.task_type == "regression":
        if cfg.loss_name == "mse":
            return lambda pred, y: F.mse_loss(pred, y)
        if cfg.loss_name == "huber":
            return lambda pred, y: F.smooth_l1_loss(pred, y, beta=cfg.huber_delta)
        if cfg.loss_name == "quantile_loss":
            return lambda pred, y: quantile_loss(pred, y, tau=cfg.tau)
        # IC/RankIC 损失函数（需要按日batch训练）
        if cfg.loss_name == "pearson_ic":
            return lambda pred, y: pearson_ic_loss(pred, y)
        if cfg.loss_name == "sigmoid_rankic":
            return lambda pred, y: sigmoid_rankic_loss(pred, y, tau=cfg.softrank_tau)
        if cfg.loss_name == "neuralsort_rankic":
            return lambda pred, y: neuralsort_rankic_loss(pred, y, tau=cfg.softrank_tau)
        raise ValueError(f"Unsupported loss function: {cfg.loss_name}")
    
    if cfg.task_type == "classification":
        if cfg.loss_name == "bce":
            return lambda pred, y: F.binary_cross_entropy_with_logits(pred, y)
        raise ValueError(f"Unsupported loss function: {cfg.loss_name}")
    
    raise ValueError(f"Unsupported task type: {cfg.task_type}")