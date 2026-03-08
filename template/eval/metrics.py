"""
eval/metrics.py - 跨模型通用的评估指标模块

包含:
- 基础损失函数的 numpy 实现 (mae, mse, huber, quantile)
- IC/RankIC 评价指标 (numpy 版本)
- 按日期分组的 IC 计算 (核心研究指标，最强泛化能力)
- 统一的损失计算接口

设计原则:
- 所有函数为纯函数，不依赖全局状态
- 可用于任意模型的评估，不限于 LSTM
- by_date 版本的 IC 指标最具研究价值，推荐优先使用
"""
# pyright: reportMissingTypeArgument=false, reportArgumentType=false
from __future__ import annotations

from typing import Protocol, runtime_checkable
import numpy as np
import pandas as pd

from .schema import ensure_pred_df, require_columns

__all__ = [
    # 核心研究指标（按日期分组，最强泛化能力，推荐优先使用）
    "rank_ic_by_date",
    "pearson_ic_by_date",
    # 基础损失函数
    "mae_np",
    "mse_np",
    "r2_score_np",
    "huber_loss_np",
    "quantile_loss_np",
    # 统一损失计算
    "compute_loss_np",
    # IC 工具函数
    "pearson_ic_np",
    "spearman_rankic_np",
    # IC 序列与汇总
    "ic_series_by_date",
    "ic_summary",
    # 配置协议
    "LossConfig",
]


# ============================================================
# 配置协议定义（支持任意配置类，只要有必要属性即可）
# ============================================================
@runtime_checkable
class LossConfig(Protocol):
    """compute_loss_np 所需的配置协议，任何具有这些属性的对象都可使用"""
    loss_name: str
    huber_delta: float
    tau: float


def _compute_daily_ic(
    group: pd.DataFrame,
    score_col: str,
    target_col: str,
    min_samples: int,
) -> pd.Series:
    valid = group[[score_col, target_col]].dropna()
    if len(valid) > 0:
        mask = np.isfinite(valid[score_col]) & np.isfinite(valid[target_col])
        valid = valid[mask]
    if len(valid) < min_samples:
        return pd.Series({"ic": np.nan, "rank_ic": np.nan})
    try:
        ic = valid[score_col].corr(valid[target_col])
        rank_ic = valid[score_col].corr(valid[target_col], method="spearman")
    except Exception:
        ic = np.nan
        rank_ic = np.nan
    return pd.Series({"ic": ic, "rank_ic": rank_ic})


def ic_series_by_date(
    pred_df: pd.DataFrame,
    date_col: str = "date",
    score_col: str = "pred",
    target_col: str = "y",
    min_samples: int = 10,
) -> pd.DataFrame:
    """
    计算日度 IC / RankIC 序列
    """
    ensure_pred_df(pred_df, date_col=date_col, score_col=score_col, target_col=target_col)
    metrics_df = pred_df.groupby(date_col).apply(
        lambda g: _compute_daily_ic(g, score_col, target_col, min_samples),
        include_groups=False,
    ).reset_index()
    return metrics_df


def ic_summary(ic_df: pd.DataFrame) -> dict:
    """
    汇总 IC / RankIC 指标
    """
    require_columns(ic_df, ["ic", "rank_ic"], df_name="ic_df")
    if len(ic_df) == 0:
        return {
            "ic_mean": np.nan,
            "ic_std": np.nan,
            "ic_ir": np.nan,
            "rank_ic_mean": np.nan,
            "rank_ic_std": np.nan,
            "rank_ic_ir": np.nan,
        }

    ic_mean = ic_df["ic"].mean()
    ic_std = ic_df["ic"].std()
    rank_ic_mean = ic_df["rank_ic"].mean()
    rank_ic_std = ic_df["rank_ic"].std()

    ic_ir = ic_mean / ic_std if ic_std > 0 else np.nan
    rank_ic_ir = rank_ic_mean / rank_ic_std if rank_ic_std > 0 else np.nan

    return {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ic_ir": ic_ir,
        "rank_ic_mean": rank_ic_mean,
        "rank_ic_std": rank_ic_std,
        "rank_ic_ir": rank_ic_ir,
    }


# ============================================================
# 核心研究指标：按日期分组的 IC 计算（最强泛化能力）
# ============================================================
def rank_ic_by_date(
    pred_df: pd.DataFrame,
    date_col: str = "date",
    score_col: str = "pred",
    target_col: str = "y"
) -> float:
    """
    按日期分组计算 Spearman RankIC 的平均值
    
    这是量化研究中最核心的评价指标之一，用于评估因子/模型的选股能力。
    通过按日期分组，消除了不同日期间收益率分布差异的影响。
    
    Args:
        pred_df: 包含预测值和真实值的 DataFrame
        date_col: 日期列名
        score_col: 预测分数列名
        target_col: 真实收益列名
    
    Returns:
        float: RankIC 均值（正值，越大越好）
    
    Example:
        >>> df = pd.DataFrame({
        ...     "date": [1,1,1,2,2,2],
        ...     "pred": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        ...     "y": [0.05, 0.15, 0.25, 0.35, 0.45, 0.55]
        ... })
        >>> rank_ic_by_date(df)
        1.0
    """
    vals = []
    for _, g in pred_df.groupby(date_col):
        if len(g) < 3:
            continue
        vals.append(g[score_col].rank().corr(g[target_col].rank(), method="spearman"))
    vals = [v for v in vals if v == v]  # 过滤 nan
    return float(np.mean(vals)) if vals else float("nan")


def pearson_ic_by_date(
    pred_df: pd.DataFrame,
    date_col: str = "date",
    score_col: str = "pred",
    target_col: str = "y"
) -> float:
    """
    按日期分组计算 Pearson IC 的平均值
    
    与 RankIC 相比，Pearson IC 保留了原始数值的线性关系，
    对异常值更敏感，但也能捕捉到更多的数值信息。
    
    Args:
        pred_df: 包含预测值和真实值的 DataFrame
        date_col: 日期列名
        score_col: 预测分数列名
        target_col: 真实收益列名
    
    Returns:
        float: Pearson IC 均值（正值，越大越好）
    """
    vals = []
    for _, g in pred_df.groupby(date_col):
        if len(g) < 3:
            continue
        corr = g[score_col].corr(g[target_col], method="pearson")
        if corr == corr:  # not nan
            vals.append(corr)
    return float(np.mean(vals)) if vals else float("nan")


# ============================================================
# 基础损失函数（numpy 实现，用于评估，不可微）
# ============================================================
def mae_np(a: np.ndarray, b: np.ndarray) -> float:
    """
    Mean Absolute Error (MAE)
    
    Args:
        a: 预测值数组
        b: 真实值数组
    
    Returns:
        float: MAE 值（越小越好）
    """
    return float(np.mean(np.abs(a - b)))


def r2_score_np(pred: np.ndarray, y: np.ndarray) -> float:
    """R-squared (coefficient of determination).

    R^2 = 1 - SS_res / SS_tot, where
    SS_res = sum((y - pred)^2), SS_tot = sum((y - mean(y))^2).
    Perfect prediction gives 1.0; always-predicting-mean gives 0.0.
    """
    p = pred.flatten().astype(np.float64)
    t = y.flatten().astype(np.float64)
    if len(t) < 2:
        return float("nan")
    ss_res = float(np.sum((t - p) ** 2))
    ss_tot = float(np.sum((t - np.mean(t)) ** 2))
    if ss_tot < 1e-15:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def mse_np(a: np.ndarray, b: np.ndarray) -> float:
    """
    Mean Squared Error (MSE)
    
    Args:
        a: 预测值数组
        b: 真实值数组
    
    Returns:
        float: MSE 值（越小越好）
    """
    return float(np.mean((a - b) ** 2))


def huber_loss_np(pred: np.ndarray, y: np.ndarray, delta: float = 1.0) -> float:
    """
    Huber 损失的 numpy 实现
    
    结合了 MSE 和 MAE 的优点：
    - 对于小误差（|error| <= delta），使用平方损失
    - 对于大误差（|error| > delta），使用线性损失
    
    Args:
        pred: 预测值数组
        y: 真实值数组
        delta: 阈值参数，控制平方损失和线性损失的切换点
    
    Returns:
        float: Huber 损失值（越小越好）
    """
    error = np.abs(y - pred)
    quadratic = np.minimum(error, delta)
    linear = error - quadratic
    return float(np.mean(0.5 * quadratic ** 2 + delta * linear))


def quantile_loss_np(pred: np.ndarray, y: np.ndarray, tau: float = 0.9) -> float:
    """
    分位数损失 (Quantile Loss / Pinball Loss)
    
    用于分位数回归，可以预测条件分位数而非条件均值。
    tau=0.9 时，模型倾向于预测 90% 分位数。
    
    Args:
        pred: 预测值数组
        y: 真实值数组
        tau: 分位数参数，范围 (0, 1)
    
    Returns:
        float: 分位数损失值（越小越好）
    """
    error = y - pred
    loss = np.maximum(tau * error, (tau - 1) * error)
    return float(np.mean(loss))


# ============================================================
# IC/RankIC 工具函数（numpy 版本，用于评估）
# ============================================================
def pearson_ic_np(pred: np.ndarray, y: np.ndarray) -> float:
    """
    Pearson IC（相关系数）的 numpy 实现
    
    计算预测值和真实值之间的 Pearson 相关系数。
    注意：这是全局计算，不分组。对于因子研究，推荐使用 pearson_ic_by_date。
    
    Args:
        pred: 预测值数组
        y: 真实值数组
    
    Returns:
        float: Pearson 相关系数（正值，越大越好）
    """
    from scipy.stats import pearsonr
    
    pred_flat = pred.flatten()
    y_flat = y.flatten()
    
    if len(pred_flat) < 3:
        return float("nan")
    
    try:
        result = pearsonr(pred_flat, y_flat)
        corr = float(result[0])  # type: ignore
        return float(corr) if not np.isnan(corr) else float("nan")
    except Exception:
        return float("nan")


def spearman_rankic_np(pred: np.ndarray, y: np.ndarray) -> float:
    """
    Spearman RankIC（秩相关系数）的 numpy 实现
    
    计算预测值和真实值之间的 Spearman 秩相关系数。
    相比 Pearson IC，RankIC 对异常值更鲁棒。
    注意：这是全局计算，不分组。对于因子研究，推荐使用 rank_ic_by_date。
    
    Args:
        pred: 预测值数组
        y: 真实值数组
    
    Returns:
        float: Spearman 秩相关系数（正值，越大越好）
    """
    from scipy.stats import spearmanr
    
    pred_flat = pred.flatten()
    y_flat = y.flatten()
    
    if len(pred_flat) < 3:
        return float("nan")
    
    try:
        result = spearmanr(pred_flat, y_flat)
        corr = float(result[0]) if isinstance(result, tuple) else float(result.correlation)  # type: ignore
        return corr if not np.isnan(corr) else float("nan")
    except Exception:
        return float("nan")


# ============================================================
# 统一损失计算接口
# ============================================================
def compute_loss_np(pred: np.ndarray, y: np.ndarray, cfg: LossConfig) -> float:
    """
    根据配置计算损失函数的 numpy 版本
    
    这是一个统一的接口，根据配置中的 loss_name 选择相应的损失函数。
    用于评估阶段，与训练时的 PyTorch 损失函数保持一致。
    
    Args:
        pred: 预测值数组
        y: 真实值数组
        cfg: 配置对象，需要有 loss_name, huber_delta, tau 属性
             可使用任何符合 LossConfig 协议的对象
    
    Returns:
        float: 损失值
        
    Supported loss_name:
        - "mse": Mean Squared Error
        - "huber": Huber Loss
        - "quantile_loss": Quantile Loss
        - "pearson_ic", "sigmoid_rankic", "neuralsort_rankic": IC 类损失（返回负 RankIC）
    
    Raises:
        ValueError: 当 loss_name 不支持时
    """
    if cfg.loss_name == "mse":
        return mse_np(pred, y)
    elif cfg.loss_name == "huber":
        return huber_loss_np(pred, y, cfg.huber_delta)
    elif cfg.loss_name == "quantile_loss":
        return quantile_loss_np(pred, y, cfg.tau)
    elif cfg.loss_name in ["pearson_ic", "sigmoid_rankic", "neuralsort_rankic"]:
        # IC 类损失返回负 IC（越小越好），所以这里返回负值
        return -spearman_rankic_np(pred, y)
    else:
        raise ValueError(f"Unsupported loss function: {cfg.loss_name}")
