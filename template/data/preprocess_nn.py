"""
Preprocessing V2 — 可扩展的标准化策略模块。

设计理念：
    1. 标准化策略与数据存储/加载完全解耦
    2. 每个 Standardizer 只负责一件事：对截面 block [n_stocks, n_features] 做变换
    3. fit_medians_streaming 只负责从流式数据中计算中位数用于填充
    4. 工厂函数 get_standardizer 统一创建入口

使用方式：
    # 在 memmap_v2 的 derived cache 构建中
    standardizer = get_standardizer("rank_gauss")
    for each (dateid, timeid):
        block = raw_cache[:, d, t, :]        # [n_stocks, n_features]
        normed = standardizer.transform_cross_section(block)
        derived_cache[:, d, t, :] = normed
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from scipy.special import ndtri
from scipy.stats import rankdata


# ============================================================
# Standardizer Protocol
# ============================================================

@runtime_checkable
class Standardizer(Protocol):
    """标准化策略接口。

    每个实现只需提供 transform_cross_section 方法，
    输入为同一截面（dateid, timeid）的所有股票特征矩阵，
    输出为标准化后的同形状矩阵。
    """

    def transform_cross_section(self, block: np.ndarray) -> np.ndarray:
        """对截面 block 做标准化。

        Args:
            block: shape [n_stocks, n_features], float32

        Returns:
            shape [n_stocks, n_features], float32
        """
        ...


# ============================================================
# RankGauss Standardizer
# ============================================================

class RankGaussStandardizer:
    """截面 RankGauss 标准化。

    对每个特征独立地：
        1. 在截面内对所有股票排名（处理并列用 average 方法）
        2. 将排名映射到 (0, 1) 开区间
        3. 应用逆正态变换（ndtri / erfinv）
        4. Clip 到 [-clip_val, clip_val] 防止极端值

    适用场景：
        金融截面数据，特征分布不规则，需要消除截面内的分布差异。
    """

    def __init__(self, clip_val: float = 6.0):
        self.clip_val = clip_val

    def transform_cross_section(self, block: np.ndarray) -> np.ndarray:
        n_stocks, n_features = block.shape

        # 边界：单只或零只股票无法排名
        if n_stocks <= 1:
            return np.zeros_like(block, dtype=np.float32)

        # 1. 截面排名，axis=0 → 对每列（每个特征）在所有股票间排名
        #    method='average' 处理并列（median 填充会产生大量并列值）
        ranks = rankdata(block, axis=0, method="average")

        # 2. 映射到 (0, 1) 开区间：p = (rank - 0.5) / n
        p = (ranks - 0.5) / n_stocks

        # 3. 逆正态变换
        out = ndtri(p)

        # 4. 处理全同值列：rankdata 对全同值返回相同 rank → p 相同 → ndtri 返回同一个值
        #    检测方差为 0 的列，置零
        #    用 float64 计算避免 float32 下 (x-mean)² 求和 overflow
        col_std = np.std(block, axis=0, dtype=np.float64)
        zero_std_mask = col_std < 1e-12
        if np.any(zero_std_mask):
            out[:, zero_std_mask] = 0.0

        # 5. Clip 极端值
        np.clip(out, -self.clip_val, self.clip_val, out=out)

        return out.astype(np.float32)


# ============================================================
# Global Z-Score Standardizer
# ============================================================

class GlobalZScoreStandardizer:
    """全量 Z-Score 标准化。

    使用预计算的全局均值和标准差做标准化：(x - mean) / std。
    作为 fallback / legacy 模式。

    Args:
        means: shape [n_features], 全局均值
        stds:  shape [n_features], 全局标准差（std <= 1e-12 的位置应预设为 1.0）
    """

    def __init__(self, means: np.ndarray, stds: np.ndarray):
        self.means = np.asarray(means, dtype=np.float32).ravel()
        self.stds = np.asarray(stds, dtype=np.float32).ravel()
        # 防止除零
        self.stds = np.where(self.stds > 1e-12, self.stds, 1.0)

    def transform_cross_section(self, block: np.ndarray) -> np.ndarray:
        # 广播：block [n_stocks, n_features] - means [n_features]
        return ((block - self.means) / self.stds).astype(np.float32)


# ============================================================
# NoOp Standardizer
# ============================================================

class NoOpStandardizer:
    """不做任何标准化，直接返回原值。用于测试或调试。"""

    def transform_cross_section(self, block: np.ndarray) -> np.ndarray:
        return block.astype(np.float32)


# ============================================================
# Factory
# ============================================================

def get_standardizer(method: str, **kwargs) -> Standardizer:
    """根据方法名创建标准化器实例。

    Args:
        method: "rank_gauss" | "global_zscore" | "none"
        **kwargs: 各方法需要的额外参数
            - rank_gauss: clip_val (float, default 6.0)
            - global_zscore: means (np.ndarray), stds (np.ndarray)

    Returns:
        Standardizer instance
    """
    if method == "rank_gauss":
        clip_val = kwargs.get("clip_val", 6.0)
        return RankGaussStandardizer(clip_val=clip_val)

    if method == "global_zscore":
        means = kwargs.get("means")
        stds = kwargs.get("stds")
        if means is None or stds is None:
            raise ValueError("global_zscore requires 'means' and 'stds' arrays")
        return GlobalZScoreStandardizer(means=means, stds=stds)

    if method == "none":
        return NoOpStandardizer()

    raise ValueError(f"Unknown standardization method: {method!r}")


# ============================================================
# Streaming Fit Result
# ============================================================

@dataclass
class StreamingFitResult:
    """fit_medians_streaming 的返回值。"""
    medians: dict[str, float]          # 各特征的近似中位数（水库采样）
    global_means: dict[str, float]     # 各特征的全局均值（Welford 精确）
    global_stds: dict[str, float]      # 各特征的全局标准差（Welford 精确）


# ============================================================
# Streaming Median Fit (for missing value imputation)
# ============================================================

def fit_medians_streaming(
    train_batches: Iterable[pd.DataFrame],
    feature_cols: Sequence[str],
    reservoir_size: int = 200_000,
    on_batch: Callable[[int, int], None] | None = None,
    compute_global_stats: bool = False,
) -> StreamingFitResult:
    """流式计算训练集各特征的近似中位数（用于缺失值填充）。

    使用 Reservoir Sampling 避免全量数据加载：
        - 前 reservoir_size 行直接存入
        - 之后的行以递减概率随机替换
        - 最终对 reservoir 计算精确中位数

    Args:
        train_batches: 训练数据批次迭代器（DataFrame）
        feature_cols: 特征列名列表
        reservoir_size: 水库大小，越大越精确，默认 200K
        on_batch: 进度回调 (batches_done, rows_seen)
        compute_global_stats: 是否计算全局均值/标准差（仅 global_zscore 需要）

    Returns:
        StreamingFitResult (medians, global_means, global_stds)
    """
    cols = list(feature_cols)
    n_feat = len(cols)

    reservoir = np.empty((reservoir_size, n_feat), dtype=np.float32)
    res_fill = 0
    total_seen = 0
    batches_done = 0

    # Welford 累加器：仅 global_zscore 标准化需要
    if compute_global_stats:
        running_count = np.zeros(n_feat, dtype=np.float64)
        running_mean = np.zeros(n_feat, dtype=np.float64)
        running_m2 = np.zeros(n_feat, dtype=np.float64)

    for batch_df in train_batches:
        if len(batch_df) == 0:
            continue

        values = batch_df[cols].to_numpy(dtype=np.float32)
        n_batch = len(values)

        # ── Welford 并行合并（计算全局均值/方差）── 仅按需计算
        if compute_global_stats:
            values_f64 = values.astype(np.float64)
            valid_mask = ~np.isnan(values_f64)
            b_count = valid_mask.sum(axis=0).astype(np.float64)
            safe_vals = np.where(valid_mask, values_f64, 0.0)
            b_sum = safe_vals.sum(axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                b_mean = np.where(b_count > 0, b_sum / b_count, 0.0)
            b_dev = np.where(valid_mask, values_f64 - b_mean, 0.0)
            b_m2 = (b_dev ** 2).sum(axis=0)

            total = running_count + b_count
            delta = b_mean - running_mean
            with np.errstate(invalid="ignore", divide="ignore"):
                new_mean = np.where(
                    total > 0,
                    (running_count * running_mean + b_count * b_mean) / total,
                    0.0,
                )
                new_m2 = running_m2 + b_m2 + np.where(
                    total > 0, delta ** 2 * running_count * b_count / total, 0.0,
                )
            running_count, running_mean, running_m2 = total, new_mean, new_m2

        # 填充阶段：直接写入 reservoir
        if res_fill < reservoir_size:
            take = min(n_batch, reservoir_size - res_fill)
            reservoir[res_fill : res_fill + take] = values[:take]
            res_fill += take
            rest_start = take
        else:
            rest_start = 0

        # 替换阶段：以 reservoir_size / (total_seen + i) 的概率替换
        n_rest = n_batch - rest_start
        if n_rest > 0:
            base = total_seen + rest_start
            totals = np.arange(base + 1, base + 1 + n_rest, dtype=np.float64)
            rand = np.random.random(n_rest)
            indices = (rand * totals).astype(np.int64)
            mask = indices < reservoir_size
            if mask.any():
                reservoir[indices[mask]] = values[rest_start:][mask]

        total_seen += n_batch
        batches_done += 1
        if on_batch is not None:
            on_batch(batches_done, total_seen)

    # 从 reservoir 计算中位数
    medians: dict[str, float] = {}
    n_res = min(res_fill, reservoir_size)
    if n_res > 0:
        for j, col in enumerate(cols):
            col_vals = reservoir[:n_res, j]
            valid = col_vals[~np.isnan(col_vals)]
            medians[col] = float(np.median(valid)) if len(valid) > 0 else 0.0
    else:
        for col in cols:
            medians[col] = 0.0

    # 从 Welford 累加器计算全局均值/标准差（仅在需要时）
    if compute_global_stats:
        final_std = np.sqrt(running_m2 / np.maximum(running_count, 1))
        global_means = {col: float(running_mean[j]) for j, col in enumerate(cols)}
        global_stds = {
            col: float(final_std[j]) if final_std[j] > 1e-12 else 1.0
            for j, col in enumerate(cols)
        }
    else:
        global_means = {}
        global_stds = {}

    return StreamingFitResult(
        medians=medians,
        global_means=global_means,
        global_stds=global_stds,
    )
