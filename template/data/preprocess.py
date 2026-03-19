from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Sequence

import numpy as np
import pandas as pd



@dataclass
class PreprocessorState:
    fillna_strategy: str            # 缺失值填充策略
    standardize: bool               # 是否标准化 (True 等同于 standardize_mode="global")
    clip_quantile: float | None     # 是否采用分位数裁剪，None表示不裁剪
    medians: dict[str, float]       # 每列中位数（用于填充）
    means: dict[str, float]         # 每列均值（用于标准化）
    stds: dict[str, float]          # 每列标准差（用于标准化）
    clip_ranges: dict[str, tuple[float, float]]  # 每列分位数裁剪区间
    standardize_mode: str = "none"  # "none"/"global"/"cross_section"/"date"/"time_rolling"
    date_col: str = "dateid"
    time_col: str = "timeid"
    z_window: int = 60


def fit_preprocessor(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    fillna_strategy: str = "none",
    standardize: bool = False,
    clip_quantile: float | None = None,
) -> PreprocessorState:
    """
    计算训练集的预处理统计参数（如中位数、均值、标准差、分位点），并封装为PreprocessorState用于后续特征处理。
    """
    cols = list(feature_cols)
    medians: dict[str, float] = {}
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    clip_ranges: dict[str, tuple[float, float]] = {}

    if fillna_strategy == "median":
        medians = {
            col: float(train_df[col].median(skipna=True))
            for col in cols
        }

    if standardize:
        means = {
            col: float(train_df[col].mean(skipna=True))
            for col in cols
        }
        stds = {}
        for col in cols:
            std_v = float(train_df[col].std(skipna=True))
            stds[col] = std_v if std_v > 1e-12 else 1.0

    if clip_quantile is not None:
        q = float(clip_quantile)
        low_q = (1.0 - q) / 2.0
        high_q = 1.0 - low_q
        clip_ranges = {
            col: (
                float(train_df[col].quantile(low_q)),
                float(train_df[col].quantile(high_q)),
            )
            for col in cols
        }

    return PreprocessorState(
        fillna_strategy=fillna_strategy,
        standardize=standardize,
        clip_quantile=clip_quantile,
        medians=medians,
        means=means,
        stds=stds,
        clip_ranges=clip_ranges,
    )


def transform_features(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    state: PreprocessorState,
    *,
    copy_frame: bool = True,
) -> pd.DataFrame:
    """
    按照提供的PreprocessorState，对指定特征列进行缺失值填充、分位数裁剪和标准化等处理。
    """
    out = df.copy() if copy_frame else df
    cols = list(feature_cols)

    if state.fillna_strategy == "zero":
        out[cols] = out[cols].fillna(0.0)
    elif state.fillna_strategy == "median":
        out[cols] = out[cols].fillna(value=state.medians)

    if state.clip_quantile is not None:
        for col in cols:
            low, high = state.clip_ranges[col]
            out[col] = out[col].clip(lower=low, upper=high)

    mode = state.standardize_mode
    if mode == "none" and state.standardize:
        mode = "global"

    if mode == "global" and state.means and state.stds:
        for col in cols:
            out[col] = (out[col] - state.means[col]) / state.stds[col]
    elif mode == "cross_section":
        group_cols = [c for c in [state.date_col, state.time_col] if c in out.columns]
        if group_cols:
            grp = out.groupby(group_cols, sort=False)[cols]
            mu = grp.transform("mean")
            # Use built-in "std" string instead of lambda for C-level speed
            sd = grp.transform("std")
            sd = sd.where(sd.abs() > 1e-12, 1.0)
            out[cols] = (out[cols] - mu) / sd
    elif mode == "date":
        if state.date_col in out.columns:
            grp = out.groupby(state.date_col, sort=False)[cols]
            mu = grp.transform("mean")
            sd = grp.transform(lambda x: x.std(ddof=0))
            sd = sd.where(sd.abs() > 1e-12, 1.0)
            out[cols] = (out[cols] - mu) / sd
    elif mode == "time_rolling":
        id_col = "stockid"
        if id_col in out.columns and state.date_col in out.columns:
            sort_c = [id_col] + [c for c in [state.date_col, state.time_col] if c in out.columns]
            out = out.sort_values(sort_c)
            grp = out.groupby(id_col, sort=False)[cols]
            mu = grp.transform(lambda x: x.rolling(state.z_window, min_periods=1).mean())
            sd = grp.transform(lambda x: x.rolling(state.z_window, min_periods=1).std(ddof=0))
            sd = sd.where(sd.abs() > 1e-12, 1.0)
            out[cols] = (out[cols] - mu) / sd

    out[cols] = out[cols].astype(np.float32, copy=False)
    return out


def fit_transform_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    feature_cols: Sequence[str],
    fillna_strategy: str = "none",
    standardize: bool = False,
    clip_quantile: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, PreprocessorState]:
    """
    依次对训练集、验证集、留出集拟合并变换特征，返回处理后数据和预处理状态参数。
    """
    state = fit_preprocessor(
        train_df=train_df,
        feature_cols=feature_cols,
        fillna_strategy=fillna_strategy,
        standardize=standardize,
        clip_quantile=clip_quantile,
    )
    train_t = transform_features(train_df, feature_cols, state)
    val_t = transform_features(val_df, feature_cols, state)
    holdout_t = transform_features(holdout_df, feature_cols, state)
    return train_t, val_t, holdout_t, state
