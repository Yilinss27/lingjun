from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
import pandas as pd


@dataclass
class PreprocessorState:
    fillna_strategy: str
    standardize: bool
    clip_quantile: float | None
    medians: dict[str, float]
    means: dict[str, float]
    stds: dict[str, float]
    clip_ranges: dict[str, tuple[float, float]]


def fit_preprocessor(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    fillna_strategy: str = "none",
    standardize: bool = False,
    clip_quantile: float | None = None,
) -> PreprocessorState:
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

    if state.standardize:
        for col in cols:
            mean_v = state.means[col]
            std_v = state.stds[col]
            out[col] = (out[col] - mean_v) / std_v

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
