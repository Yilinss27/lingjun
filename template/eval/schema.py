from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


def require_columns(df: pd.DataFrame, columns: Iterable[str], df_name: str = "df") -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{df_name} missing required columns: {missing}")


def ensure_pred_df(
    pred_df: pd.DataFrame,
    *,
    date_col: str = "date",
    score_col: str = "pred",
    target_col: str = "y",
    id_col: str | None = None,
) -> None:
    columns = [date_col, score_col, target_col]
    if id_col:
        columns.append(id_col)
    require_columns(pred_df, columns, df_name="pred_df")
    pred_df[date_col] = pd.to_datetime(pred_df[date_col])


def ensure_score_return_df(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    score_col: str = "pred",
    return_col: str = "return",
) -> None:
    require_columns(df, [date_col, score_col, return_col], df_name="score_return_df")
    df[date_col] = pd.to_datetime(df[date_col])


def ensure_backtest_results(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    decile_col: str = "decile",
    return_col: str = "return",
    n_stocks_col: str = "n_stocks",
) -> None:
    required = [date_col, decile_col, return_col, n_stocks_col]
    require_columns(df, required, df_name="backtest_results")
    df[date_col] = pd.to_datetime(df[date_col])


@dataclass(frozen=True)
class PredColumns:
    date_col: str = "date"
    score_col: str = "pred"
    target_col: str = "y"
    id_col: str | None = "index_id"


@dataclass(frozen=True)
class ScoreReturnColumns:
    date_col: str = "date"
    score_col: str = "pred"
    return_col: str = "return"


@dataclass(frozen=True)
class BacktestResultColumns:
    date_col: str = "date"
    decile_col: str = "decile"
    return_col: str = "return"
    n_stocks_col: str = "n_stocks"
