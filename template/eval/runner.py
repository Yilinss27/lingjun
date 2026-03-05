"""
eval/runner.py - 一站式评估管线
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import pandas as pd

from eval.backtest import (
    BacktestConfig,
    backtest_decile_strategy,
    backtest_grouped_returns,
    compute_backtest_summary,
    compute_long_short_metrics,
)
from eval.experiment_log import ExperimentConfig, log_experiment_full, log_experiment_full_csv
from eval.metrics import (
    ic_series_by_date,
    mae_np,
    mse_np,
    pearson_ic_by_date,
    rank_ic_by_date,
)
from eval.plots import plot_backtest_results, plot_ic_series, plot_long_short_curve
from eval.schema import ensure_pred_df, require_columns


@dataclass
class EvalReport:
    metrics: Dict[str, float] = field(default_factory=dict)
    ic_series: Optional[pd.DataFrame] = None
    backtest_results: Optional[pd.DataFrame] = None
    backtest_summary: Optional[pd.DataFrame] = None
    long_short_metrics: Optional[Dict[str, float]] = None
    plot_paths: Dict[str, str] = field(default_factory=dict)
    log_paths: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class _PriceConfig:
    id_col: str
    date_col: str


def run_evaluation(
    pred_df: pd.DataFrame,
    *,
    price_df: Optional[pd.DataFrame] = None,
    cfg: Optional[BacktestConfig] = None,
    date_col: str = "date",
    id_col: str | None = "index_id",
    score_col: str = "pred",
    target_col: str = "y",
    return_col: Optional[str] = None,
    price_id_col: str = "thscode",
    price_date_col: str = "date",
    n_groups: int = 10,
    risk_free_rate: float = 0.0,
    long_decile: int = 1,
    short_decile: int = 10,
    start_date: Optional[str] = None,
    backtest_tag: str = "total",
    save_dir: str = "res",
    model_name: str = "",
    plot: bool = True,
    show: bool = False,
    ic_min_samples: int = 10,
    log_cfg: Optional[ExperimentConfig] = None,
    log_dir: str = "log/experiments",
    experiment_name: Optional[str] = None,
    extra_info: Optional[Dict[str, Any]] = None,
    log_full_csv: bool = False,
) -> EvalReport:
    """
    一站式评估：指标 -> IC序列 -> 回测 -> 可视化 -> 日志
    """
    ensure_pred_df(pred_df, date_col=date_col, score_col=score_col, target_col=target_col, id_col=id_col)

    report = EvalReport()
    valid = pred_df[[score_col, target_col]].dropna()
    if len(valid) > 0:
        pred = valid[score_col].to_numpy()
        y = valid[target_col].to_numpy()
        report.metrics = {
            "mse": mse_np(pred, y),
            "mae": mae_np(pred, y),
            "pearson_ic": pearson_ic_by_date(pred_df, date_col=date_col, score_col=score_col, target_col=target_col),
            "rank_ic": rank_ic_by_date(pred_df, date_col=date_col, score_col=score_col, target_col=target_col),
        }
    else:
        report.metrics = {
            "mse": float("nan"),
            "mae": float("nan"),
            "pearson_ic": float("nan"),
            "rank_ic": float("nan"),
        }

    report.ic_series = ic_series_by_date(
        pred_df,
        date_col=date_col,
        score_col=score_col,
        target_col=target_col,
        min_samples=ic_min_samples,
    )

    backtest_results = None
    if price_df is not None:
        if id_col is None:
            raise ValueError("id_col is required when price_df is provided")
        price_cfg = cfg or _PriceConfig(id_col=price_id_col, date_col=price_date_col)
        require_columns(price_df, [price_cfg.id_col, price_cfg.date_col, "open", "close"], df_name="price_df")
        backtest_results = backtest_decile_strategy(
            pred_df,
            price_df,
            price_cfg, #type:ignore
            n_groups=n_groups,
            pred_col=score_col,
            index_col=id_col,
            date_col=date_col,
        )
    else:
        return_col = return_col or target_col
        backtest_results = backtest_grouped_returns(
            pred_df,
            score_col=score_col,
            return_col=return_col,
            date_col=date_col,
            n_groups=n_groups,
        )

    report.backtest_results = backtest_results
    report.backtest_summary = compute_backtest_summary(backtest_results, risk_free_rate) if len(backtest_results) > 0 else None
    report.long_short_metrics = (
        compute_long_short_metrics(backtest_results, long_decile=long_decile, short_decile=short_decile)
        if len(backtest_results) > 0
        else None
    )

    if plot:
        if len(backtest_results) > 0:
            report.plot_paths["backtest"] = plot_backtest_results(
                backtest_results,
                type=backtest_tag,
                save_dir=save_dir,
                start_date=start_date,
                show=show,
                model_name=model_name,
            )
            report.plot_paths["long_short"] = plot_long_short_curve(
                backtest_results,
                long_decile=long_decile,
                short_decile=short_decile,
                save_dir=save_dir,
                start_date=start_date,
                show=show,
                model_name=model_name,
            )
        report.plot_paths["ic_series"] = plot_ic_series(
            pred_df,
            date_col=date_col,
            score_col=score_col,
            target_col=target_col,
            save_dir=save_dir,
            show=show,
            model_name=model_name,
        )

    if log_cfg is not None:
        log_metrics: Dict[str, float] = dict(report.metrics)
        if report.long_short_metrics:
            for k, v in report.long_short_metrics.items():
                log_metrics[f"ls_{k}"] = float(v)
        report.log_paths = log_experiment_full(
            log_cfg,
            metrics=log_metrics,
            log_dir=log_dir,
            experiment_name=experiment_name,
            extra_info=extra_info,
        )
        if log_full_csv:
            csv_path = log_experiment_full_csv(
                log_cfg,
                metrics=log_metrics,
                log_dir=log_dir,
                experiment_name=experiment_name,
                extra_info=extra_info,
            )
            report.log_paths["csv"] = csv_path

    return report
