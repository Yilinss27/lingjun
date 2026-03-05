"""
eval/factor_backtest.py - 因子截面回测与评估工具
"""
from __future__ import annotations

from typing import Dict, List, Optional
import os
import pandas as pd

from eval.backtest import backtest_grouped_returns, compute_long_short_metrics
from eval.metrics import ic_series_by_date, ic_summary
from eval.plots import plot_backtest_results, plot_long_short_curve, plot_ic_series


def backtest_single_factor(
    df: pd.DataFrame,
    factor_col: str,
    return_col: str = "target",
    date_col: str = "date",
    n_groups: int = 10,
    min_samples_per_group: int = 2,
) -> pd.DataFrame:
    """
    单因子截面分层回测
    """
    return backtest_grouped_returns(
        df,
        score_col=factor_col,
        return_col=return_col,
        date_col=date_col,
        n_groups=n_groups,
        min_samples_per_group=min_samples_per_group,
        higher_is_better=True,
    )


def backtest_all_factors(
    df: pd.DataFrame,
    factor_cols: List[str],
    return_col: str = "target",
    date_col: str = "date",
    n_groups: int = 10,
    save_dir: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    批量回测所有因子
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    results: Dict[str, pd.DataFrame] = {}

    total = len(factor_cols)
    for i, factor in enumerate(factor_cols):
        if total > 0:
            print(f"\r回测进度: {i+1}/{total} - {factor[:30]:<30}", end="")
        bt_result = backtest_single_factor(
            df,
            factor_col=factor,
            return_col=return_col,
            date_col=date_col,
            n_groups=n_groups,
        )
        if len(bt_result) == 0:
            continue
        results[factor] = bt_result
        if save_dir:
            bt_result.to_excel(f"{save_dir}/{factor}_backtest.xlsx", index=False)

    if total > 0:
        print(f"\n回测完成: {len(results)}/{total} 个因子有效")
    return results


def calc_factor_ic(
    df: pd.DataFrame,
    factor_col: str,
    return_col: str = "target",
    date_col: str = "date",
    min_samples: int = 10,
) -> pd.DataFrame:
    """
    计算因子的每日 IC 和 RankIC
    """
    return ic_series_by_date(
        df,
        date_col=date_col,
        score_col=factor_col,
        target_col=return_col,
        min_samples=min_samples,
    )


def calc_factor_summary(
    backtest_result: pd.DataFrame,
    ic_df: pd.DataFrame,
    long_decile: int = 1,
    short_decile: int = 10,
) -> Dict[str, float]:
    """
    计算因子回测汇总指标
    """
    ic_stats = ic_summary(ic_df)
    ls_stats = compute_long_short_metrics(
        backtest_result,
        long_decile=long_decile,
        short_decile=short_decile,
    )

    return {
        "IC均值": ic_stats["ic_mean"],
        "IC标准差": ic_stats["ic_std"],
        "ICIR": ic_stats["ic_ir"],
        "RankIC均值": ic_stats["rank_ic_mean"],
        "RankIC标准差": ic_stats["rank_ic_std"],
        "RankICIR": ic_stats["rank_ic_ir"],
        "多头累计收益": ls_stats["long_return"],
        "空头累计收益": ls_stats["short_return"],
        "多空累计收益": ls_stats["long_short_return"],
        "多均累计收益": ls_stats["long_avg_return"],
        "单调性": ls_stats["monotonicity"],
        "实际做多组": ls_stats["actual_long_decile"],
        "实际做空组": ls_stats["actual_short_decile"],
    }


def evaluate_factor(
    df: pd.DataFrame,
    factor_col: str,
    backtest_result: pd.DataFrame,
    save_dir: str,
    return_col: str = "target",
    date_col: str = "date",
    plot_start_date: Optional[str] = None,
) -> Dict[str, float]:
    """
    评估单个因子并绘制图表
    """
    os.makedirs(save_dir, exist_ok=True)
    ic_df = calc_factor_ic(df, factor_col, return_col, date_col)
    summary = calc_factor_summary(backtest_result, ic_df)
    summary["因子"] = factor_col

    if len(backtest_result) > 0:
        plot_backtest_results(
            backtest_result,
            type="total",
            save_dir=save_dir,
            start_date=plot_start_date,
            show=False,
            model_name=factor_col[:20],
        )
        plot_long_short_curve(
            backtest_result,
            type="longaverage",
            save_dir=save_dir,
            start_date=plot_start_date,
            show=False,
            model_name=factor_col[:20],
        )
        plot_long_short_curve(
            backtest_result,
            type="longshort",
            save_dir=save_dir,
            start_date=plot_start_date,
            show=False,
            model_name=factor_col[:20],
        )

    if len(ic_df) > 0:
        plot_ic_series(
            df,
            date_col=date_col,
            score_col=factor_col,
            target_col=return_col,
            save_dir=save_dir,
            show=False,
            model_name=factor_col[:20],
        )

    return summary


def evaluate_all_factors(
    df: pd.DataFrame,
    factor_cols: List[str],
    backtest_results: Dict[str, pd.DataFrame],
    save_dir: Optional[str] = None,
    return_col: str = "target",
    date_col: str = "date",
    top_n: Optional[int] = None,
    plot_start_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    评估所有因子并生成汇总报告
    """
    save_dir = save_dir or "res/factor_backtest"
    os.makedirs(save_dir, exist_ok=True)
    summaries = []
    factors_to_eval = factor_cols[:top_n] if top_n else factor_cols

    total = len(factors_to_eval)
    for i, factor in enumerate(factors_to_eval):
        if total > 0:
            print(f"\r评估进度: {i+1}/{total} - {factor[:30]:<30}", end="")
        if factor not in backtest_results:
            continue
        summary = evaluate_factor(
            df,
            factor,
            backtest_results[factor],
            save_dir,
            return_col,
            date_col,
            plot_start_date,
        )
        summaries.append(summary)

    if total > 0:
        print(f"\n评估完成: {len(summaries)} 个因子")

    summary_df = pd.DataFrame(summaries)
    if "ICIR" in summary_df.columns:
        summary_df = summary_df.sort_values("ICIR", ascending=False, key=abs)

    cols = ["因子"] + [c for c in summary_df.columns if c != "因子"]
    summary_df = summary_df[cols].reset_index(drop=True)

    summary_df.to_excel(f"{save_dir}/factor_summary.xlsx", index=False, engine="openpyxl")
    print(f"汇总报告已保存: {save_dir}/factor_summary.xlsx")
    return summary_df
