"""
eval/backtest.py - 跨模型通用的策略回测模块

提供基于模型预测结果的策略回测功能，包括：
- 十分位组合回测（Decile Portfolio）
- 多空组合回测（Long-Short Portfolio）
- 回测指标计算（夏普比率、最大回撤等）

设计原则:
- 所有函数为纯函数，不依赖全局状态
- 可用于任意模型的回测，不限于 LSTM
- 支持灵活的配置协议
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable, Optional, Dict, List, Tuple
import numpy as np
import pandas as pd

from eval.schema import ensure_backtest_results, ensure_score_return_df

__all__ = [
    # 核心回测函数
    "backtest_decile_strategy",
    "backtest_grouped_returns",
    # 回测指标计算
    "calc_sharpe_ratio",
    "calc_max_drawdown",
    "calc_annual_return",
    "select_long_short_deciles",
    "compute_long_short_metrics",
    # 配置协议
    "BacktestConfig",
]


# ============================================================
# 配置协议定义
# ============================================================
@runtime_checkable
class BacktestConfig(Protocol):
    """回测模块所需的配置协议，任何具有这些属性的对象都可使用"""
    id_col: str
    date_col: str


# ============================================================
# 回测指标计算函数
# ============================================================
def calc_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 244
) -> float:
    """
    计算夏普比率
    
    Args:
        returns: 收益率序列
        risk_free_rate: 无风险利率（年化）
        periods_per_year: 每年交易日数量
    
    Returns:
        float: 年化夏普比率
    """
    if len(returns) < 2 or returns.std() == 0:
        return float("nan")
    
    excess_return = returns.mean() - risk_free_rate / periods_per_year
    return float(excess_return / returns.std() * np.sqrt(periods_per_year))


def calc_max_drawdown(cumulative_returns: pd.Series) -> float:
    """
    计算最大回撤
    
    Args:
        cumulative_returns: 累积收益率序列
    
    Returns:
        float: 最大回撤（正值，如 0.2 表示 20% 回撤）
    """
    if len(cumulative_returns) < 2:
        return float("nan")
    
    # 累积净值
    wealth = 1 + cumulative_returns
    # 历史最高点
    running_max = wealth.cummax()
    # 回撤序列
    drawdown = (running_max - wealth) / running_max
    return float(drawdown.max())


def calc_annual_return(
    cumulative_return: float,
    num_days: int,
    periods_per_year: int = 244
) -> float:
    """
    计算年化收益率
    
    Args:
        cumulative_return: 累积收益率
        num_days: 交易日数量
        periods_per_year: 每年交易日数量
    
    Returns:
        float: 年化收益率
    """
    if num_days <= 0:
        return float("nan")
    
    years = num_days / periods_per_year
    return float((1 + cumulative_return) ** (1 / years) - 1)


# ============================================================
# 核心回测函数
# ============================================================
def backtest_decile_strategy(
    pred_df: pd.DataFrame,
    original_df: pd.DataFrame,
    cfg: BacktestConfig,
    n_groups: int = 10,
    pred_col: str = "pred",
    index_col: str = "index_id",
    date_col: str = "date",
    target_period: int = 1
) -> pd.DataFrame:
    """
    基于预测结果的分组策略回测
    
    将股票按预测分数排序，等分为N组（默认10组），计算每组的收益率。
    第1组为预测分数最高的股票（多头组），第N组为预测分数最低的股票（空头组）。
    
    策略逻辑：
    - 预测日 T：模型输出预测分数
    - 买入日 T+1：以开盘价买入
    - 卖出日 T+target_period+1：以收盘价卖出
    
    Args:
        pred_df: 预测结果 DataFrame，需包含 date_col, index_col, pred_col, 'y' 列
        original_df: 原始价格数据，需包含 cfg.id_col, cfg.date_col, 'open', 'close' 列
        cfg: 配置对象，需符合 BacktestConfig 协议
        n_groups: 分组数量，默认 10（十分位）
        pred_col: 预测分数列名
        index_col: 股票/指数ID列名
        date_col: 日期列名
        target_period: 持仓周期（天数），默认 1
    
    Returns:
        pd.DataFrame: 回测结果，包含以下列：
            - date: 收益率对应的日期（卖出日）
            - decile: 分组编号（1=最高分组，N=最低分组）
            - return: 该组当日平均收益率
            - n_stocks: 该组当日股票数量
    
    Example:
        >>> from eval.backtest import backtest_decile_strategy
        >>> results = backtest_decile_strategy(test_df, df_panel, cfg)
        >>> # 查看多头组（第1组）的累积收益
        >>> top_group = results[results['decile'] == 1]
        >>> print(f"多头组累积收益: {top_group['return'].sum():.2%}")
    """
    # 复制数据，避免修改原始数据
    test_df = pred_df.copy()
    test_df = test_df.sort_values([index_col, date_col]).reset_index(drop=True)
    
    # 计算买入日和卖出日
    # 预测日T，买入日T+1（开盘），卖出日T+1+target_period（收盘）
    test_df['buy_date'] = test_df.groupby(index_col)[date_col].shift(-1)
    test_df['sell_date'] = test_df.groupby(index_col)[date_col].shift(-1 - target_period)
    
    # 构建价格查询表并合并（向量化替代逐行查询）
    price_data = original_df[[cfg.id_col, cfg.date_col, 'open', 'close']].copy()
    price_data[cfg.date_col] = pd.to_datetime(price_data[cfg.date_col])
    price_data = price_data.rename(columns={
        cfg.id_col: index_col,
        cfg.date_col: 'date'
    })

    buy_price = price_data.rename(columns={
        'date': 'buy_date',
        'open': 'buy_price'
    })[[index_col, 'buy_date', 'buy_price']]

    sell_price = price_data.rename(columns={
        'date': 'sell_date',
        'close': 'sell_price'
    })[[index_col, 'sell_date', 'sell_price']]

    test_df = test_df.merge(buy_price, on=[index_col, 'buy_date'], how='left')
    test_df = test_df.merge(sell_price, on=[index_col, 'sell_date'], how='left')
    
    # 计算收益率
    test_df['return'] = test_df['sell_price'] / test_df['buy_price'] - 1
    
    # 按预测日分组，计算每组收益
    decile_returns: List[Dict] = []
    
    for pred_date, group in test_df.groupby(date_col):
        # 过滤无效数据
        group = group.dropna(subset=['return', pred_col])
        if len(group) < n_groups:
            continue
        
        # 按预测分数降序排序
        group = group.sort_values(pred_col, ascending=False)
        
        # 使用排名等分为N组
        group['rank'] = range(len(group))
        try:
            group['decile'] = pd.qcut(
                group['rank'], 
                n_groups, 
                labels=range(1, n_groups + 1), 
                duplicates='drop'
            )
        except ValueError:
            # 如果无法等分（数据量不足），跳过
            continue
        
        # 计算每组的平均收益率
        for decile in range(1, n_groups + 1):
            decile_data = group[group['decile'] == decile]
            if len(decile_data) > 0:
                avg_return = decile_data['return'].mean()
                sell_date = decile_data['sell_date'].iloc[0]
                if pd.notna(sell_date):
                    decile_returns.append({
                        'date': sell_date,
                        'decile': decile,
                        'return': avg_return,
                        'n_stocks': len(decile_data)
                    })
    
    result_df = pd.DataFrame(decile_returns)
    
    # 排序并重置索引
    if len(result_df) > 0:
        result_df = result_df.sort_values(['date', 'decile']).reset_index(drop=True)
    
    return result_df


def backtest_grouped_returns(
    df: pd.DataFrame,
    score_col: str,
    return_col: str,
    date_col: str = "date",
    n_groups: int = 10,
    min_samples_per_group: int = 1,
    higher_is_better: bool = True,
) -> pd.DataFrame:
    """
    按 score 分组回测（适用于已有收益率列的场景）
    """
    ensure_score_return_df(df, date_col=date_col, score_col=score_col, return_col=return_col)

    decile_returns: List[Dict] = []
    min_total = n_groups * max(1, min_samples_per_group)

    for pred_date, group in df.groupby(date_col):
        group = group.dropna(subset=[score_col, return_col])
        if len(group) < min_total:
            continue

        group = group.sort_values(score_col, ascending=not higher_is_better)
        group = group.assign(rank=range(len(group))) #type:ignore
        try:
            group["decile"] = pd.qcut(
                group["rank"],
                n_groups,
                labels=range(1, n_groups + 1),
                duplicates="drop",
            )
        except ValueError:
            continue

        for decile in range(1, n_groups + 1):
            decile_data = group[group["decile"] == decile]
            if len(decile_data) == 0:
                continue
            avg_return = decile_data[return_col].mean()
            decile_returns.append({
                "date": pred_date,
                "decile": decile,
                "return": avg_return,
                "n_stocks": len(decile_data),
            })

    result_df = pd.DataFrame(decile_returns)
    if len(result_df) > 0:
        result_df = result_df.sort_values(["date", "decile"]).reset_index(drop=True)
    return result_df


def select_long_short_deciles(
    backtest_results: pd.DataFrame,
    long_decile: int = 1,
    short_decile: int = 10,
) -> tuple[int, int]:
    """
    根据累计收益决定真实的多头/空头分组
    """
    ensure_backtest_results(backtest_results)

    long_returns = backtest_results[backtest_results["decile"] == long_decile]["return"]
    short_returns = backtest_results[backtest_results["decile"] == short_decile]["return"]

    long_cum = long_returns.cumsum().iloc[-1] if len(long_returns) > 0 else float("-inf")
    short_cum = short_returns.cumsum().iloc[-1] if len(short_returns) > 0 else float("-inf")

    if long_cum >= short_cum:
        return long_decile, short_decile
    return short_decile, long_decile


def compute_long_short_metrics(
    backtest_results: pd.DataFrame,
    long_decile: int = 1,
    short_decile: int = 10,
) -> Dict[str, float]:
    """
    计算多空相关指标（累计收益、单调性等）
    """
    ensure_backtest_results(backtest_results)
    if len(backtest_results) == 0:
        return {
            "long_return": float("nan"),
            "short_return": float("nan"),
            "long_short_return": float("nan"),
            "long_avg_return": float("nan"),
            "monotonicity": float("nan"),
            "actual_long_decile": float("nan"),
            "actual_short_decile": float("nan"),
        }

    decile_returns = backtest_results.groupby("decile")["return"].sum()
    actual_long, actual_short = select_long_short_deciles(
        backtest_results,
        long_decile=long_decile,
        short_decile=short_decile,
    )

    long_return = float(decile_returns.get(actual_long, np.nan))
    short_return = float(decile_returns.get(actual_short, np.nan))
    long_short_return = (
        long_return - short_return
        if not np.isnan(long_return) and not np.isnan(short_return)
        else float("nan")
    )

    other_returns = decile_returns[decile_returns.index != actual_long]
    other_avg_return = float(other_returns.mean()) if len(other_returns) > 0 else float("nan")
    long_avg_return = (
        long_return - other_avg_return
        if not np.isnan(long_return) and not np.isnan(other_avg_return)
        else float("nan")
    )

    monotonicity = float("nan")
    if len(decile_returns) > 1:
        monotonicity = decile_returns.sort_index().reset_index(drop=True).corr(
            pd.Series(range(1, len(decile_returns) + 1)),
            method="spearman",
        )

    return {
        "long_return": long_return,
        "short_return": short_return,
        "long_short_return": long_short_return,
        "long_avg_return": long_avg_return,
        "monotonicity": monotonicity,
        "actual_long_decile": float(actual_long),
        "actual_short_decile": float(actual_short),
    }


def compute_backtest_summary(
    backtest_results: pd.DataFrame,
    risk_free_rate: float = 0.0
) -> pd.DataFrame:
    """
    计算回测结果的汇总统计
    
    Args:
        backtest_results: backtest_decile_strategy 的输出
        risk_free_rate: 无风险利率（年化）
    
    Returns:
        pd.DataFrame: 每个分组的汇总统计，包含：
            - decile: 分组编号
            - total_return: 累积收益率
            - annual_return: 年化收益率
            - sharpe_ratio: 夏普比率
            - max_drawdown: 最大回撤
            - num_trades: 交易次数
    """
    if len(backtest_results) == 0:
        return pd.DataFrame()
    
    summaries: List[Dict] = []
    
    for decile in sorted(backtest_results['decile'].unique()):
        decile_data = backtest_results[backtest_results['decile'] == decile].copy()
        decile_data = decile_data.sort_values('date')
        
        returns = decile_data['return']
        cumulative = returns.cumsum()
        
        total_return = float(cumulative.iloc[-1]) if len(cumulative) > 0 else 0.0
        num_days = len(decile_data)
        
        summaries.append({
            'decile': decile,
            'total_return': total_return,
            'annual_return': calc_annual_return(total_return, num_days),
            'sharpe_ratio': calc_sharpe_ratio(returns, risk_free_rate),
            'max_drawdown': calc_max_drawdown(cumulative),
            'num_trades': num_days,
            'avg_daily_return': float(returns.mean()),
            'volatility': float(returns.std()),
        })
    
    return pd.DataFrame(summaries)
