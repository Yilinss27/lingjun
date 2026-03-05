"""
eval/plots.py - 跨模型通用的可视化模块

提供训练过程和回测结果的可视化功能，包括：
- 训练曲线（损失函数、评估指标）
- 回测结果可视化（分组收益曲线、多空净值曲线）
- IC/RankIC 时序图

设计原则:
- 所有函数为纯函数，不依赖全局状态
- 可用于任意模型的可视化，不限于 LSTM
- 支持灵活的配置协议
- 默认保存图片到 res/ 目录
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable, Optional, List, Dict, Union, cast
import time
import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from eval.backtest import select_long_short_deciles
from eval.metrics import ic_series_by_date

__all__ = [
    # 训练曲线
    "plot_training_curve",
    "plot_training_metrics",
    # 回测可视化
    "plot_backtest_results",
    "plot_long_short_curve",
    # IC 可视化
    "plot_ic_series",
    # 配置协议
    "PlotConfig",
]


# ============================================================
# 配置协议定义
# ============================================================
@runtime_checkable
class PlotConfig(Protocol):
    """可视化模块所需的配置协议，任何具有这些属性的对象都可使用"""
    loss_name: str


# ============================================================
# 训练曲线可视化
# ============================================================
def plot_training_curve(
    train_losses: List[float],
    val_losses: List[float],
    cfg: PlotConfig,
    save_dir: str = "res",
    show: bool = False,
    figsize: tuple = (10, 6),
    dpi: int = 150,
    model_name: str = ""
) -> str:
    """
    绘制训练曲线（训练损失 vs 验证损失）
    
    Args:
        train_losses: 每个 epoch 的训练损失列表
        val_losses: 每个 epoch 的验证损失列表
        cfg: 配置对象，需符合 PlotConfig 协议（主要用于获取 loss_name）
        save_dir: 图片保存目录
        show: 是否显示图片
        figsize: 图片尺寸
    
    Returns:
        str: 保存的图片路径
    
    Example:
        >>> from eval.plots import plot_training_curve
        >>> plot_training_curve(train_losses, val_losses, cfg)
    """
    epochs = range(1, len(train_losses) + 1)
    loss_name = cfg.loss_name.upper()
    
    plt.figure(figsize=figsize)
    
    # 绘制训练和验证损失
    plt.plot(
        epochs, train_losses, 'b-', 
        label=f'Training {loss_name}', 
        linewidth=2, marker='o', markersize=4
    )
    plt.plot(
        epochs, val_losses, 'r-', 
        label=f'Validation {loss_name}', 
        linewidth=2, marker='s', markersize=4
    )
    
    # 标注最佳验证点
    best_epoch = int(np.argmin(val_losses)) + 1
    best_val = min(val_losses)
    plt.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.5, label=f'Best Epoch ({best_epoch})')
    plt.scatter([best_epoch], [best_val], color='green', s=100, zorder=5, marker='*')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(f'{loss_name} Loss', fontsize=12)
    plt.title(f'Training Curve: {loss_name} Loss (Train vs Val)', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图片
    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    prefix = f"{model_name}_" if model_name else ""
    filename = f"{save_dir}/{prefix}training_curve_{timestamp}.png"
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    print(f"训练曲线已保存到: {filename}")
    return filename


def plot_training_metrics(
    metrics_history: Dict[str, List[float]],
    cfg: PlotConfig,
    save_dir: str = "res",
    show: bool = False,
    figsize: tuple = (15, 10),
    dpi: int = 150,
    model_name: str = ""
) -> str:
    """
    绘制多个训练指标的曲线（支持多子图）
    
    Args:
        metrics_history: 指标历史字典，格式如 {'train_loss': [...], 'val_loss': [...], 'val_ic': [...]}
        cfg: 配置对象
        save_dir: 图片保存目录
        show: 是否显示图片
        figsize: 图片尺寸
    
    Returns:
        str: 保存的图片路径
    """
    # 按指标类型分组
    loss_metrics = {k: v for k, v in metrics_history.items() if 'loss' in k.lower()}
    ic_metrics = {k: v for k, v in metrics_history.items() if 'ic' in k.lower()}
    other_metrics = {k: v for k, v in metrics_history.items() 
                     if 'loss' not in k.lower() and 'ic' not in k.lower()}
    
    n_plots = sum([bool(loss_metrics), bool(ic_metrics), bool(other_metrics)])
    if n_plots == 0:
        print("警告: 没有可绘制的指标")
        return ""
    
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # 绘制损失曲线
    if loss_metrics:
        ax = axes[plot_idx]
        for name, values in loss_metrics.items():
            epochs = range(1, len(values) + 1)
            ax.plot(epochs, values, label=name, linewidth=2, marker='o', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'Training & Validation Loss ({cfg.loss_name})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # 绘制 IC 曲线
    if ic_metrics:
        ax = axes[plot_idx]
        for name, values in ic_metrics.items():
            epochs = range(1, len(values) + 1)
            ax.plot(epochs, values, label=name, linewidth=2, marker='s', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('IC')
        ax.set_title('IC / RankIC Metrics')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        plot_idx += 1
    
    # 绘制其他指标
    if other_metrics:
        ax = axes[plot_idx]
        for name, values in other_metrics.items():
            epochs = range(1, len(values) + 1)
            ax.plot(epochs, values, label=name, linewidth=2, marker='^', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.set_title('Other Metrics')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    prefix = f"{model_name}_" if model_name else ""
    filename = f"{save_dir}/{prefix}training_metrics_{timestamp}.png"
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    print(f"训练指标曲线已保存到: {filename}")
    return filename


# ============================================================
# 回测结果可视化
# ============================================================
def plot_backtest_results(
    backtest_results: pd.DataFrame,
    train_end: Optional[pd.Timestamp] = None,
    val_end: Optional[pd.Timestamp] = None,
    type: str = "total",
    save_dir: str = "res",
    show: bool = False,
    figsize: tuple = (15, 10),
    start_date: Optional[str] = None,
    dpi: int = 150,
    model_name: str = ""
) -> str:
    """
    绘制回测结果图（分组累积收益曲线）
    
    Args:
        backtest_results: backtest_decile_strategy 的输出 DataFrame
        train_end: 训练集结束日期（用于绘制分界线）
        val_end: 验证集结束日期（用于绘制分界线）
        type: 回测类型标签，用于文件命名（如 'total', 'test', 'val'）
        save_dir: 图片保存目录
        show: 是否显示图片
        figsize: 图片尺寸
        start_date: 起始日期筛选（格式如 '2023-01-01'）
        dpi: 图片分辨率（默认 150，降低可加速保存）
    
    Returns:
        str: 保存的图片路径
    
    Example:
        >>> from eval.plots import plot_backtest_results
        >>> plot_backtest_results(backtest_results, train_end, val_end, type="total")
    """
    if len(backtest_results) == 0:
        print("警告: 回测结果为空，无法绘图")
        return ""
    
    # 数据预处理（避免不必要的 copy）
    results = backtest_results
    if not pd.api.types.is_datetime64_any_dtype(results["date"]):
        results = backtest_results.copy()
        results["date"] = pd.to_datetime(results["date"])
    
    # 日期筛选
    if start_date is not None:
        results = results[results["date"] >= pd.Timestamp(start_date)]
    
    if len(results) == 0:
        print("警告: 筛选后回测结果为空，无法绘图")
        return ""
    
    plt.figure(figsize=figsize)
    
    # 获取分组数量
    deciles = sorted(results['decile'].unique())
    n_deciles = len(deciles)
    
    # 使用颜色映射
    colors = plt.cm.RdYlGn_r(np.linspace(0, 1, n_deciles))  # type: ignore
    
    # 预排序并使用 groupby + cumsum 向量化计算（比逐组循环快 2-3 倍）
    results_sorted = results.sort_values(['decile', 'date'])
    results_sorted['cumulative_return'] = results_sorted.groupby('decile')['return'].cumsum()
    
    # 绘制每组的累积收益曲线
    decile_to_color = {d: colors[i] for i, d in enumerate(deciles)}
    for decile, decile_data in results_sorted.groupby('decile'):
        label = f'Decile {decile}' if n_deciles <= 10 else f'Group {decile}'
        if decile == 1:
            label += ' (Top)'
        elif decile == n_deciles:
            label += ' (Bottom)'
        
        plt.plot(
            decile_data['date'], 
            decile_data['cumulative_return'],
            label=label,
            linewidth=2,
            color=decile_to_color[decile]
        )
    
    # 添加训练集和验证集分界线
    if train_end is not None:
        plt.axvline(x=train_end, color='red', linestyle='--', linewidth=2, alpha=0.8)# type: ignore
        ymin, ymax = plt.ylim()
        plt.text(
            train_end, ymax * 0.95, 'Train End', # type: ignore
            rotation=90, verticalalignment='top', 
            fontsize=10, color='red', fontweight='bold'
        )
    
    if val_end is not None:
        plt.axvline(x=val_end, color='orange', linestyle='--', linewidth=2, alpha=0.8)# type: ignore
        ymin, ymax = plt.ylim()
        plt.text(
            val_end, ymax * 0.95, 'Val End', # type: ignore
            rotation=90, verticalalignment='top', 
            fontsize=10, color='orange', fontweight='bold'
        )
    
    # 添加零线
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    
    plt.title(f'Decile Portfolio Cumulative Returns ({type.upper()})', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # 保存图片
    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    prefix = f"{model_name}_" if model_name else ""
    filename = f"{save_dir}/{prefix}backtest_decile_returns_{type}_{timestamp}.png"
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    print(f"回测结果图已保存到: {filename}")
    return filename

def _select_longshort(
    results: pd.DataFrame,
    long_decile: int,
    short_decile: int
) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    """
    根据累计收益决定谁是真正的 Long，谁是 short（用于 short 或用于对照组）。
    返回：long_data, short_data, real_long_decile, real_short_decile

    """
    real_long_decile, real_short_decile = select_long_short_deciles(
        results,
        long_decile=long_decile,
        short_decile=short_decile,
    )

    long = results[results["decile"] == real_long_decile][["date", "return"]].copy()
    short = results[results["decile"] == real_short_decile][["date", "return"]].copy()

    return long, short, real_long_decile, real_short_decile

def _prepare_long_short_data(results: pd.DataFrame, long_decile: int, short_decile: int) -> Optional[pd.DataFrame]:
    """准备多头-空头模式的数据。如果long_decile累计收益小于short_decile，则自动调整多空方向。"""
    long_raw, short_raw, real_long_decile, real_short_decile = _select_longshort(
        results, long_decile, short_decile
    )

    long_data = long_raw.rename(columns={"return": "long_return"})
    short_data = short_raw.rename(columns={"return": "benchmark_return"})

    merged = pd.merge(long_data, short_data, on="date", how="inner")
    if len(merged) == 0:
        return None

    merged = merged.sort_values("date")
    merged["excess_return"] = merged["long_return"] - merged["benchmark_return"]

    # 关键：把真实 Long/Short 都存起来，后面画图用
    merged.attrs = {
        "long_decile": real_long_decile,
        "benchmark_decile": real_short_decile,
    }
    return merged


def _prepare_long_average_data(results: pd.DataFrame, long_decile: int) -> Optional[pd.DataFrame]:
    """
    准备多头-其余组平均模式的数据。
    首先筛选出真正的long_data，判断标准与_prepare_long_short_data一致（即哪一组累计收益更高就作为long），
    然后再去计算其余组的均值
    """

    short_decile =  10

    # ✅ 只改这里：用统一函数选择真正 long
    long_data, short_data, real_long_decile, real_short_decile = _select_longshort(
        results, long_decile, short_decile
    )

    # 其余组平均（除了 real_long_decile）
    other_data = results[results["decile"] != real_long_decile].groupby("date")["return"].mean().reset_index()

    long_data_ren = long_data.rename(columns={"return": "long_return"})
    other_data_ren = other_data.rename(columns={"return": "benchmark_return"})

    merged = pd.merge(long_data_ren, other_data_ren, on="date", how="inner")
    if len(merged) == 0:
        return None

    merged = merged.sort_values("date")
    merged["excess_return"] = merged["long_return"] - merged["benchmark_return"]
    merged.attrs = {
        "long_decile": real_long_decile,
        "average_as_long": True,
    }
    return merged


def plot_long_short_curve(
    backtest_results: pd.DataFrame,
    long_decile: int = 1,
    short_decile: int = 10,
    type: str = "longshort",
    save_dir: str = "res",
    show: bool = False,
    start_date: Optional[str] = None,
    figsize: tuple = (12, 6),
    dpi: int = 150,
    model_name: str = ""
) -> str:
    """
    绘制多空组合净值曲线
    
    Args:
        backtest_results: backtest_decile_strategy 的输出 DataFrame
        long_decile: 多头组编号（默认 1，最高分组）
        short_decile: 空头组编号（默认 10，最低分组），仅 type="longshort" 时使用
        type: 绘图模式
            - "longshort": 多头 vs 空头 vs 多空差值
            - "longaverage": 多头 vs 其余组平均 vs 超额收益
        save_dir: 图片保存目录
        show: 是否显示图片
        figsize: 图片尺寸
        dpi: 图片分辨率
        model_name: 模型名称（用于文件前缀）
    
    Returns:
        str: 保存的图片路径
    """
    if len(backtest_results) == 0:
        print("警告: 回测结果为空，无法绘图")
        return ""
    
    # 数据预处理
    results = backtest_results
    if not pd.api.types.is_datetime64_any_dtype(results["date"]):
        results = backtest_results.copy()
        results["date"] = pd.to_datetime(results["date"])
        # 日期筛选
    if start_date is not None:
        results = results[results["date"] >= pd.Timestamp(start_date)]
    
    if len(results) == 0:
        print("警告: 筛选后回测结果为空，无法绘图")
        return ""

    # === 数据准备 ===
    if type == "longaverage":
        merged = _prepare_long_average_data(results, long_decile)
        title = "Long vs Average Portfolio Cumulative Returns"
        excess_label = "Long - Average"
    else:
        merged = _prepare_long_short_data(results, long_decile, short_decile)
        title = "Long-Short Portfolio Cumulative Returns"
        excess_label = "Long - Short"

    if merged is None:
        print("警告: 无法合并数据")
        return ""

    # === 累计收益 ===
    merged["long_cumsum"] = merged["long_return"].cumsum()
    merged["benchmark_cumsum"] = merged["benchmark_return"].cumsum()
    merged["excess_cumsum"] = merged["excess_return"].cumsum()

    # === 从 attrs 读取真实方向（核心修复点）===
    real_long_decile = merged.attrs.get("long_decile", long_decile)

    if type == "longaverage":
        benchmark_label_plot = "Average (Others)"
    else:
        real_benchmark_decile = merged.attrs.get("benchmark_decile", short_decile)
        benchmark_label_plot = f"Short (Decile {real_benchmark_decile})"

    # === 画图 ===
    plt.figure(figsize=figsize)

    plt.plot(
        merged["date"], merged["long_cumsum"],
        label=f"Long (Decile {real_long_decile})",
        linewidth=2, color="green"
    )

    plt.plot(
        merged["date"], merged["benchmark_cumsum"],
        label=benchmark_label_plot,
        linewidth=2, color="red"
    )

    plt.plot(
        merged["date"], merged["excess_cumsum"],
        label=excess_label,
        linewidth=2, color="blue", linestyle="--"
    )

    plt.axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.5)

    plt.title(title, fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Cumulative Return", fontsize=12)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # === 保存 ===
    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    prefix = f"{model_name}_" if model_name else ""
    filename = f"{save_dir}/{prefix}long_short_curve_{type}_{timestamp}.png"

    plt.savefig(filename, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    print(f"多空曲线图已保存到: {filename}")
    return filename


# ============================================================
# IC 可视化
# ============================================================
def plot_ic_series(
    pred_df: pd.DataFrame,
    date_col: str = "date",
    score_col: str = "pred",
    target_col: str = "y",
    save_dir: str = "res",
    show: bool = False,
    figsize: tuple = (14, 8),
    dpi: int = 150,
    model_name: str = ""
) -> str:
    """
    绘制 IC/RankIC 时序图
    
    Args:
        pred_df: 预测结果 DataFrame，需包含 date_col, score_col, target_col 列
        date_col: 日期列名
        score_col: 预测分数列名
        target_col: 真实收益列名
        save_dir: 图片保存目录
        show: 是否显示图片
        figsize: 图片尺寸
        dpi: 图片分辨率（默认 150，降低可加速保存）
    
    Returns:
        str: 保存的图片路径
    """
    metrics_df = ic_series_by_date(
        pred_df,
        date_col=date_col,
        score_col=score_col,
        target_col=target_col,
        min_samples=10,
    )
    
    # 过滤无效值
    metrics_df = metrics_df.dropna(subset=['ic', 'rank_ic'], how='all')
    
    if len(metrics_df) == 0:
        print("警告: 无法计算日度 IC")
        return ""
    
    metrics_df[date_col] = pd.to_datetime(metrics_df[date_col])
    metrics_df = metrics_df.sort_values(date_col).rename(columns={date_col: 'date'})
    
    # 计算累计 IC
    metrics_df['ic_cumsum'] = metrics_df['ic'].cumsum()
    metrics_df['rank_ic_cumsum'] = metrics_df['rank_ic'].cumsum()
    
    # 计算 ICIR（IC均值 / IC标准差）
    ic_mean = metrics_df['ic'].mean()
    ic_std = metrics_df['ic'].std()
    ic_ir = ic_mean / ic_std if ic_std > 0 else np.nan
    
    rank_ic_mean = metrics_df['rank_ic'].mean()
    rank_ic_std = metrics_df['rank_ic'].std()
    rank_ic_ir = rank_ic_mean / rank_ic_std if rank_ic_std > 0 else np.nan
    
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    # Pearson IC: 柱状图 + 累计曲线（双Y轴）
    ax1 = axes[0]
    ax1.bar(metrics_df['date'], metrics_df['ic'], alpha=0.3, color='blue', label='Daily IC')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_ylabel('Daily Pearson IC', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_title(f'Pearson IC (Mean={ic_mean:.4f}, ICIR={ic_ir:.2f})')
    ax1.grid(True, alpha=0.3)
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(metrics_df['date'], metrics_df['ic_cumsum'], 
                  color='red', linewidth=2, label='Cumulative IC')
    ax1_twin.set_ylabel('Cumulative IC', color='red')
    ax1_twin.tick_params(axis='y', labelcolor='red')
    
    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Spearman RankIC: 柱状图 + 累计曲线（双Y轴）
    ax2 = axes[1]
    ax2.bar(metrics_df['date'], metrics_df['rank_ic'], alpha=0.3, color='green', label='Daily RankIC')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Daily Spearman RankIC', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.set_title(f'Spearman RankIC (Mean={rank_ic_mean:.4f}, ICIR={rank_ic_ir:.2f})')
    ax2.grid(True, alpha=0.3)
    
    ax2_twin = ax2.twinx()
    ax2_twin.plot(metrics_df['date'], metrics_df['rank_ic_cumsum'], 
                  color='red', linewidth=2, label='Cumulative RankIC')
    ax2_twin.set_ylabel('Cumulative RankIC', color='red')
    ax2_twin.tick_params(axis='y', labelcolor='red')
    
    # 合并图例
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    
    # 保存图片
    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    prefix = f"{model_name}_" if model_name else ""
    filename = f"{save_dir}/{prefix}ic_series_{timestamp}.png"
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    print(f"IC 时序图已保存到: {filename}")
    return filename
