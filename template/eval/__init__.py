# pyright: reportMissingImports=false
"""
eval - 跨模型通用的评估模块

提供模型评估所需的各种功能，包括指标计算、策略回测、可视化和实验记录。
这些函数设计为纯函数，可用于任意模型（LSTM、Transformer、XGBoost 等）的评估。

主要模块:
- metrics: 评估指标函数（IC、RankIC、损失函数等）
- backtest: 策略回测（十组分层、多空组合等）
- plots: 训练曲线与回测结果可视化
- experiment_log: 实验结果落盘与对比

推荐用法:
    # 1. 评估指标
    from eval.metrics import rank_ic_by_date, pearson_ic_by_date
    ic = rank_ic_by_date(pred_df, date_col="date", score_col="pred", target_col="y")
    
    # 2. 策略回测
    from eval.backtest import backtest_decile_strategy
    results = backtest_decile_strategy(test_df, original_df, cfg)
    
    # 3. 可视化
    from eval.plots import plot_training_curve, plot_backtest_results
    plot_training_curve(train_losses, val_losses, cfg)
    plot_backtest_results(results, train_end, val_end)
    
    # 4. 实验记录
    from eval.experiment_log import log_experiment_results
    log_experiment_results(cfg, test_loss, test_mse, test_mae, test_ric, best_val_loss)
"""
# ============================================================
# metrics 模块
# ============================================================
from .metrics import (
    # 核心研究指标
    rank_ic_by_date,
    pearson_ic_by_date,
    # 基础损失函数
    mae_np,
    mse_np,
    r2_score_np,
    huber_loss_np,
    quantile_loss_np,
    # 统一损失计算
    compute_loss_np,
    # IC 工具函数
    pearson_ic_np,
    spearman_rankic_np,
    # IC 序列与汇总
    ic_series_by_date,
    ic_summary,
    # 配置协议
    LossConfig,
)

# ============================================================
# backtest 模块
# ============================================================
from .backtest import (
    # 核心回测函数
    backtest_decile_strategy,
    backtest_grouped_returns,
    # 回测指标计算
    calc_sharpe_ratio,
    calc_max_drawdown,
    calc_annual_return,
    compute_backtest_summary,
    select_long_short_deciles,
    compute_long_short_metrics,
    # 配置协议
    BacktestConfig,
)

# ============================================================
# plots 模块
# ============================================================
from .plots import (
    # 训练曲线
    plot_training_curve,
    plot_training_metrics,
    # 回测可视化
    plot_backtest_results,
    plot_long_short_curve,
    # IC 可视化
    plot_ic_series,
    # 配置协议
    PlotConfig,
)

# ============================================================
# factor_backtest 模块
# ============================================================
from .factor_backtest import (
    backtest_single_factor,
    backtest_all_factors,
    calc_factor_ic,
    calc_factor_summary,
    evaluate_factor,
    evaluate_all_factors,
)

# ============================================================
# runner 模块
# ============================================================
from .runner import (
    run_evaluation,
    EvalReport,
)

# ============================================================
# schema 模块
# ============================================================
from .schema import (
    require_columns,
    ensure_pred_df,
    ensure_score_return_df,
    ensure_backtest_results,
    PredColumns,
    ScoreReturnColumns,
    BacktestResultColumns,
)

# ============================================================
# experiment_log 模块
# ============================================================
from .experiment_log import (
    # 核心记录函数
    log_experiment_results,
    # 高级功能
    log_experiment_full,
    log_experiment_full_csv,
    load_experiment_history,
    compare_experiments,
    get_best_experiment,
    # 配置协议
    ExperimentConfig,
)
from .reporting import save_json_report, log_metrics_bundle

__all__ = [
    # === metrics ===
    "rank_ic_by_date",
    "pearson_ic_by_date",
    "mae_np",
    "mse_np",
    "r2_score_np",
    "huber_loss_np",
    "quantile_loss_np",
    "compute_loss_np",
    "pearson_ic_np",
    "spearman_rankic_np",
    "ic_series_by_date",
    "ic_summary",
    "LossConfig",
    # === backtest ===
    "backtest_decile_strategy",
    "backtest_grouped_returns",
    "calc_sharpe_ratio",
    "calc_max_drawdown",
    "calc_annual_return",
    "compute_backtest_summary",
    "select_long_short_deciles",
    "compute_long_short_metrics",
    "BacktestConfig",
    # === plots ===
    "plot_training_curve",
    "plot_training_metrics",
    "plot_backtest_results",
    "plot_long_short_curve",
    "plot_ic_series",
    "PlotConfig",
    # === factor_backtest ===
    "backtest_single_factor",
    "backtest_all_factors",
    "calc_factor_ic",
    "calc_factor_summary",
    "evaluate_factor",
    "evaluate_all_factors",
    # === experiment_log ===
    "log_experiment_results",
    "log_experiment_full",
    "log_experiment_full_csv",
    "load_experiment_history",
    "compare_experiments",
    "get_best_experiment",
    "ExperimentConfig",
    "save_json_report",
    "log_metrics_bundle",
    # === runner ===
    "run_evaluation",
    "EvalReport",
    # === schema ===
    "require_columns",
    "ensure_pred_df",
    "ensure_score_return_df",
    "ensure_backtest_results",
    "PredColumns",
    "ScoreReturnColumns",
    "BacktestResultColumns",
]
