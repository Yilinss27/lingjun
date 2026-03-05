# eval

跨模型通用评估模块，包含指标计算、回测、可视化、实验记录、因子回测与一站式评估管线。

## metrics
- `rank_ic_by_date`: 按日期分组计算 RankIC 均值
- `pearson_ic_by_date`: 按日期分组计算 Pearson IC 均值
- `mae_np` / `mse_np` / `huber_loss_np` / `quantile_loss_np`: numpy 评估损失
- `compute_loss_np`: 按配置统一计算损失
- `pearson_ic_np` / `spearman_rankic_np`: 全局 IC/RankIC
- `ic_series_by_date`: 生成日度 IC/RankIC 序列
- `ic_summary`: 汇总 IC/RankIC 统计
- `LossConfig`: 损失配置协议

## backtest
- `backtest_decile_strategy`: 基于预测分数的分层回测
- `backtest_grouped_returns`: 基于已有收益列的分层回测
- `calc_sharpe_ratio` / `calc_max_drawdown` / `calc_annual_return`: 回测指标
- `compute_backtest_summary`: 分层回测汇总表
- `select_long_short_deciles`: 判定真实多空分组
- `compute_long_short_metrics`: 多空相关指标汇总
- `BacktestConfig`: 回测配置协议

## plots
- `plot_training_curve`: 训练/验证损失曲线
- `plot_training_metrics`: 多指标训练曲线
- `plot_backtest_results`: 分组累计收益曲线
- `plot_long_short_curve`: 多空/多均净值曲线
- `plot_ic_series`: IC/RankIC 时序图
- `PlotConfig`: 可视化配置协议

## experiment_log
- `log_experiment_results`: 文本日志记录
- `log_experiment_full`: JSON/TXT/CSV 全量记录
- `log_experiment_full_csv`: 全量 CSV 记录
- `load_experiment_history`: 读取实验历史
- `compare_experiments`: 实验对比
- `get_best_experiment`: 按指标选最佳实验
- `ExperimentConfig`: 实验配置协议

## factor_backtest
- `backtest_single_factor` / `backtest_all_factors`: 因子分层回测
- `calc_factor_ic` / `calc_factor_summary`: 因子 IC 与汇总指标
- `evaluate_factor` / `evaluate_all_factors`: 评估与出图

## runner
- `run_evaluation`: 一站式评估管线
- `EvalReport`: 统一输出结构

## schema
- `require_columns`: 列缺失校验
- `ensure_pred_df` / `ensure_score_return_df` / `ensure_backtest_results`: 常用表结构校验
- `PredColumns` / `ScoreReturnColumns` / `BacktestResultColumns`: 列名协议
