# pyright: reportGeneralTypeIssues=false, reportAttributeAccessIssue=false, reportMissingImports=false, reportArgumentType=false, reportConstantRedefinition=false, reportMissingTypeArgument=false, reportReturnType=false, reportFunctionMemberAccess=false
"""
LightGBM 模型训练入口

使用模块化架构：
- data_pipeline: 数据预处理（复用 LSTM 的因子工程和标准化）
- eval: 评估、回测、可视化、实验记录

支持两种模式：
- Ranker: 按日分组的排序学习（推荐，更贴合 RankIC 评估）
- Regressor: 直接回归未来收益
"""

from __future__ import annotations
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Literal

import numpy as np
import pandas as pd
import lightgbm as lgb
import itertools
from dataclasses import replace

from utils import check_data, init_reproducibility, get_lgbm_seed_params, get_seed, is_reproducible
from eval import (
    # 指标
    rank_ic_by_date,
    pearson_ic_by_date,
    # 回测
    backtest_decile_strategy,
    compute_backtest_summary,
    # 可视化
    plot_backtest_results,
    plot_long_short_curve,
    plot_ic_series,
    # 实验记录
    log_experiment_full,
    log_experiment_full_csv,
)

_RUN_LOG_FILE = None

__all__ = [
    "prepare_lgbm_data",
    "split_data",
    "print_sanity_checks",
    "as_float",
]


def setup_run_log(log_dir: str = "log", prefix: str = "run_lgbm") -> str:
    """Redirect stdout/stderr to both console and a log file."""
    global _RUN_LOG_FILE
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{prefix}_{timestamp}.log")

    _RUN_LOG_FILE = open(log_path, "a", buffering=1, encoding="utf-8")

    class TeeStream:
        def __init__(self, stream, log_file):
            self._stream = stream
            self._log_file = log_file

        def write(self, data):
            if data:
                self._stream.write(data)
                self._log_file.write(data)

        def flush(self):
            self._stream.flush()
            self._log_file.flush()

        def isatty(self):
            return False

    sys.stdout = TeeStream(sys.stdout, _RUN_LOG_FILE)
    sys.stderr = TeeStream(sys.stderr, _RUN_LOG_FILE)
    print(f"[Log] Streaming console output to: {log_path}")
    return log_path


# ============================================================
# LightGBM 配置
# ============================================================
@dataclass
class LGBMConfig:
    """LightGBM 专用配置"""
    # 数据配置（与 Config 保持一致）
    date_col: str = "date"
    id_col: str = "thscode"
    target_col: str = "target"
    
    # 因子管道（复用 LSTM 的因子加载与目标列；若都非 None 则走因子管道）
    json_path: Optional[str] = None
    factor_path: Optional[str] = None
    daily_data_path: str = r"C:\Users\Administrator\Desktop\dixi_thsgn\data\ths_concept_daily.f"
    target_period_start: int = 1
    target_period_end: int = 2
    
    # 特征配置
    feature_cols: List[str] = field(default_factory=list)
    
    # 数据分割
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    start_date: datetime = field(default_factory=lambda: datetime(2020, 1, 1))
    end_date: datetime = field(default_factory=lambda: datetime(2026, 1, 9))
    
    # 模型配置
    mode: Literal["ranker", "regressor"] = "ranker"  # ranker 更贴合排序任务
    
    # LightGBM 超参数
    num_leaves: int = 31
    max_depth: int = -1
    learning_rate: float = 0.05
    n_estimators: int = 500
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1

    # 设备配置
    device_type: Literal["cpu", "gpu"] = "cpu"
    
    # 早停配置
    early_stopping_rounds: int = 50
    # 前 warmup 轮不触发早停（warmup 之后才开始计数/判断）
    early_stopping_warmup_rounds: int = 0
    
    # 输出配置
    verbose: int = 50
    save_pred: bool = False
    
    # 兼容 eval 模块的属性（ExperimentConfig 协议）
    loss_name: str = "lgbm"
    lr: float = 0.05  # 映射 learning_rate
    epochs: int = 500  # 映射 n_estimators
    batch_size: int = 0  # 树模型无 batch 概念
    lookback: int = 1  # 树模型无 lookback 概念
    
    def __post_init__(self):
        """同步 lr 和 learning_rate"""
        self.lr = self.learning_rate
        self.epochs = self.n_estimators

    def model(self) -> str:
        return "lgbm"

    def run_tag(self) -> str:
        lr_tag = str(self.learning_rate).replace(".", "p")
        depth_tag = "full" if self.max_depth == -1 else str(self.max_depth)
        return f"{self.mode}_nl{self.num_leaves}_md{depth_tag}_lr{lr_tag}"
    
    def to_lgbm_params(self) -> Dict:
        """转换为 LightGBM 参数字典"""
        params = {
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "min_child_samples": self.min_child_samples,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "device_type": self.device_type,
            "n_jobs": -1,
            "verbose": -1,
        }

        params.update(get_lgbm_seed_params())
        
        if self.mode == "ranker":
            params["objective"] = "lambdarank"
            params["metric"] = "ndcg"
            # 不放入 params，改由 fit() 的 eval_at 传入，避免 sklearn 接口重复警告
        else:
            params["objective"] = "regression"
            params["metric"] = "mse"
        
        return params


# ============================================================
# 数据预处理（复用 data_pipeline 的核心逻辑）
# ============================================================
def prepare_lgbm_data(
    df_panel: pd.DataFrame,
    cfg: LGBMConfig,
) -> Tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp, int, int]:
    """
    LightGBM 数据预处理
    
    复用 LSTM pipeline 的因子计算和标准化，但不构建序列样本。
    树模型直接使用单日特征。
    当 cfg.json_path 与 cfg.factor_path 均非 None 时走因子管道，否则用 df_panel 简单预处理。
    
    Returns:
        df: 处理后的面板数据
        train_end: 训练集截止日期
        val_end: 验证集截止日期
        num_factors: 特征数量（自动从数据中提取）
        num_stocks: 股票数量
    """
    print("=" * 60)
    print("LGBM DATA PIPELINE - 开始")
    print("=" * 60)
    
    use_factor_pipeline = (cfg.json_path is not None and cfg.factor_path is not None)
    
    if use_factor_pipeline:
        assert cfg.json_path is not None and cfg.factor_path is not None  # 类型收窄
        from predata.data_pipeline import load_selected_factors, add_target_column

        # 1. 加载选定的因子
        print("\n[1/2] 加载选定因子...")
        df, missing_factors = load_selected_factors(
            json_path=cfg.json_path,
            factor_path=cfg.factor_path,
            date_col=cfg.date_col,
            id_col=cfg.id_col,
            verbose=True
        )
        if missing_factors:
            print(f"  [警告] 缺失因子: {missing_factors}")
        
        print(f"  加载后: {len(df)} 行, {df[cfg.date_col].nunique()} 天, {df[cfg.id_col].nunique()} 只股票")

        # 2. 添加目标列
        print("[2/2] 计算目标列...")
        df = add_target_column(
            df,
            data_path=cfg.daily_data_path,
            time_col=cfg.date_col,
            id_col=cfg.id_col,
            target_period_start=cfg.target_period_start,
            target_period_end=cfg.target_period_end
        )
        print(f"  目标周期: T+{cfg.target_period_start}open-T+{cfg.target_period_end}close")
    else:
        # 简单预处理：假设数据已包含所需特征
        df = df_panel.copy()
        df[cfg.date_col] = pd.to_datetime(df[cfg.date_col])
        df = df.sort_values([cfg.id_col, cfg.date_col]).reset_index(drop=True)
    
    # 过滤起始与结束日期
    df = df[(df[cfg.date_col] >= cfg.start_date) & (df[cfg.date_col] <= cfg.end_date)]
    print(f"\n过滤后: {len(df)} 行 (start_date >= {cfg.start_date.date()}, end_date <= {cfg.end_date.date()})")
    
    # 删除目标列缺失的行
    df = df.dropna(subset=[cfg.target_col])
    print(f"删除缺失目标后: {len(df)} 行")
    
    # 自动提取特征列（除了 date, id, target 之外的所有列）
    num_factors = len([col for col in df.columns if col not in {cfg.date_col, cfg.id_col, cfg.target_col}])
    num_stocks = df[cfg.id_col].nunique()
    print(f"\n特征信息: {num_factors} 个特征, {num_stocks} 只股票")
    
    # 时间切分
    dates = sorted(df[cfg.date_col].unique())
    n = len(dates)
    train_end = pd.to_datetime(dates[int(n * cfg.train_ratio) - 1])
    val_end = pd.to_datetime(dates[int(n * (cfg.train_ratio + cfg.val_ratio)) - 1])
    
    print(f"时间切分: train_end={train_end.date()}, val_end={val_end.date()}")
    print("=" * 60)
    print("LGBM DATA PIPELINE - 完成")
    print("=" * 60 + "\n")
    
    return df, train_end, val_end, num_factors, num_stocks


def split_data(
    df: pd.DataFrame,
    cfg: LGBMConfig,
    train_end: pd.Timestamp,
    val_end: pd.Timestamp
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按时间切分数据集"""
    train_df = df[df[cfg.date_col] <= train_end].copy()
    val_df = df[(df[cfg.date_col] > train_end) & (df[cfg.date_col] <= val_end)].copy()
    test_df = df[df[cfg.date_col] > val_end].copy()
    
    print(f"[数据集] train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
    return train_df, val_df, test_df


def print_sanity_checks(
    df: pd.DataFrame,
    cfg: LGBMConfig,
    feature_cols: List[str]
) -> None:
    """训练前快速检查数据分布"""
    if df.empty:
        print("[Sanity] train_df 为空，跳过检查")
        return

    print("\n" + "=" * 60)
    print("SANITY CHECK")
    print("=" * 60)

    daily_counts = df.groupby(cfg.date_col).size()
    print("[Sanity] Daily sample count distribution:")
    print(daily_counts.describe().to_string())

    if feature_cols:
        missing_rate = df[feature_cols].isna().mean().sort_values(ascending=False)
        print("\n[Sanity] Feature missing rate (top 10):")
        print(missing_rate.head(10).to_string())

    target = df[cfg.target_col].dropna()
    if len(target) > 0:
        print("\n[Sanity] Target mean:")
        print(f"{target.mean():.6f}")
        print("[Sanity] Target quantiles:")
        print(target.quantile([0.0, 0.25, 0.5, 0.75, 1.0]).to_string())

    print("=" * 60 + "\n")


def as_float(value: object, default: float = float("nan")) -> float:
    if isinstance(value, (int, float, np.floating)):
        return float(value)
    return default


# ============================================================
# LightGBM 训练器
# ============================================================
class LGBMTrainer:
    """LightGBM 训练器（支持 Ranker 和 Regressor）"""
    
    def __init__(self, cfg: LGBMConfig):
        self.cfg = cfg
        self.model: Optional[lgb.LGBMModel] = None
        self.feature_importance: Optional[pd.DataFrame] = None
    
    def _prepare_ranker_groups(self, df: pd.DataFrame) -> np.ndarray:
        """为 Ranker 模式准备分组信息（每日为一组）"""
        group_sizes = df.groupby(self.cfg.date_col).size().values
        return group_sizes #type: ignore
    
    def _convert_to_relevance(self, df: pd.DataFrame, target_col: str, n_levels: int = 10) -> np.ndarray:
        """
        将收益率转换为相关性级别（用于 Ranker 模式）
        
        按日期分组，将收益率分桶为 0 ~ n_levels-1 的相关性级别
        级别越高表示收益越高（更相关）
        
        Args:
            df: 数据
            target_col: 目标列
            n_levels: 相关性级别数（默认10，即0-9）
        """
        def to_relevance(series: pd.Series) -> pd.Series:
            # 使用 qcut 分成 n_levels 个桶，收益越高级别越高
            try:
                labels = pd.qcut(series, q=n_levels, labels=range(n_levels), duplicates='drop')
            except ValueError:
                # 如果数据太少或重复太多，退化为简单排名
                labels = pd.cut(series.rank(method='first'), bins=n_levels, labels=range(n_levels))
            return labels
        
        relevance = df.groupby(self.cfg.date_col)[target_col].transform(to_relevance)
        return np.asarray(relevance.fillna(0).astype(np.int32))
    
    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame
    ) -> Dict[str, float | List[float]]:
        """
        训练模型
        
        Returns:
            history: 训练历史（用于可视化）
        """
        feature_cols = self.cfg.feature_cols
        target_col = self.cfg.target_col
        date_col = self.cfg.date_col
        
        # 过滤特征列中的缺失值
        train_df = train_df.dropna(subset=feature_cols)
        val_df = val_df.dropna(subset=feature_cols)
        
        # [关键] Ranker 模式必须按日期排序，确保同一日期的样本连续排列
        # 否则 group 数组与数据行会错位，导致排序训练失真
        if self.cfg.mode == "ranker":
            train_df = train_df.sort_values(date_col).reset_index(drop=True)
            val_df = val_df.sort_values(date_col).reset_index(drop=True)
        
        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].values
        X_val = val_df[feature_cols].values
        y_val = val_df[target_col].values
        
        params = self.cfg.to_lgbm_params()

        def delayed_early_stopping(
            stopping_rounds: int,
            warmup_rounds: int,
            verbose: bool = True,
        ):
            """
            自定义“延迟早停”：在 warmup_rounds 之前绝不早停；从 warmup_rounds 开始才按 stopping_rounds 判断。

            注：这里不复用 LightGBM 内置 early_stopping callback，避免“中途创建 callback”导致内部状态长度不一致报错。
            """
            # (best_iter, best_score, is_higher_better, announced)
            state: Dict[str, object] = {
                "best_iter": None,
                "best_score": None,
                "higher_better": None,
                "announced": False,
                "metric_name": None,
                "best_evals": None,
            }

            def _callback(env):  # LightGBM callback signature
                cur_iter = int(getattr(env, "iteration", 0))
                evals = getattr(env, "evaluation_result_list", None)

                # warmup 前：不早停，且不初始化 best（保证 warmup 之后才开始计数/比较）
                if cur_iter < int(warmup_rounds):
                    return

                if not state["announced"] and verbose:
                    print(
                        f"[EarlyStopping] warmup={warmup_rounds} 已结束，从第 {warmup_rounds} 轮开始启用早停（patience={stopping_rounds}）"
                    )
                    state["announced"] = True

                if not evals:
                    return

                # 选择第一个评估指标（与 LightGBM 默认行为一致）
                # tuple: (data_name, eval_name, result, is_higher_better)
                data_name, eval_name, result, is_higher_better = evals[0]
                metric_name = f"{data_name}-{eval_name}"
                score = float(result)
                higher = bool(is_higher_better)

                if state["best_iter"] is None:
                    state["best_iter"] = cur_iter
                    state["best_score"] = score
                    state["higher_better"] = higher
                    state["metric_name"] = metric_name
                    state["best_evals"] = list(evals)
                    return

                best_score = float(state["best_score"])  # type: ignore[arg-type]
                best_iter = int(state["best_iter"])  # type: ignore[arg-type]
                higher_better = bool(state["higher_better"])  # type: ignore[arg-type]

                improved = (score > best_score) if higher_better else (score < best_score)
                if improved:
                    state["best_score"] = score
                    state["best_iter"] = cur_iter
                    state["best_evals"] = list(evals)
                    return

                # 未提升：检查 patience
                if cur_iter - best_iter >= int(stopping_rounds):
                    from lightgbm.callback import EarlyStopException

                    if verbose:
                        print(
                            f"[EarlyStopping] Stopping. Best {state['metric_name']}={best_score:.6f} "
                            f"at iteration {best_iter} (warmup={warmup_rounds}, patience={stopping_rounds})."
                        )
                    # LightGBM 使用抛异常来中断训练；best_iter 是 0-based
                    best_evals = state["best_evals"]
                    # EarlyStopException 在不同版本里对 best_score 类型要求不同；这里传入评估结果列表更兼容
                    raise EarlyStopException(best_iter, best_evals)  # type: ignore[arg-type]

            _callback.order = 30  # 与 LightGBM 内置 early_stopping 相近的顺序
            return _callback

        warmup = int(getattr(self.cfg, "early_stopping_warmup_rounds", 0) or 0)
        if warmup > 0:
            es_cb = delayed_early_stopping(
                stopping_rounds=int(self.cfg.early_stopping_rounds),
                warmup_rounds=warmup,
                verbose=True,
            )
        else:
            es_cb = lgb.early_stopping(int(self.cfg.early_stopping_rounds), verbose=True)

        callbacks = [es_cb, lgb.log_evaluation(self.cfg.verbose)]
        
        if self.cfg.mode == "ranker":
            # 排序模式：按日分组，将收益率转换为整数排名
            # 注意：此时 train_df/val_df 已按日期排序，group 与数据行对齐
            train_groups = self._prepare_ranker_groups(train_df)
            val_groups = self._prepare_ranker_groups(val_df)
            
            # 将收益率转换为相关性级别（0-9，越高越好）
            y_train_rank = self._convert_to_relevance(train_df, target_col)
            y_val_rank = self._convert_to_relevance(val_df, target_col)
            
            self.model = lgb.LGBMRanker(**params)
            self.model.fit(
                X_train, y_train_rank,
                group=train_groups,
                eval_set=[(X_val, y_val_rank)],
                eval_group=[val_groups],
                eval_metric="ndcg",
                eval_at=[10, 20, 50],
                callbacks=callbacks,
            )
        else:
            # 回归模式
            self.model = lgb.LGBMRegressor(**params)
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=callbacks,
            )
        
        # 特征重要性
        self.feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        # 构造训练历史（简化版，主要用于记录）
        best_iter = self.model.best_iteration_ if self.model.best_iteration_ is not None else 0

        best_ndcg = float("nan")
        best_val = float("nan")
        evals_result = getattr(self.model, "evals_result_", {})
        if isinstance(evals_result, dict) and len(evals_result) > 0:
            valid_metrics = next(iter(evals_result.values()))
            if self.cfg.mode == "ranker":
                ndcg_vals = []
                for metric_name, values in valid_metrics.items():
                    if str(metric_name).startswith("ndcg"):
                        ndcg_vals.append(np.nanmax(values))
                if ndcg_vals:
                    best_ndcg = float(np.nanmax(ndcg_vals))
                    best_val = best_ndcg
            else:
                metric_name = next(iter(valid_metrics.keys()), None)
                if metric_name is not None:
                    best_val = float(np.nanmin(valid_metrics[metric_name]))

        best_ic = float("nan")
        best_rankic = float("nan")
        try:
            val_pred_df = self.predict(val_df)
            best_ic = pearson_ic_by_date(val_pred_df, date_col='date', score_col='pred', target_col='y')
            best_rankic = rank_ic_by_date(val_pred_df, date_col='date', score_col='pred', target_col='y')
        except Exception:
            pass

        history = {
            'train_loss': [],
            'val_loss': [],
            'best_iteration': int(best_iter),
            'best_val': float(best_val),
            'best_ndcg': float(best_ndcg),
            'best_ic': float(best_ic),
            'best_rankic': float(best_rankic),
        }

        return history
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        预测并返回标准格式的 DataFrame
        
        Returns:
            DataFrame: 包含 date, index_id, pred, y 列（兼容 eval 模块）
        """
        if self.model is None:
            raise RuntimeError("模型未训练，请先调用 fit()")
        
        feature_cols = self.cfg.feature_cols
        target_col = self.cfg.target_col
        
        # 过滤缺失值
        valid_mask = df[feature_cols].notna().all(axis=1)
        valid_df = df[valid_mask].copy()
        
        X = valid_df[feature_cols].values
        preds = self.model.predict(X)
        
        result = pd.DataFrame({
            'date': valid_df[self.cfg.date_col].values,
            'index_id': valid_df[self.cfg.id_col].values,
            'pred': preds,
            'y': valid_df[target_col].values,
        })
        
        return result
    
    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        """评估模型"""
        pred_df = self.predict(df)
        
        ic = pearson_ic_by_date(pred_df, date_col='date', score_col='pred', target_col='y')
        rank_ic = rank_ic_by_date(pred_df, date_col='date', score_col='pred', target_col='y')
        
        # MSE/MAE/Loss
        mse = float(np.mean((pred_df['pred'] - pred_df['y']) ** 2))
        mae = float(np.mean(np.abs(pred_df['pred'] - pred_df['y'])))
        loss = mse  # 对于回归任务，loss 通常是 MSE
        
        return {
            'loss': loss,
            'mse': mse,
            'mae': mae,
            'ic': ic,
            'rankic': rank_ic,
        }


# ============================================================
# 主函数
# ============================================================
def run_lgbm(
    lgbm_cfg: LGBMConfig,
    reproducible: bool = True,
    verbose: bool = True,
    save_test_pred: bool = False
) -> None:
    """
    LightGBM 模型训练全流程（完全按照 run_lstm.py 的逻辑）
    
    Args:
        lgbm_cfg: LightGBM 配置（含因子管道参数 json_path/factor_path 等）
        reproducible: 是否启用可复现模式
        verbose: 是否显示详细信息
        save_test_pred: 是否保存测试集预测结果
    """
    # 可复现性设置
    init_reproducibility(reproducible=reproducible, seed=42)

    model_name = lgbm_cfg.model()
    run_tag = lgbm_cfg.run_tag()
    model_name_tag = f"{model_name}_{run_tag}"
    res_dir = os.path.join("res", model_name)
    
    # 1. 数据预处理
    use_factor_pipeline = (lgbm_cfg.json_path is not None and lgbm_cfg.factor_path is not None)
    daily_df = pd.read_feather(lgbm_cfg.daily_data_path) if use_factor_pipeline else pd.DataFrame()
    df, train_end, val_end, num_factors, num_stocks = prepare_lgbm_data(pd.DataFrame(), lgbm_cfg)
    
    # 自动提取特征列（除了 date, id, target）
    feature_cols = [col for col in df.columns if col not in {lgbm_cfg.date_col, lgbm_cfg.id_col, lgbm_cfg.target_col}]
    lgbm_cfg.feature_cols = feature_cols
    print(f"[特征列] 自动提取 {len(feature_cols)} 个特征")
    
    train_df, val_df, test_df = split_data(df, lgbm_cfg, train_end, val_end)

    print_sanity_checks(train_df, lgbm_cfg, feature_cols)
    
    # 2. 构建模型 + 训练器
    trainer = LGBMTrainer(lgbm_cfg)
    
    # 3. 训练
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)
    history = trainer.fit(train_df, val_df)
    
    # 4. 测试集评估
    print("=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    
    test_metrics = trainer.evaluate(test_df)
    print(f"[Test] loss={test_metrics['loss']:.6f}, mse={test_metrics['mse']:.6f}, "
          f"mae={test_metrics['mae']:.6f}")
    print(f"[Test] IC={test_metrics['ic']:.4f}, RankIC={test_metrics['rankic']:.4f}")
    
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")

    # 5. 可视化 - 训练曲线（LightGBM 特有：特征重要性）
    # 注：由于 LightGBM 没有像神经网络那样的训练曲线，我们展示特征重要性
    feature_importance = trainer.feature_importance
    if feature_importance is not None:
        print("\n[Feature Importance - Top 20]")
        print(feature_importance.head(20).to_string(index=False))
        os.makedirs(res_dir, exist_ok=True)
        feature_importance.to_csv(
            os.path.join(res_dir, f"feature_importance_{run_tag}_{run_timestamp}.csv"),
            index=False
        )
    
    # 6. 实验记录 - 完整记录（JSON + CSV + TXT）
    best_iteration = int(as_float(history.get('best_iteration'), 0.0))
    best_val = as_float(history.get('best_val'))
    best_ndcg = as_float(history.get('best_ndcg'))
    best_ic = as_float(history.get('best_ic'))
    best_rankic = as_float(history.get('best_rankic'))
    
    exp_name = f"{model_name}_{lgbm_cfg.mode}_{lgbm_cfg.num_leaves}leaf_{run_timestamp}"

    log_dir = os.path.join("log", "experiments", model_name)
    
    lgbm_params = lgbm_cfg.to_lgbm_params()
    seed_value = get_seed() if is_reproducible() else None

    log_experiment_full(
        lgbm_cfg,
        metrics={
            'test_loss': test_metrics['loss'],
            'test_mse': test_metrics['mse'],
            'test_mae': test_metrics['mae'],
            'test_ic': test_metrics['ic'],
            'test_rankic': test_metrics['rankic'],
            'best_val': best_val,
            'best_ndcg': best_ndcg,
            'best_ic': best_ic,
            'best_rankic': best_rankic,
            'best_epoch': float(best_iteration),
        },
        log_dir=log_dir,
        experiment_name=exp_name,
        extra_info={
            'train_end': str(train_end.date()),
            'val_end': str(val_end.date()),
            'n_features': num_factors,
            'n_stocks': num_stocks,
            'mode': lgbm_cfg.mode,
            'seed': seed_value,
            'n_jobs': lgbm_params.get('n_jobs'),
            'deterministic': lgbm_params.get('deterministic', False),
        }
    )

    log_experiment_full_csv(
        lgbm_cfg,
        metrics={
            'test_loss': test_metrics['loss'],
            'test_mse': test_metrics['mse'],
            'test_mae': test_metrics['mae'],
            'test_ic': test_metrics['ic'],
            'test_rankic': test_metrics['rankic'],
            'best_val': best_val,
            'best_ndcg': best_ndcg,
            'best_ic': best_ic,
            'best_rankic': best_rankic,
            'best_epoch': float(best_iteration),
        },
        log_dir=log_dir,
        experiment_name=exp_name,
        extra_info={
            'train_end': str(train_end.date()),
            'val_end': str(val_end.date()),
            'n_features': num_factors,
            'n_stocks': num_stocks,
            'mode': lgbm_cfg.mode,
            'seed': seed_value,
            'n_jobs': lgbm_params.get('n_jobs'),
            'deterministic': lgbm_params.get('deterministic', False),
        }
    )
    
    # 7. 策略回测
    print("\n" + "=" * 60)
    print("STRATEGY BACKTEST")
    print("=" * 60)
    
    train_df_pred = trainer.predict(train_df)
    val_df_pred = trainer.predict(val_df)
    test_df_pred = trainer.predict(test_df)
    
    if save_test_pred or lgbm_cfg.save_pred:
        test_pred_path = f"log/{exp_name}_{run_tag}_test_pred.feather"
        test_df_pred.to_feather(test_pred_path)
        print(f"[Saved] test predictions -> {test_pred_path}")
    
    total_df = pd.concat([train_df_pred, val_df_pred, test_df_pred])
    
    # 全量回测
    backtest_total = backtest_decile_strategy(total_df, daily_df, lgbm_cfg)
    plot_backtest_results(backtest_total, train_end, val_end, "total", save_dir=res_dir, model_name=model_name_tag)
    
    # 测试集回测
    backtest_test = backtest_decile_strategy(test_df_pred, daily_df, lgbm_cfg)
    plot_backtest_results(backtest_test, type="test", save_dir=res_dir, model_name=model_name_tag)
    
    # 8. 增强分析
    print("\n" + "=" * 60)
    print("ENHANCED ANALYSIS")
    print("=" * 60)
    
    # 多空净值曲线
    plot_long_short_curve(backtest_test, save_dir=res_dir, model_name=model_name_tag)
    
    # IC 时序图
    plot_ic_series(test_df_pred, save_dir=res_dir, model_name=model_name_tag)
    
    # 回测汇总统计
    summary = compute_backtest_summary(backtest_test)
    print("\n[Backtest Summary - Test Set]")
    print(summary.to_string(index=False))

def dict_product(grid):
    keys = list(grid.keys())
    for values in itertools.product(*grid.values()):
        yield dict(zip(keys, values))
# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    setup_run_log()
    GRID = {
        "mode": ["regressor"],
        "num_leaves":        [31, 127, 255],
        "max_depth":         [-1, 6, 12],
        "learning_rate":     [0.02, 0.04, 0.08],
        "n_estimators":      [1000, 6000],
        "min_child_samples": [10, 60],
        "subsample":         [0.6, 0.9],
        "colsample_bytree":  [0.6, 0.9],
        "early_stopping_warmup_rounds": [0, 100],
    }
    # LightGBM 配置（特征列会在 run_lgbm 中自动提取）
    # 设置 json_path / factor_path 即走因子管道；不设则需自行提供 df_panel
    for method in ["all", "selection_多均累计收益", "selection_IC均值"]:
        for standardized_method in ["cs_gauss"]:
            print("=" * 60)
            print(f"正在标准化: {standardized_method}, 筛选方法: {method}")
            print("=" * 60)
            json_path = fr"C:\Users\Administrator\Desktop\dixi_thsgn\res\factor_backtest\factor_{method}_{standardized_method}.json"
            factor_path = fr"C:\Users\Administrator\Desktop\dixi_thsgn\res\features_all_{standardized_method}.f"
            base_cfg = LGBMConfig(
                json_path=rf"C:\Users\Administrator\Desktop\dixi_thsgn\res\factor_backtest\factor_selection_IC均值_cs_zscore.json",
                factor_path=rf"C:\Users\Administrator\Desktop\dixi_thsgn\res\features_all_cs_zscore.f",
                daily_data_path=r"C:\Users\Administrator\Desktop\dixi_thsgn\data\ths_concept_daily.f",
                target_period_start=1,
                target_period_end=2,
                feature_cols=[],
                reg_alpha=0.1,
                reg_lambda=0.1,
                early_stopping_rounds=100,
            )
            # ===== 网格循环 =====
            for i, hp in enumerate(dict_product(GRID), 1):
                print(f"\n========== RUN {i} / {len(list(dict_product(GRID)))} ==========")
                print("HP:", hp)

                cfg = replace(base_cfg, **hp)  # dataclass 的安全拷贝修改

                try:
                    run_lgbm(cfg)
                except Exception as e:
                    print("❌ FAILED:", e)

    
    # 运行
    # run_lgbm(lgbm_cfg)
    

