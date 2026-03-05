# pyright: reportGeneralTypeIssues=false, reportAttributeAccessIssue=false, reportMissingImports=false, reportArgumentType=false, reportMissingTypeArgument=false, reportReturnType=false
"""
XGBoost 模型训练入口

复用 run_lgbm 的数据管道与评估/回测流程，最少改动实现 XGBoost 训练。
"""

from __future__ import annotations
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Literal, Union

import numpy as np
import pandas as pd
import xgboost as xgb

from utils import init_reproducibility, get_seed, is_reproducible
from legacy.run_lgbm_legacy import prepare_lgbm_data, split_data, print_sanity_checks, as_float
from eval import (
    rank_ic_by_date,
    pearson_ic_by_date,
    backtest_decile_strategy,
    compute_backtest_summary,
    plot_backtest_results,
    plot_long_short_curve,
    plot_ic_series,
    log_experiment_full,
    log_experiment_full_csv,
)


@dataclass
class XGBConfig:
    """XGBoost 专用配置（对齐 LGBMConfig）"""
    # 数据配置
    date_col: str = "date"
    id_col: str = "thscode"
    target_col: str = "target"

    # 因子管道
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
    mode: Literal["ranker", "regressor"] = "ranker"

    # XGBoost 超参数
    max_depth: int = 6
    learning_rate: float = 0.05
    n_estimators: int = 500
    min_child_weight: float = 1.0
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1

    # 设备配置
    device_type: Literal["cpu", "gpu"] = "cpu"

    # 早停配置
    early_stopping_rounds: int = 50

    # 输出配置
    verbose: int = 50
    save_pred: bool = False

    # 兼容 eval 模块的属性（ExperimentConfig 协议）
    loss_name: str = "xgb"
    lr: float = 0.05
    epochs: int = 500
    batch_size: int = 0
    lookback: int = 1

    def __post_init__(self) -> None:
        self.lr = self.learning_rate
        self.epochs = self.n_estimators

    def model(self) -> str:
        return "xgb"

    def run_tag(self) -> str:
        lr_tag = str(self.learning_rate).replace(".", "p")
        return f"{self.mode}_md{self.max_depth}_lr{lr_tag}"

    def to_xgb_params(self) -> Dict:
        tree_method = "gpu_hist" if self.device_type == "gpu" else "hist"
        params = {
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "tree_method": tree_method,
            "n_jobs": -1,
            "early_stopping_rounds": int(self.early_stopping_rounds),
        }

        seed_value = get_seed() if is_reproducible() else None
        if seed_value is not None:
            params["random_state"] = int(seed_value)

        if self.mode == "ranker":
            params["objective"] = "rank:ndcg"
            # XGBoost ranker 会自动使用 ndcg 作为评估指标
            # 如果需要指定多个 ndcg@k，可以在初始化时设置 eval_metric
            params["eval_metric"] = "ndcg@10"
        else:
            params["objective"] = "reg:squarederror"
            params["eval_metric"] = "rmse"

        return params


class XGBTrainer:
    """XGBoost 训练器（支持 Ranker 和 Regressor）"""

    def __init__(self, cfg: XGBConfig):
        self.cfg = cfg
        self.model: Optional[Union[xgb.XGBRanker, xgb.XGBRegressor]] = None
        self.feature_importance: Optional[pd.DataFrame] = None

    def _prepare_ranker_groups(self, df: pd.DataFrame) -> np.ndarray:
        group_sizes = df.groupby(self.cfg.date_col).size().values
        return group_sizes  # type: ignore

    def _convert_to_relevance(self, df: pd.DataFrame, target_col: str, n_levels: int = 10) -> np.ndarray:
        def to_relevance(series: pd.Series) -> pd.Series:
            try:
                labels = pd.qcut(series, q=n_levels, labels=range(n_levels), duplicates='drop')
            except ValueError:
                labels = pd.cut(series.rank(method='first'), bins=n_levels, labels=range(n_levels))
            return labels

        relevance = df.groupby(self.cfg.date_col)[target_col].transform(to_relevance)
        return np.asarray(relevance.fillna(0).astype(np.int32))

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict[str, float | List[float]]:
        feature_cols = self.cfg.feature_cols
        target_col = self.cfg.target_col
        date_col = self.cfg.date_col

        train_df = train_df.dropna(subset=feature_cols)
        val_df = val_df.dropna(subset=feature_cols)

        if self.cfg.mode == "ranker":
            train_df = train_df.sort_values(date_col).reset_index(drop=True)
            val_df = val_df.sort_values(date_col).reset_index(drop=True)

        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].values
        X_val = val_df[feature_cols].values
        y_val = val_df[target_col].values

        params = self.cfg.to_xgb_params()

        if self.cfg.mode == "ranker":
            train_groups = self._prepare_ranker_groups(train_df)
            val_groups = self._prepare_ranker_groups(val_df)
            y_train_rank = self._convert_to_relevance(train_df, target_col)
            y_val_rank = self._convert_to_relevance(val_df, target_col)

            # XGBRanker 的参数（包括 early_stopping_rounds 和 eval_metric）需要在初始化时设置
            self.model = xgb.XGBRanker(**params)
            self.model.fit(
                X_train,
                y_train_rank,
                group=train_groups,
                eval_set=[(X_val, y_val_rank)],
                eval_group=[val_groups],
                verbose=self.cfg.verbose,
            )
        else:
            # XGBRegressor 的参数（包括 early_stopping_rounds 和 eval_metric）需要在初始化时设置
            self.model = xgb.XGBRegressor(**params)
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=self.cfg.verbose,
            )

        importances = getattr(self.model, "feature_importances_", None)
        if importances is not None:
            self.feature_importance = pd.DataFrame({
                "feature": feature_cols,
                "importance": importances,
            }).sort_values("importance", ascending=False)

        best_iter = getattr(self.model, "best_iteration", None)
        best_score = getattr(self.model, "best_score", None)

        best_ic = float("nan")
        best_rankic = float("nan")
        try:
            val_pred_df = self.predict(val_df)
            best_ic = pearson_ic_by_date(val_pred_df, date_col="date", score_col="pred", target_col="y")
            best_rankic = rank_ic_by_date(val_pred_df, date_col="date", score_col="pred", target_col="y")
        except Exception:
            pass

        history = {
            "train_loss": [],
            "val_loss": [],
            "best_iteration": int(best_iter) if best_iter is not None else 0,
            "best_val": float(best_score) if best_score is not None else float("nan"),
            "best_ndcg": float(best_score) if self.cfg.mode == "ranker" and best_score is not None else float("nan"),
            "best_ic": float(best_ic),
            "best_rankic": float(best_rankic),
        }
        return history

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("模型未训练，请先调用 fit()")

        feature_cols = self.cfg.feature_cols
        target_col = self.cfg.target_col

        valid_mask = df[feature_cols].notna().all(axis=1)
        valid_df = df[valid_mask].copy()

        X = valid_df[feature_cols].values
        preds = self.model.predict(X)

        return pd.DataFrame({
            "date": valid_df[self.cfg.date_col].values,
            "index_id": valid_df[self.cfg.id_col].values,
            "pred": preds,
            "y": valid_df[target_col].values,
        })

    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        pred_df = self.predict(df)
        ic = pearson_ic_by_date(pred_df, date_col="date", score_col="pred", target_col="y")
        rank_ic = rank_ic_by_date(pred_df, date_col="date", score_col="pred", target_col="y")
        mse = float(np.mean((pred_df["pred"] - pred_df["y"]) ** 2))
        mae = float(np.mean(np.abs(pred_df["pred"] - pred_df["y"])) )
        loss = mse
        return {"loss": loss, "mse": mse, "mae": mae, "ic": ic, "rankic": rank_ic}


def run_xgb(
    xgb_cfg: XGBConfig,
    reproducible: bool = True,
    verbose: bool = True,
    save_test_pred: bool = False,
) -> None:
    init_reproducibility(reproducible=reproducible, seed=42)

    model_name = xgb_cfg.model()
    run_tag = xgb_cfg.run_tag()
    model_name_tag = f"{model_name}_{run_tag}"
    res_dir = os.path.join("res", model_name)

    use_factor_pipeline = (xgb_cfg.json_path is not None and xgb_cfg.factor_path is not None)
    daily_df = pd.read_feather(xgb_cfg.daily_data_path) if use_factor_pipeline else pd.DataFrame()
    # XGBConfig 与 LGBMConfig 结构兼容，可以安全传递
    df, train_end, val_end, num_factors, num_stocks = prepare_lgbm_data(pd.DataFrame(), xgb_cfg)  # type: ignore

    feature_cols = [
        col for col in df.columns
        if col not in {xgb_cfg.date_col, xgb_cfg.id_col, xgb_cfg.target_col}
    ]
    xgb_cfg.feature_cols = feature_cols
    print(f"[特征列] 自动提取 {len(feature_cols)} 个特征")

    # XGBConfig 与 LGBMConfig 结构兼容，可以安全传递
    train_df, val_df, test_df = split_data(df, xgb_cfg, train_end, val_end)  # type: ignore
    print_sanity_checks(train_df, xgb_cfg, feature_cols)  # type: ignore

    trainer = XGBTrainer(xgb_cfg)

    print("=" * 60)
    print("TRAINING")
    print("=" * 60)
    history = trainer.fit(train_df, val_df)

    print("=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    test_metrics = trainer.evaluate(test_df)
    print(f"[Test] loss={test_metrics['loss']:.6f}, mse={test_metrics['mse']:.6f}, "
          f"mae={test_metrics['mae']:.6f}")
    print(f"[Test] IC={test_metrics['ic']:.4f}, RankIC={test_metrics['rankic']:.4f}")

    run_timestamp = time.strftime("%Y%m%d_%H%M%S")

    feature_importance = trainer.feature_importance
    if feature_importance is not None:
        print("\n[Feature Importance - Top 20]")
        print(feature_importance.head(20).to_string(index=False))
        os.makedirs(res_dir, exist_ok=True)
        feature_importance.to_csv(
            os.path.join(res_dir, f"feature_importance_{run_tag}_{run_timestamp}.csv"),
            index=False,
        )

    best_iteration = int(as_float(history.get("best_iteration"), 0.0))
    best_val = as_float(history.get("best_val"))
    best_ndcg = as_float(history.get("best_ndcg"))
    best_ic = as_float(history.get("best_ic"))
    best_rankic = as_float(history.get("best_rankic"))

    exp_name = f"{model_name}_{xgb_cfg.mode}_{xgb_cfg.max_depth}depth_{run_timestamp}"
    log_dir = os.path.join("log", "experiments", model_name)
    seed_value = get_seed() if is_reproducible() else None

    log_experiment_full(
        xgb_cfg,
        metrics={
            "test_loss": test_metrics["loss"],
            "test_mse": test_metrics["mse"],
            "test_mae": test_metrics["mae"],
            "test_ic": test_metrics["ic"],
            "test_rankic": test_metrics["rankic"],
            "best_val": best_val,
            "best_ndcg": best_ndcg,
            "best_ic": best_ic,
            "best_rankic": best_rankic,
            "best_epoch": float(best_iteration),
        },
        log_dir=log_dir,
        experiment_name=exp_name,
        extra_info={
            "train_end": str(train_end.date()),
            "val_end": str(val_end.date()),
            "n_features": num_factors,
            "n_stocks": num_stocks,
            "mode": xgb_cfg.mode,
            "seed": seed_value,
            "n_jobs": -1,
            "device_type": xgb_cfg.device_type,
        },
    )

    log_experiment_full_csv(
        xgb_cfg,
        metrics={
            "test_loss": test_metrics["loss"],
            "test_mse": test_metrics["mse"],
            "test_mae": test_metrics["mae"],
            "test_ic": test_metrics["ic"],
            "test_rankic": test_metrics["rankic"],
            "best_val": best_val,
            "best_ndcg": best_ndcg,
            "best_ic": best_ic,
            "best_rankic": best_rankic,
            "best_epoch": float(best_iteration),
        },
        log_dir=log_dir,
        experiment_name=exp_name,
        extra_info={
            "train_end": str(train_end.date()),
            "val_end": str(val_end.date()),
            "n_features": num_factors,
            "n_stocks": num_stocks,
            "mode": xgb_cfg.mode,
            "seed": seed_value,
            "n_jobs": -1,
            "device_type": xgb_cfg.device_type,
        },
    )

    print("\n" + "=" * 60)
    print("STRATEGY BACKTEST")
    print("=" * 60)

    train_df_pred = trainer.predict(train_df)
    val_df_pred = trainer.predict(val_df)
    test_df_pred = trainer.predict(test_df)

    if save_test_pred or xgb_cfg.save_pred:
        test_pred_path = f"log/{exp_name}_{run_tag}_test_pred.feather"
        test_df_pred.to_feather(test_pred_path)
        print(f"[Saved] test predictions -> {test_pred_path}")

    total_df = pd.concat([train_df_pred, val_df_pred, test_df_pred])

    backtest_total = backtest_decile_strategy(total_df, daily_df, xgb_cfg)
    plot_backtest_results(backtest_total, train_end, val_end, "total", save_dir=res_dir, model_name=model_name_tag)

    backtest_test = backtest_decile_strategy(test_df_pred, daily_df, xgb_cfg)
    plot_backtest_results(backtest_test, type="test", save_dir=res_dir, model_name=model_name_tag)

    print("\n" + "=" * 60)
    print("ENHANCED ANALYSIS")
    print("=" * 60)

    plot_long_short_curve(backtest_test, save_dir=res_dir, model_name=model_name_tag)
    plot_ic_series(test_df_pred, save_dir=res_dir, model_name=model_name_tag)

    summary = compute_backtest_summary(backtest_test)
    print("\n[Backtest Summary - Test Set]")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    # 示例配置（需自行修改 json_path/factor_path）
    cfg = XGBConfig(
        json_path=r"C:\Users\Administrator\Desktop\dixi_thsgn\res\factor_backtest\factor_selection_IC均值_cs_zscore.json",
        factor_path=r"C:\Users\Administrator\Desktop\dixi_thsgn\res\features_all_cs_zscore.f",
        daily_data_path=r"C:\Users\Administrator\Desktop\dixi_thsgn\data\ths_concept_daily.f",
        target_period_start=1,
        target_period_end=2,
        mode="regressor",
        device_type="cpu",
    )
    run_xgb(cfg)
