# pyright: reportMissingImports=false
from __future__ import annotations

from template.configs.pipeline_config import PipelineConfig


def make_baseline_xgb() -> PipelineConfig:
    """
    基础 XGBoost 配置。仅设置模型相关参数。

    GBDT 不需要标准化和缺失值填充（内部自处理），因此覆盖预处理为关闭。
    非模型参数由 pipeline_config.py 默认值控制。
    """
    cfg = PipelineConfig()

    # ── 模型家族 ──
    cfg.model.family = "xgb"
    cfg.model.objective = "regression"
    cfg.model.params = {
        "learning_rate": 0.05,
        "n_estimators": 1000,
        "max_depth": 8,
        "min_child_weight": 1.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "early_stopping_rounds": 80,
        "device_type": "cpu",
    }

    # ── GBDT 不需要预处理 ──
    cfg.preprocess.fillna_strategy = "none"
    cfg.preprocess.standardize = False

    return cfg
