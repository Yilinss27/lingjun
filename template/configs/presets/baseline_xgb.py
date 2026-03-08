# pyright: reportMissingImports=false
from __future__ import annotations

from template.configs.pipeline_config import PipelineConfig


def make_baseline_xgb() -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.model.family = "xgb"
    cfg.model.name = "xgb"
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
    cfg.preprocess.fillna_strategy = "none"
    cfg.preprocess.standardize = False
    cfg.runner.output_dir = "res/pipeline_xgb"
    return cfg
