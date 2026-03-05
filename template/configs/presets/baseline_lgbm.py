# pyright: reportMissingImports=false
from __future__ import annotations

try:
    from template.configs.pipeline_config import PipelineConfig
except ImportError:
    from configs.pipeline_config import PipelineConfig


def make_baseline_lgbm() -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.model.family = "lgbm"
    cfg.model.name = "lgbm"
    cfg.model.objective = "regression"
    cfg.model.params = {
        "learning_rate": 0.05,
        "n_estimators": 1200,
        "num_leaves": 127,
        "max_depth": -1,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "early_stopping_rounds": 80,
        "device_type": "cpu",
    }
    cfg.preprocess.fillna_strategy = "none"
    cfg.preprocess.standardize = False
    cfg.runner.output_dir = "res/pipeline_lgbm"
    return cfg
