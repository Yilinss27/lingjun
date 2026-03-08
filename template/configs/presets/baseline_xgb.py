'''
Author: Young yyyao75@163.com
Date: 2026-03-05 19:56:14
LastEditors: Young yyyao75@163.com
LastEditTime: 2026-03-08 19:35:43
FilePath: \lingjun\template\configs\presets\baseline_xgb.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# pyright: reportMissingImports=false
from __future__ import annotations

from template.configs.pipeline_config import PipelineConfig


def make_baseline_xgb() -> PipelineConfig:
    cfg = PipelineConfig()
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
    cfg.preprocess.fillna_strategy = "none"
    cfg.preprocess.standardize = False
    cfg.runner.output_dir = "res/pipeline"
    return cfg
