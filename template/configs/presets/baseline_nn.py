# pyright: reportMissingImports=false
from __future__ import annotations

from template.configs.pipeline_config import PipelineConfig


def make_baseline_nn() -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.model.family = "nn"
    cfg.model.objective = "regression"
    cfg.preprocess.fillna_strategy = "median"
    cfg.preprocess.standardize = True
    cfg.runner.output_dir = "res/pipeline"

    cfg.nn.model_name = "lstm"
    cfg.nn.lookback = 32
    cfg.nn.hidden = 128
    cfg.nn.layers = 2
    cfg.nn.drop_out = 0.2
    cfg.nn.loss_name = "mse"
    cfg.nn.lr = 1e-3
    cfg.nn.batch_size = 256
    cfg.nn.epochs = 8
    cfg.nn.use_amp = False
    return cfg
