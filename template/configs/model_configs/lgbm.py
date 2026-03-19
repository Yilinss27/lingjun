"""LightGBM 模型配置。"""
from ._base import GBDTModelConfig

config = GBDTModelConfig(
    family="lgbm",
    objective="regression",
    params={
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
    },
)
