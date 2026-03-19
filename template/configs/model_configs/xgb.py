"""XGBoost 模型配置。"""
from ._base import GBDTModelConfig

config = GBDTModelConfig(
    family="xgb",
    objective="regression",
    params={
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
    },
)
