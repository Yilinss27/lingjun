"""模型配置加载器。

使用 load_model_config(name) 根据模型名加载对应配置。
模型名对应 model_configs/ 下的文件名。
"""
from __future__ import annotations

import importlib
from typing import Union

from ._base import NNModelConfig, GBDTModelConfig, nn_config_from_meta

# 已注册的 NN 模型
_NN_MODELS = {"autoformer_encoder", "lstm", "attention_gru"}
# 已注册的 GBDT 模型
_GBDT_MODELS = {"lgbm", "xgb"}


def load_model_config(model_name: str) -> Union[NNModelConfig, GBDTModelConfig]:
    """根据模型名加载对应的模型配置。

    Args:
        model_name: 模型名称，对应 model_configs/ 下的文件名
                    NN:   "autoformer_encoder", "lstm", "attention_gru"
                    GBDT: "lgbm", "xgb"

    Returns:
        NNModelConfig 或 GBDTModelConfig 实例
    """
    all_models = _NN_MODELS | _GBDT_MODELS
    if model_name not in all_models:
        raise ValueError(
            f"Unknown model: {model_name!r}. "
            f"Available: {sorted(all_models)}"
        )
    module = importlib.import_module(f".{model_name}", package=__package__)
    return module.config


def get_model_family(model_name: str) -> str:
    """根据模型名返回模型家族标识。

    Returns:
        "lgbm" / "xgb" (GBDT) 或 "nn" (神经网络)
    """
    if model_name in _GBDT_MODELS:
        return model_name
    return "nn"


def is_nn_model(model_name: str) -> bool:
    return model_name in _NN_MODELS


def is_gbdt_model(model_name: str) -> bool:
    return model_name in _GBDT_MODELS


__all__ = [
    "NNModelConfig",
    "GBDTModelConfig",
    "nn_config_from_meta",
    "load_model_config",
    "get_model_family",
    "is_nn_model",
    "is_gbdt_model",
]
