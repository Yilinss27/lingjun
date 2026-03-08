# pyright: reportMissingImports=false
from __future__ import annotations

from typing import Callable

from .nn import (
    build_lstm_model,
    build_gru_model,
    build_attention_gru_model,
    build_tcn_model,
    build_transformer_model,
    build_transformer_enc_model,
    build_autoformer_model,
    build_autoformer_encoder_model,
)


ModelBuilder = Callable[..., object]


NN_MODEL_REGISTRY = {
    "lstm": build_lstm_model,
    "gru": build_gru_model,
    "attention_gru": build_attention_gru_model,
    "tcn": build_tcn_model,
    "transformer": build_transformer_model,
    "transformer_encoder": build_transformer_enc_model,
    "autoformer": build_autoformer_model,
    "autoformer_encoder": build_autoformer_encoder_model,
}


def get_nn_builder(model_name: str) -> ModelBuilder:
    name = model_name.strip().lower().replace("-", "_")
    if name not in NN_MODEL_REGISTRY:
        supported = ", ".join(sorted(NN_MODEL_REGISTRY.keys()))
        raise ValueError(f"Unsupported nn model '{model_name}', supported: {supported}")
    return NN_MODEL_REGISTRY[name]


def build_nn_model(model_name: str, cfg: object, num_indices: int, num_features: int) -> object:
    builder = get_nn_builder(model_name)
    return builder(cfg, num_indices, num_features)
