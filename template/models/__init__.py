# pyright: reportMissingImports=false
"""统一模型导出（NN + GBDT + registry）。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .nn import (
    LSTMWithIndexEmbedding,
    build_lstm_model,
    GRUWithIndexEmbedding,
    build_gru_model,
    AttentionGRUWithIndexEmbedding,
    build_attention_gru_model,
    TCNWithIndexEmbedding,
    build_tcn_model,
    TransformerModel,
    build_transformer_model,
    TransformerEncModel,
    build_transformer_enc_model,
    AutoformerModel,
    build_autoformer_model,
    Autoformer_encoderModel,
    build_autoformer_encoder_model,
)
from .registry import NN_MODEL_REGISTRY, get_nn_builder, build_nn_model

if TYPE_CHECKING:
    from .gbdt import LGBMModelWrapper, LGBMParams, XGBModelWrapper, XGBParams

__all__ = [
    "LSTMWithIndexEmbedding",
    "build_lstm_model",
    "GRUWithIndexEmbedding",
    "build_gru_model",
    "AttentionGRUWithIndexEmbedding",
    "build_attention_gru_model",
    "TCNWithIndexEmbedding",
    "build_tcn_model",
    "TransformerModel",
    "build_transformer_model",
    "TransformerEncModel",
    "build_transformer_enc_model",
    "AutoformerModel",
    "build_autoformer_model",
    "Autoformer_encoderModel",
    "build_autoformer_encoder_model",
    "LGBMParams",
    "LGBMModelWrapper",
    "XGBParams",
    "XGBModelWrapper",
    "NN_MODEL_REGISTRY",
    "get_nn_builder",
    "build_nn_model",
]


def __getattr__(name: str) -> Any:
    if name in {"LGBMParams", "LGBMModelWrapper", "XGBParams", "XGBModelWrapper"}:
        from .gbdt import LGBMModelWrapper, LGBMParams, XGBModelWrapper, XGBParams

        return {
            "LGBMParams": LGBMParams,
            "LGBMModelWrapper": LGBMModelWrapper,
            "XGBParams": XGBParams,
            "XGBModelWrapper": XGBModelWrapper,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
