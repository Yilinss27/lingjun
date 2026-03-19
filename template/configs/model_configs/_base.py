# pyright: reportMissingImports=false
"""模型配置基类定义。

NNModelConfig  — 神经网络模型的全部可调参数（结构 + 训练 + 损失 + 加速 + 早停）
GBDTModelConfig — GBDT 模型的全部可调参数（家族 + 目标 + 超参字典）

用户在 model_configs/xxx.py 中实例化这些类来定义具体模型的参数。
管线参数（数据路径、预处理、运行设置等）在 pipeline_config.py 中设置。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class NNModelConfig:
    """神经网络模型配置。

    包含模型结构、训练超参、损失函数、GPU 加速、Checkpoint 等所有模型相关参数。
    管线无关参数（device、dataloader_workers、缓存目录等）在 pipeline_config.RunnerConfig 中。
    """

    # ── 模型结构 ──
    model_name: str = "lstm"
    lookback: int = 32
    emb_dim: int = 0
    hidden: int = 128
    layers: int = 2
    drop_out: float = 0.2

    # ── 损失 ──
    loss_name: str = "mse"
    huber_delta: float = 1.0
    tau: float = 0.9
    softrank_tau: float = 1.0

    # ── 优化器 / 训练 ──
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 2048
    epochs: int = 12

    # ── GPU 加速 ──
    use_amp: bool = True
    amp_dtype: str = "bfloat16"
    compile_model: bool = False
    compile_mode: str = "reduce-overhead"

    # ── Checkpoint / 早停 ──
    checkpoint_enabled: bool = False
    checkpoint_metric: str = "val_loss"
    checkpoint_strategy: str = "combined"
    checkpoint_patience: int = 16
    checkpoint_warmup_epochs: int = 16
    checkpoint_min_delta: float = 0.0
    checkpoint_ema_alpha: float = 0.0
    checkpoint_save_to_disk: bool = False
    checkpoint_save_path: str = "checkpoints/"


@dataclass
class GBDTModelConfig:
    """GBDT 模型配置（LightGBM / XGBoost）。"""

    family: Literal["lgbm", "xgb"] = "lgbm"
    objective: Literal["regression", "ranking"] = "regression"
    params: dict[str, Any] = field(default_factory=dict)


def nn_config_from_meta(d: dict) -> NNModelConfig:
    """从 meta['nn_config'] 字典恢复 NNModelConfig，供 predict 端使用。"""
    return NNModelConfig(
        model_name=str(d.get("model_name", "lstm")),
        lookback=int(d.get("lookback", 32)),
        emb_dim=int(d.get("emb_dim", 0)),
        hidden=int(d.get("hidden", 128)),
        layers=int(d.get("layers", 2)),
        drop_out=float(d.get("drop_out", 0.2)),
        loss_name=str(d.get("loss_name", "mse")),
        huber_delta=float(d.get("huber_delta", 1.0)),
        tau=float(d.get("tau", 0.9)),
        softrank_tau=float(d.get("softrank_tau", 1.0)),
        lr=float(d.get("lr", 1e-3)),
        weight_decay=float(d.get("weight_decay", 1e-5)),
        batch_size=int(d.get("batch_size", 2048)),
        epochs=int(d.get("epochs", 12)),
        use_amp=bool(d.get("use_amp", True)),
        amp_dtype=str(d.get("amp_dtype", "bfloat16")),
        compile_model=bool(d.get("compile_model", False)),
        compile_mode=str(d.get("compile_mode", "reduce-overhead")),
        checkpoint_enabled=bool(d.get("checkpoint_enabled", False)),
        checkpoint_metric=str(d.get("checkpoint_metric", "val_loss")),
        checkpoint_strategy=str(d.get("checkpoint_strategy", "combined")),
        checkpoint_patience=int(d.get("checkpoint_patience", 16)),
        checkpoint_warmup_epochs=int(d.get("checkpoint_warmup_epochs", 16)),
        checkpoint_min_delta=float(d.get("checkpoint_min_delta", 0.0)),
        checkpoint_ema_alpha=float(d.get("checkpoint_ema_alpha", 0.0)),
        checkpoint_save_to_disk=bool(d.get("checkpoint_save_to_disk", False)),
        checkpoint_save_path=str(d.get("checkpoint_save_path", "checkpoints/")),
    )
