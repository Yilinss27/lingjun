# pyright: reportMissingTypeArgument=false
"""NN 内部运行时配置 — 供 TorchTrainer、losses、checkpoint 和模型 builder 消费。

不再作为对外主配置；对外主配置请使用 pipeline_config.PipelineConfig。
本文件只保留 NN 运行时真正需要的扁平字段，以及两个辅助适配函数。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ── 损失函数名称常量（losses.py / torch_trainer.py 可能导入）─────────────
IC_LOSS_NAMES = ("pearson_ic", "sigmoid_rankic", "neuralsort_rankic")
REGRESSION_LOSS_NAMES = ("mse", "huber", "quantile_loss", *IC_LOSS_NAMES)
CLASSIFICATION_LOSS_NAMES = ("bce",)


@dataclass
class NNRuntimeConfig:
    """NN 运行时扁平配置，所有 NN builder / TorchTrainer / losses 通过属性访问消费。

    模型专属字段（transformer_* / autoformer_* / encoder_* / tcn_* / gru_*）
    不在此 dataclass 中声明，由 ``build_nn_runtime_config`` 按需设为实例属性。
    builder 中的 ``getattr(cfg, "transformer_d_model", cfg.hidden)`` 等惯用写法
    在字段不存在时自然回退到公共字段，与原 Config 行为一致。
    """

    # ── 模型结构（公共）──────────────────────────────────────────────────
    model_name: str = "lstm"
    emb_dim: int = 0
    hidden: int = 128
    layers: int = 2
    drop_out: float = 0.2
    lookback: int = 32

    # ── 训练 ─────────────────────────────────────────────────────────────
    task_type: str = "regression"
    loss_name: str = "mse"
    huber_delta: float = 1.0
    tau: float = 0.9
    softrank_tau: float = 1.0
    use_daily_batch: bool = False
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    epochs: int = 12

    # ── 设备 / 性能 ──────────────────────────────────────────────────────
    device: str = "auto"
    use_amp: bool = False
    amp_dtype: str = "float16"
    compile_model: bool = False
    compile_mode: str = "reduce-overhead"
    dataloader_workers: int = 0
    prefetch_factor: int = 2
    pin_memory: bool = False

    # ── Checkpoint / 早停 ────────────────────────────────────────────────
    checkpoint_enabled: bool = False
    checkpoint_metric: str = "val_loss"
    checkpoint_strategy: str = "combined"
    checkpoint_warmup_epochs: int = 16
    checkpoint_patience: int = 16
    checkpoint_min_delta: float = 0.0
    checkpoint_ema_alpha: float = 0.0
    checkpoint_save_to_disk: bool = False
    checkpoint_save_path: str = "checkpoints/"

    # ── 梯度范数日志（高级）───────────────────────────────────────────────
    log_grad_norm: bool = False
    grad_norm_log_freq: int = 50
    grad_norm_group_level: int = 3
    grad_norm_topk: int = 200
    grad_norm_log_dir: str = "log/grad_norms"

    def __post_init__(self) -> None:
        if self.loss_name in IC_LOSS_NAMES:
            self.use_daily_batch = True


# 向后兼容：允许 ``from template.configs.model_runtime_config import Config``
Config = NNRuntimeConfig


# ── 适配函数 ─────────────────────────────────────────────────────────────

def build_nn_runtime_config(pipeline_cfg: Any) -> NNRuntimeConfig:
    """从 ``PipelineConfig`` 构建 ``NNRuntimeConfig``。

    模型专属覆盖字段（transformer_* 等）按公共字段预填充为实例属性，
    builder 中的 ``getattr(cfg, "xxx", cfg.hidden)`` 惯用写法仍可正常工作。
    """
    nn = pipeline_cfg.nn
    objective = getattr(pipeline_cfg.model, "objective", "regression")

    rc = NNRuntimeConfig(
        model_name=nn.model_name,
        emb_dim=nn.emb_dim,
        hidden=nn.hidden,
        layers=nn.layers,
        drop_out=nn.drop_out,
        lookback=nn.lookback,
        task_type=objective,
        loss_name=nn.loss_name,
        huber_delta=getattr(nn, "huber_delta", 1.0),
        tau=getattr(nn, "tau", 0.9),
        softrank_tau=getattr(nn, "softrank_tau", 1.0),
        lr=nn.lr,
        weight_decay=nn.weight_decay,
        batch_size=nn.batch_size,
        epochs=nn.epochs,
        device=nn.device,
        use_amp=nn.use_amp,
        amp_dtype=getattr(nn, "amp_dtype", "float16"),
        compile_model=getattr(nn, "compile_model", False),
        compile_mode=getattr(nn, "compile_mode", "reduce-overhead"),
        dataloader_workers=getattr(nn, "dataloader_workers", 0),
        prefetch_factor=getattr(nn, "prefetch_factor", 2),
        pin_memory=getattr(nn, "pin_memory", False),
        checkpoint_enabled=nn.checkpoint_enabled,
        checkpoint_metric=getattr(nn, "checkpoint_metric", "val_loss"),
        checkpoint_strategy=getattr(nn, "checkpoint_strategy", "combined"),
        checkpoint_warmup_epochs=getattr(nn, "checkpoint_warmup_epochs", 16),
        checkpoint_patience=getattr(nn, "checkpoint_patience", 16),
        checkpoint_min_delta=getattr(nn, "checkpoint_min_delta", 0.0),
        checkpoint_ema_alpha=getattr(nn, "checkpoint_ema_alpha", 0.0),
    )
    return rc


def build_nn_runtime_config_from_dict(d: dict) -> NNRuntimeConfig:
    """从 ``meta.json`` 中序列化的 ``nn_config`` 字典重建 ``NNRuntimeConfig``。"""
    rc = NNRuntimeConfig(
        model_name=str(d.get("model_name", "lstm")),
        emb_dim=int(d.get("emb_dim", 0)),
        hidden=int(d.get("hidden", 128)),
        layers=int(d.get("layers", 2)),
        drop_out=float(d.get("drop_out", 0.2)),
        lookback=int(d.get("lookback", 32)),
        loss_name=str(d.get("loss_name", "mse")),
        huber_delta=float(d.get("huber_delta", 1.0)),
        tau=float(d.get("tau", 0.9)),
        softrank_tau=float(d.get("softrank_tau", 1.0)),
        lr=float(d.get("lr", 1e-3)),
        weight_decay=float(d.get("weight_decay", 1e-5)),
        batch_size=int(d.get("batch_size", 256)),
        epochs=int(d.get("epochs", 12)),
        device=str(d.get("device", "auto")),
        use_amp=bool(d.get("use_amp", False)),
        amp_dtype=str(d.get("amp_dtype", "float16")),
        checkpoint_enabled=bool(d.get("checkpoint_enabled", False)),
        checkpoint_metric=str(d.get("checkpoint_metric", "val_loss")),
        checkpoint_strategy=str(d.get("checkpoint_strategy", "combined")),
        checkpoint_patience=int(d.get("checkpoint_patience", 16)),
        checkpoint_warmup_epochs=int(d.get("checkpoint_warmup_epochs", 16)),
        checkpoint_min_delta=float(d.get("checkpoint_min_delta", 0.0)),
        checkpoint_ema_alpha=float(d.get("checkpoint_ema_alpha", 0.0)),
    )
    return rc
