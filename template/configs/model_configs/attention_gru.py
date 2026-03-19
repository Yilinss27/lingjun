"""Attention-GRU 模型配置。"""
from ._base import NNModelConfig

config = NNModelConfig(
    # ── 模型结构 ──
    model_name="attention_gru",
    lookback=32,
    hidden=128,
    layers=2,
    drop_out=0.2,

    # ── 训练参数 ──
    lr=1e-3,
    batch_size=2048,
    epochs=12,

    # ── 损失函数 ──
    loss_name="mse",

    # ── GPU 加速 ──
    use_amp=True,
    amp_dtype="bfloat16",
)
