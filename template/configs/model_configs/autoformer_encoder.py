"""Autoformer Encoder 模型配置。

修改此文件来调整模型结构和训练参数。
管线参数（数据路径、预处理、输出目录等）在 pipeline_config.py 中设置。
"""
from ._base import NNModelConfig

config = NNModelConfig(
    # ── 模型结构 ──
    model_name="autoformer_encoder",
    lookback=32,
    hidden=128,
    layers=2,
    drop_out=0.3,

    # ── 训练参数 ──
    lr=5e-4,
    batch_size=4096,             # 从 512 增大：RTX 5080 (16GB) 仅占 ~4GB VRAM，迭代次数 /8 → 减少数据加载开销
    epochs=4,

    # ── 损失函数 ──
    loss_name="mse",

    # ── GPU 加速 ──
    use_amp=True,
    amp_dtype="bfloat16",
)
