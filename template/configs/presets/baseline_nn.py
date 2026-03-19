# pyright: reportMissingImports=false
from __future__ import annotations

from template.configs.pipeline_config import PipelineConfig


def make_baseline_nn() -> PipelineConfig:
    """
    基础神经网络配置。仅设置模型相关参数。

    非模型参数（预处理、数据、输出目录等）由 pipeline_config.py 默认值控制。

    模型参数说明：
    - cfg.nn.model_name: 网络结构 ("autoformer_encoder", "lstm", "mlp", "gru" 等)
    - cfg.nn.lookback: 回看窗口长度
    - cfg.nn.hidden: 隐藏层单元数
    - cfg.nn.layers: 网络层数
    - cfg.nn.drop_out: Dropout 比例 (0~0.5)
    - cfg.nn.loss_name: 损失函数 ("mse", "mae", "huber", "ic", "rankic")
    - cfg.nn.lr: 学习率
    - cfg.nn.batch_size: 每批样本数
    - cfg.nn.epochs: 训练轮数
    - cfg.nn.use_amp: 混合精度训练
    """

    cfg = PipelineConfig()

    # ── 模型家族 ──
    cfg.model.family = "nn"
    cfg.model.objective = "regression"

    # ── 网络结构 ──
    cfg.nn.model_name = "lstm"
    cfg.nn.lookback = 32
    cfg.nn.hidden = 128
    cfg.nn.layers = 2
    cfg.nn.drop_out = 0.2

    # ── 训练超参 ──
    cfg.nn.loss_name = "mse"
    cfg.nn.lr = 1e-3
    cfg.nn.batch_size = 512
    cfg.nn.epochs = 8

    # ── GPU 性能 ──
    cfg.nn.use_amp = True
    cfg.nn.amp_dtype = "bfloat16"

    return cfg
