'''
Author: Young yyyao75@163.com
Date: 2026-03-05 19:56:27
LastEditors: Young yyyao75@163.com
LastEditTime: 2026-03-09 14:30:02
FilePath: \lingjun\template\configs\presets\baseline_nn.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# pyright: reportMissingImports=false
from __future__ import annotations

from template.configs.pipeline_config import PipelineConfig


def make_baseline_nn() -> PipelineConfig:
    """
    构建一个基础神经网络流水线配置（PipelineConfig）。

    参数说明与可选项：
    - cfg.model.family: 模型家族类型，神经网络请设为 "nn"。（可选项: "nn", "lgbm", "xgb"）
    - cfg.model.objective: 任务目标类型，回归或排序。（可选项："regression", "ranking"）
    - cfg.preprocess.fillna_strategy: 缺失值填充策略。可选 "none"（不填充）、"zero"（填充0）、"median"（按中位数填充）。
    - cfg.preprocess.standardize: 是否标准化特征。True为按指定standardize_mode标准化（如z-score）。
    - cfg.runner.output_dir: 输出目录，保存模型、预测等结果。

    神经网络相关参数（cfg.nn）：
    - cfg.nn.model_name: 神经网络结构名称。（常用: "autoformer_encoder", "lstm", "mlp", "gru" 等）
    - cfg.nn.lookback: 时间步长窗口长度，决定每个样本看多久历史。
    - cfg.nn.hidden: 隐藏层单元数（神经元个数）。
    - cfg.nn.layers: 网络层数。
    - cfg.nn.drop_out: Dropout比例，防止过拟合，通常在0~0.5之间。
    - cfg.nn.loss_name: 损失函数类型。可选 "mse"（均方误差）、"mae"、"huber"、"ic"、"rankic" 等。
    - cfg.nn.lr: 学习率。常用取值1e-3~1e-5。
    - cfg.nn.batch_size: 训练时每批样本数。
    - cfg.nn.epochs: 训练轮数。
    - cfg.nn.use_amp: 是否启用混合精度训练（True/False），加快训练速度，节省GPU显存。默认为False。

    其他参数可参考 pipeline_config.py 注释说明进行高级配置。
    """

    cfg = PipelineConfig()
    cfg.model.family = "nn"                       # [必选] 模型家族: 神经网络
    cfg.model.objective = "regression"            # [必选] 任务目标: 回归
    cfg.preprocess.fillna_strategy = "median"     # [推荐] 缺失值用中位数填充, 可选: "none", "zero", "median"
    cfg.preprocess.standardize = True             # [推荐] 标准化特征（z-score）
    cfg.runner.output_dir = "res/pipeline"        # [可选] 输出保存目录

    cfg.nn.model_name = "autoformer_encoder"      # [可选] 神经网络结构名称, 如 "autoformer_encoder", "transformer"
    cfg.nn.lookback = 32                          # [可选] 序列回看窗口长度
    cfg.nn.hidden = 128                           # [可选] 隐藏层单元数
    cfg.nn.layers = 2                             # [可选] 网络层数
    cfg.nn.drop_out = 0.2                         # [可选] Dropout 比例, 一般 0.0 ~ 0.5
    cfg.nn.loss_name = "mse"                      # [可选] 损失函数, 可选 "mse"、"mae"、"huber"、"ic"、"rankic"
    cfg.nn.lr = 1e-3                              # [可选] 学习率
    cfg.nn.batch_size = 512                       # [推荐] RTX 5080 16GB VRAM可支持512-1024
    cfg.nn.epochs = 8                             # [可选] 最大训练轮数

    # ── GPU 性能优化 ──
    cfg.nn.use_amp = True                         # [推荐] 启用混合精度训练, 加速~1.5-2x
    cfg.nn.amp_dtype = "bfloat16"                 # [推荐] RTX 5080 原生支持 bfloat16, 比 float16 数值更稳定
    cfg.nn.dataloader_workers = 4                 # [推荐] Windows 下 4 个 worker 是最优平衡
    cfg.nn.prefetch_factor = 4                    # [推荐] 每个 worker 预取 4 个 batch
    cfg.nn.pin_memory = True                      # [推荐] 锁页内存加速 CPU→GPU 传输

    # ── 全量数据 memmap 缓存 ──
    cfg.nn.memmap_cache_dir = "cache/memmap"      # [可选] memmap 缓存目录
    cfg.nn.cross_date_windows = False             # [推荐] 不跨日期构建窗口 (金融数据日间有跳空)

    return cfg
