'''
Author: Young yyyao75@163.com
Date: 2026-01-08 15:32:28
LastEditors: Young yyyao75@163.com
Description: 配置模块 —— 多子 dataclass 分组 + 总 Config 扁平兼容访问

结构：
  DataConfig            ← 数据路径、日期、特征列、序列长度等
  ModelCommonConfig     ← 所有模型共用（名称、hidden、layers、drop_out 等）
  TransformerConfig     ← Transformer / Transformer_Enc 专用
  AutoformerConfig      ← Autoformer 专用
  AutoformerEncoderConfig ← Autoformer_Encoder 专用
  TrainConfig           ← 训练、损失、设备、GPU 优化、推理等
  Config                ← 总入口；分组访问 cfg.data.xxx / 扁平访问 cfg.xxx 均可
'''

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# 损失函数名称常量
# ──────────────────────────────────────────────────────────────────────────────
IC_LOSS_NAMES = ("pearson_ic", "sigmoid_rankic", "neuralsort_rankic")

REGRESSION_LOSS_NAMES = ("mse", "huber", "quantile_loss", *IC_LOSS_NAMES)
CLASSIFICATION_LOSS_NAMES = ("bce",)


# ══════════════════════════════════════════════════════════════════════════════
# 1. 数据子配置
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class DataConfig:
    """数据读取与预处理相关配置（路径、日期范围、特征列、序列长度等）。
    factor_path 和 json_path 由 __post_init__ 根据其他字段自动推导，
    也可通过 cfg.factor_path = "..." 事后覆盖。
    """
    daily_data_path: str = r"C:\Users\Administrator\Desktop\dixi_thsgn\data\ths_concept_daily.f"
    start_date: datetime = field(default_factory=lambda: datetime(2020, 1, 1))
    end_date: datetime = field(default_factory=lambda: datetime(2026, 1, 9))
    feature_cols: List[str] = field(default_factory=lambda: [
        "open_ts", "high_ts", "low_ts", "close_ts",
        "volume_ts", "amount_ts", "mmt_ts", "turnF_ts", "volume_momentum_ts",
    ])
    lookback: int = 60                    # 回看窗口（可变）
    target_period: int = 1                # 预测未来 N 天收益
    underlying: str = "index"             # "index" / "stock"
    target_period_start: int = 1
    target_period_end: int = 2
    date_col: str = "date"
    id_col: str = "thscode"

    # 数据分组
    train_ratio: float = 0.70
    val_ratio: float = 0.15

    # 标准化与因子筛选
    z_window: int = 120
    is_standardized: bool = False
    standardized_methods: str = "ts_zscore"
    filter_method: str = "多均累计收益"
    is_all_factors: bool = False

    def __post_init__(self) -> None:
        # 由 standardized_methods / filter_method / is_all_factors 自动推导路径
        self.factor_path: str = (
            rf"C:\Users\Administrator\Desktop\dixi_thsgn\res\features_all_{self.standardized_methods}.f"
        )
        self.json_path: str
        if self.is_all_factors:
            self.json_path = (
                rf"C:\Users\Administrator\Desktop\dixi_thsgn\res\factor_backtest"
                rf"\factor_all_{self.standardized_methods}.json"
            )
        else:
            self.json_path = (
                rf"C:\Users\Administrator\Desktop\dixi_thsgn\res\factor_backtest"
                rf"\factor_selection_{self.filter_method}_{self.standardized_methods}.json"
            )


# ══════════════════════════════════════════════════════════════════════════════
# 2. 通用模型子配置
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class ModelCommonConfig:
    """所有模型共用配置（名称、嵌入维度、隐层、层数、dropout、序列长度等）。
    seq_len / pred_len / label_len 为 None 时，各 build_* 函数会回退到
    lookback 或内置默认值，与原有逻辑一致。
    """
    model_name: str = "autoformer_encoder"
    emb_dim: int = 0
    hidden: int = 256
    layers: int = 2
    drop_out: float = 0.3
    # 序列/解码长度（None → build 内部用 lookback 推导）
    seq_len: Optional[int] = None
    pred_len: Optional[int] = None
    label_len: Optional[int] = None


# ══════════════════════════════════════════════════════════════════════════════
# 3. Transformer 专用子配置
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class TransformerConfig:
    """Transformer / Transformer_Enc 专用配置。
    通过 cfg.transformer_xxx 扁平访问，例如：
      cfg.transformer_dec_input  →  cfg.transformer.dec_input
      cfg.transformer_d_model    →  cfg.transformer.d_model

    说明：
    - 直接访问分组字段（cfg.transformer.d_model）时，None 表示“使用公共 model.hidden”等；
    - 通过扁平名 + getattr(cfg, "transformer_d_model", cfg.hidden) 访问时，
      若该字段仍为 None 且用户未显式设置，则在 Config.__getattr__ 中抛出 AttributeError，
      使 getattr 使用默认值（与重构前行为保持一致）。
    """
    dec_input: str = "standard"           # transformer_dec_input（'standard'/'legacy'）
    d_model: Optional[int] = None         # 未显式设置时，经由扁平访问会回退到 cfg.hidden
    n_heads: int = 8
    e_layers: Optional[int] = None        # 未显式设置时，经由扁平访问会回退到 cfg.layers
    d_layers: int = 1
    dropout: Optional[float] = None       # 未显式设置时，经由扁平访问会回退到 cfg.drop_out
    embed: str = "fixed"
    freq: str = "d"
    d_ff: Optional[int] = None
    activation: str = "gelu"
    factor: int = 1
    pos_emb_type: str = "fixed"
    norm_type: str = "ln"
    task_name: str = "short_term_forecast"
    num_class: int = 2


# ══════════════════════════════════════════════════════════════════════════════
# 4. Autoformer 专用子配置
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class AutoformerConfig:
    """Autoformer 专用配置。
    通过 cfg.autoformer_xxx 扁平访问，例如：
      cfg.autoformer_d_model  →  cfg.autoformer.d_model

    说明：
    - 直接访问分组字段（cfg.autoformer.d_model）时，None 表示“使用公共 model.hidden”等；
    - 通过扁平名 + getattr(cfg, "autoformer_d_model", cfg.hidden) 访问时，
      若该字段仍为 None 且用户未显式设置，则在 Config.__getattr__ 中抛出 AttributeError，
      使 getattr 使用默认值（与重构前行为保持一致）。
    """
    moving_avg: int = 5
    d_model: Optional[int] = None         # 未显式设置时，经由扁平访问会回退到 cfg.hidden
    n_heads: int = 8
    e_layers: Optional[int] = None        # 未显式设置时，经由扁平访问会回退到 cfg.layers
    d_layers: int = 1
    dropout: Optional[float] = None       # 未显式设置时，经由扁平访问会回退到 cfg.drop_out
    embed: str = "fixed"
    freq: str = "d"
    d_ff: Optional[int] = None
    activation: str = "gelu"
    factor: int = 1
    task_name: str = "short_term_forecast"
    num_class: int = 2


# ══════════════════════════════════════════════════════════════════════════════
# 5. AutoformerEncoder 专用子配置
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class AutoformerEncoderConfig:
    """Autoformer_Encoder 专用配置。
    扁平访问规则：
      cfg.encoder_hidden   →  cfg.autoformer_encoder.hidden
      cfg.encoder_layers   →  cfg.autoformer_encoder.layers
      cfg.encoder_drop_out →  cfg.autoformer_encoder.drop_out
      cfg.pe_freq          →  cfg.autoformer_encoder.pe_freq
      cfg.encoder_kernel_size → cfg.autoformer_encoder.kernel_size
      cfg.encoder_n_heads  →  cfg.autoformer_encoder.n_heads

    说明：
    - 直接访问分组字段（cfg.autoformer_encoder.hidden）时，None 表示“使用公共 model.hidden”等；
    - 通过扁平名 + getattr(cfg, "encoder_hidden", cfg.hidden) 访问时，
      若该字段仍为 None 且用户未显式设置，则在 Config.__getattr__ 中抛出 AttributeError，
      使 getattr 使用默认值（与重构前行为保持一致）。
    """
    hidden: Optional[int] = None          # 未显式设置时，经由扁平访问会回退到 cfg.hidden
    layers: Optional[int] = None          # 未显式设置时，经由扁平访问会回退到 cfg.layers
    drop_out: Optional[float] = None      # 未显式设置时，经由扁平访问会回退到 cfg.drop_out
    pe_freq: float = 10000.0
    kernel_size: int = 5
    n_heads: int = 4


# ══════════════════════════════════════════════════════════════════════════════
# 6. 训练子配置
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class TrainConfig:
    """训练流程、损失函数、设备选择、GPU 优化及推理相关配置。"""
    # 任务类型与损失函数
    task_type: Literal["regression", "classification"] = "regression"
    loss_name: str = "huber"
    huber_delta: float = 1.0
    tau: float = 0.9                      # quantile_loss 的 tau
    softrank_tau: float = 1.0             # soft-rank 温度参数
    use_daily_batch: bool = False         # IC/RankIC 损失时由 Config._post_init 自动置 True

    # 训练超参
    seed: int = 42
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 128
    epochs: int = 48

    # 梯度范数日志
    log_grad_norm: bool = False
    grad_norm_log_freq: int = 50
    grad_norm_group_level: int = 3
    grad_norm_topk: int = 200
    grad_norm_log_dir: str = "log/grad_norms"

    # 检查点与早停
    checkpoint_enabled: bool = True
    checkpoint_strategy: str = "combined"    # 'simple'/'warmup'/'ema'/'patience'/'combined'
    checkpoint_metric: str = "val_loss"      # 'val_loss'/'val_rankic'/'val_ic'
    checkpoint_warmup_epochs: int = 16
    checkpoint_patience: int = 16
    checkpoint_min_delta: float = 0.0
    checkpoint_ema_alpha: float = 0.0
    checkpoint_save_to_disk: bool = False
    checkpoint_save_path: str = "checkpoints/"

    # 设备
    device: str = "auto"                     # 'auto'/'cpu'/'gpu'

    # GPU 优化
    use_amp: bool = True
    amp_dtype: str = "float16"
    compile_model: bool = False
    compile_mode: str = "reduce-overhead"
    dataloader_workers: int = 8
    prefetch_factor: int = 4
    pin_memory: bool = True
    cudnn_benchmark: bool = True

    # 推理
    topk: int = 20


# ══════════════════════════════════════════════════════════════════════════════
# 扁平名 → (子配置属性名, 子配置字段名) 映射表
# 用于 Config.__getattr__ / __setattr__ 路由，以及 to_dict 遍历。
# ══════════════════════════════════════════════════════════════════════════════
_FLAT_TO_SUB: Dict[str, Tuple[str, str]] = {
    # ── DataConfig ──────────────────────────────────────────────────────────
    "daily_data_path":        ("data", "daily_data_path"),
    "start_date":             ("data", "start_date"),
    "end_date":               ("data", "end_date"),
    "feature_cols":           ("data", "feature_cols"),
    "lookback":               ("data", "lookback"),
    "target_period":          ("data", "target_period"),
    "underlying":             ("data", "underlying"),
    "target_period_start":    ("data", "target_period_start"),
    "target_period_end":      ("data", "target_period_end"),
    "date_col":               ("data", "date_col"),
    "id_col":                 ("data", "id_col"),
    "train_ratio":            ("data", "train_ratio"),
    "val_ratio":              ("data", "val_ratio"),
    "z_window":               ("data", "z_window"),
    "is_standardized":        ("data", "is_standardized"),
    "standardized_methods":   ("data", "standardized_methods"),
    "filter_method":          ("data", "filter_method"),
    "is_all_factors":         ("data", "is_all_factors"),
    # factor_path / json_path 由 __post_init__ 推导（非声明字段），单独路由
    "factor_path":            ("data", "factor_path"),
    "json_path":              ("data", "json_path"),

    # ── ModelCommonConfig ────────────────────────────────────────────────────
    "model_name":             ("model", "model_name"),
    "emb_dim":                ("model", "emb_dim"),
    "hidden":                 ("model", "hidden"),
    "layers":                 ("model", "layers"),
    "drop_out":               ("model", "drop_out"),
    "seq_len":                ("model", "seq_len"),
    "pred_len":               ("model", "pred_len"),
    "label_len":              ("model", "label_len"),

    # ── TrainConfig ──────────────────────────────────────────────────────────
    "task_type":              ("train", "task_type"),
    "loss_name":              ("train", "loss_name"),
    "huber_delta":            ("train", "huber_delta"),
    "tau":                    ("train", "tau"),
    "softrank_tau":           ("train", "softrank_tau"),
    "use_daily_batch":        ("train", "use_daily_batch"),
    "seed":                   ("train", "seed"),
    "lr":                     ("train", "lr"),
    "weight_decay":           ("train", "weight_decay"),
    "batch_size":             ("train", "batch_size"),
    "epochs":                 ("train", "epochs"),
    "log_grad_norm":          ("train", "log_grad_norm"),
    "grad_norm_log_freq":     ("train", "grad_norm_log_freq"),
    "grad_norm_group_level":  ("train", "grad_norm_group_level"),
    "grad_norm_topk":         ("train", "grad_norm_topk"),
    "grad_norm_log_dir":      ("train", "grad_norm_log_dir"),
    "checkpoint_enabled":     ("train", "checkpoint_enabled"),
    "checkpoint_strategy":    ("train", "checkpoint_strategy"),
    "checkpoint_metric":      ("train", "checkpoint_metric"),
    "checkpoint_warmup_epochs": ("train", "checkpoint_warmup_epochs"),
    "checkpoint_patience":    ("train", "checkpoint_patience"),
    "checkpoint_min_delta":   ("train", "checkpoint_min_delta"),
    "checkpoint_ema_alpha":   ("train", "checkpoint_ema_alpha"),
    "checkpoint_save_to_disk": ("train", "checkpoint_save_to_disk"),
    "checkpoint_save_path":   ("train", "checkpoint_save_path"),
    "device":                 ("train", "device"),
    "use_amp":                ("train", "use_amp"),
    "amp_dtype":              ("train", "amp_dtype"),
    "compile_model":          ("train", "compile_model"),
    "compile_mode":           ("train", "compile_mode"),
    "dataloader_workers":     ("train", "dataloader_workers"),
    "prefetch_factor":        ("train", "prefetch_factor"),
    "pin_memory":             ("train", "pin_memory"),
    "cudnn_benchmark":        ("train", "cudnn_benchmark"),
    "topk":                   ("train", "topk"),

    # ── TransformerConfig（扁平名 = "transformer_" + 字段名）────────────────
    "transformer_dec_input":   ("transformer", "dec_input"),
    "transformer_d_model":     ("transformer", "d_model"),
    "transformer_n_heads":     ("transformer", "n_heads"),
    "transformer_e_layers":    ("transformer", "e_layers"),
    "transformer_d_layers":    ("transformer", "d_layers"),
    "transformer_dropout":     ("transformer", "dropout"),
    "transformer_embed":       ("transformer", "embed"),
    "transformer_freq":        ("transformer", "freq"),
    "transformer_d_ff":        ("transformer", "d_ff"),
    "transformer_activation":  ("transformer", "activation"),
    "transformer_factor":      ("transformer", "factor"),
    "transformer_pos_emb_type": ("transformer", "pos_emb_type"),
    "transformer_norm_type":   ("transformer", "norm_type"),
    "transformer_task_name":   ("transformer", "task_name"),
    "transformer_num_class":   ("transformer", "num_class"),

    # ── AutoformerConfig（扁平名 = "autoformer_" + 字段名）──────────────────
    "autoformer_moving_avg":   ("autoformer", "moving_avg"),
    "autoformer_d_model":      ("autoformer", "d_model"),
    "autoformer_n_heads":      ("autoformer", "n_heads"),
    "autoformer_e_layers":     ("autoformer", "e_layers"),
    "autoformer_d_layers":     ("autoformer", "d_layers"),
    "autoformer_dropout":      ("autoformer", "dropout"),
    "autoformer_embed":        ("autoformer", "embed"),
    "autoformer_freq":         ("autoformer", "freq"),
    "autoformer_d_ff":         ("autoformer", "d_ff"),
    "autoformer_activation":   ("autoformer", "activation"),
    "autoformer_factor":       ("autoformer", "factor"),
    "autoformer_task_name":    ("autoformer", "task_name"),
    "autoformer_num_class":    ("autoformer", "num_class"),

    # ── AutoformerEncoderConfig（扁平名 = "encoder_xxx" 或独立名）───────────
    "encoder_hidden":          ("autoformer_encoder", "hidden"),
    "encoder_layers":          ("autoformer_encoder", "layers"),
    "encoder_drop_out":        ("autoformer_encoder", "drop_out"),
    "pe_freq":                 ("autoformer_encoder", "pe_freq"),
    "encoder_kernel_size":     ("autoformer_encoder", "kernel_size"),
    "encoder_n_heads":         ("autoformer_encoder", "n_heads"),
}

# 子配置属性名集合，用于 __setattr__ 判断是否直接存储
_SUB_CONFIG_NAMES: frozenset = frozenset({
    "data", "model", "train", "transformer", "autoformer", "autoformer_encoder",
})


# ══════════════════════════════════════════════════════════════════════════════
# 总配置入口
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(init=False)
class Config:
    """总配置入口，支持两种访问方式：

    分组访问（新写法）：
        cfg.data.lookback
        cfg.transformer.d_model
        cfg.train.lr

    扁平访问（向后兼容）：
        cfg.lookback          →  cfg.data.lookback
        cfg.hidden            →  cfg.model.hidden
        cfg.transformer_dec_input  →  cfg.transformer.dec_input
        cfg.loss_name         →  cfg.train.loss_name

    兼容构造（与原 Config 完全一致）：
        Config(lookback=60, hidden=256, loss_name="huber", ...)
    或分组构造：
        Config(data=DataConfig(lookback=60), model=ModelCommonConfig(hidden=256), ...)
    两种方式可混用。
    """

    # 声明子配置字段（dataclass 字段，仅用于类型标注与 IDE 友好）
    data: DataConfig
    model: ModelCommonConfig
    train: TrainConfig
    transformer: TransformerConfig
    autoformer: AutoformerConfig
    autoformer_encoder: AutoformerEncoderConfig

    def __init__(
        self,
        data: Optional[DataConfig] = None,
        model: Optional[ModelCommonConfig] = None,
        train: Optional[TrainConfig] = None,
        transformer: Optional[TransformerConfig] = None,
        autoformer: Optional[AutoformerConfig] = None,
        autoformer_encoder: Optional[AutoformerEncoderConfig] = None,
        **kwargs: Any,
    ) -> None:
        # ── 1. 将扁平 kwargs 按子配置分类 ────────────────────────────────────
        sub_kwargs: Dict[str, Dict[str, Any]] = {s: {} for s in _SUB_CONFIG_NAMES}
        for k, v in kwargs.items():
            if k not in _FLAT_TO_SUB:
                raise ValueError(
                    f"Config() 收到未知参数 '{k}'。"
                    f"请检查参数名是否在 _FLAT_TO_SUB 映射中。"
                )
            sub_attr, field_name = _FLAT_TO_SUB[k]
            sub_kwargs[sub_attr][field_name] = v

        # ── 2. 构造（或沿用）各子配置 ─────────────────────────────────────────
        # 若调用方已传入子配置对象，则与扁平 kwargs 中对应部分合并（replace）；
        # 若只有扁平 kwargs，则直接用它们构造新子配置。
        def _build(existing, cls, upd: Dict[str, Any]):
            if existing is None:
                return cls(**upd)
            if upd:
                # replace 会重新调用 __init__ → __post_init__，确保推导字段更新
                from dataclasses import replace as dc_replace
                return dc_replace(existing, **upd)
            return existing

        # 用 object.__setattr__ 绕过自定义 __setattr__，避免循环
        object.__setattr__(self, "data",
            _build(data, DataConfig, sub_kwargs["data"]))
        object.__setattr__(self, "model",
            _build(model, ModelCommonConfig, sub_kwargs["model"]))
        object.__setattr__(self, "train",
            _build(train, TrainConfig, sub_kwargs["train"]))
        object.__setattr__(self, "transformer",
            _build(transformer, TransformerConfig, sub_kwargs["transformer"]))
        object.__setattr__(self, "autoformer",
            _build(autoformer, AutoformerConfig, sub_kwargs["autoformer"]))
        object.__setattr__(self, "autoformer_encoder",
            _build(autoformer_encoder, AutoformerEncoderConfig, sub_kwargs["autoformer_encoder"]))

        # ── 3. 后处理与验证 ───────────────────────────────────────────────────
        self._post_init()

    # ── 后处理 ────────────────────────────────────────────────────────────────
    def _post_init(self) -> None:
        # IC/RankIC 损失：强制按日 batch 模式
        if self.train.loss_name in IC_LOSS_NAMES:
            # 直接写入 TrainConfig 实例（不触发 Config.__setattr__）
            object.__setattr__(self.train, "use_daily_batch", True)
        self.validate()

    # ── 扁平读取路由（仅在普通属性查找失败时触发）────────────────────────────
    def __getattr__(self, name: str) -> Any:
        # 注意：此方法只在 __dict__ 及类层面找不到属性时才被调用
        if name in _FLAT_TO_SUB:
            sub_attr, field_name = _FLAT_TO_SUB[name]
            # 用 object.__getattribute__ 避免递归
            sub_obj = object.__getattribute__(self, sub_attr)
            value = getattr(sub_obj, field_name)
            # 关键兼容逻辑：
            # - 对于默认值为 None 的字段（如 transformer_d_model / autoformer_d_model /
            #   encoder_hidden / transformer_e_layers 等），旧版 Config 中根本不存在
            #   这些属性，getattr(cfg, "xxx", fallback) 会返回 fallback；
            # - 若我们在此返回 None，则 getattr(cfg, "xxx", fallback) 会得到 None，
            #   导致后续 int(None) / float(None) 报错。
            # 因此：当值为 None 时，这里主动抛出 AttributeError，让外层 getattr
            # 走到它自己的默认值逻辑，从而与重构前行为保持一致。
            if value is None:
                raise AttributeError(f"'Config' attribute '{name}' is logically unset (None)")
            return value
        raise AttributeError(f"'Config' 没有属性 '{name}'")

    # ── 扁平写入路由（拦截所有赋值）──────────────────────────────────────────
    def __setattr__(self, name: str, value: Any) -> None:
        if name in _SUB_CONFIG_NAMES:
            # 直接存储子配置对象
            object.__setattr__(self, name, value)
        elif name in _FLAT_TO_SUB:
            # 路由到对应子配置的字段
            sub_attr, field_name = _FLAT_TO_SUB[name]
            sub_obj = object.__getattribute__(self, sub_attr)
            setattr(sub_obj, field_name, value)
        else:
            # 未知字段：直接存为实例属性（兼容外部动态赋值，如 cfg.factor_path = ...）
            object.__setattr__(self, name, value)

    # ── 属性：按日模式判断 ─────────────────────────────────────────────────────
    @property
    def use_daily_mode(self) -> bool:
        """训练/评估是否走按日 batch 模式（建议 runner 统一用它）"""
        return self.train.use_daily_batch or (self.train.loss_name in IC_LOSS_NAMES)

    # ── 工具方法 ──────────────────────────────────────────────────────────────
    def copy_with(self, **kwargs: Any) -> "Config":
        """调参用：返回改了部分字段的新 cfg（扁平 kwargs）。
        实现方式：深拷贝当前 cfg，再逐一通过 __setattr__ 应用 kwargs，
        最后重跑 _post_init（重新触发 IC 检测与 validate）。
        """
        new_cfg = copy.deepcopy(self)
        for k, v in kwargs.items():
            if k not in _FLAT_TO_SUB:
                raise ValueError(f"copy_with() 收到未知字段: '{k}'")
            setattr(new_cfg, k, v)
        # 重跑后处理（IC 检测 + validate）
        new_cfg._post_init()
        return new_cfg

    def to_dict(self) -> Dict[str, Any]:
        """返回扁平 dict，与原 asdict(cfg) 结构兼容。
        factor_path / json_path 作为推导字段一并包含。
        """
        result: Dict[str, Any] = {}
        for flat_name, (sub_attr, field_name) in _FLAT_TO_SUB.items():
            try:
                sub_obj = object.__getattribute__(self, sub_attr)
                result[flat_name] = getattr(sub_obj, field_name)
            except AttributeError:
                pass  # 跳过未初始化的推导字段
        return result

    def validate(self) -> None:
        """校验各字段合法性（与原 Config.validate 逻辑完全一致）。"""
        if self.lookback < 2:
            raise ValueError(f"lookback must be >= 2, got {self.lookback}")
        if self.target_period < 1:
            raise ValueError(f"target_period must be >= 1, got {self.target_period}")
        if not (0.0 < self.train_ratio < 1.0):
            raise ValueError(f"train_ratio must be in (0,1), got {self.train_ratio}")
        if not (0.0 < self.val_ratio < 1.0):
            raise ValueError(f"val_ratio must be in (0,1), got {self.val_ratio}")
        if self.train_ratio + self.val_ratio >= 1.0:
            raise ValueError(
                f"train_ratio + val_ratio must be < 1, got {self.train_ratio + self.val_ratio}"
            )
        if not (0.0 <= self.drop_out < 1.0):
            raise ValueError(f"drop_out must be in [0,1), got {self.drop_out}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")

        # task / loss 合法性
        if self.task_type == "regression":
            if self.loss_name not in REGRESSION_LOSS_NAMES:
                raise ValueError(
                    f"Unsupported regression loss_name={self.loss_name}, "
                    f"available={REGRESSION_LOSS_NAMES}"
                )
        elif self.task_type == "classification":
            if self.loss_name not in CLASSIFICATION_LOSS_NAMES:
                raise ValueError(
                    f"Unsupported classification loss_name={self.loss_name}, "
                    f"available={CLASSIFICATION_LOSS_NAMES}"
                )
        else:
            raise ValueError(f"Unsupported task_type={self.task_type}")

        # checkpoint 指标合法性
        if self.checkpoint_metric not in ("val_loss", "val_rankic", "val_ic"):
            raise ValueError(f"Unsupported checkpoint_metric={self.checkpoint_metric}")

    def __repr__(self) -> str:
        return (
            f"Config(\n"
            f"  data={self.data!r},\n"
            f"  model={self.model!r},\n"
            f"  train={self.train!r},\n"
            f"  transformer={self.transformer!r},\n"
            f"  autoformer={self.autoformer!r},\n"
            f"  autoformer_encoder={self.autoformer_encoder!r},\n"
            f")"
        )
