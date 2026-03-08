# pyright: reportMissingImports=false
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from template.data.parquet_adapter import LargeParquetConfig


@dataclass
class DataConfig:
    train_parquet_path: str = "data/train.parquet"
    test_parquet_path: str = "data/test.parquet"
    label_col: str = "LabelA"
    id_cols: tuple[str, ...] = ("stockid", "dateid", "timeid", "exchangeid")
    date_col: str = "dateid"
    # feature_mode参数用于指定特征选择模式，具体含义如下：
    # "all"：使用所有可用特征; "no_missing"：只使用无缺失值的特征; "top_ic"：使用IC（信息系数）排名前列的特征;"custom"：使用用户自定义的特征列表（custom_features字段给出）
    feature_mode: Literal["all", "no_missing", "top_ic", "custom"] = "no_missing"
    custom_features: list[str] = field(default_factory=list)

    train_ratio: float = 0.70
    val_ratio: float = 0.15

    max_train_rows: int | None = None    # 训练集最多读取的行数，None表示不限制
    max_test_rows: int | None = None          # 测试集最多读取的行数，None表示不限制
    batch_size: int = 200_000                 # 读取数据时每批次的最大行数
    row_group_stride: int = 1                 # parquet row group 步长度，等于1表示每次加载一个row group
    row_group_offset: int = 0                 # 读取parquet时跳过前多少个row group

    def to_large_parquet_config(self) -> LargeParquetConfig:
        return LargeParquetConfig(
            train_path=self.train_parquet_path,
            test_path=self.test_parquet_path,
            label_col=self.label_col,
            id_cols=self.id_cols,
            date_col=self.date_col,
            feature_mode=self.feature_mode,
            custom_features=self.custom_features,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            max_train_rows=self.max_train_rows,
            max_test_rows=self.max_test_rows,
            batch_size=self.batch_size,
            row_group_stride=self.row_group_stride,
            row_group_offset=self.row_group_offset,
        )


@dataclass
class PreprocessConfig:
    fillna_strategy: Literal["none", "zero", "median"] = "none"
    standardize: bool = False
    standardize_mode: Literal["none", "global", "cross_section", "date", "time_rolling"] = "none"
    clip_quantile: float | None = None
    z_window: int = 60


@dataclass
class ModelConfig:
    family: Literal["lgbm", "xgb", "nn"] = "lgbm"
    objective: Literal["regression", "ranking"] = "regression"
    name: str = "lgbm"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunnerConfig:
    seed: int = 42
    output_dir: str = "res/pipeline"
    save_model: bool = True   # 是否保存模型权重/参数（model.pkl 或 model.pt）
    save_test_pred: bool = True
    predict_batch_size: int = 1_000_000


@dataclass
class NNConfig:
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
    batch_size: int = 256
    epochs: int = 12

    # ── 设备 / 性能 ──
    device: Literal["auto", "cpu", "gpu"] = "auto"
    use_amp: bool = False
    amp_dtype: str = "float16"
    compile_model: bool = False
    compile_mode: str = "reduce-overhead"
    dataloader_workers: int = 0
    prefetch_factor: int = 2
    pin_memory: bool = False

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


def nn_config_from_meta(d: dict) -> NNConfig:
    """从 meta['nn_config'] 字典恢复 NNConfig，供 predict 端使用。"""
    return NNConfig(
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
        batch_size=int(d.get("batch_size", 256)),
        epochs=int(d.get("epochs", 12)),
        device=d.get("device", "auto") if isinstance(d.get("device"), str) else "auto",
        use_amp=bool(d.get("use_amp", False)),
        amp_dtype=str(d.get("amp_dtype", "float16")),
        compile_model=bool(d.get("compile_model", False)),
        compile_mode=str(d.get("compile_mode", "reduce-overhead")),
        dataloader_workers=int(d.get("dataloader_workers", 0)),
        prefetch_factor=int(d.get("prefetch_factor", 2)),
        pin_memory=bool(d.get("pin_memory", False)),
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


@dataclass
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)
    nn: NNConfig = field(default_factory=NNConfig)
