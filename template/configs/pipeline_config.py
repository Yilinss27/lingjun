# pyright: reportMissingImports=false
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

try:
    from template.predata.parquet_adapter import LargeParquetConfig
except ImportError:
    from predata.parquet_adapter import LargeParquetConfig


@dataclass
class DataConfig:
    train_parquet_path: str = "data/train.parquet"
    test_parquet_path: str = "data/test.parquet"
    label_col: str = "LabelA"
    id_cols: tuple[str, ...] = ("stockid", "dateid", "timeid", "exchangeid")
    date_col: str = "dateid"
    feature_mode: Literal["all", "no_missing", "top_ic", "custom"] = "no_missing"
    custom_features: list[str] = field(default_factory=list)

    train_ratio: float = 0.70
    val_ratio: float = 0.15

    max_train_rows: int | None = 3_000_000
    max_test_rows: int | None = None
    batch_size: int = 200_000
    row_group_stride: int = 1
    row_group_offset: int = 0

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
    clip_quantile: float | None = None


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
    save_test_pred: bool = True
    predict_batch_size: int = 1_000_000


@dataclass
class NNConfig:
    model_name: str = "lstm"
    lookback: int = 32
    emb_dim: int = 0
    hidden: int = 128
    layers: int = 2
    drop_out: float = 0.2
    loss_name: str = "mse"
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    epochs: int = 12
    device: Literal["auto", "cpu", "gpu"] = "auto"
    use_amp: bool = False
    checkpoint_enabled: bool = False


@dataclass
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)
    nn: NNConfig = field(default_factory=NNConfig)
