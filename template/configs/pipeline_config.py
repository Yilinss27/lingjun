# pyright: reportMissingImports=false
"""管线配置（模型无关参数）。

模型相关参数（结构/训练/损失/checkpoint）在 model_configs/ 下对应文件中设置。
运行时通过 load_model_config(model_name) 加载。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from template.data.parquet_adapter import LargeParquetConfig


@dataclass
class DataConfig:
    train_parquet_path: str = "data/train.parquet"
    test_parquet_path: str = "data/test.parquet"
    label_col: str = "LabelA"
    id_cols: tuple[str, ...] = ("stockid", "dateid", "timeid", "exchangeid")
    date_col: str = "dateid"
    # feature_mode: "all" 全部特征 | "no_missing" 无缺失 | "top_ic" IC 排名 | "custom" 自定义
    feature_mode: Literal["all", "no_missing", "top_ic", "custom"] = "all"
    custom_features: list[str] = field(default_factory=list)

    train_ratio: float = 0.75   # 训练集 75%
    val_ratio: float = 0.10     # 验证集 10%（测试集 = 剩余 15%）

    max_train_rows: int | None = None
    max_test_rows: int | None = None
    batch_size: int = 200_000                 # Parquet 读取批大小
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
    fillna_strategy: Literal["none", "zero", "median"] = "median"
    # ── V2 管线标准化配置 ──
    standardize_method: Literal["rank_gauss", "global_zscore", "none"] = "rank_gauss"
    # ── 以下为旧 V1 管线保留字段 ──
    standardize: bool = True
    standardize_mode: Literal["none", "global", "cross_section", "date", "time_rolling"] = "global"
    clip_quantile: float | None = None
    z_window: int = 60


@dataclass
class RunnerConfig:
    seed: int = 42
    output_dir: str = "res/pipeline"
    save_model: bool = True
    save_test_pred: bool = True
    predict_batch_size: int = 1_000_000
    smoke_test: bool = False

    # ── 模块独立开关 ──
    do_test_eval: bool = True
    do_submission_predict: bool = False

    # ── 设备 / 数据加载（模型无关）──
    device: Literal["auto", "cpu", "gpu"] = "auto"
    dataloader_workers: int = 2      # 降至 2：在 RAM 99% 场景下减少 worker 进程内存开销 (~1GB)，为 page cache 让出空间
    prefetch_factor: int = 2          # 降至 2：减少 prefetch 缓冲区内存 (~400MB)，配合 ChunkedShuffleSampler 顺序 I/O 已足够
    pin_memory: bool = True

    # ── 缓存目录 ──
    memmap_cache_dir: str = "cache/memmap"
    raw_cache_dir: str = "cache/raw"
    derived_cache_dir: str = "cache/derived"

    # ── 数据管线版本 ──
    use_v2_pipeline: bool = True
    cross_date_windows: bool = False


@dataclass
class ProgressConfig:
    enabled: bool = True
    every_n_batches: int = 10
    every_n_steps: int = 50
    min_interval_sec: float = 10.0
    jsonl_path: str = ""


@dataclass
class PipelineConfig:
    """管线配置（模型无关参数）。

    使用方式：
        1. 在此处设置 model_name 选择模型
        2. 在 model_configs/<model_name>.py 中设置模型参数
        3. 运行 train.py
    """
    data: DataConfig = field(default_factory=DataConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)
    progress: ProgressConfig = field(default_factory=ProgressConfig)

    # ── 模型选择 ──
    # 对应 template/configs/model_configs/ 下的文件名
    # NN:   "autoformer_encoder", "lstm", "attention_gru"
    # GBDT: "lgbm", "xgb"
    model_name: str = "autoformer_encoder"
