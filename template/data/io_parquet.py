# pyright: reportMissingImports=false
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterator, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from template.data.parquet_adapter import (
    LargeParquetConfig,
    discover_factor_columns,
    iter_parquet_batches,
    prepare_large_parquet_train_frame,
    select_feature_columns,
    split_dates_by_ratio,
)
# 本文件用于封装针对大型Parquet文件的数据读取与预处理相关函数，包括：
# - 加载训练面板数据（load_train_panel）
# - 按指定批次迭代读取测试数据（iter_test_batches）
# - 用生成器流式写出预测结果到Parquet文件（stream_predictions_to_parquet）
# 依赖template.data.parquet_adapter中的配置与基础工具，支撑大规模数据集内存友好型批处理。


@dataclass
class TrainPanelBundle:
    panel: pd.DataFrame
    feature_cols: list[str]
    train_end: object
    val_end: object


def load_train_panel(cfg: LargeParquetConfig) -> TrainPanelBundle:
    panel, feature_cols, train_end, val_end = prepare_large_parquet_train_frame(cfg)
    return TrainPanelBundle(
        panel=panel,
        feature_cols=feature_cols,
        train_end=train_end,
        val_end=val_end,
    )


def iter_test_batches(
    cfg: LargeParquetConfig,
    feature_cols: Sequence[str],
    batch_size: int,
) -> Iterator[pd.DataFrame]:
    id_cols = list(cfg.id_cols)
    read_cols = id_cols + list(feature_cols)
    yield from iter_parquet_batches(
        parquet_path=cfg.test_path or "",
        columns=read_cols,
        batch_size=batch_size,
        row_group_stride=cfg.row_group_stride,
        row_group_offset=cfg.row_group_offset,
        max_rows=cfg.max_test_rows,
    )


def scan_train_metadata(
    cfg: LargeParquetConfig,
) -> tuple[list[str], object, object]:
    """Discover feature columns and date boundaries by scanning only metadata
    and the date column — never loads the full training panel."""
    all_features = discover_factor_columns(cfg.train_path)
    feature_cols = select_feature_columns(
        all_features, cfg.feature_mode, cfg.custom_features,
    )

    all_dates: set = set()
    total_rows = 0
    for batch in iter_parquet_batches(
        parquet_path=cfg.train_path,
        columns=[cfg.date_col],
        batch_size=cfg.batch_size,
        row_group_stride=cfg.row_group_stride,
        row_group_offset=cfg.row_group_offset,
        max_rows=cfg.max_train_rows,
    ):
        all_dates.update(batch[cfg.date_col].unique())
        total_rows += len(batch)

    unique_dates = sorted(all_dates)
    if (cfg.max_train_rows is not None
            and total_rows >= int(cfg.max_train_rows)
            and len(unique_dates) > 3):
        unique_dates = unique_dates[:-1]

    train_end, val_end = split_dates_by_ratio(
        unique_dates, cfg.train_ratio, cfg.val_ratio,
    )
    return feature_cols, train_end, val_end


def stream_predictions_to_parquet(
    output_path: str,
    id_cols: Sequence[str],
    batch_iter: Iterator[pd.DataFrame],
    predict_fn,
    feature_cols: Sequence[str],
) -> str:
    writer: pq.ParquetWriter | None = None
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        for batch in batch_iter:
            preds = predict_fn(batch[list(feature_cols)].to_numpy(dtype=np.float32, copy=False))
            preds_arr = np.asarray(preds, dtype=np.float32).reshape(-1)
            if len(preds_arr) != len(batch):
                raise ValueError(f"Prediction length mismatch: pred={len(preds_arr)}, batch={len(batch)}")

            out = batch[list(id_cols)].copy()
            out["pred"] = preds_arr
            table = pa.Table.from_pandas(out, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(str(out_file), table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    return str(out_file)
