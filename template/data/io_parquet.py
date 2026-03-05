# pyright: reportMissingImports=false
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterator, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from template.predata.parquet_adapter import (
        LargeParquetConfig,
        iter_parquet_batches,
        prepare_large_parquet_train_frame,
    )
except ImportError:
    from predata.parquet_adapter import (
        LargeParquetConfig,
        iter_parquet_batches,
        prepare_large_parquet_train_frame,
    )


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
