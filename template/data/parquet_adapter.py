from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Iterator, Sequence
from typing import Any, Literal
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# 本文件主要用于封装对大型Parquet数据文件的批量读取配置，
# 包括用于训练和测试的数据路径、特征选择模式、批处理相关参数等，
# 并提供在大数据量下分批次加载和处理数据的能力以节省内存资源。


_FACTOR_PATTERN = re.compile(r"^f(\d+)$")

# Derived from 数据探索报告.md (train/test no-missing overlap, excluding labels)
NO_MISSING_FEATURES: tuple[str, ...] = (
    "f29", "f41", "f46", "f47", "f48", "f58", "f59", "f61",
    "f170", "f171", "f172", "f173", "f219", "f223", "f230", "f235",
    "f236", "f237", "f238", "f258", "f259", "f260", "f261", "f262",
    "f263", "f264", "f265", "f266", "f270", "f273", "f274", "f279",
    "f280", "f281", "f282", "f283", "f286", "f287", "f290", "f292",
    "f297", "f298", "f299", "f300", "f313", "f314", "f315", "f316",
    "f317", "f318", "f319", "f320", "f321", "f322", "f323", "f324",
    "f325", "f334", "f335", "f336", "f338", "f339", "f340", "f341",
    "f342", "f343", "f344", "f345", "f346", "f350", "f354", "f355",
    "f356", "f357", "f358", "f359", "f360", "f361", "f362", "f366",
    "f367", "f368", "f369", "f370", "f371", "f372", "f373",
)

TOP_IC_FEATURES: tuple[str, ...] = (
    "f285", "f36", "f267", "f229", "f129", "f298", "f241", "f281",
)


@dataclass
class LargeParquetConfig:
    train_path: str
    test_path: str | None = None
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


def _factor_sort_key(name: str) -> tuple[int, str]:
    """
    为因子列名排序生成排序键，如f29、f113；其它列排后。
    """
    match = _FACTOR_PATTERN.match(name)
    if match is None:
        return (10_000_000, name)
    return (int(match.group(1)), name)


def discover_factor_columns(parquet_path: str) -> list[str]:
    """
    发现parquet文件中的全部因子特征列（如fXXX），按编号排序返回。
    """
    schema = pq.read_schema(parquet_path)
    factors = [name for name in schema.names if _FACTOR_PATTERN.match(name)]
    factors.sort(key=_factor_sort_key)
    return factors


def select_feature_columns(
    all_features: Sequence[str],
    feature_mode: Literal["all", "no_missing", "top_ic", "custom"],
    custom_features: Sequence[str] | None = None,
) -> list[str]:
    """
    按feature_mode选择有效特征列。
    - all: 全部
    - no_missing: 只选无缺失的列
    - top_ic: 只选IC分数高的列
    - custom: 用户指定
    """
    if feature_mode == "all":
        return list(all_features)
    if feature_mode == "no_missing":
        selected = [f for f in all_features if f in NO_MISSING_FEATURES]
        return selected if selected else list(all_features)
    if feature_mode == "top_ic":
        selected = [f for f in all_features if f in TOP_IC_FEATURES]
        return selected if selected else list(all_features)
    if feature_mode == "custom":
        custom = set(custom_features or [])
        selected = [f for f in all_features if f in custom]
        if not selected:
            raise ValueError("feature_mode='custom' requires valid custom_features")
        return selected
    raise ValueError(f"Unsupported feature_mode={feature_mode}")


def _optimize_numeric_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """
    降低DataFrame中数值列的存储类型（float64->float32, 可降int64->int32）。
    """
    out = frame
    float64_cols = out.select_dtypes(include=["float64"]).columns
    if len(float64_cols) > 0:
        out[float64_cols] = out[float64_cols].astype(np.float32)

    int64_cols = out.select_dtypes(include=["int64"]).columns
    if len(int64_cols) > 0:
        for col in int64_cols:
            values = out[col]
            min_v = values.min()
            max_v = values.max()
            if min_v >= np.iinfo(np.int32).min and max_v <= np.iinfo(np.int32).max:
                out[col] = values.astype(np.int32)
    return out


def iter_parquet_batches(
    parquet_path: str,
    columns: Sequence[str],
    batch_size: int,
    row_group_stride: int = 1,
    row_group_offset: int = 0,
    max_rows: int | None = None,
    dropna_label_col: str | None = None,
) -> Iterator[pd.DataFrame]:
    """
    按batch逐批迭代读取parquet文件，支持指定列、行数、行组步长等，自动类型优化。
    row_group_stride: 行组步长，等于1表示每次加载一个row group
    row_group_offset: 读取parquet时跳过前多少个row group
    这两个参数常用于多进程分任务,冒烟测试等
    """
    pf = pq.ParquetFile(parquet_path)
    stride = max(1, int(row_group_stride))
    offset = max(0, int(row_group_offset))

    emitted_rows = 0
    for row_group_idx in range(offset, pf.num_row_groups, stride):
        batches = pf.iter_batches(
            batch_size=batch_size,
            row_groups=[row_group_idx],
            columns=list(columns),
            use_threads=True,
        )
        for batch in batches:
            frame = batch.to_pandas()
            frame = _optimize_numeric_dtypes(frame)
            if dropna_label_col and dropna_label_col in frame.columns:
                frame = frame[frame[dropna_label_col].notna()]
            if max_rows is not None:
                remain = int(max_rows) - emitted_rows
                if remain <= 0:
                    return
                if len(frame) > remain:
                    frame = frame.iloc[:remain].copy()
            emitted_rows += len(frame)
            if len(frame) > 0:
                yield frame


def load_parquet_frame(
    parquet_path: str,
    columns: Sequence[str],
    batch_size: int,
    row_group_stride: int = 1,
    row_group_offset: int = 0,
    max_rows: int | None = None,
    dropna_label_col: str | None = None,
) -> pd.DataFrame:
    """
    批量读取parquet全量（按columns组合多批次），返回单个DataFrame。
    """
    parts: list[pd.DataFrame] = []
    for part in iter_parquet_batches(
        parquet_path=parquet_path,
        columns=columns,
        batch_size=batch_size,
        row_group_stride=row_group_stride,
        row_group_offset=row_group_offset,
        max_rows=max_rows,
        dropna_label_col=dropna_label_col,
    ):
        parts.append(part)
    if not parts:
        return pd.DataFrame(columns=list(columns))
    return pd.concat(parts, axis=0, ignore_index=True, copy=False)


def split_dates_by_ratio(
    dates: Sequence[object] | Any,
    train_ratio: float,
    val_ratio: float,
) -> tuple[object, object]:
    """
    根据给定的train/val比例，将所有unique date分为train、val、test区间并返回train、val分界date。
    用于基于时间的序列分割，保证时间顺序不被破坏。
    :param dates: 数据中所有的日期序列
    :param train_ratio: 训练集占比 [0, 1]
    :param val_ratio: 验证集占比 [0, 1]
    :return: (train区间最后一天, val区间最后一天)
    """
    unique_dates = sorted(pd.unique(np.asarray(dates)))
    if len(unique_dates) < 3:
        raise ValueError("Need at least 3 unique dates for train/val/test split")

    train_idx = max(1, int(len(unique_dates) * train_ratio)) - 1
    val_idx = max(train_idx + 1, int(len(unique_dates) * (train_ratio + val_ratio)) - 1)
    val_idx = min(val_idx, len(unique_dates) - 2)
    return unique_dates[train_idx], unique_dates[val_idx]


def prepare_large_parquet_train_frame(
    cfg: LargeParquetConfig,
) -> tuple[pd.DataFrame, list[str], object, object]:
    """
    按照配置读取大parquet训练集，返回全部数据、选用的特征列表、train/val分界date。
    - 负责特征选择、筛除最后不完整日期、并排序
    :param cfg: LargeParquetConfig 配置对象
    :return: (DataFrame, 特征列表, train区间最后一天, val区间最后一天)
    """
    all_features = discover_factor_columns(cfg.train_path)
    feature_cols = select_feature_columns(
        all_features=all_features,
        feature_mode=cfg.feature_mode,
        custom_features=cfg.custom_features,
    )

    required_columns: list[str] = list(cfg.id_cols) + feature_cols + [cfg.label_col]
    frame = load_parquet_frame(
        parquet_path=cfg.train_path,
        columns=required_columns,
        batch_size=cfg.batch_size,
        row_group_stride=cfg.row_group_stride,
        row_group_offset=cfg.row_group_offset,
        max_rows=cfg.max_train_rows,
        dropna_label_col=cfg.label_col,
    )

    sort_cols = [col for col in (cfg.date_col, "timeid", "stockid") if col in frame.columns]
    if sort_cols:
        frame = frame.sort_values(sort_cols).reset_index(drop=True)

    if cfg.max_train_rows is not None and len(frame) >= int(cfg.max_train_rows):
        last_date = frame[cfg.date_col].iloc[-1]
        frame = frame[frame[cfg.date_col] != last_date].reset_index(drop=True)
        if len(frame) == 0:
            raise ValueError("max_train_rows too small after trimming incomplete last date")

    train_end, val_end = split_dates_by_ratio(
        frame[cfg.date_col].to_numpy(copy=False),
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
    )
    return frame, feature_cols, train_end, val_end


def prepare_large_parquet_test_frame(
    cfg: LargeParquetConfig,
    feature_cols: Sequence[str],
    sort_rows: bool = False,
) -> pd.DataFrame:
    """
    按照配置读取大parquet测试集，返回全部测试数据，可以选择是否排序。
    :param cfg: LargeParquetConfig 配置对象
    :param feature_cols: 测试集需要的特征列表
    :param sort_rows: 是否按时间/股票id排序
    :return: DataFrame
    """
    if not cfg.test_path:
        return pd.DataFrame()
    columns = list(cfg.id_cols) + list(feature_cols)
    frame = load_parquet_frame(
        parquet_path=cfg.test_path,
        columns=columns,
        batch_size=cfg.batch_size,
        row_group_stride=cfg.row_group_stride,
        row_group_offset=cfg.row_group_offset,
        max_rows=cfg.max_test_rows,
    )
    sort_cols = [col for col in (cfg.date_col, "timeid", "stockid") if col in frame.columns]
    if sort_rows and sort_cols:
        frame = frame.sort_values(sort_cols).reset_index(drop=True)
    return frame
