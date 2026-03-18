from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Sequence

import numpy as np
import pandas as pd


def build_nn_samples_from_panel(
    frame: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    label_col: str,
    id_col: str,
    date_col: str,
    time_col: str | None,
    lookback: int,
    train_end: object,
    val_end: object,
) -> dict[str, dict[str, object]]:
    """
    根据面板数据构建神经网络训练样本，支持滑动窗口特征序列的生成，并按日期分割为train/val/test集合。

    参数:
        frame: 原始面板型数据（DataFrame），包含特征、标签及分组信息；
        feature_cols: 用作输入的特征列名序列；
        label_col: 标签列名；
        id_col: 个体ID列名（如股票代码）；
        date_col: 日期列名（可与time_col配合实现更细粒度时序）；
        time_col: 可选，时间字段（如分钟）；
        lookback: 滑窗长度；
        train_end, val_end: 训练/验证集的日期边界。
    
    返回:
        dict，键为"train"、"val"、"test"，值均为dict，各包含
          X: shape=(N, lookback, F)，特征窗口数组
          id: shape=(N,)，整数ID
          y: shape=(N,)，标签
          date: shape=(N,)，窗口末端的日期
          index: shape=(N,)，原始ID（如股票代码）
    """
    sort_cols = [date_col]
    if time_col and time_col in frame.columns:
        sort_cols.append(time_col)
    sort_cols.append(id_col)

    work = frame[[id_col, date_col, label_col, *feature_cols, *( [time_col] if time_col and time_col in frame.columns else [] )]].copy()
    work = work.sort_values(sort_cols).reset_index(drop=True)

    unique_ids = sorted(pd.unique(work[id_col]))
    id_map = {v: i for i, v in enumerate(unique_ids)}

    buckets = {
        "train": {"X": [], "id": [], "y": [], "date": [], "index": []},
        "val": {"X": [], "id": [], "y": [], "date": [], "index": []},
        "test": {"X": [], "id": [], "y": [], "date": [], "index": []},
    }

    for raw_id, group in work.groupby(id_col, sort=False):
        seq = group.sort_values(sort_cols)
        x_all = seq[list(feature_cols)].to_numpy(dtype=np.float32)
        y_all = seq[label_col].to_numpy(dtype=np.float32)
        d_all = seq[date_col].to_numpy()

        if len(seq) < lookback:
            continue

        int_id = id_map[raw_id]

        for idx in range(lookback - 1, len(seq)):
            x_win = x_all[idx - lookback + 1 : idx + 1]
            y_v = y_all[idx]
            if np.any(np.isnan(x_win)) or np.any(np.isinf(x_win)):
                continue
            if np.isnan(y_v) or np.isinf(y_v):
                continue

            d_v = d_all[idx]
            split = "train" if d_v <= train_end else ("val" if d_v <= val_end else "test")
            buckets[split]["X"].append(x_win)
            buckets[split]["id"].append(int_id)
            buckets[split]["y"].append(float(y_v))
            buckets[split]["date"].append(d_v)
            buckets[split]["index"].append(raw_id)

    feature_count = len(feature_cols)

    def _pack(split: str) -> dict[str, object]:
        data = buckets[split]
        n = len(data["X"])
        if n == 0:
            return {
                "X": np.zeros((0, lookback, feature_count), dtype=np.float32),
                "id": np.zeros((0,), dtype=np.int64),
                "y": np.zeros((0,), dtype=np.float32),
                "date": np.asarray([], dtype=object),
                "index": np.asarray([], dtype=object),
            }
        return {
            "X": np.asarray(data["X"], dtype=np.float32),
            "id": np.asarray(data["id"], dtype=np.int64),
            "y": np.asarray(data["y"], dtype=np.float32),
            "date": np.asarray(data["date"], dtype=object),
            "index": np.asarray(data["index"], dtype=object),
        }

    return {"train": _pack("train"), "val": _pack("val"), "test": _pack("test")}


def build_nn_samples_streaming(
    batch_iter: Iterable[pd.DataFrame],
    feature_cols: Sequence[str],
    *,
    label_col: str,
    id_col: str,
    date_col: str,
    time_col: str | None,
    lookback: int,
    train_end: object,
    val_end: object,
    on_batch: Callable[[int, int], None] | None = None,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[object, int], int]:
    """Build NN sliding-window samples by streaming pre-transformed batches.

    Unlike ``build_nn_samples_from_panel`` this function never needs the full
    panel in memory.  It maintains per-stock ring buffers of length *lookback*
    and emits a window as soon as a buffer is full.

    ``on_batch(batches_done, rows_seen)`` is called after each non-empty
    batch for optional progress tracking.

    Returns
    -------
    samples : dict  ``{"train": {...}, "val": {...}, "test": {...}}``
    id_map  : dict  raw stock id -> integer index
    num_ids : int   total unique stock ids seen
    """
    sort_cols = [date_col]
    if time_col:
        sort_cols.append(time_col)
    sort_cols.append(id_col)

    # Per-stock ring buffers: sid -> deque of (x_row, y, date, is_invalid)
    buffers: dict[object, deque] = {}
    nan_counts: dict[object, int] = {}
    seen_ids: set = set()

    buckets: dict[str, dict[str, list]] = {
        "train": {"X": [], "id": [], "y": [], "date": [], "index": []},
        "val":   {"X": [], "id": [], "y": [], "date": [], "index": []},
        "test":  {"X": [], "id": [], "y": [], "date": [], "index": []},
    }

    batches_done = 0
    rows_seen = 0

    for batch_df in batch_iter:
        if len(batch_df) == 0:
            continue
        avail = [c for c in sort_cols if c in batch_df.columns]
        if avail:
            batch_df = batch_df.sort_values(avail).reset_index(drop=True)

        feat = batch_df[list(feature_cols)].to_numpy(dtype=np.float32)
        ids  = batch_df[id_col].to_numpy()
        dates = batch_df[date_col].to_numpy()
        labels = batch_df[label_col].to_numpy(dtype=np.float32)

        for i in range(len(batch_df)):
            sid = ids[i]
            seen_ids.add(sid)
            x_row = feat[i]
            y_val = labels[i]
            d_val = dates[i]

            bad = bool(np.any(np.isnan(x_row)) or np.any(np.isinf(x_row))
                       or np.isnan(y_val) or np.isinf(y_val))

            if sid not in buffers:
                buffers[sid] = deque(maxlen=lookback)
                nan_counts[sid] = 0

            buf = buffers[sid]
            if len(buf) == lookback:
                old_bad = buf[0][3]
                if old_bad:
                    nan_counts[sid] -= 1

            buf.append((x_row, y_val, d_val, bad))
            if bad:
                nan_counts[sid] += 1

            if len(buf) == lookback and nan_counts[sid] == 0:
                x_win = np.stack([e[0] for e in buf], axis=0)
                split = "train" if d_val <= train_end else ("val" if d_val <= val_end else "test")
                buckets[split]["X"].append(x_win)
                buckets[split]["id"].append(sid)
                buckets[split]["y"].append(float(y_val))
                buckets[split]["date"].append(d_val)
                buckets[split]["index"].append(sid)

        batches_done += 1
        rows_seen += len(batch_df)
        if on_batch is not None:
            on_batch(batches_done, rows_seen)

    id_map: dict[object, int] = {v: i for i, v in enumerate(sorted(seen_ids))}
    num_ids = len(id_map)
    n_feat = len(feature_cols)

    def _pack(split: str) -> dict[str, np.ndarray]:
        data = buckets[split]
        n = len(data["X"])
        if n == 0:
            return {
                "X": np.zeros((0, lookback, n_feat), dtype=np.float32),
                "id": np.zeros((0,), dtype=np.int64),
                "y": np.zeros((0,), dtype=np.float32),
                "date": np.asarray([], dtype=object),
                "index": np.asarray([], dtype=object),
            }
        return {
            "X": np.asarray(data["X"], dtype=np.float32),
            "id": np.asarray([id_map[s] for s in data["id"]], dtype=np.int64),
            "y": np.asarray(data["y"], dtype=np.float32),
            "date": np.asarray(data["date"], dtype=object),
            "index": np.asarray(data["index"], dtype=object),
        }

    samples = {"train": _pack("train"), "val": _pack("val"), "test": _pack("test")}
    return samples, id_map, num_ids
