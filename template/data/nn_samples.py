from __future__ import annotations

from collections.abc import Sequence

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
