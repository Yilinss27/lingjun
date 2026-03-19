"""
Memmap V2 — 双层缓存数据管线。

架构：
    raw cache   → 只做稳定预处理（缺失值填充），不做标准化，只建一次
    derived cache → 从 raw 做标准化（RankGauss 等），改方法时分钟级重建

存储布局（stock-major 4D）：
    X_raw.dat:     [n_stock, n_step, n_feature]  float32  （n_step = n_date * n_time）
    labels.dat:    [n_stock, n_step]              float32
    X_derived.dat: [n_stock, n_step, n_feature]  float32  （标准化后）

窗口采样：
    一只股票的连续时间步在 memmap 中是连续内存 → lookback 窗口 O(1) 读取

截面标准化：
    X_raw[:, d*n_time + t, :] 可取出同一截面所有股票的特征（stride 跳跃但只在构建时用）
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, DataLoader

from template.data.preprocess_nn import Standardizer


# ============================================================
# Data Structures
# ============================================================

@dataclass
class RawCacheMeta:
    """raw cache 的元信息。"""
    cache_dir: str
    x_raw_path: str          # X_raw.dat 路径
    labels_path: str         # labels.dat 路径
    n_stock: int
    n_date: int
    n_time: int
    n_feature: int
    n_step: int              # = n_date * n_time
    stock_id_map: dict[int, int]   # raw_stockid → int index
    date_list: list[int]     # 排序后的 unique dateids
    train_end: int           # 训练集最后日期
    val_end: int             # 验证集最后日期
    feature_cols: list[str]
    fillna_medians: dict[str, float]  # 用于填充的中位数
    fingerprint: str
    global_means: dict[str, float] | None = None   # 全局均值（global_zscore 用）
    global_stds: dict[str, float] | None = None    # 全局标准差（global_zscore 用）


@dataclass
class DerivedCacheMeta:
    """derived cache 的元信息。"""
    cache_dir: str
    x_derived_path: str      # X_derived.dat 路径
    raw_meta: RawCacheMeta
    standardize_method: str
    idx_train_path: str      # 窗口索引
    idx_val_path: str
    idx_test_path: str
    train_windows: int
    val_windows: int
    test_windows: int
    fingerprint: str


# ============================================================
# Raw Cache Builder
# ============================================================

def _raw_fingerprint(
    parquet_path: str,
    feature_cols: Sequence[str],
    label_col: str,
    fillna_strategy: str,
) -> str:
    h = hashlib.sha256()
    h.update(parquet_path.encode())
    h.update(",".join(feature_cols).encode())
    h.update(label_col.encode())
    h.update(fillna_strategy.encode())
    return h.hexdigest()[:16]


def try_load_raw_cache(
    parquet_path: str,
    feature_cols: list[str],
    label_col: str,
    cache_dir: str = "cache/raw",
) -> Optional[RawCacheMeta]:
    """检查 raw cache 是否命中。命中返回 RawCacheMeta，否则返回 None。

    用于在 train.py 中提前判断是否需要执行 fit_medians 等耗时操作。
    """
    fp = _raw_fingerprint(parquet_path, feature_cols, label_col, "median")
    fp_path = os.path.join(cache_dir, "fingerprint.txt")
    meta_path = os.path.join(cache_dir, "meta.json")

    if not (os.path.exists(fp_path) and os.path.exists(meta_path)):
        return None

    stored_fp = open(fp_path, "r").read().strip()
    if stored_fp != fp:
        return None

    with open(meta_path, "r", encoding="utf-8") as f:
        cached = json.load(f)

    x_path = cached["x_raw_path"]
    l_path = cached["labels_path"]
    if not (os.path.exists(x_path) and os.path.exists(l_path)):
        return None

    return RawCacheMeta(
        cache_dir=cache_dir,
        x_raw_path=x_path,
        labels_path=l_path,
        n_stock=cached["n_stock"],
        n_date=cached["n_date"],
        n_time=cached["n_time"],
        n_feature=cached["n_feature"],
        n_step=cached["n_step"],
        stock_id_map={int(k): v for k, v in cached["stock_id_map"].items()},
        date_list=cached["date_list"],
        train_end=cached["train_end"],
        val_end=cached["val_end"],
        feature_cols=cached["feature_cols"],
        fillna_medians=cached["fillna_medians"],
        fingerprint=fp,
        global_means=cached.get("global_means"),
        global_stds=cached.get("global_stds"),
    )


def build_raw_cache(
    parquet_path: str,
    feature_cols: list[str],
    label_col: str,
    id_col: str,
    date_col: str,
    time_col: str,
    fillna_medians: dict[str, float],
    train_end: int,
    val_end: int,
    cache_dir: str = "cache/raw",
    on_progress: Optional[Callable[[int, int, int], None]] = None,
    global_means: Optional[dict[str, float]] = None,
    global_stds: Optional[dict[str, float]] = None,
    prediction_mode: bool = False,
) -> RawCacheMeta:
    """构建 raw cache：从 Parquet 读取，填充缺失值，写入 stock-major memmap。

    Parquet 假设：每个 row group = 1 只股票，内部按 (dateid, timeid) 排序。

    输出文件：
        X_raw.dat:  shape [n_stock, n_step, n_feature], float32
        labels.dat: shape [n_stock, n_step], float32
        meta.json:  元信息

    Returns:
        RawCacheMeta
    """
    os.makedirs(cache_dir, exist_ok=True)

    # 缓存命中检查（复用 try_load_raw_cache）
    cached = try_load_raw_cache(parquet_path, feature_cols, label_col, cache_dir)
    if cached is not None:
        print(f"[RawCache] Cache hit (fp={cached.fingerprint[:8]}), skipping rebuild.")
        return cached

    fp = _raw_fingerprint(parquet_path, feature_cols, label_col, "median")
    fp_path = os.path.join(cache_dir, "fingerprint.txt")
    meta_path = os.path.join(cache_dir, "meta.json")

    # ── 扫描 Parquet 结构 ──
    pf = pq.ParquetFile(parquet_path)
    n_row_groups = pf.num_row_groups

    print(f"[RawCache] Scanning {n_row_groups} row groups...")
    t0 = time.time()

    # 第一遍：快速扫描维度（不假设 row group = 1 stock）
    all_stocks: set[int] = set()
    all_dates: set[int] = set()
    all_times: set[int] = set()

    for rg_idx in range(n_row_groups):
        table = pf.read_row_group(rg_idx, columns=[id_col, date_col, time_col])
        df = table.to_pandas()
        all_stocks.update(df[id_col].unique().tolist())
        all_dates.update(df[date_col].unique().tolist())
        all_times.update(df[time_col].unique().tolist())

    stock_list = sorted(all_stocks)
    date_list = sorted(all_dates)
    time_list = sorted(all_times)
    n_stock = len(stock_list)
    n_date = len(date_list)
    n_time = len(time_list)
    n_step = n_date * n_time
    n_feature = len(feature_cols)

    # 建立映射
    stock_id_map = {sid: idx for idx, sid in enumerate(stock_list)}
    date_map = {d: idx for idx, d in enumerate(date_list)}
    time_map = {t: idx for idx, t in enumerate(time_list)}

    print(f"[RawCache] Dimensions: {n_stock} stocks × {n_date} dates × {n_time} times × {n_feature} features")
    print(f"[RawCache] Scan done in {time.time() - t0:.1f}s")

    # ── 分配 memmap ──
    x_raw_path = os.path.join(cache_dir, "X_raw.dat")
    labels_path = os.path.join(cache_dir, "labels.dat")

    x_size = n_stock * n_step * n_feature * 4
    l_size = n_stock * n_step * 4
    print(f"[RawCache] Allocating X_raw (~{x_size / (1024**3):.1f} GB) + labels (~{l_size / (1024**3):.2f} GB)")

    # 预分配文件
    for fpath, size in [(x_raw_path, x_size), (labels_path, l_size)]:
        with open(fpath, "wb") as f:
            f.seek(size - 1)
            f.write(b"\0")

    X_raw = np.memmap(x_raw_path, dtype=np.float32, mode="r+",
                      shape=(n_stock, n_step, n_feature))
    labels = np.memmap(labels_path, dtype=np.float32, mode="r+",
                       shape=(n_stock, n_step))

    # 用 NaN 初始化（标记未填充位置）
    X_raw[:] = np.nan
    if prediction_mode:
        # 预测模式：labels 初始化为 0.0（非 NaN），确保所有窗口都被包含
        labels[:] = 0.0
    else:
        labels[:] = np.nan

    # ── 填充数据 ──
    print(f"[RawCache] Writing data...")
    t1 = time.time()
    # 预测模式下，parquet 可能没有 label 列
    has_label_col = label_col in [f.name for f in pf.schema]
    if prediction_mode and not has_label_col:
        read_cols = [id_col, date_col, time_col] + feature_cols
    else:
        read_cols = [id_col, date_col, time_col] + feature_cols + [label_col]

    # 构建中位数填充数组（按特征列顺序）
    median_arr = np.array([fillna_medians.get(c, 0.0) for c in feature_cols], dtype=np.float32)

    for rg_idx in range(n_row_groups):
        table = pf.read_row_group(rg_idx, columns=read_cols)
        df = table.to_pandas()

        # 提取数据
        feats = df[feature_cols].to_numpy(dtype=np.float32)
        if prediction_mode and not has_label_col:
            labs = np.zeros(len(df), dtype=np.float32)
        else:
            labs = df[label_col].to_numpy(dtype=np.float32)
            if prediction_mode:
                # 预测模式下将 NaN 标签替换为 0.0，避免窗口索引过滤
                np.nan_to_num(labs, copy=False, nan=0.0)
        stockids = df[id_col].to_numpy()
        dateids = df[date_col].to_numpy()
        timeids = df[time_col].to_numpy()

        # 中位数填充
        nan_mask = np.isnan(feats)
        if np.any(nan_mask):
            fill_vals = np.broadcast_to(median_arr, feats.shape)
            feats = np.where(nan_mask, fill_vals, feats)

        # 残余 NaN/Inf 清理
        np.nan_to_num(feats, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # 向量化写入（每个 row group 通常 1 只股票，批量写入替代逐行赋值）
        s_indices = np.array([stock_id_map.get(int(s), -1) for s in stockids])
        d_indices = np.array([date_map.get(int(d), -1) for d in dateids])
        t_indices = np.array([time_map.get(int(t), -1) for t in timeids])
        valid = (s_indices >= 0) & (d_indices >= 0) & (t_indices >= 0)

        if valid.any():
            sv = s_indices[valid]
            steps = d_indices[valid] * n_time + t_indices[valid]
            feats_v = feats[valid]
            labs_v = labs[valid]

            # 按股票分组批量写入（通常每 row group 只有 1 只股票）
            for sid in np.unique(sv):
                mask = sv == sid
                X_raw[sid, steps[mask], :] = feats_v[mask]
                labels[sid, steps[mask]] = labs_v[mask]

        # 进度
        elapsed = time.time() - t1
        if on_progress is not None:
            on_progress(rg_idx + 1, n_row_groups, (rg_idx + 1))
        if (rg_idx + 1) % max(1, n_row_groups // 20) == 0 or rg_idx + 1 == n_row_groups:
            pct = (rg_idx + 1) / n_row_groups
            eta = elapsed / pct - elapsed if pct > 0 else 0
            print(f"[RawCache]   {rg_idx + 1}/{n_row_groups} row groups "
                  f"({100*pct:.0f}%), elapsed={elapsed:.0f}s, ETA~{eta:.0f}s",
                  flush=True)

    # 逐股票清理 X_raw 残余 NaN（避免一次性分配 ~15GB bool 数组导致 OOM）
    # labels 中的 NaN 保留：标记"该位置无真实数据"，供 derived cache 构建窗口索引时过滤
    print(f"[RawCache] Cleaning residual NaN (per-stock chunked)...")
    for s_idx in range(n_stock):
        stock_x = np.array(X_raw[s_idx])   # copy to RAM (~27MB per stock at n_step=17280, n_feature=384)
        nan_mask = np.isnan(stock_x)
        if np.any(nan_mask):
            stock_x[nan_mask] = 0.0
            X_raw[s_idx] = stock_x

    X_raw.flush()
    labels.flush()
    del X_raw, labels

    elapsed_total = time.time() - t0
    print(f"[RawCache] Done in {elapsed_total:.1f}s")

    # ── 保存元信息 ──
    meta = RawCacheMeta(
        cache_dir=cache_dir,
        x_raw_path=x_raw_path,
        labels_path=labels_path,
        n_stock=n_stock,
        n_date=n_date,
        n_time=n_time,
        n_feature=n_feature,
        n_step=n_step,
        stock_id_map=stock_id_map,
        date_list=date_list,
        train_end=int(train_end),
        val_end=int(val_end),
        feature_cols=feature_cols,
        fillna_medians=fillna_medians,
        fingerprint=fp,
        global_means=global_means,
        global_stds=global_stds,
    )

    meta_dict = {
        "x_raw_path": x_raw_path,
        "labels_path": labels_path,
        "n_stock": n_stock,
        "n_date": n_date,
        "n_time": n_time,
        "n_feature": n_feature,
        "n_step": n_step,
        "stock_id_map": {str(k): v for k, v in stock_id_map.items()},
        "date_list": date_list,
        "train_end": int(train_end),
        "val_end": int(val_end),
        "feature_cols": feature_cols,
        "fillna_medians": {k: float(v) for k, v in fillna_medians.items()},
        "fingerprint": fp,
        "build_time_sec": round(elapsed_total, 1),
        "global_means": {k: float(v) for k, v in global_means.items()} if global_means else None,
        "global_stds": {k: float(v) for k, v in global_stds.items()} if global_stds else None,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2, ensure_ascii=False)
    with open(fp_path, "w") as f:
        f.write(fp)

    return meta


# ============================================================
# Derived Cache Builder
# ============================================================

def _derived_fingerprint(raw_fp: str, standardize_method: str, lookback: int,
                         cross_date: bool, train_end: int, val_end: int) -> str:
    h = hashlib.sha256()
    h.update(raw_fp.encode())
    h.update(standardize_method.encode())
    h.update(str(lookback).encode())
    h.update(str(cross_date).encode())
    h.update(str(train_end).encode())
    h.update(str(val_end).encode())
    return h.hexdigest()[:16]


def build_derived_cache(
    raw_meta: RawCacheMeta,
    standardizer: Standardizer,
    standardize_method: str,
    lookback: int,
    cross_date_windows: bool = False,
    cache_dir: str = "cache/derived",
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> DerivedCacheMeta:
    """从 raw cache 构建 derived cache：逐日截面标准化 + 窗口索引。

    流程：
        1. 打开 X_raw memmap
        2. 逐日期：取 X_raw[:, d*n_time:(d+1)*n_time, :] → 逐 timeid 做截面标准化
        3. 写入 X_derived memmap
        4. 构建窗口索引（train/val/test）

    Returns:
        DerivedCacheMeta
    """
    os.makedirs(cache_dir, exist_ok=True)

    fp = _derived_fingerprint(
        raw_meta.fingerprint, standardize_method, lookback,
        cross_date_windows, raw_meta.train_end, raw_meta.val_end,
    )
    fp_path = os.path.join(cache_dir, "fingerprint.txt")
    meta_path = os.path.join(cache_dir, "meta.json")

    # 缓存命中
    if os.path.exists(fp_path) and os.path.exists(meta_path):
        stored_fp = open(fp_path, "r").read().strip()
        if stored_fp == fp:
            with open(meta_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            all_exist = all(
                os.path.exists(cached.get(k, ""))
                for k in ("x_derived_path", "idx_train_path", "idx_val_path", "idx_test_path")
            )
            if all_exist:
                print(f"[DerivedCache] Cache hit (fp={fp[:8]}), skipping rebuild.")
                return DerivedCacheMeta(
                    cache_dir=cache_dir,
                    x_derived_path=cached["x_derived_path"],
                    raw_meta=raw_meta,
                    standardize_method=standardize_method,
                    idx_train_path=cached["idx_train_path"],
                    idx_val_path=cached["idx_val_path"],
                    idx_test_path=cached["idx_test_path"],
                    train_windows=cached["train_windows"],
                    val_windows=cached["val_windows"],
                    test_windows=cached["test_windows"],
                    fingerprint=fp,
                )

    n_stock = raw_meta.n_stock
    n_date = raw_meta.n_date
    n_time = raw_meta.n_time
    n_step = raw_meta.n_step
    n_feature = raw_meta.n_feature
    date_list = raw_meta.date_list
    train_end = raw_meta.train_end
    val_end = raw_meta.val_end

    # Bug 5: lookback > n_time 错误检查
    if not cross_date_windows and lookback > n_time:
        raise ValueError(f"[DerivedCache] ERROR: lookback ({lookback}) > n_time ({n_time}) "
                         f"with cross_date_windows=False — 无日内窗口，训练样本将为 0")

    # ── 打开 raw memmap ──
    X_raw = np.memmap(raw_meta.x_raw_path, dtype=np.float32, mode="r",
                      shape=(n_stock, n_step, n_feature))

    # ── 分配 derived memmap ──
    x_derived_path = os.path.join(cache_dir, "X_derived.dat")
    x_size = n_stock * n_step * n_feature * 4
    print(f"[DerivedCache] Allocating X_derived (~{x_size / (1024**3):.1f} GB)")

    with open(x_derived_path, "wb") as f:
        f.seek(x_size - 1)
        f.write(b"\0")

    X_derived = np.memmap(x_derived_path, dtype=np.float32, mode="r+",
                          shape=(n_stock, n_step, n_feature))

    # ── 逐日期做截面标准化 ──
    print(f"[DerivedCache] Applying {standardize_method} standardization...")
    t0 = time.time()

    for d_idx, date_val in enumerate(date_list):
        step_start = d_idx * n_time
        step_end = step_start + n_time

        # 取一天的数据：[n_stock, n_time, n_feature]
        day_block = np.array(X_raw[:, step_start:step_end, :])  # copy to RAM

        # 逐 timeid 做截面标准化
        for t_idx in range(n_time):
            cross_section = day_block[:, t_idx, :]  # [n_stock, n_feature]
            day_block[:, t_idx, :] = standardizer.transform_cross_section(cross_section)

        # 写回 derived
        X_derived[:, step_start:step_end, :] = day_block

        if on_progress is not None:
            on_progress(d_idx + 1, n_date)
        if (d_idx + 1) % max(1, n_date // 20) == 0 or d_idx + 1 == n_date:
            elapsed = time.time() - t0
            pct = (d_idx + 1) / n_date
            eta = elapsed / pct - elapsed if pct > 0 else 0
            print(f"[DerivedCache]   {d_idx + 1}/{n_date} dates "
                  f"({100*pct:.0f}%), elapsed={elapsed:.0f}s, ETA~{eta:.0f}s",
                  flush=True)

    X_derived.flush()
    elapsed_std = time.time() - t0
    print(f"[DerivedCache] Standardization done in {elapsed_std:.1f}s")

    # ── 构建窗口索引 ──
    print(f"[DerivedCache] Building window indices (lookback={lookback})...")
    t1 = time.time()

    labels = np.memmap(raw_meta.labels_path, dtype=np.float32, mode="r",
                       shape=(n_stock, n_step))

    train_indices: list[tuple[int, int]] = []   # (stock_idx, end_step)
    val_indices: list[tuple[int, int]] = []
    test_indices: list[tuple[int, int]] = []

    for s_idx in range(n_stock):
        # 复制一次该股票的标签到 RAM，避免反复 memmap 随机读
        stock_labels = np.array(labels[s_idx])  # [n_step]

        if cross_date_windows:
            # 允许跨日期窗口
            all_end_steps = np.arange(lookback - 1, n_step)
            valid_mask = ~np.isnan(stock_labels[all_end_steps])
            valid_steps = all_end_steps[valid_mask]

            for end_step in valid_steps:
                end_date = date_list[int(end_step) // n_time]

                if end_date <= train_end:
                    train_indices.append((s_idx, int(end_step)))
                elif end_date <= val_end:
                    val_indices.append((s_idx, int(end_step)))
                else:
                    test_indices.append((s_idx, int(end_step)))
        else:
            # 窗口不跨日期边界（向量化 NaN 过滤 + extend 批量添加）
            for d_idx in range(n_date):
                date_val = date_list[d_idx]
                day_start = d_idx * n_time

                end_steps = np.arange(day_start + lookback - 1, day_start + n_time)
                valid_mask = ~np.isnan(stock_labels[end_steps])
                valid_steps = end_steps[valid_mask]

                if len(valid_steps) == 0:
                    continue

                pairs = [(s_idx, int(es)) for es in valid_steps]
                if date_val <= train_end:
                    train_indices.extend(pairs)
                elif date_val <= val_end:
                    val_indices.extend(pairs)
                else:
                    test_indices.extend(pairs)

    del labels, X_raw, X_derived

    # 保存索引为 numpy 数组：[N, 2] (stock_idx, end_step)
    idx_train_path = os.path.join(cache_dir, "idx_train.npy")
    idx_val_path = os.path.join(cache_dir, "idx_val.npy")
    idx_test_path = os.path.join(cache_dir, "idx_test.npy")

    idx_train = np.array(train_indices, dtype=np.int64).reshape(-1, 2) if train_indices else np.zeros((0, 2), dtype=np.int64)
    idx_val = np.array(val_indices, dtype=np.int64).reshape(-1, 2) if val_indices else np.zeros((0, 2), dtype=np.int64)
    idx_test = np.array(test_indices, dtype=np.int64).reshape(-1, 2) if test_indices else np.zeros((0, 2), dtype=np.int64)

    np.save(idx_train_path, idx_train)
    np.save(idx_val_path, idx_val)
    np.save(idx_test_path, idx_test)

    elapsed_idx = time.time() - t1
    print(f"[DerivedCache] Index built in {elapsed_idx:.1f}s: "
          f"train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}")

    # ── 保存元信息 ──
    derived_meta = DerivedCacheMeta(
        cache_dir=cache_dir,
        x_derived_path=x_derived_path,
        raw_meta=raw_meta,
        standardize_method=standardize_method,
        idx_train_path=idx_train_path,
        idx_val_path=idx_val_path,
        idx_test_path=idx_test_path,
        train_windows=len(idx_train),
        val_windows=len(idx_val),
        test_windows=len(idx_test),
        fingerprint=fp,
    )

    meta_dict = {
        "x_derived_path": x_derived_path,
        "raw_cache_dir": raw_meta.cache_dir,
        "standardize_method": standardize_method,
        "lookback": lookback,
        "cross_date_windows": cross_date_windows,
        "idx_train_path": idx_train_path,
        "idx_val_path": idx_val_path,
        "idx_test_path": idx_test_path,
        "train_windows": len(idx_train),
        "val_windows": len(idx_val),
        "test_windows": len(idx_test),
        "fingerprint": fp,
        "build_time_sec": round(time.time() - t0, 1),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2, ensure_ascii=False)
    with open(fp_path, "w") as f:
        f.write(fp)

    return derived_meta


# ============================================================
# PyTorch Dataset
# ============================================================

class StockWindowDataset(Dataset):
    """从 derived cache 读取滑动窗口样本。

    每个样本 = (X, stock_id, y):
        X:   [lookback, n_features] float32  — 连续内存读取
        ids: [1] int64                       — stock integer id
        y:   [1] float32                     — label

    index 格式：[N, 2]，每行是 (stock_idx, end_step)。
    """

    def __init__(
        self,
        x_derived_path: str,
        labels_path: str,
        x_shape: tuple[int, int, int],    # (n_stock, n_step, n_feature)
        labels_shape: tuple[int, int],     # (n_stock, n_step)
        index_path: str,
        lookback: int,
    ):
        self.x_path = x_derived_path
        self.labels_path = labels_path
        self.x_shape = x_shape
        self.labels_shape = labels_shape
        self.lookback = lookback
        self.indices = np.load(index_path)   # [N, 2]

        # Lazy init for Windows spawn
        self._X: Optional[np.memmap] = None
        self._labels: Optional[np.memmap] = None

    def _ensure_mm(self):
        if self._X is None:
            self._X = np.memmap(self.x_path, dtype=np.float32, mode="r",
                                shape=self.x_shape)
            self._labels = np.memmap(self.labels_path, dtype=np.float32, mode="r",
                                     shape=self.labels_shape)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._ensure_mm()
        assert self._X is not None and self._labels is not None

        stock_idx, end_step = int(self.indices[idx, 0]), int(self.indices[idx, 1])
        start_step = end_step - self.lookback + 1

        # 连续内存读取：X[stock, start:end+1, :] 是 C-contiguous
        X = self._X[stock_idx, start_step:end_step + 1, :].copy()  # [lookback, F]
        y = np.array([self._labels[stock_idx, end_step]], dtype=np.float32)
        ids = np.array([stock_idx], dtype=np.int64)

        return X, ids, y


class StockWindowPredictionDataset(Dataset):
    """Like StockWindowDataset but also returns dateid for evaluation metrics."""

    def __init__(
        self,
        x_derived_path: str,
        labels_path: str,
        x_shape: tuple[int, int, int],
        labels_shape: tuple[int, int],
        index_path: str,
        lookback: int,
        n_time: int,
        date_list: list[int],
    ):
        self.x_path = x_derived_path
        self.labels_path = labels_path
        self.x_shape = x_shape
        self.labels_shape = labels_shape
        self.lookback = lookback
        self.n_time = n_time
        self.date_list = date_list
        self.indices = np.load(index_path)

        self._X: Optional[np.memmap] = None
        self._labels: Optional[np.memmap] = None

    def _ensure_mm(self):
        if self._X is None:
            self._X = np.memmap(self.x_path, dtype=np.float32, mode="r",
                                shape=self.x_shape)
            self._labels = np.memmap(self.labels_path, dtype=np.float32, mode="r",
                                     shape=self.labels_shape)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        self._ensure_mm()
        assert self._X is not None and self._labels is not None

        stock_idx, end_step = int(self.indices[idx, 0]), int(self.indices[idx, 1])
        start_step = end_step - self.lookback + 1

        X = self._X[stock_idx, start_step:end_step + 1, :].copy()
        y_val = float(self._labels[stock_idx, end_step])
        date_idx = end_step // self.n_time
        date_val = float(self.date_list[date_idx]) if date_idx < len(self.date_list) else 0.0

        return (
            X,
            np.array([stock_idx], dtype=np.int64),
            np.array([y_val], dtype=np.float32),
            date_val,
            float(stock_idx),
        )


# ============================================================
# Collate Functions
# ============================================================

def collate_fn(
    batch: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    Xb, idb, yb = zip(*batch)
    X = torch.from_numpy(np.stack(Xb, axis=0))
    ids = torch.from_numpy(np.concatenate(idb, axis=0))
    y = torch.from_numpy(np.stack(yb, axis=0))
    return X, ids, y


def pred_collate_fn(batch):
    Xb, idb, yb, dates, stocks = zip(*batch)
    X = torch.from_numpy(np.stack(Xb, axis=0))
    ids = torch.from_numpy(np.concatenate(idb, axis=0))
    y = torch.from_numpy(np.stack(yb, axis=0))
    return X, ids, y, np.array(dates), np.array(stocks)


# ============================================================
# Sampler — Memmap-Friendly Chunked Shuffle
# ============================================================

class ChunkedShuffleSampler(torch.utils.data.Sampler):
    """Memmap 友好的分块 shuffle 采样器。

    derived cache 的索引天然按 (stock_idx, end_step) 排序，对应 memmap 的
    物理布局 [n_stock, n_step, n_feature]。本采样器保持块内顺序（顺序 I/O），
    仅在块间 shuffle（保留训练随机性）。

    效果：
        - 块内连续读取 → OS page cache 预读命中，page fault 降低 10-50x
        - 块间 shuffle → 每个 epoch 顺序不同，训练仍有随机性
        - 适用于 MSE 等 sample-independent loss

    注意：
        对于 IC/RankIC 等截面损失函数，每个 batch 需要跨股票的多样性。
        此时应使用 shuffle_mode="full" 回退到标准随机 shuffle。
    """

    def __init__(self, num_samples: int, chunk_size: int = 2048, seed: int = 42):
        self._num_samples = num_samples
        self.chunk_size = chunk_size
        self.n_chunks = (num_samples + chunk_size - 1) // chunk_size
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        chunk_order = self.rng.permutation(self.n_chunks)
        for ci in chunk_order:
            start = int(ci) * self.chunk_size
            end = min(start + self.chunk_size, self._num_samples)
            yield from range(start, end)

    def __len__(self) -> int:
        return self._num_samples


# ============================================================
# DataLoader Factory
# ============================================================

def make_v2_dataloader(
    derived_meta: DerivedCacheMeta,
    split: str,
    lookback: int,
    batch_size: int = 512,
    shuffle: bool = True,
    num_workers: int = 2,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    shuffle_mode: str = "chunked",
) -> DataLoader:
    """创建 DataLoader。

    Args:
        derived_meta: DerivedCacheMeta
        split: "train" / "val" / "test"
        lookback: 窗口长度
        batch_size: 批大小
        shuffle: 是否打乱顺序（仅 train 需要）
        shuffle_mode: "chunked"（默认，memmap 友好分块 shuffle）或
                      "full"（标准随机 shuffle，适用于截面 loss 如 IC）
    """
    import sys

    raw = derived_meta.raw_meta
    index_path = {
        "train": derived_meta.idx_train_path,
        "val": derived_meta.idx_val_path,
        "test": derived_meta.idx_test_path,
    }[split]

    ds = StockWindowDataset(
        x_derived_path=derived_meta.x_derived_path,
        labels_path=raw.labels_path,
        x_shape=(raw.n_stock, raw.n_step, raw.n_feature),
        labels_shape=(raw.n_stock, raw.n_step),
        index_path=index_path,
        lookback=lookback,
    )

    if sys.platform == "win32" and num_workers > 4:
        num_workers = 4

    actual_pin = pin_memory and torch.cuda.is_available()

    # 选择采样策略
    sampler = None
    effective_shuffle = shuffle
    if shuffle and shuffle_mode == "chunked":
        sampler = ChunkedShuffleSampler(len(ds), chunk_size=batch_size)
        effective_shuffle = False  # DataLoader 要求 sampler 和 shuffle 不能同时启用

    if num_workers > 0:
        return DataLoader(
            ds, batch_size=batch_size, shuffle=effective_shuffle,
            sampler=sampler,
            num_workers=num_workers, collate_fn=collate_fn,
            pin_memory=actual_pin, persistent_workers=True,
            prefetch_factor=prefetch_factor, drop_last=False,
        )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=effective_shuffle,
        sampler=sampler,
        num_workers=0, collate_fn=collate_fn,
        pin_memory=actual_pin, persistent_workers=False,
        drop_last=False,
    )


def make_v2_pred_dataloader(
    derived_meta: DerivedCacheMeta,
    split: str,
    lookback: int,
    batch_size: int = 512,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """创建用于评估的 DataLoader（返回 dateid）。"""
    import sys

    raw = derived_meta.raw_meta
    index_path = {
        "train": derived_meta.idx_train_path,
        "val": derived_meta.idx_val_path,
        "test": derived_meta.idx_test_path,
    }[split]

    ds = StockWindowPredictionDataset(
        x_derived_path=derived_meta.x_derived_path,
        labels_path=raw.labels_path,
        x_shape=(raw.n_stock, raw.n_step, raw.n_feature),
        labels_shape=(raw.n_stock, raw.n_step),
        index_path=index_path,
        lookback=lookback,
        n_time=raw.n_time,
        date_list=raw.date_list,
    )

    if sys.platform == "win32" and num_workers > 4:
        num_workers = 4

    actual_pin = pin_memory and torch.cuda.is_available()

    if num_workers > 0:
        return DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=pred_collate_fn,
            pin_memory=actual_pin, persistent_workers=True,
        )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=pred_collate_fn,
        pin_memory=actual_pin, persistent_workers=False,
    )
