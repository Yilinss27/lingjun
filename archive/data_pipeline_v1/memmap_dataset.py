"""
Memory-mapped dataset for large-scale NN training.

Core idea:
    1. One-pass preprocessing writes ALL rows (preprocessed features + labels + metadata)
       into a flat memory-mapped file, contiguous per stock and sorted by (dateid, timeid).
    2. A lightweight index records valid sliding-window start positions per split.
    3. A PyTorch Dataset reads windows lazily via memmap — O(1) random access,
       ~10KB per read, OS-managed paging keeps RAM usage at ~2-4GB.

This replaces build_nn_samples_streaming() which materialised ALL windows in RAM (~450GB
for full data — completely infeasible).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from template.data.preprocess import PreprocessorState, transform_features


# ============================================================
# Config & Helpers
# ============================================================

@dataclass
class MemmapCacheConfig:
    """Configuration for the memmap cache."""
    cache_dir: str = "cache/memmap"
    lookback: int = 32
    # If True, windows are allowed to span overnight boundaries (date changes).
    # For financial data this is usually False.
    cross_date_windows: bool = False


def _cache_key(
    parquet_path: str,
    feature_cols: Sequence[str],
    label_col: str,
    lookback: int,
    cross_date: bool,
    prep_state_hash: str,
) -> str:
    """Generate a deterministic cache key from config."""
    h = hashlib.sha256()
    h.update(parquet_path.encode())
    h.update(",".join(feature_cols).encode())
    h.update(label_col.encode())
    h.update(str(lookback).encode())
    h.update(str(cross_date).encode())
    h.update(prep_state_hash.encode())
    return h.hexdigest()[:16]


def _prep_state_hash(state: PreprocessorState) -> str:
    """Hash of preprocessor state for cache invalidation."""
    h = hashlib.sha256()
    h.update(state.fillna_strategy.encode())
    h.update(str(state.standardize).encode())
    h.update(state.standardize_mode.encode())
    h.update(str(state.clip_quantile).encode())
    # Include mean/std summaries
    if state.means:
        vals = sorted(state.means.items())
        h.update(str(vals[:5]).encode())
    if state.stds:
        vals = sorted(state.stds.items())
        h.update(str(vals[:5]).encode())
    return h.hexdigest()[:12]


# ============================================================
# Memmap Builder
# ============================================================

def build_memmap_cache(
    parquet_path: str,
    feature_cols: list[str],
    label_col: str,
    id_col: str,
    date_col: str,
    time_col: str,
    prep_state: PreprocessorState,
    lookback: int,
    train_end: object,
    val_end: object,
    cache_dir: str = "cache/memmap",
    cross_date_windows: bool = False,
    on_progress: Optional[Callable[[int, int, int], None]] = None,
) -> dict:
    """Build a memory-mapped preprocessed data file and window indices.

    Args:
        parquet_path: Path to the parquet file (e.g. train.parquet).
        feature_cols: List of feature column names.
        label_col: Label column name.
        id_col: Stock ID column name (e.g. 'stockid').
        date_col: Date column name.
        time_col: Time column name.
        prep_state: Fitted PreprocessorState from streaming pass.
        lookback: Sliding window length.
        train_end: Last date for training split.
        val_end: Last date for validation split.
        cache_dir: Directory to store memmap and index files.
        cross_date_windows: Whether windows can span dates.
        on_progress: Callback (row_group_idx, total_row_groups, rows_written).

    Returns:
        dict with keys: mm_path, mm_shape, n_features, n_stocks,
                        idx_train_path, idx_val_path, idx_test_path,
                        stock_id_map, meta_path
    """
    os.makedirs(cache_dir, exist_ok=True)

    # ── Cache hit detection ──────────────────────────────────────────────────
    # Compute a fingerprint of the current build config.
    # If it matches the stored fingerprint, skip the expensive rebuild.
    ph = _prep_state_hash(prep_state)
    fingerprint = _cache_key(
        parquet_path=parquet_path,
        feature_cols=feature_cols,
        label_col=label_col,
        lookback=lookback,
        cross_date=cross_date_windows,
        prep_state_hash=ph,
    )
    # Also encode split boundaries into fingerprint
    fingerprint_full = hashlib.sha256(
        (fingerprint + str(train_end) + str(val_end)).encode()
    ).hexdigest()[:16]

    meta_path = os.path.join(cache_dir, "meta.json")
    fp_path = os.path.join(cache_dir, "fingerprint.txt")

    if os.path.exists(fp_path) and os.path.exists(meta_path):
        stored_fp = open(fp_path, "r").read().strip()
        if stored_fp == fingerprint_full:
            with open(meta_path, "r", encoding="utf-8") as f:
                cached_meta = json.load(f)
            # Verify that all referenced files actually exist
            all_exist = all(
                os.path.exists(cached_meta.get(k, ""))
                for k in ("mm_path", "idx_train_path", "idx_val_path", "idx_test_path")
            )
            if all_exist:
                print(f"[Memmap] Cache hit (fingerprint={fingerprint_full}), skipping rebuild.")
                return cached_meta
            else:
                print(f"[Memmap] Cache fingerprint matches but files missing, rebuilding...")
        else:
            print(f"[Memmap] Config changed (old={stored_fp[:8]}, new={fingerprint_full[:8]}), rebuilding...")
    # ────────────────────────────────────────────────────────────────────────

    pf = pq.ParquetFile(parquet_path)
    n_row_groups = pf.num_row_groups
    total_rows = pf.metadata.num_rows
    n_features = len(feature_cols)

    # Memmap layout per row: [features(n_features) | label(1) | stock_id_int(1) | dateid(1) | timeid(1)]
    # Total columns = n_features + 4  (added timeid for cross-section standardization)
    n_cols = n_features + 4
    mm_path = os.path.join(cache_dir, "preprocessed.dat")

    print(f"[Memmap] Building cache: {total_rows} rows, {n_features} features, "
          f"{n_row_groups} stocks → {mm_path}")
    print(f"[Memmap] Initializing memory-mapped file (~{total_rows * n_cols * 4 / (1024**3):.1f} GB)...")
    t0 = time.time()

    # Create memmap file incrementally using binary file I/O (faster on Windows)
    # Step 1: Pre-allocate file on disk without loading everything into memory
    with open(mm_path, 'wb') as f:
        f.write(np.float32(0))  # Write a single zero
        f.seek(total_rows * n_cols * 4 - 1)  # Jump to end
        f.write(b'\0')  # Finalize file size
    print(f"[Memmap] File pre-allocated in {time.time() - t0:.1f}s")
    
    # Step 2: Open as memmap
    mm = np.memmap(mm_path, dtype=np.float32, mode="r+", shape=(total_rows, n_cols))

    # Track per-stock info for index building
    # Each row group = one stock; row groups are sorted by stockid
    stock_info: list[dict] = []  # [{stock_id, start_row, n_rows, dateids}]
    stock_id_map: dict[int, int] = {}  # raw_stockid -> integer index
    row_offset = 0

    read_cols = [id_col, date_col, time_col] + feature_cols + [label_col]

    for rg_idx in range(n_row_groups):
        table = pf.read_row_group(rg_idx, columns=read_cols)
        df = table.to_pandas()

        # Sort within stock by (dateid, timeid)
        df = df.sort_values([date_col, time_col]).reset_index(drop=True)

        # Get stock ID (should be uniform within a row group)
        raw_stock_id = int(df[id_col].iloc[0])
        if raw_stock_id not in stock_id_map:
            stock_id_map[raw_stock_id] = len(stock_id_map)
        int_stock_id = stock_id_map[raw_stock_id]

        # Apply preprocessing (fillna, clip only — NO standardization).
        # 
        # Rationale: We want to decouple standardization from memmap building
        # to support flexible standardization strategies (cross-section, rank-gauss, etc)
        # that are applied dynamically in DataLoader, not baked into memmap.
        #
        # Temporarily disable standardization for memmap building:
        _orig_standardize = prep_state.standardize
        _orig_mode = prep_state.standardize_mode
        prep_state.standardize = False
        prep_state.standardize_mode = "none"
        
        df = transform_features(df, feature_cols, prep_state, copy_frame=False)
        
        prep_state.standardize = _orig_standardize  # restore
        prep_state.standardize_mode = _orig_mode

        # Extract arrays
        feats = df[feature_cols].to_numpy(dtype=np.float32)
        labels = df[label_col].to_numpy(dtype=np.float32).reshape(-1, 1)
        dateids = df[date_col].to_numpy()
        timeids = df[time_col].to_numpy()  # ← New: extract timeid
        stock_ids_col = np.full((len(df), 1), int_stock_id, dtype=np.float32)
        dateids_f32 = dateids.astype(np.float32).reshape(-1, 1)
        timeids_f32 = timeids.astype(np.float32).reshape(-1, 1)  # ← New: convert to float32

        # Fill any remaining NaN in features with 0
        np.nan_to_num(feats, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # Write to memmap: [features | label | stock_id | dateid | timeid]
        n_rows = len(df)
        mm[row_offset:row_offset + n_rows, :n_features] = feats
        mm[row_offset:row_offset + n_rows, n_features] = labels.ravel()
        mm[row_offset:row_offset + n_rows, n_features + 1] = stock_ids_col.ravel()
        mm[row_offset:row_offset + n_rows, n_features + 2] = dateids_f32.ravel()
        mm[row_offset:row_offset + n_rows, n_features + 3] = timeids_f32.ravel()  # ← New: store timeid

        stock_info.append({
            "stock_id": int_stock_id,
            "raw_stock_id": raw_stock_id,
            "start_row": row_offset,
            "n_rows": n_rows,
            "dateids": dateids,  # keep as numpy array (avoid slow .tolist())
        })

        row_offset += n_rows

        # Progress: time-based (every 30s) + milestone-based (every 5%)
        elapsed = time.time() - t0
        pct = (rg_idx + 1) / n_row_groups
        _last_print = getattr(build_memmap_cache, '_last_print', 0.0)
        if elapsed - _last_print >= 30 or (rg_idx + 1) % max(1, n_row_groups // 20) == 0 or rg_idx + 1 == n_row_groups:
            eta = elapsed / pct - elapsed if pct > 0 else 0
            print(f"[Memmap] Pass3 write: {rg_idx + 1}/{n_row_groups} stocks "
                  f"({100*pct:.0f}%), rows={row_offset/1e6:.1f}M, "
                  f"elapsed={elapsed:.0f}s, ETA~{eta:.0f}s", flush=True)
            build_memmap_cache._last_print = elapsed  # type: ignore[attr-defined]

        if on_progress is not None:
            on_progress(rg_idx + 1, n_row_groups, row_offset)

    mm.flush()
    del mm

    # ── Build window indices ──
    print(f"[Memmap] Data written in {time.time() - t0:.1f}s. Building window indices...")
    t1 = time.time()

    train_indices = []
    val_indices = []
    test_indices = []

    for info in stock_info:
        start = info["start_row"]
        dateids = np.asarray(info["dateids"])  # already numpy from build loop
        n = info["n_rows"]

        if cross_date_windows:
            # Allow windows to span dates
            for w in range(n - lookback + 1):
                last_date = dateids[w + lookback - 1]
                # Check label is valid (NaN check done later in __getitem__ if needed)
                if last_date <= train_end:
                    train_indices.append(start + w)
                elif last_date <= val_end:
                    val_indices.append(start + w)
                else:
                    test_indices.append(start + w)
        else:
            # Windows must stay within same date
            unique_dates = np.unique(dateids)
            for d in unique_dates:
                day_mask = dateids == d
                day_positions = np.where(day_mask)[0]
                n_day = len(day_positions)
                if n_day < lookback:
                    continue

                # Verify contiguity (rows within a date should be consecutive)
                assert np.all(np.diff(day_positions) == 1), \
                    f"Non-contiguous rows for stock {info['raw_stock_id']}, date {d}"

                day_start = day_positions[0]
                for w in range(n_day - lookback + 1):
                    global_start = start + day_start + w
                    if d <= train_end:
                        train_indices.append(global_start)
                    elif d <= val_end:
                        val_indices.append(global_start)
                    else:
                        test_indices.append(global_start)

    # Save indices
    idx_train = np.array(train_indices, dtype=np.int64)
    idx_val = np.array(val_indices, dtype=np.int64)
    idx_test = np.array(test_indices, dtype=np.int64)

    idx_train_path = os.path.join(cache_dir, "idx_train.npy")
    idx_val_path = os.path.join(cache_dir, "idx_val.npy")
    idx_test_path = os.path.join(cache_dir, "idx_test.npy")

    np.save(idx_train_path, idx_train)
    np.save(idx_val_path, idx_val)
    np.save(idx_test_path, idx_test)

    elapsed = time.time() - t0
    print(f"[Memmap] Index built in {time.time() - t1:.1f}s. "
          f"Train={len(idx_train):,}, Val={len(idx_val):,}, Test={len(idx_test):,}")
    print(f"[Memmap] Total cache build: {elapsed:.1f}s")

    # Save metadata
    meta = {
        "mm_path": mm_path,
        "mm_shape": [total_rows, n_cols],
        "n_features": n_features,
        "n_stocks": len(stock_id_map),
        "lookback": lookback,
        "label_col": label_col,
        "feature_cols": feature_cols,
        "stock_id_map": {str(k): v for k, v in stock_id_map.items()},
        "idx_train_path": idx_train_path,
        "idx_val_path": idx_val_path,
        "idx_test_path": idx_test_path,
        "cross_date_windows": cross_date_windows,
        "total_rows": total_rows,
        "train_windows": len(idx_train),
        "val_windows": len(idx_val),
        "test_windows": len(idx_test),
        "build_time_sec": round(elapsed, 1),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # Write fingerprint so next run can skip rebuild
    with open(fp_path, "w") as f:
        f.write(fingerprint_full)

    return meta


# ============================================================
# PyTorch Dataset — lazy memmap window reader
# ============================================================

class MemmapWindowDataset(Dataset):
    """Lazy sliding-window dataset backed by memory-mapped file with optional standardization.

    Each __getitem__ reads a contiguous (lookback, n_features) window
    from the memmap file — ~10KB per read, O(1) random access.

    Memory usage: ~150MB for index + OS-managed page cache for memmap.
    Compatible with DataLoader(num_workers>0) on Windows (spawn).
    
    Optional: Apply standardization dynamically (e.g., rank-gauss per cross-section).
    """

    def __init__(
        self,
        mm_path: str,
        mm_shape: tuple[int, int],
        index_path: str,
        lookback: int,
        n_features: int,
        standardizer: Optional[object] = None,
    ):
        # np.memmap is lazy — opening is instant, pages faulted on access
        # Each spawned worker on Windows will create its own memmap handle
        self.mm_path = mm_path
        self.mm_shape = mm_shape
        self.mm: Optional[np.memmap] = None  # Lazy init for Windows spawn compatibility
        self.indices = np.load(index_path)  # int64 array of window start positions
        self.lookback = lookback
        self.nf = n_features
        self.standardizer = standardizer  # Optional RankGaussTransformer or similar

    def _ensure_mm(self) -> np.memmap:
        """Lazy memmap initialization — safe for Windows spawn workers."""
        if self.mm is None:
            self.mm = np.memmap(
                self.mm_path, dtype=np.float32, mode="r",
                shape=tuple(self.mm_shape),
            )
        return self.mm

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mm = self._ensure_mm()
        start = int(self.indices[idx])
        # Read lookback contiguous rows — great cache locality
        window = mm[start:start + self.lookback].copy()  # .copy() detaches from mmap

        X = window[:, :self.nf]                              # [lookback, n_features]
        y = window[-1, self.nf:self.nf + 1]                  # [1,] label
        stock_id = int(window[-1, self.nf + 1])               # scalar
        dateid = int(window[-1, self.nf + 2])                 # scalar ← New: dateid
        timeid = int(window[-1, self.nf + 3])                 # scalar ← New: timeid
        
        # Optional: Apply standardization (e.g., rank-gauss)
        if self.standardizer is not None:
            X = self.standardizer.transform(X, dateid=dateid, timeid=timeid)
        
        return X, np.array([stock_id], dtype=np.int64), y.astype(np.float32), np.array([dateid, timeid], dtype=np.int64)


def memmap_collate_fn(
    batch: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for MemmapWindowDataset.
    
    Note: Ignores dateid/timeid (already used for standardization in __getitem__).
    """
    Xb, idb, yb, _ = zip(*batch)  # ← ignore cross-section ids
    X = torch.from_numpy(np.stack(Xb, axis=0))       # [B, lookback, n_features]
    ids = torch.from_numpy(np.concatenate(idb, axis=0))  # [B,]
    y = torch.from_numpy(np.stack(yb, axis=0))        # [B, 1]
    return X, ids, y


# ============================================================
# Evaluation helper — extract predictions with IDs for metrics
# ============================================================

class MemmapPredictionDataset(Dataset):
    """Like MemmapWindowDataset but also returns dateid for metric grouping."""

    def __init__(
        self,
        mm_path: str,
        mm_shape: tuple[int, int],
        index_path: str,
        lookback: int,
        n_features: int,
    ):
        self.mm_path = mm_path
        self.mm_shape = mm_shape
        self.mm: Optional[np.memmap] = None
        self.indices = np.load(index_path)
        self.lookback = lookback
        self.nf = n_features

    def _ensure_mm(self) -> np.memmap:
        if self.mm is None:
            self.mm = np.memmap(
                self.mm_path, dtype=np.float32, mode="r",
                shape=tuple(self.mm_shape),
            )
        return self.mm

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        mm = self._ensure_mm()
        start = int(self.indices[idx])
        window = mm[start:start + self.lookback].copy()

        X = window[:, :self.nf]
        y_val = float(window[-1, self.nf])
        stock_id = int(window[-1, self.nf + 1])
        date_val = float(window[-1, self.nf + 2])

        return X, np.array([stock_id], dtype=np.int64), np.array([y_val], dtype=np.float32), date_val, float(stock_id)


def pred_collate_fn(batch):
    """Collate for prediction dataset (includes date and stock info)."""
    Xb, idb, yb, dates, stocks = zip(*batch)
    X = torch.from_numpy(np.stack(Xb, axis=0))
    ids = torch.from_numpy(np.concatenate(idb, axis=0))
    y = torch.from_numpy(np.stack(yb, axis=0))
    return X, ids, y, np.array(dates), np.array(stocks)


# ============================================================
# DataLoader factory
# ============================================================

def make_memmap_dataloader(
    mm_path: str,
    mm_shape: tuple[int, int],
    index_path: str,
    lookback: int,
    n_features: int,
    batch_size: int = 512,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 4,
    standardizer: Optional[object] = None,
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader backed by memmap dataset.
    
    Args:
        standardizer: Optional RankGaussTransformer or similar for dynamic standardization.
    """
    import sys
    from torch.utils.data import DataLoader

    ds = MemmapWindowDataset(
        mm_path=mm_path,
        mm_shape=mm_shape,
        index_path=index_path,
        lookback=lookback,
        n_features=n_features,
        standardizer=standardizer,
    )

    # Limit workers on Windows
    if sys.platform == "win32" and num_workers > 4:
        num_workers = 4

    actual_pin = pin_memory and torch.cuda.is_available()

    if num_workers > 0:
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=memmap_collate_fn,
            pin_memory=actual_pin,
            persistent_workers=True,
            prefetch_factor=prefetch_factor,
            drop_last=False,
        )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=memmap_collate_fn,
        pin_memory=actual_pin,
        persistent_workers=False,
        drop_last=False,
    )
