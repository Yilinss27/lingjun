"""Unified prediction runner for persisted GBDT and NN models."""

# pyright: reportMissingImports=false, reportMissingTypeArgument=false, reportArgumentType=false
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, cast
import os
import sys

_TEMPLATE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_TEMPLATE_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _TEMPLATE_DIR)

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn


from template.data.io_parquet import iter_test_batches
from template.data.parquet_adapter import LargeParquetConfig
from template.models.gbdt.lgbm_model import LGBMModelWrapper
from template.models.gbdt.xgb_model import XGBModelWrapper
from template.configs.model_configs import nn_config_from_meta
from template.models import build_nn_model

# 直接运行本脚本时使用的默认参数，请按需修改 model_path / meta_path
DEFAULT_PREDICT = {
    "family": "nn",
    "model_path": "",
    "meta_path": "",
    "output_path": "res/pipeline/predict_output.parquet",
    "test_parquet": "",
}


def _load_wrapper(family: str, model_path: str):
    if family == "lgbm":
        return LGBMModelWrapper.load(model_path)
    return XGBModelWrapper.load(model_path)


def _make_composite_key(df: pd.DataFrame, key_cols: list[str]) -> pd.Series:
    """Join *key_cols* values with ``|`` to form a single composite key column."""
    return df[key_cols].astype(str).agg("|".join, axis=1)


# ============================================================
# GBDT Prediction (unchanged)
# ============================================================

def _predict_gbdt(
    family: str,
    model_path: str,
    meta: dict,
    output_path: str,
    test_parquet: str,
) -> None:
    from template.data.preprocess import PreprocessorState, transform_features

    def _state_from_dict(raw: dict) -> PreprocessorState:
        return PreprocessorState(
            fillna_strategy=str(raw.get("fillna_strategy", "none")),
            standardize=bool(raw.get("standardize", False)),
            clip_quantile=raw.get("clip_quantile"),
            medians={k: float(v) for k, v in raw.get("medians", {}).items()},
            means={k: float(v) for k, v in raw.get("means", {}).items()},
            stds={k: float(v) for k, v in raw.get("stds", {}).items()},
            clip_ranges={k: (float(v[0]), float(v[1])) for k, v in raw.get("clip_ranges", {}).items()},
            standardize_mode=str(raw.get("standardize_mode", "none")),
            date_col=str(raw.get("date_col", "dateid")),
            time_col=str(raw.get("time_col", "timeid")),
            z_window=int(raw.get("z_window", 60)),
        )

    feature_cols = list(meta["feature_cols"])
    data_cfg = dict(meta["data_config"])
    preprocess_state = _state_from_dict(dict(meta["preprocess_state"]))

    if test_parquet:
        data_cfg["test_parquet_path"] = test_parquet
    test_parquet_path = str(data_cfg.get("test_parquet_path", ""))
    if not test_parquet_path or not os.path.exists(test_parquet_path):
        raise FileNotFoundError(f"test parquet not found: {test_parquet_path}")

    large_cfg = LargeParquetConfig(
        train_path=test_parquet_path,
        test_path=test_parquet_path,
        label_col=str(data_cfg["label_col"]),
        id_cols=tuple(data_cfg["id_cols"]),
        date_col=str(data_cfg["date_col"]),
        feature_mode=cast(Literal["all", "no_missing", "top_ic", "custom"], str(data_cfg["feature_mode"])),
        custom_features=list(data_cfg.get("custom_features", [])),
        train_ratio=float(data_cfg["train_ratio"]),
        val_ratio=float(data_cfg["val_ratio"]),
        max_train_rows=None,
        max_test_rows=data_cfg.get("max_test_rows"),
        batch_size=int(data_cfg["batch_size"]),
        row_group_stride=int(data_cfg["row_group_stride"]),
        row_group_offset=int(data_cfg["row_group_offset"]),
    )

    wrapper = _load_wrapper(family, model_path)
    key_cols = ["stockid", "dateid", "timeid"]

    writer: pq.ParquetWriter | None = None
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        for batch in iter_test_batches(large_cfg, feature_cols, batch_size=int(large_cfg.batch_size)):
            batch = transform_features(batch, feature_cols, preprocess_state, copy_frame=False)
            preds = wrapper.predict(batch[feature_cols].to_numpy(dtype=np.float32, copy=False))
            avail_keys = [c for c in key_cols if c in batch.columns]
            out = pd.DataFrame({
                "stockid|dateid|timeid": _make_composite_key(batch, avail_keys),
                "prediction": np.asarray(preds, dtype=np.float32).reshape(-1),
            })
            table = pa.Table.from_pandas(out, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(str(out_file), table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    print(f"[Done] GBDT prediction saved to {out_file}")


# ============================================================
# NN Prediction — V2 Memmap Pipeline
# ============================================================

def _predict_nn(
    family: str,
    model_path: str,
    meta: dict,
    output_path: str,
    test_parquet: str,
) -> None:
    """V2 NN 预测：使用 memmap cache 管线确保与训练一致的预处理。

    流程：
        1. 从 meta 恢复训练配置（特征列、fillna 中位数、标准化方法、ID 映射）
        2. 为 test parquet 构建 raw cache（填充缺失值、prediction_mode）
        3. 构建 derived cache（RankGauss 截面标准化）
        4. 用 DataLoader 批量推理
        5. 输出 predictions parquet
    """
    from template.data.memmap_cache import (
        build_raw_cache, build_derived_cache, try_load_raw_cache,
        make_v2_pred_dataloader,
    )
    from template.data.preprocess_nn import get_standardizer

    # ── 1. 恢复训练配置 ──
    feature_cols = list(meta["feature_cols"])
    data_cfg = dict(meta["data_config"])
    nn_cfg_raw = dict(meta.get("nn_config", {}))
    nn_cfg = nn_config_from_meta(nn_cfg_raw)
    fillna_medians = {k: float(v) for k, v in meta["fillna_medians"].items()}
    std_method = str(meta.get("standardize_method", "rank_gauss"))
    training_id_map = {int(k): int(v) for k, v in meta.get("id_map", {}).items()}
    num_ids = int(meta.get("num_ids", len(training_id_map)))

    # 确定 test parquet 路径
    if test_parquet:
        test_path = test_parquet
    else:
        test_path = str(data_cfg.get("test_parquet_path", "data/test.parquet"))
    if not test_path or not os.path.exists(test_path):
        raise FileNotFoundError(f"test parquet not found: {test_path}")

    label_col = str(data_cfg.get("label_col", "LabelA"))
    id_col = str(data_cfg.get("id_cols", ["stockid"])[0])
    date_col = str(data_cfg.get("date_col", "dateid"))
    time_col = "timeid"

    print(f"[Predict] V2 NN prediction: model={nn_cfg.model_name}, test={test_path}")
    print(f"[Predict] features={len(feature_cols)}, lookback={nn_cfg.lookback}, "
          f"standardize={std_method}")

    # ── 2. 构建 raw cache ──
    predict_raw_dir = "cache/predict_raw"
    print(f"[Predict] Building raw cache for test data...")
    raw_meta = build_raw_cache(
        parquet_path=test_path,
        feature_cols=feature_cols,
        label_col=label_col,
        id_col=id_col,
        date_col=date_col,
        time_col=time_col,
        fillna_medians=fillna_medians,
        train_end=-1,       # 所有数据都归入 "test" split
        val_end=-1,
        cache_dir=predict_raw_dir,
        prediction_mode=True,
    )
    n_stock = raw_meta.n_stock
    n_date = raw_meta.n_date
    n_time = raw_meta.n_time
    n_step = raw_meta.n_step
    date_list = raw_meta.date_list
    print(f"[Predict] Raw cache: {n_stock} stocks x {n_date} dates x {n_time} times")

    # ── 3. 构建 derived cache（跨日窗口 = 全覆盖）──
    predict_derived_dir = "cache/predict_derived"
    standardizer = get_standardizer(std_method)
    print(f"[Predict] Building derived cache ({std_method}, cross_date_windows=True)...")
    derived_meta = build_derived_cache(
        raw_meta=raw_meta,
        standardizer=standardizer,
        standardize_method=std_method,
        lookback=nn_cfg.lookback,
        cross_date_windows=True,     # 跨日窗口：允许用前一天数据填充 lookback
        cache_dir=predict_derived_dir,
    )
    print(f"[Predict] Derived cache: {derived_meta.test_windows} prediction windows "
          f"(full grid = {n_stock * n_step})")

    if derived_meta.test_windows == 0:
        print("[Predict] WARNING: 0 prediction windows. Check lookback vs n_time.")
        return

    # ── 4. 加载模型 ──
    num_features = len(feature_cols)
    model = build_nn_model(nn_cfg.model_name, nn_cfg, num_ids, num_features)
    model = cast(nn.Module, model)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"[Predict] Model loaded on {device}")

    # ── 5. 构建 stock ID 映射 ──
    pred_idx_to_stockid = {v: k for k, v in raw_meta.stock_id_map.items()}
    stockid_to_training_idx = training_id_map

    # ── 6. Stock-sequential 推理 ──
    # 绕过 DataLoader：逐股票顺序读取 memmap → RAM，然后从 RAM 构建窗口批量推理
    # 优势：每只股票 ~35MB 一次顺序读取，不再有 page fault 问题
    test_indices = np.load(derived_meta.idx_test_path)  # [N, 2]: (stock_idx, end_step)
    cache_stock_indices = test_indices[:, 0]
    end_steps = test_indices[:, 1]

    # 预计算 per-stock 边界（indices 已按 stock_idx 排序）
    stock_boundaries = np.searchsorted(cache_stock_indices, np.arange(n_stock + 1))

    # 预计算 training ids
    training_ids_all = np.array([
        stockid_to_training_idx.get(pred_idx_to_stockid.get(int(si), -1), 0)
        for si in cache_stock_indices
    ], dtype=np.int64)

    infer_batch_size = max(nn_cfg.batch_size, 8192)

    # AMP 推理
    use_amp = nn_cfg.use_amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if getattr(nn_cfg, "amp_dtype", "float16") == "bfloat16" else torch.float16
    amp_label = "bfloat16" if use_amp and amp_dtype == torch.bfloat16 else "float16" if use_amp else "off"

    print(f"[Predict] Stock-sequential inference: {len(test_indices)} windows, "
          f"{n_stock} stocks (batch={infer_batch_size}, amp={amp_label})")

    X_derived = np.memmap(derived_meta.x_derived_path, dtype=np.float32, mode="r",
                          shape=(n_stock, n_step, num_features))

    preds_arr = np.empty(len(test_indices), dtype=np.float32)
    lookback = nn_cfg.lookback
    import time as _time
    t_infer = _time.time()

    with torch.no_grad():
        for s_idx in range(n_stock):
            s_start = int(stock_boundaries[s_idx])
            s_end = int(stock_boundaries[s_idx + 1])
            n_win = s_end - s_start
            if n_win == 0:
                continue

            # 一次顺序读取整只股票的 derived 数据到 RAM (~35MB)
            stock_data = np.array(X_derived[s_idx])  # [n_step, F]
            stock_end_steps = end_steps[s_start:s_end]
            stock_t_ids = training_ids_all[s_start:s_end]

            # 分批构建窗口并推理
            for b_start in range(0, n_win, infer_batch_size):
                b_end = min(b_start + infer_batch_size, n_win)
                bs = b_end - b_start

                # 从 RAM 构建窗口（无 memmap 访问）
                batch_ends = stock_end_steps[b_start:b_end]
                windows = np.empty((bs, lookback, num_features), dtype=np.float32)
                for j in range(bs):
                    es = int(batch_ends[j])
                    windows[j] = stock_data[es - lookback + 1: es + 1]

                X_batch = torch.from_numpy(windows).to(device, non_blocking=True)
                ids_batch = torch.from_numpy(stock_t_ids[b_start:b_end].copy()).to(device, non_blocking=True)

                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    pred = model(X_batch, ids_batch)
                preds_arr[s_start + b_start: s_start + b_end] = pred.detach().float().cpu().numpy().reshape(-1)

            if (s_idx + 1) % 100 == 0 or s_idx + 1 == n_stock:
                elapsed = _time.time() - t_infer
                eta = elapsed / (s_idx + 1) * (n_stock - s_idx - 1)
                print(f"[Predict]   {s_idx + 1}/{n_stock} stocks, "
                      f"elapsed={elapsed:.0f}s, ETA~{eta:.0f}s")

    del X_derived
    print(f"[Predict] Inference done: {len(preds_arr)} predictions in {_time.time()-t_infer:.1f}s")

    # ── 7. 构建全量输出网格 ──
    pred_grid = np.zeros((n_stock, n_step), dtype=np.float32)
    pred_grid[cache_stock_indices, end_steps] = preds_arr

    # ── 7b. Warmup 位置：用第一个有效预测值向前填充 ──
    # cross_date_windows=True 已覆盖 day 1+ 的所有 timeid
    # 仅剩第一天的前 lookback-1 个 time step 没有预测值
    # 策略：每只股票用 timeid=lookback-1（第一个完整窗口）的预测值填充前面所有位置
    lookback = nn_cfg.lookback
    warmup_steps = min(lookback - 1, n_step)  # 通常 = 31

    if warmup_steps > 0:
        first_valid_step = lookback - 1  # 第一个有完整窗口的 step
        for s_idx in range(n_stock):
            fill_val = pred_grid[s_idx, first_valid_step]
            pred_grid[s_idx, :warmup_steps] = fill_val

        print(f"[Predict] Warmup: {n_stock * warmup_steps} positions filled "
              f"(each stock's timeid 0~{warmup_steps-1} ← timeid {first_valid_step}'s prediction)")

    total = n_stock * n_step
    print(f"[Predict] Full grid: {total} cells "
          f"({len(preds_arr)} from DataLoader, "
          f"{n_stock * warmup_steps} forward-filled)")

    # ── 8. 向量化生成全量输出 ──
    stockid_arr = np.array([pred_idx_to_stockid[i] for i in range(n_stock)], dtype=np.int64)

    stock_flat = np.repeat(stockid_arr, n_step)
    step_flat = np.tile(np.arange(n_step), n_stock)
    date_flat = np.array(date_list, dtype=np.int64)[step_flat // n_time]
    time_flat = (step_flat % n_time).astype(np.int64)
    pred_flat = pred_grid.reshape(-1)

    out_df = pd.DataFrame({
        "stockid": stock_flat,
        "dateid": date_flat,
        "timeid": time_flat,
        "prediction": pred_flat,
    })
    out_df["Uid"] = (out_df["stockid"].astype(str) + "|"
                     + out_df["dateid"].astype(str) + "|"
                     + out_df["timeid"].astype(str))

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # 根据扩展名自动选择格式
    submission = out_df[["Uid", "prediction"]]
    if out_file.suffix == ".csv":
        submission.to_csv(str(out_file), index=False)
    else:
        submission.to_parquet(str(out_file), index=False, compression="zstd")
    print(f"[Done] Submission saved to {out_file} ({len(submission)} rows)")


# ============================================================
# Entry Point
# ============================================================

def run_predict(
    family: str,
    model_path: str,
    meta_path: str,
    output_path: str = "res/pipeline/predict_output.parquet",
    test_parquet: str = "",
) -> None:
    """Run prediction: load meta from meta_path, then run GBDT or NN prediction."""
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if family in ("lgbm", "xgb"):
        _predict_gbdt(family, model_path, meta, output_path, test_parquet)
    else:
        _predict_nn(family, model_path, meta, output_path, test_parquet)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Run prediction with trained model")
    parser.add_argument("--family", type=str, default=None,
                        help="Model family: lgbm, xgb, nn (auto-detected from meta if omitted)")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model file (model.pt or model.txt)")
    parser.add_argument("--meta", type=str, default=None,
                        help="Path to meta.json")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Run directory (auto-finds model.pt and meta.json)")
    parser.add_argument("--test-parquet", type=str, default="",
                        help="Override test parquet path")
    parser.add_argument("--output", type=str, default="",
                        help="Output parquet path")
    args = parser.parse_args()

    # 从 run-dir 自动推断 model 和 meta 路径
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not args.model:
            args.model = str(run_dir / "model.pt")
        if not args.meta:
            args.meta = str(run_dir / "meta.json")

    if not args.model or not args.meta:
        # Fallback: 使用 DEFAULT_PREDICT
        opts = DEFAULT_PREDICT
        if not opts["model_path"] or not opts["meta_path"]:
            print("Usage: python predict.py --run-dir <path/to/run_dir> [--test-parquet <path>]")
            print("   or: python predict.py --model <model.pt> --meta <meta.json>")
            return
        args.model = opts["model_path"]
        args.meta = opts["meta_path"]
        if not args.family:
            args.family = opts["family"]

    # 自动检测 family
    if not args.family:
        with open(args.meta, "r", encoding="utf-8") as f:
            meta = json.load(f)
        args.family = str(meta.get("family", "nn"))

    # 自动生成输出路径（默认 parquet，提交时用 --output submission.csv）
    if not args.output:
        run_dir = Path(args.model).parent
        args.output = str(run_dir / "submission.parquet")

    run_predict(
        family=args.family,
        model_path=args.model,
        meta_path=args.meta,
        output_path=args.output,
        test_parquet=args.test_parquet,
    )


if __name__ == "__main__":
    main()
