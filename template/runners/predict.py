"""Unified prediction runner for persisted GBDT and NN models."""

# pyright: reportMissingImports=false, reportMissingTypeArgument=false, reportArgumentType=false
from __future__ import annotations

import json
from collections import deque
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

from template.data.io_parquet import iter_test_batches, stream_predictions_to_parquet
from template.data.preprocess import PreprocessorState, transform_features
from template.data.parquet_adapter import LargeParquetConfig, iter_parquet_batches
from template.models.gbdt.lgbm_model import LGBMModelWrapper
from template.models.gbdt.xgb_model import XGBModelWrapper
from template.configs.pipeline_config import nn_config_from_meta
from template.models import build_nn_model

# 直接运行本脚本时使用的默认参数，请按需修改 model_path / meta_path
DEFAULT_PREDICT = {
    "family": "lgbm",
    "model_path": "",
    "meta_path": "",
    "output_path": "res/pipeline/predict_output.parquet",
    "test_parquet": "",
}


def _load_wrapper(family: str, model_path: str):
    if family == "lgbm":
        return LGBMModelWrapper.load(model_path)
    return XGBModelWrapper.load(model_path)


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


def _make_composite_key(df: pd.DataFrame, key_cols: list[str]) -> pd.Series:
    """Join *key_cols* values with ``|`` to form a single composite key column."""
    return df[key_cols].astype(str).agg("|".join, axis=1)


def _predict_gbdt(
    family: str,
    model_path: str,
    meta: dict,
    output_path: str,
    test_parquet: str,
) -> None:
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
    print(f"[Done] prediction saved to {out_file}")


def _predict_nn(
    family: str,
    model_path: str,
    meta: dict,
    output_path: str,
    test_parquet: str,
) -> None:
    feature_cols = list(meta["feature_cols"])
    data_cfg = dict(meta["data_config"])
    nn_cfg_raw = dict(meta.get("nn_config", {}))
    preprocess_state = _state_from_dict(dict(meta["preprocess_state"]))
    id_map_raw = meta.get("id_map", {})
    id_map = {k: int(v) for k, v in id_map_raw.items()}
    num_ids = int(meta.get("num_ids", len(id_map)))
    lookback = int(nn_cfg_raw.get("lookback", 32))

    if test_parquet:
        test_path = test_parquet
    else:
        test_path = str(data_cfg.get("test_parquet_path", ""))
    if not test_path or not os.path.exists(test_path):
        raise FileNotFoundError(f"test parquet not found: {test_path}")

    nn_cfg = nn_config_from_meta(nn_cfg_raw)
    num_features = len(feature_cols)
    model = build_nn_model(nn_cfg.model_name, nn_cfg, num_ids, num_features)
    model = cast(nn.Module, model)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    id_col = str(data_cfg.get("id_cols", ["stockid"])[0])
    date_col = str(data_cfg.get("date_col", "dateid"))
    key_cols = ["stockid", "dateid", "timeid"]
    read_cols = list(set(list(data_cfg.get("id_cols", [])) + feature_cols))

    buffers: dict[object, deque] = {}
    batch_size = int(data_cfg.get("batch_size", 200_000))

    writer: pq.ParquetWriter | None = None
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        for batch in iter_parquet_batches(
            parquet_path=test_path,
            columns=read_cols,
            batch_size=batch_size,
        ):
            batch = transform_features(batch, feature_cols, preprocess_state, copy_frame=False)
            sort_c = [c for c in [date_col, "timeid", id_col] if c in batch.columns]
            if sort_c:
                batch = batch.sort_values(sort_c).reset_index(drop=True)

            feat = batch[feature_cols].to_numpy(dtype=np.float32)
            ids_arr = batch[id_col].to_numpy()
            pred_keys: list[str] = []
            pred_vals: list[float] = []

            for i in range(len(batch)):
                sid = ids_arr[i]
                x_row = feat[i]

                if sid not in buffers:
                    buffers[sid] = deque(maxlen=lookback)
                buffers[sid].append(x_row)

                if len(buffers[sid]) == lookback:
                    x_win = np.stack(list(buffers[sid]), axis=0)
                    if np.any(np.isnan(x_win)) or np.any(np.isinf(x_win)):
                        continue
                    x_t = torch.tensor(x_win, dtype=torch.float32).unsqueeze(0).to(device)
                    sid_str = str(sid)
                    int_id = id_map.get(sid_str, 0)
                    id_t = torch.tensor([int_id], dtype=torch.long).to(device)
                    with torch.no_grad():
                        pred = model(x_t, id_t).cpu().item()

                    avail = [c for c in key_cols if c in batch.columns]
                    row_vals = [str(batch.iloc[i][c]) for c in avail]
                    pred_keys.append("|".join(row_vals))
                    pred_vals.append(pred)

            if pred_keys:
                out = pd.DataFrame({
                    "stockid|dateid|timeid": pred_keys,
                    "prediction": np.asarray(pred_vals, dtype=np.float32),
                })
                table = pa.Table.from_pandas(out, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(str(out_file), table.schema, compression="zstd")
                writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    print(f"[Done] prediction saved to {out_file}")


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
    opts = DEFAULT_PREDICT
    if not opts["model_path"] or not opts["meta_path"]:
        print("Set DEFAULT_PREDICT['model_path'] and DEFAULT_PREDICT['meta_path'] in predict.py (or call run_predict(...) with explicit args).")
        return
    run_predict(
        family=opts["family"],
        model_path=opts["model_path"],
        meta_path=opts["meta_path"],
        output_path=opts["output_path"],
        test_parquet=opts["test_parquet"],
    )


if __name__ == "__main__":
    main()
