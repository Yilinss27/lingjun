# pyright: reportMissingImports=false, reportCallIssue=false, reportMissingTypeArgument=false, reportAttributeAccessIssue=false, reportGeneralTypeIssues=false
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time
import os
import sys
from typing import cast

_TEMPLATE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_TEMPLATE_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _TEMPLATE_DIR)

import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from template.configs.pipeline_config import PipelineConfig
from template.configs.presets import make_baseline_lgbm, make_baseline_xgb, make_baseline_nn
from template.data.io_parquet import (
    load_train_panel, iter_test_batches, stream_predictions_to_parquet,
    scan_train_metadata,
)
from template.data.parquet_adapter import iter_parquet_batches
from template.data.split import split_frame_by_date
from template.data.preprocess import (
    fit_transform_splits, transform_features, fit_preprocessor_streaming,
)
from template.data.nn_samples import build_nn_samples_streaming
from template.eval import save_json_report
from template.models import build_nn_model
from template.train import GBDTTrainer, TorchTrainer, get_torch_device
from template.utils import init_reproducibility


def _load_base_config(family: str):
    if family == "lgbm":
        return make_baseline_lgbm()
    if family == "xgb":
        return make_baseline_xgb()
    return make_baseline_nn()


def _make_run_dir(base: str, family: str) -> tuple[Path, str]:
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base) / family / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, run_id


# ── GBDT ────────────────────────────────────────────────────────────────────

def _train_gbdt(cfg) -> dict:
    large_cfg = cfg.data.to_large_parquet_config()
    bundle = load_train_panel(large_cfg)
    full_panel = bundle.panel
    feature_cols = bundle.feature_cols
    train_end = bundle.train_end
    val_end = bundle.val_end

    std_mode = getattr(cfg.preprocess, "standardize_mode", "none")
    train_df, val_df, holdout_df = split_frame_by_date(full_panel, cfg.data.date_col, train_end, val_end)
    train_df, val_df, holdout_df, prep_state = fit_transform_splits(
        train_df, val_df, holdout_df, feature_cols,
        fillna_strategy=cfg.preprocess.fillna_strategy,
        standardize=cfg.preprocess.standardize,
        clip_quantile=cfg.preprocess.clip_quantile,
    )
    if std_mode not in ("none", "global"):
        prep_state.standardize_mode = std_mode
        prep_state.z_window = getattr(cfg.preprocess, "z_window", 60)
        train_df = transform_features(train_df, feature_cols, prep_state, copy_frame=False)
        val_df = transform_features(val_df, feature_cols, prep_state, copy_frame=False)
        holdout_df = transform_features(holdout_df, feature_cols, prep_state, copy_frame=False)

    trainer = GBDTTrainer(
        family=cfg.model.family,
        objective=cfg.model.objective,
        model_params=cfg.model.params,
        date_col=cfg.data.date_col,
        label_col=cfg.data.label_col,
    )
    trainer.fit(train_df, val_df, feature_cols)
    metrics = trainer.evaluate(holdout_df, feature_cols)

    run_dir, run_id = _make_run_dir(cfg.runner.output_dir, cfg.model.family)
    model_path = run_dir / f"model.pkl"
    trainer.save_model(str(model_path))

    pred_path = ""
    if cfg.runner.save_test_pred and cfg.data.test_parquet_path:
        def _iter_transformed_batches():
            for batch in iter_test_batches(large_cfg, feature_cols, cfg.runner.predict_batch_size):
                yield transform_features(batch, feature_cols, prep_state, copy_frame=False)

        pred_file = run_dir / "test_pred.parquet"
        pred_path = stream_predictions_to_parquet(
            output_path=str(pred_file),
            id_cols=cfg.data.id_cols,
            batch_iter=_iter_transformed_batches(),
            predict_fn=trainer.model.predict,
            feature_cols=feature_cols,
        )

    report = {
        "run_id": run_id,
        "family": cfg.model.family,
        "objective": cfg.model.objective,
        "metrics": metrics,
        "model_path": str(model_path),
        "test_pred_path": pred_path,
        "feature_cols": feature_cols,
        "train_end": train_end,
        "val_end": val_end,
        "preprocess_state": asdict(prep_state),
        "data_config": asdict(cfg.data),
        "model_config": asdict(cfg.model),
    }
    save_json_report(str(run_dir / "meta.json"), report)
    return report


# ── NN (streaming — low memory) ────────────────────────────────────────────

def _train_nn(cfg) -> dict:
    large_cfg = cfg.data.to_large_parquet_config()

    # ── Pass 1: lightweight scan for feature columns & date boundaries ──
    feature_cols, train_end, val_end = scan_train_metadata(large_cfg)

    # ── Pass 2: fit preprocessor by streaming training rows ──
    read_cols = list(large_cfg.id_cols) + feature_cols + [large_cfg.label_col]

    std_mode = getattr(cfg.preprocess, "standardize_mode", "none")

    def _train_batches():
        for batch in iter_parquet_batches(
            parquet_path=large_cfg.train_path,
            columns=read_cols,
            batch_size=large_cfg.batch_size,
            row_group_stride=large_cfg.row_group_stride,
            row_group_offset=large_cfg.row_group_offset,
            max_rows=large_cfg.max_train_rows,
            dropna_label_col=large_cfg.label_col,
        ):
            mask = batch[large_cfg.date_col] <= train_end
            if mask.any():
                yield batch[mask]

    prep_state = fit_preprocessor_streaming(
        _train_batches(),
        feature_cols,
        fillna_strategy=cfg.preprocess.fillna_strategy,
        standardize=cfg.preprocess.standardize,
        standardize_mode=std_mode,
        clip_quantile=cfg.preprocess.clip_quantile,
        date_col=cfg.data.date_col,
        time_col="timeid",
        z_window=getattr(cfg.preprocess, "z_window", 60),
    )

    # ── Pass 3: stream all data → transform → build sliding windows ──
    time_col = "timeid" if "timeid" in large_cfg.id_cols else None

    def _transformed_batches():
        for batch in iter_parquet_batches(
            parquet_path=large_cfg.train_path,
            columns=read_cols,
            batch_size=large_cfg.batch_size,
            row_group_stride=large_cfg.row_group_stride,
            row_group_offset=large_cfg.row_group_offset,
            max_rows=large_cfg.max_train_rows,
            dropna_label_col=large_cfg.label_col,
        ):
            yield transform_features(batch, feature_cols, prep_state, copy_frame=False)

    samples, id_map, num_ids = build_nn_samples_streaming(
        _transformed_batches(),
        feature_cols,
        label_col=large_cfg.label_col,
        id_col=large_cfg.id_cols[0],
        date_col=large_cfg.date_col,
        time_col=time_col,
        lookback=cfg.nn.lookback,
        train_end=train_end,
        val_end=val_end,
    )

    # ── Build model & trainer ──
    num_features = len(feature_cols)
    model = build_nn_model(cfg.nn.model_name, cfg.nn, num_ids, num_features)
    model = cast(nn.Module, model)
    device = get_torch_device(cfg)
    trainer = TorchTrainer(model, cfg, device)

    samples = cast(dict[str, dict[str, np.ndarray]], samples)

    history = trainer.fit(samples["train"], samples["val"])
    metrics = trainer.evaluate(samples["test"])
    pred_df = trainer.predict(samples["test"])

    # ── Save artifacts ──
    run_dir, run_id = _make_run_dir(cfg.runner.output_dir, f"nn_{cfg.nn.model_name}")
    model_path = run_dir / "model.pt"
    torch.save(trainer.model.state_dict(), model_path)

    pred_path = run_dir / "test_pred.parquet"
    pred_df.to_parquet(pred_path, index=False)

    id_map_serializable = {str(k): v for k, v in id_map.items()}

    report = {
        "run_id": run_id,
        "family": "nn",
        "model_name": cfg.nn.model_name,
        "metrics": metrics,
        "history": {k: v[-1] if isinstance(v, list) and v else None for k, v in history.items()},
        "model_path": str(model_path),
        "test_pred_path": str(pred_path),
        "feature_cols": feature_cols,
        "train_end": train_end,
        "val_end": val_end,
        "preprocess_state": asdict(prep_state),
        "data_config": asdict(cfg.data),
        "nn_config": asdict(cfg.nn),
        "id_map": id_map_serializable,
        "num_ids": num_ids,
    }
    save_json_report(str(run_dir / "meta.json"), report)
    return report


# ── Entry point ─────────────────────────────────────────────────────────────

def run_train(cfg) -> dict:
    """Run training with the given PipelineConfig. Returns the report dict."""
    init_reproducibility(reproducible=True, seed=cfg.runner.seed)
    if cfg.model.family in {"lgbm", "xgb"}:
        report = _train_gbdt(cfg)
    else:
        report = _train_nn(cfg)
    print("[Done] train runner finished")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2, default=str))
    return report


def main() -> None:
    default_family = PipelineConfig().model.family
    cfg = _load_base_config(default_family)
    run_train(cfg)


if __name__ == "__main__":
    main()
