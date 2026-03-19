# pyright: reportMissingImports=false, reportCallIssue=false, reportMissingTypeArgument=false, reportAttributeAccessIssue=false, reportGeneralTypeIssues=false
"""Training runner — thin orchestration layer.

Responsibilities:
    - Load pipeline config and model config
    - Dispatch to GBDT or NN training path
    - Save artifacts and report
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time
import os
import sys
from typing import Literal, cast

_TEMPLATE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_TEMPLATE_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _TEMPLATE_DIR)

import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from template.configs.pipeline_config import PipelineConfig
from template.configs.model_configs import (
    load_model_config, get_model_family,
    NNModelConfig, GBDTModelConfig,
)
from template.data.io_parquet import (
    load_train_panel, iter_test_batches, stream_predictions_to_parquet,
    scan_train_metadata,
)
from template.data.parquet_adapter import iter_parquet_batches
from template.data.split import split_frame_by_date
from template.data.preprocess import (
    fit_transform_splits, transform_features,
)
from template.data.preprocess_nn import (
    fit_medians_streaming,
    get_standardizer,
)
from template.data.memmap_cache import (
    build_raw_cache,
    build_derived_cache,
    make_v2_dataloader,
    try_load_raw_cache,
)
from template.eval import save_json_report
from template.eval.reporting import append_leaderboard
from template.models import build_nn_model
from template.train import GBDTTrainer, get_torch_device
from template.train.nn_trainer import NNTrainer, evaluate_v2
from template.utils import init_reproducibility
from template.utils.progress import ProgressTracker


def _make_run_dir(base: str, family: str) -> tuple[Path, str]:
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base) / family / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, run_id


# ── GBDT ────────────────────────────────────────────────────────────────────

def _train_gbdt(cfg, model_cfg: GBDTModelConfig) -> dict:
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

    family = model_cfg.family
    if family not in ("lgbm", "xgb"):
        raise ValueError(f"Invalid GBDT family: {family!r}")
    objective = model_cfg.objective
    if objective not in ("regression", "ranking"):
        raise ValueError(f"Invalid GBDT objective: {objective!r}")

    trainer = GBDTTrainer(
        family=cast(Literal["lgbm", "xgb"], family),
        objective=cast(Literal["regression", "ranking"], objective),
        model_params=model_cfg.params,
        date_col=cfg.data.date_col,
        label_col=cfg.data.label_col,
    )
    trainer.fit(train_df, val_df, feature_cols)
    metrics = trainer.evaluate(holdout_df, feature_cols)

    run_dir, run_id = _make_run_dir(cfg.runner.output_dir, model_cfg.family)
    model_path = run_dir / "model.pkl"
    if cfg.runner.save_model:
        trainer.save_model(str(model_path))
    else:
        model_path = ""

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
        "family": model_cfg.family,
        "objective": model_cfg.objective,
        "metrics": metrics,
        "model_path": str(model_path) if model_path else "",
        "test_pred_path": pred_path,
        "feature_cols": feature_cols,
        "train_end": train_end,
        "val_end": val_end,
        "preprocess_state": asdict(prep_state),
        "data_config": asdict(cfg.data),
        "model_config": asdict(model_cfg),
    }
    save_json_report(str(run_dir / "meta.json"), report)
    return report


# ── NN (streaming — low memory) ────────────────────────────────────────────

def _train_nn(cfg, nn_cfg: NNModelConfig) -> dict:
    tracker = ProgressTracker(cfg)
    try:
        return _train_nn_v2(cfg, nn_cfg, tracker)
    finally:
        tracker.close()


def _train_nn_v2(cfg, nn_cfg: NNModelConfig, tracker: ProgressTracker) -> dict:
    """V2 数据管线：raw cache + derived cache（支持截面 RankGauss）。"""
    large_cfg = cfg.data.to_large_parquet_config()

    # ── Pass 1: scan metadata ──
    tracker.stage_start("Pass1: scan_train_metadata")
    feature_cols, train_end, val_end = scan_train_metadata(large_cfg)
    tracker.stage_end("Pass1: scan_train_metadata", features=len(feature_cols))

    # ── Check raw cache ──
    raw_cache_dir = cfg.runner.raw_cache_dir
    if getattr(cfg.runner, "smoke_test", False):
        raw_cache_dir = "cache/raw_smoke"

    cached_raw = try_load_raw_cache(
        parquet_path=large_cfg.train_path,
        feature_cols=feature_cols,
        label_col=large_cfg.label_col,
        cache_dir=raw_cache_dir,
    )

    if cached_raw is not None:
        tracker._emit(
            f"[Progress] Raw cache hit — skipping Pass2 (fit_medians) & Pass3 (build_raw_cache)",
            {"event": "cache_hit", "stage": "raw_cache"},
        )
        raw_meta = cached_raw
        fillna_medians = cached_raw.fillna_medians
    else:
        # ── Pass 2: fit medians (streaming) ──
        read_cols = list(large_cfg.id_cols) + feature_cols + [large_cfg.label_col]

        def _train_batches_for_median():
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

        tracker.stage_start("Pass2: fit_medians")
        std_method = getattr(cfg.preprocess, "standardize_method", "rank_gauss")
        fit_result = fit_medians_streaming(
            _train_batches_for_median(),
            feature_cols,
            on_batch=tracker.make_streaming_callback("Pass2: fit_medians"),
            compute_global_stats=(std_method == "global_zscore"),
        )
        fillna_medians = fit_result.medians
        tracker.stage_end("Pass2: fit_medians")

        # ── Pass 3: build raw cache ──
        tracker.stage_start("Pass3: build_raw_cache")
        raw_meta = build_raw_cache(
            parquet_path=large_cfg.train_path,
            feature_cols=feature_cols,
            label_col=large_cfg.label_col,
            id_col=str(large_cfg.id_cols[0]),
            date_col=large_cfg.date_col,
            time_col="timeid",
            fillna_medians=fillna_medians,
            train_end=int(train_end), #type:ignore
            val_end=int(val_end), #type:ignore
            cache_dir=raw_cache_dir,
            global_means=fit_result.global_means,
            global_stds=fit_result.global_stds,
        )
        tracker.stage_end("Pass3: build_raw_cache",
                          stocks=raw_meta.n_stock, dates=raw_meta.n_date,
                          times=raw_meta.n_time)

    # ── Pass 4: build derived cache ──
    std_method = getattr(cfg.preprocess, "standardize_method", "rank_gauss")
    if std_method == "global_zscore":
        if raw_meta.global_means is None or raw_meta.global_stds is None:
            raise ValueError(
                "global_zscore 需要全局均值/标准差，但 raw cache 中未找到。"
                "请删除 raw cache 目录后重新运行，以便在 Pass 2 中计算全局统计量。"
            )
        means_arr = np.array([raw_meta.global_means[c] for c in feature_cols], dtype=np.float32)
        stds_arr = np.array([raw_meta.global_stds[c] for c in feature_cols], dtype=np.float32)
        standardizer = get_standardizer(std_method, means=means_arr, stds=stds_arr)
    else:
        standardizer = get_standardizer(std_method)

    derived_cache_dir = cfg.runner.derived_cache_dir
    if getattr(cfg.runner, "smoke_test", False):
        derived_cache_dir = "cache/derived_smoke"

    tracker.stage_start("Pass4: build_derived_cache")
    derived_meta = build_derived_cache(
        raw_meta=raw_meta,
        standardizer=standardizer,
        standardize_method=std_method,
        lookback=nn_cfg.lookback,
        cross_date_windows=cfg.runner.cross_date_windows,
        cache_dir=derived_cache_dir,
    )
    tracker.stage_end("Pass4: build_derived_cache",
                      train=derived_meta.train_windows,
                      val=derived_meta.val_windows,
                      test=derived_meta.test_windows)

    # ── Build model ──
    num_features = len(feature_cols)
    num_ids = raw_meta.n_stock
    id_map = raw_meta.stock_id_map
    model = build_nn_model(nn_cfg.model_name, nn_cfg, num_ids, num_features)
    model = cast(nn.Module, model)
    device = get_torch_device(getattr(cfg.runner, "device", "auto"))

    tracker._emit(
        f"[Progress] Model built (model={nn_cfg.model_name}, "
        f"features={num_features}, ids={num_ids}, lookback={nn_cfg.lookback}, "
        f"device={device})",
        {"event": "model_built", "model": nn_cfg.model_name,
         "features": num_features, "ids": num_ids, "device": str(device)},
    )

    # ── Create DataLoaders ──
    num_workers = cfg.runner.dataloader_workers
    pin_mem = cfg.runner.pin_memory and torch.cuda.is_available()
    prefetch = cfg.runner.prefetch_factor

    train_loader = make_v2_dataloader(
        derived_meta, split="train", lookback=nn_cfg.lookback,
        batch_size=nn_cfg.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_mem,
        prefetch_factor=prefetch,
    )
    val_loader = make_v2_dataloader(
        derived_meta, split="val", lookback=nn_cfg.lookback,
        batch_size=nn_cfg.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_mem,
        prefetch_factor=prefetch,
    )

    # ── Train ──
    tracker.stage_start("Training")
    trainer = NNTrainer(model, nn_cfg, device)

    def _step_cb_factory(ep):
        return tracker.make_train_step_callback(ep, nn_cfg.epochs)

    history = trainer.fit(train_loader, val_loader, on_step_factory=_step_cb_factory)
    tracker.stage_end("Training")

    # ── Evaluate on test split ──
    do_test_eval = getattr(cfg.runner, "do_test_eval", True)
    metrics: dict = {}
    pred_df = pd.DataFrame()

    if do_test_eval and derived_meta.test_windows > 0:
        tracker.stage_start("Evaluate")
        metrics = evaluate_v2(
            trainer, derived_meta, nn_cfg.lookback, num_features,
            nn_cfg.batch_size, num_workers, pin_mem,
        )
        tracker.stage_end("Evaluate")

        # Prediction DataFrame for artifact saving
        test_loader = make_v2_dataloader(
            derived_meta, split="test", lookback=nn_cfg.lookback,
            batch_size=nn_cfg.batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_mem,
            prefetch_factor=prefetch,
        )
        pred_df = trainer.predict_loader(test_loader)
    else:
        tracker._emit("[Progress] Skipping test evaluation", {})

    # ── Save artifacts ──
    tracker.stage_start("Save artifacts")
    run_dir, run_id = _make_run_dir(cfg.runner.output_dir, f"nn_{nn_cfg.model_name}")
    model_path = run_dir / "model.pt"
    if cfg.runner.save_model:
        torch.save(trainer.model.state_dict(), model_path)
    else:
        model_path = ""

    pred_path = ""
    if do_test_eval and not pred_df.empty:
        pred_path_obj = run_dir / "test_pred.parquet"
        pred_df.to_parquet(pred_path_obj, index=False)
        pred_path = str(pred_path_obj)

    id_map_serializable = {str(k): v for k, v in id_map.items()}

    report = {
        "run_id": run_id,
        "family": "nn",
        "model_name": nn_cfg.model_name,
        "metrics": metrics,
        "history": {k: v[-1] if isinstance(v, list) and v else None for k, v in history.items()},
        "model_path": str(model_path) if model_path else "",
        "test_pred_path": pred_path,
        "feature_cols": feature_cols,
        "train_end": train_end,
        "val_end": val_end,
        "fillna_medians": {k: float(v) for k, v in fillna_medians.items()},
        "standardize_method": std_method,
        "data_config": asdict(cfg.data),
        "nn_config": asdict(nn_cfg),
        "id_map": id_map_serializable,
        "num_ids": num_ids,
        "pipeline_version": "v2",
    }
    save_json_report(str(run_dir / "meta.json"), report)
    tracker.stage_end("Save artifacts")

    return report


# ── Entry point ─────────────────────────────────────────────────────────────

def run_train(cfg, model_cfg=None) -> dict:
    """Run training with the given PipelineConfig. Returns the report dict."""
    if model_cfg is None:
        model_cfg = load_model_config(cfg.model_name)

    init_reproducibility(reproducible=True, seed=cfg.runner.seed)
    if getattr(cfg.runner, "smoke_test", False):
        cfg.data.train_parquet_path = "data/train_first5days.parquet"
        cfg.data.test_parquet_path = "data/test_first5days.parquet"
        cfg.runner.memmap_cache_dir = "cache/smoke_test"
        cfg.runner.raw_cache_dir = "cache/raw_smoke"
        cfg.runner.derived_cache_dir = "cache/derived_smoke"

    family = get_model_family(cfg.model_name)
    if family in ("lgbm", "xgb"):
        if not isinstance(model_cfg, GBDTModelConfig):
            raise TypeError(f"Expected GBDTModelConfig for family={family!r}, got {type(model_cfg).__name__}")
        report = _train_gbdt(cfg, model_cfg)
    else:
        if not isinstance(model_cfg, NNModelConfig):
            raise TypeError(f"Expected NNModelConfig for family='nn', got {type(model_cfg).__name__}")
        report = _train_nn(cfg, model_cfg)
    # ── 自动追加到排行榜 ──
    try:
        append_leaderboard(report, cfg=cfg)
    except Exception as e:
        print(f"[Warning] 排行榜写入失败: {e}")

    print("[Done] train runner finished")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2, default=str))
    return report


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Train NN or GBDT models")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (overrides pipeline_config.model_name)")
    args = parser.parse_args()

    cfg = PipelineConfig()
    if args.model:
        cfg.model_name = args.model
    model_cfg = load_model_config(cfg.model_name)
    run_train(cfg, model_cfg)


if __name__ == "__main__":
    main()
