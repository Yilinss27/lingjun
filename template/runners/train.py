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
from template.data.memmap_dataset import (
    build_memmap_cache,
    make_memmap_dataloader,
    MemmapWindowDataset,
    MemmapPredictionDataset,
    memmap_collate_fn,
    pred_collate_fn,
)
from template.eval import save_json_report
from template.models import build_nn_model
from template.train import GBDTTrainer, TorchTrainer, get_torch_device
from template.utils import init_reproducibility


def _fmt_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{int(m)}m{s:.0f}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h{int(m)}m{s:.0f}s"


class _ProgressTracker:
    """Low-overhead progress tracker — prints stage boundaries / heartbeats
    and optionally writes every event as a JSONL line."""

    def __init__(self, cfg):
        prog = getattr(cfg, "progress", None)
        self.enabled = prog.enabled if prog else True
        self.every_n_batches = prog.every_n_batches if prog else 10
        self.every_n_steps = prog.every_n_steps if prog else 50
        self.min_interval = prog.min_interval_sec if prog else 10.0
        self._jsonl_path = prog.jsonl_path if prog else ""
        self._jsonl_fh = None
        self._stage_t0: dict[str, float] = {}

    # ── core emit ──

    def _emit(self, msg: str, record: dict | None = None):
        if not self.enabled:
            return
        print(msg, flush=True)
        if self._jsonl_path and record is not None:
            if self._jsonl_fh is None:
                self._jsonl_fh = open(self._jsonl_path, "a", encoding="utf-8")
            record.setdefault("ts", time.time())
            self._jsonl_fh.write(
                json.dumps(record, ensure_ascii=False, default=str) + "\n"
            )
            self._jsonl_fh.flush()

    # ── stage boundary helpers ──

    def stage_start(self, name: str):
        self._stage_t0[name] = time.time()
        self._emit(f"[Progress] {name} ...", {"event": "stage_start", "stage": name})

    def stage_end(self, name: str, **extra):
        elapsed = time.time() - self._stage_t0.pop(name, time.time())
        parts = [f"[Progress] {name} done in {_fmt_elapsed(elapsed)}"]
        if extra:
            details = ", ".join(f"{k}={v}" for k, v in extra.items())
            parts.append(f"({details})")
        self._emit(
            " ".join(parts),
            {"event": "stage_end", "stage": name,
             "elapsed_sec": round(elapsed, 1), **extra},
        )

    # ── streaming callback factory ──

    def make_streaming_callback(self, stage: str):
        """Return a lightweight (batches_done, rows_seen) callback, or None."""
        if not self.enabled:
            return None
        last_t = [time.time()]
        stage_t0 = self._stage_t0
        every_n = self.every_n_batches
        min_iv = self.min_interval
        emit = self._emit

        def _cb(batches_done: int, rows_seen: int):
            now = time.time()
            if batches_done % every_n == 0 and (now - last_t[0]) >= min_iv:
                elapsed = now - stage_t0.get(stage, now)
                emit(
                    f"[Progress]   {stage}: batch {batches_done}, "
                    f"rows {rows_seen:,}, elapsed {_fmt_elapsed(elapsed)}",
                    {"event": "heartbeat", "stage": stage,
                     "batches_done": batches_done, "rows_seen": rows_seen,
                     "elapsed_sec": round(elapsed, 1)},
                )
                last_t[0] = now
        return _cb

    # ── training step callback factory (per-epoch) ──

    def make_train_step_callback(self, epoch: int, total_epochs: int):
        """Return a (step, total_steps, avg_loss) callback for one epoch."""
        if not self.enabled:
            return None
        last_t = [time.time()]
        epoch_t0 = [time.time()]
        every_n = self.every_n_steps
        min_iv = self.min_interval
        emit = self._emit

        def _cb(step: int, total_steps: int, avg_loss: float):
            now = time.time()
            if step % every_n == 0 and (now - last_t[0]) >= min_iv:
                elapsed = now - epoch_t0[0]
                eta = elapsed / step * (total_steps - step) if step > 0 else 0
                emit(
                    f"  [Epoch {epoch:02d}/{total_epochs}] "
                    f"step {step}/{total_steps}, avg_loss={avg_loss:.6f}, "
                    f"elapsed {_fmt_elapsed(elapsed)}, eta {_fmt_elapsed(eta)}",
                    {"event": "train_step", "epoch": epoch, "step": step,
                     "total_steps": total_steps, "avg_loss": round(avg_loss, 6),
                     "elapsed_sec": round(elapsed, 1), "eta_sec": round(eta, 1)},
                )
                last_t[0] = now
        return _cb

    def close(self):
        if self._jsonl_fh:
            self._jsonl_fh.close()
            self._jsonl_fh = None


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
        "family": cfg.model.family,
        "objective": cfg.model.objective,
        "metrics": metrics,
        "model_path": str(model_path) if model_path else "",
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
    tracker = _ProgressTracker(cfg)
    try:
        return _train_nn_impl(cfg, tracker)
    finally:
        tracker.close()


def _train_nn_impl(cfg, tracker: _ProgressTracker) -> dict:
    large_cfg = cfg.data.to_large_parquet_config()

    # ── Pass 1: lightweight scan for feature columns & date boundaries ──
    tracker.stage_start("Pass1: scan_train_metadata")
    feature_cols, train_end, val_end = scan_train_metadata(large_cfg)
    tracker.stage_end("Pass1: scan_train_metadata", features=len(feature_cols))

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

    tracker.stage_start("Pass2: fit_preprocessor")
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
        on_batch=tracker.make_streaming_callback("Pass2: fit_preprocessor"),
    )
    tracker.stage_end("Pass2: fit_preprocessor")

    # ── Pass 3: build memmap cache (preprocessed rows + window indices) ──
    cache_dir = getattr(cfg.nn, "memmap_cache_dir", "cache/memmap")
    cross_date = getattr(cfg.nn, "cross_date_windows", False)

    def _memmap_progress(rg_done: int, rg_total: int, rows_written: int):
        if rg_done % 50 == 0 or rg_done == rg_total:
            tracker._emit(
                f"[Progress]   Memmap: {rg_done}/{rg_total} stocks, "
                f"{rows_written:,} rows written",
                {"event": "heartbeat", "stage": "Pass3: build_memmap",
                 "stocks_done": rg_done, "rows_written": rows_written},
            )

    tracker.stage_start("Pass3: build_memmap_cache")
    cache_meta = build_memmap_cache(
        parquet_path=large_cfg.train_path,
        feature_cols=feature_cols,
        label_col=large_cfg.label_col,
        id_col=large_cfg.id_cols[0],
        date_col=large_cfg.date_col,
        time_col="timeid",
        prep_state=prep_state,
        lookback=cfg.nn.lookback,
        train_end=train_end,
        val_end=val_end,
        cache_dir=cache_dir,
        cross_date_windows=cross_date,
        on_progress=_memmap_progress,
    )
    tracker.stage_end(
        "Pass3: build_memmap_cache",
        train=cache_meta["train_windows"],
        val=cache_meta["val_windows"],
        test=cache_meta["test_windows"],
    )

    # ── Build model ──
    num_features = len(feature_cols)
    num_ids = cache_meta["n_stocks"]
    id_map = {int(k): v for k, v in cache_meta["stock_id_map"].items()}
    model = build_nn_model(cfg.nn.model_name, cfg.nn, num_ids, num_features)
    model = cast(nn.Module, model)
    device = get_torch_device(cfg)

    tracker._emit(
        f"[Progress] Model built (model={cfg.nn.model_name}, "
        f"features={num_features}, ids={num_ids}, lookback={cfg.nn.lookback}, "
        f"device={device})",
        {"event": "model_built", "model": cfg.nn.model_name,
         "features": num_features, "ids": num_ids, "device": str(device)},
    )

    # ── Create DataLoaders from memmap ──
    mm_path = cache_meta["mm_path"]
    mm_shape = tuple(cache_meta["mm_shape"])
    num_workers = getattr(cfg.nn, "dataloader_workers", 4)
    pin_mem = getattr(cfg.nn, "pin_memory", True) and torch.cuda.is_available()
    prefetch = getattr(cfg.nn, "prefetch_factor", 4)

    train_loader = make_memmap_dataloader(
        mm_path=mm_path, mm_shape=mm_shape,
        index_path=cache_meta["idx_train_path"],
        lookback=cfg.nn.lookback, n_features=num_features,
        batch_size=cfg.nn.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_mem,
        prefetch_factor=prefetch,
    )
    val_loader = make_memmap_dataloader(
        mm_path=mm_path, mm_shape=mm_shape,
        index_path=cache_meta["idx_val_path"],
        lookback=cfg.nn.lookback, n_features=num_features,
        batch_size=cfg.nn.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_mem,
        prefetch_factor=prefetch,
    )

    # ── Train with memmap DataLoaders ──
    tracker.stage_start("Training")
    trainer = _MemmapTorchTrainer(model, cfg, device)

    def _step_cb_factory(ep):
        return tracker.make_train_step_callback(ep, cfg.nn.epochs)

    history = trainer.fit(
        train_loader, val_loader,
        on_step_factory=_step_cb_factory,
    )
    tracker.stage_end("Training")

    # ── Evaluate on test split (full metrics with IC/RankIC) ──
    do_test_eval = getattr(cfg.runner, "do_test_eval", True)
    metrics: dict = {}
    pred_df = pd.DataFrame()

    if do_test_eval:
        tracker.stage_start("Evaluate")
        metrics = trainer.evaluate_loader_full(
            mm_path=mm_path, mm_shape=mm_shape,
            index_path=cache_meta["idx_test_path"],
            lookback=cfg.nn.lookback, n_features=num_features,
            batch_size=cfg.nn.batch_size,
            num_workers=num_workers, pin_memory=pin_mem,
        )
        tracker.stage_end("Evaluate")
        # also keep a simple pred_df for artifact saving
        test_loader_pred = make_memmap_dataloader(
            mm_path=mm_path, mm_shape=mm_shape,
            index_path=cache_meta["idx_test_path"],
            lookback=cfg.nn.lookback, n_features=num_features,
            batch_size=cfg.nn.batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_mem,
            prefetch_factor=prefetch,
        )
        pred_df = trainer.predict_loader(test_loader_pred)
    else:
        tracker._emit("[Progress] Skipping test evaluation (do_test_eval=False)", {})

    # ── Save artifacts ──
    tracker.stage_start("Save artifacts")
    run_dir, run_id = _make_run_dir(cfg.runner.output_dir, f"nn_{cfg.nn.model_name}")
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
        "model_name": cfg.nn.model_name,
        "metrics": metrics,
        "history": {k: v[-1] if isinstance(v, list) and v else None for k, v in history.items()},
        "model_path": str(model_path) if model_path else "",
        "test_pred_path": pred_path,
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
    tracker.stage_end("Save artifacts")

    # ── (Optional) Generate submission predictions on test.parquet ──
    do_submission = getattr(cfg.runner, "do_submission_predict", False)
    if do_submission and cfg.runner.save_model and model_path:
        tracker.stage_start("Submission predict")
        try:
            from template.runners.predict import run_predict
            submission_path = str(run_dir / "submission.csv")
            _generate_submission(
                model=trainer.model,
                model_path_str=str(model_path),
                meta_path=str(run_dir / "meta.json"),
                test_parquet=cfg.data.test_parquet_path,
                output_csv=submission_path,
                prep_state=prep_state,
                feature_cols=feature_cols,
                id_map=id_map,
                lookback=cfg.nn.lookback,
                batch_size=cfg.nn.batch_size,
                device=get_torch_device(cfg),
            )
            report["submission_path"] = submission_path
            tracker._emit(f"[Progress] Submission saved to {submission_path}", {})
        except Exception as e:
            tracker._emit(f"[Warning] Submission predict failed: {e}", {})
        tracker.stage_end("Submission predict")
    else:
        if do_submission:
            tracker._emit(
                "[Warning] do_submission_predict=True but save_model=False; "
                "cannot generate submission without saved model.", {}
            )

    return report


def _generate_submission(
    model: "nn.Module",
    model_path_str: str,
    meta_path: str,
    test_parquet: str,
    output_csv: str,
    prep_state,
    feature_cols: list,
    id_map: dict,
    lookback: int,
    batch_size: int,
    device: "torch.device",
) -> None:
    """Stream through test.parquet and write a Kaggle-format submission CSV."""
    from collections import deque
    from template.data.parquet_adapter import iter_parquet_batches
    from template.data.preprocess import transform_features as _tf

    id_col = "stockid"
    key_cols = ["stockid", "dateid", "timeid"]
    read_cols = list(dict.fromkeys([id_col, "dateid", "timeid"] + feature_cols))

    buffers: dict = {}
    rows_out: list[dict] = []

    model.eval()
    with torch.no_grad():
        for batch in iter_parquet_batches(
            parquet_path=test_parquet,
            columns=read_cols,
            batch_size=batch_size,
        ):
            batch = _tf(batch, feature_cols, prep_state, copy_frame=False)
            sort_c = [c for c in ["dateid", "timeid", id_col] if c in batch.columns]
            if sort_c:
                batch = batch.sort_values(sort_c).reset_index(drop=True)

            feat = batch[feature_cols].to_numpy(dtype=np.float32)
            ids_arr = batch[id_col].to_numpy()

            for i in range(len(batch)):
                sid = int(ids_arr[i])
                x_row = feat[i]
                if sid not in buffers:
                    buffers[sid] = deque(maxlen=lookback)
                buffers[sid].append(x_row)

                if len(buffers[sid]) < lookback:
                    continue  # not enough history yet

                x_win = np.stack(list(buffers[sid]), axis=0)
                if np.any(np.isnan(x_win)) or np.any(np.isinf(x_win)):
                    continue

                x_t = torch.tensor(x_win, dtype=torch.float32).unsqueeze(0).to(device)
                int_id = id_map.get(sid, 0)
                id_t = torch.tensor([int_id], dtype=torch.long).to(device)
                pred_val = float(model(x_t, id_t).cpu().item())

                row = batch.iloc[i]
                uid = "|".join(str(int(row[c])) for c in key_cols if c in batch.columns)
                rows_out.append({"Uid": uid, "prediction": pred_val})

    # Write CSV
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    sub_df = pd.DataFrame(rows_out)
    sub_df.to_csv(output_csv, index=False)
    print(f"[Submission] {len(sub_df):,} predictions written to {output_csv}")


# ── Memmap-compatible Trainer ──────────────────────────────────────────────

class _MemmapTorchTrainer:
    """Lightweight trainer that works with memmap DataLoaders directly.

    Unlike TorchTrainer which requires in-memory numpy dict samples,
    this works with standard DataLoaders returning (X, ids, y) tensors.
    """

    def __init__(self, model: nn.Module, cfg, device: torch.device):
        from template.train.torch_trainer import get_amp_dtype, create_scaler
        from template.train.losses import get_loss_fn

        self.cfg = cfg
        self.device = device
        nn_cfg = cfg.nn

        # Optional torch.compile
        compile_model = getattr(nn_cfg, "compile_model", False)
        if compile_model and hasattr(torch, "compile") and device.type == "cuda":
            compile_mode = getattr(nn_cfg, "compile_mode", "reduce-overhead")
            print(f"[GPU优化] 启用 torch.compile(), mode='{compile_mode}'")
            try:
                model = cast(nn.Module, torch.compile(model, mode=compile_mode))
            except Exception as e:
                print(f"[GPU优化] torch.compile() 失败: {e}, 使用原始模型")

        self.model = model.to(device)
        self.loss_fn = get_loss_fn(cfg)

        self.use_amp = getattr(nn_cfg, "use_amp", False) and device.type == "cuda"
        self.amp_dtype = get_amp_dtype(cfg) if self.use_amp else torch.float32
        self.scaler = create_scaler(cfg, device)

        if self.use_amp:
            dtype_name = "bfloat16" if self.amp_dtype == torch.bfloat16 else "float16"
            scaler_status = "enabled" if self.scaler else "disabled"
            print(f"[GPU优化] 启用混合精度训练 (AMP), dtype={dtype_name}, scaler={scaler_status}")

        # Checkpoint
        from template.train.checkpoint import CheckpointManager
        self.ckpt_mgr = CheckpointManager(cfg) if nn_cfg.checkpoint_enabled else None

        self.history: dict[str, list] = {
            "train_loss": [], "val_loss": [],
            "val_r2": [],
        }

    def fit(
        self,
        train_loader,
        val_loader,
        on_step_factory=None,
    ) -> dict[str, list]:
        import torch.optim as optim
        from torch.amp.autocast_mode import autocast
        from template.eval import r2_score_np

        nn_cfg = self.cfg.nn
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=nn_cfg.lr,
            weight_decay=nn_cfg.weight_decay,
        )

        for ep in range(1, nn_cfg.epochs + 1):
            step_cb = on_step_factory(ep) if on_step_factory else None

            # ── Train epoch ──
            self.model.train()
            total_loss = 0.0
            n = 0
            step = 0
            total_steps = len(train_loader)

            for X, ids, y in train_loader:
                X = X.to(self.device, non_blocking=True)
                ids = ids.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                    pred = self.model(X, ids)
                    loss = self.loss_fn(pred, y)

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

                total_loss += float(loss.item()) * len(X)
                n += len(X)
                step += 1
                if step_cb is not None:
                    step_cb(step, total_steps, total_loss / max(n, 1))

            tr_loss = total_loss / max(n, 1)

            # ── Validate ──
            val_loss, val_pred, val_y = self._evaluate_internal(val_loader)
            r2_val = r2_score_np(val_pred, val_y)

            self.history["train_loss"].append(tr_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_r2"].append(r2_val)

            print(f"[Epoch {ep:02d}] train_loss={tr_loss:.6f}, "
                  f"val_loss={val_loss:.6f}, val_r2={r2_val:.6f}")

            # Checkpoint
            should_stop = False
            if self.ckpt_mgr:
                epoch_metrics = {"val_loss": val_loss, "val_r2": r2_val, "train_loss": tr_loss}
                should_save, should_stop = self.ckpt_mgr.step(ep, self.model, epoch_metrics)
                if should_save:
                    print(f"  -> New best model saved")

            if should_stop:
                print(f"[Early Stop] No improvement for {nn_cfg.checkpoint_patience} epochs.")
                break

        if self.ckpt_mgr and self.ckpt_mgr.load_best(self.model):
            self.ckpt_mgr.print_summary()

        return self.history

    def _evaluate_internal(self, loader) -> tuple[float, np.ndarray, np.ndarray]:
        """Evaluate model on a DataLoader, return (avg_loss, preds, targets)."""
        self.model.eval()
        preds_list, ys_list = [], []
        total_loss = 0.0
        n = 0

        with torch.no_grad():
            for X, ids, y in loader:
                X = X.to(self.device, non_blocking=True)
                ids = ids.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                pred = self.model(X, ids)
                loss = self.loss_fn(pred, y)
                total_loss += float(loss.item()) * len(X)
                n += len(X)
                preds_list.append(pred.detach().float().cpu().numpy())
                ys_list.append(y.detach().cpu().numpy())

        pred_all = np.concatenate(preds_list, axis=0) if preds_list else np.zeros((0, 1))
        y_all = np.concatenate(ys_list, axis=0) if ys_list else np.zeros((0, 1))
        return total_loss / max(n, 1), pred_all, y_all

    def evaluate_loader(self, loader) -> dict[str, float]:
        """Evaluate and return metrics dict (loss/mse/mae/r2 only, no date info)."""
        from template.eval import mse_np, mae_np, r2_score_np
        loss, pred, y = self._evaluate_internal(loader)
        return {
            "loss": loss,
            "mse": mse_np(pred, y),
            "mae": mae_np(pred, y),
            "r2": r2_score_np(pred, y),
        }

    def evaluate_loader_full(
        self,
        mm_path: str,
        mm_shape: tuple,
        index_path: str,
        lookback: int,
        n_features: int,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
    ) -> dict[str, float]:
        """Evaluate with full metrics including IC and RankIC (requires dateid from memmap)."""
        from template.eval import (
            mse_np, mae_np, r2_score_np,
            pearson_ic_by_date, rank_ic_by_date, ic_summary, ic_series_by_date,
        )
        from template.data.memmap_dataset import MemmapPredictionDataset, pred_collate_fn
        from torch.utils.data import DataLoader
        import sys

        ds = MemmapPredictionDataset(
            mm_path=mm_path, mm_shape=mm_shape,
            index_path=index_path, lookback=lookback, n_features=n_features,
        )

        if sys.platform == "win32" and num_workers > 4:
            num_workers = 4
        actual_pin = pin_memory and torch.cuda.is_available()

        if num_workers > 0:
            pred_loader = DataLoader(
                ds, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, collate_fn=pred_collate_fn,
                pin_memory=actual_pin, persistent_workers=True,
            )
        else:
            pred_loader = DataLoader(
                ds, batch_size=batch_size, shuffle=False,
                num_workers=0, collate_fn=pred_collate_fn,
            )

        self.model.eval()
        all_preds, all_ys, all_dates = [], [], []
        total_loss, n = 0.0, 0

        with torch.no_grad():
            for X, ids, y, dates, _ in pred_loader:
                X = X.to(self.device, non_blocking=True)
                ids = ids.to(self.device, non_blocking=True)
                y_gpu = y.to(self.device, non_blocking=True)
                pred = self.model(X, ids)
                loss = self.loss_fn(pred, y_gpu)
                total_loss += float(loss.item()) * len(X)
                n += len(X)
                all_preds.append(pred.detach().float().cpu().numpy())
                all_ys.append(y.numpy())
                all_dates.append(dates)

        pred_all = np.concatenate(all_preds).reshape(-1)
        y_all = np.concatenate(all_ys).reshape(-1)
        dates_all = np.concatenate(all_dates)

        # Build pred_df for IC calculation (date column must be numeric/comparable)
        pred_df = pd.DataFrame({"date": dates_all, "pred": pred_all, "y": y_all})

        avg_loss = total_loss / max(n, 1)

        # IC / RankIC via ic_summary
        ic_df = ic_series_by_date(pred_df, date_col="date", score_col="pred", target_col="y")
        summary = ic_summary(ic_df)

        return {
            "loss": avg_loss,
            "mse": mse_np(pred_all, y_all),
            "mae": mae_np(pred_all, y_all),
            "r2": r2_score_np(pred_all, y_all),
            "ic": float(summary.get("ic_mean", float("nan"))),
            "ic_ir": float(summary.get("ic_ir", float("nan"))),
            "rankic": float(summary.get("rank_ic_mean", float("nan"))),
            "rankic_ir": float(summary.get("rank_ic_ir", float("nan"))),
        }

    def predict_loader(self, loader) -> pd.DataFrame:
        """Generate predictions as a DataFrame."""
        self.model.eval()
        preds_list, ys_list = [], []

        with torch.no_grad():
            for X, ids, y in loader:
                X = X.to(self.device, non_blocking=True)
                ids = ids.to(self.device, non_blocking=True)
                pred = self.model(X, ids)
                preds_list.append(pred.detach().float().cpu().numpy())
                ys_list.append(y.detach().cpu().numpy())

        pred_all = np.concatenate(preds_list, axis=0).reshape(-1) if preds_list else np.array([])
        y_all = np.concatenate(ys_list, axis=0).reshape(-1) if ys_list else np.array([])

        return pd.DataFrame({
            "pred": pred_all,
            "y": y_all,
        })


# ── Entry point ─────────────────────────────────────────────────────────────

def run_train(cfg) -> dict:
    """Run training with the given PipelineConfig. Returns the report dict."""
    init_reproducibility(reproducible=True, seed=cfg.runner.seed)
    if getattr(cfg.runner, "smoke_test", False):
        cfg.data.train_parquet_path = "data/train_first5days.parquet"
        cfg.data.test_parquet_path = "data/test_first5days.parquet"
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
