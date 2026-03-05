# pyright: reportMissingImports=false, reportCallIssue=false, reportMissingTypeArgument=false, reportAttributeAccessIssue=false, reportGeneralTypeIssues=false
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch

from config import Config
from configs.presets import make_baseline_lgbm, make_baseline_xgb, make_baseline_nn
from data.io_parquet import load_train_panel, iter_test_batches, stream_predictions_to_parquet
from data.split import split_frame_by_date
from data.preprocess import fit_transform_splits, transform_features
from data.nn_samples import build_nn_samples_from_panel
from eval import save_json_report
from models import build_nn_model
from train import GBDTTrainer, TorchTrainer, get_torch_device
from utils import init_reproducibility


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified training entrypoint")
    parser.add_argument("--family", choices=["lgbm", "xgb", "nn"], default="lgbm")
    parser.add_argument("--objective", choices=["regression", "ranking"], default="regression")
    parser.add_argument("--train-parquet", default="data/train.parquet")
    parser.add_argument("--test-parquet", default="data/test.parquet")
    parser.add_argument("--label-col", default="LabelA")
    parser.add_argument("--feature-mode", choices=["all", "no_missing", "top_ic", "custom"], default="no_missing")
    parser.add_argument("--custom-features", default="")
    parser.add_argument("--max-train-rows", type=int, default=3000000)
    parser.add_argument("--max-test-rows", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=200000)
    parser.add_argument("--predict-batch-size", type=int, default=200000)
    parser.add_argument("--output-dir", default="res/pipeline")
    parser.add_argument("--no-save-test-pred", action="store_true")

    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--n-estimators", type=int, default=None)

    parser.add_argument("--nn-model", default="lstm")
    parser.add_argument("--lookback", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--emb-dim", type=int, default=0)
    parser.add_argument("--nn-loss", default="mse")
    parser.add_argument("--nn-lr", type=float, default=1e-3)
    parser.add_argument("--nn-epochs", type=int, default=8)
    parser.add_argument("--nn-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _load_base_config(family: str):
    if family == "lgbm":
        return make_baseline_lgbm()
    if family == "xgb":
        return make_baseline_xgb()
    return make_baseline_nn()


def _update_config_from_args(cfg, args: argparse.Namespace) -> None:
    cfg.data.train_parquet_path = args.train_parquet
    cfg.data.test_parquet_path = args.test_parquet
    cfg.data.label_col = args.label_col
    cfg.data.feature_mode = args.feature_mode
    cfg.data.custom_features = [x.strip() for x in args.custom_features.split(",") if x.strip()]
    cfg.data.max_train_rows = args.max_train_rows
    cfg.data.max_test_rows = None if args.max_test_rows < 0 else args.max_test_rows
    cfg.data.batch_size = args.batch_size
    cfg.runner.predict_batch_size = args.predict_batch_size
    cfg.model.objective = args.objective
    cfg.runner.output_dir = args.output_dir
    cfg.runner.save_test_pred = not args.no_save_test_pred
    cfg.runner.seed = args.seed

    if args.learning_rate is not None:
        cfg.model.params["learning_rate"] = args.learning_rate
    if args.n_estimators is not None:
        cfg.model.params["n_estimators"] = args.n_estimators

    cfg.nn.model_name = args.nn_model
    cfg.nn.lookback = args.lookback
    cfg.nn.hidden = args.hidden
    cfg.nn.layers = args.layers
    cfg.nn.drop_out = args.dropout
    cfg.nn.emb_dim = args.emb_dim
    cfg.nn.loss_name = args.nn_loss
    cfg.nn.lr = args.nn_lr
    cfg.nn.epochs = args.nn_epochs
    cfg.nn.batch_size = args.nn_batch_size


def _train_gbdt(cfg) -> dict:
    large_cfg = cfg.data.to_large_parquet_config()
    bundle = load_train_panel(large_cfg)
    full_panel = bundle.panel
    feature_cols = bundle.feature_cols
    train_end = bundle.train_end
    val_end = bundle.val_end

    train_df, val_df, holdout_df = split_frame_by_date(full_panel, cfg.data.date_col, train_end, val_end)
    train_df, val_df, holdout_df, prep_state = fit_transform_splits(
        train_df,
        val_df,
        holdout_df,
        feature_cols,
        fillna_strategy=cfg.preprocess.fillna_strategy,
        standardize=cfg.preprocess.standardize,
        clip_quantile=cfg.preprocess.clip_quantile,
    )

    trainer = GBDTTrainer(
        family=cfg.model.family,
        objective=cfg.model.objective,
        model_params=cfg.model.params,
        date_col=cfg.data.date_col,
        label_col=cfg.data.label_col,
    )
    trainer.fit(train_df, val_df, feature_cols)
    metrics = trainer.evaluate(holdout_df, feature_cols)

    out_dir = Path(cfg.runner.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    model_path = out_dir / f"{cfg.model.family}_{run_id}.pkl"
    trainer.save_model(str(model_path))

    pred_path = ""
    if cfg.runner.save_test_pred and cfg.data.test_parquet_path:
        def _iter_transformed_batches():
            for batch in iter_test_batches(large_cfg, feature_cols, cfg.runner.predict_batch_size):
                yield transform_features(batch, feature_cols, prep_state, copy_frame=False)

        pred_file = out_dir / f"{cfg.model.family}_{run_id}_test_pred.parquet"
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
    save_json_report(str(out_dir / f"{cfg.model.family}_{run_id}_meta.json"), report)
    return report


def _train_nn(cfg) -> dict:
    large_cfg = cfg.data.to_large_parquet_config()
    bundle = load_train_panel(large_cfg)
    full_panel = bundle.panel
    feature_cols = bundle.feature_cols
    train_end = bundle.train_end
    val_end = bundle.val_end

    train_df, val_df, holdout_df = split_frame_by_date(full_panel, cfg.data.date_col, train_end, val_end)
    train_df, val_df, holdout_df, prep_state = fit_transform_splits(
        train_df,
        val_df,
        holdout_df,
        feature_cols,
        fillna_strategy=cfg.preprocess.fillna_strategy,
        standardize=cfg.preprocess.standardize,
        clip_quantile=cfg.preprocess.clip_quantile,
    )

    panel_t = pd.concat([train_df, val_df, holdout_df], axis=0, ignore_index=True)
    samples = build_nn_samples_from_panel(
        panel_t,
        feature_cols,
        label_col=cfg.data.label_col,
        id_col=cfg.data.id_cols[0],
        date_col=cfg.data.date_col,
        time_col="timeid" if "timeid" in panel_t.columns else None,
        lookback=cfg.nn.lookback,
        train_end=train_end,
        val_end=val_end,
    )

    nn_cfg = Config(
        model_name=cfg.nn.model_name,
        emb_dim=cfg.nn.emb_dim,
        hidden=cfg.nn.hidden,
        layers=cfg.nn.layers,
        drop_out=cfg.nn.drop_out,
        lookback=cfg.nn.lookback,
        lr=cfg.nn.lr,
        weight_decay=cfg.nn.weight_decay,
        batch_size=cfg.nn.batch_size,
        epochs=cfg.nn.epochs,
        loss_name=cfg.nn.loss_name,
        device=cfg.nn.device,
        use_amp=cfg.nn.use_amp,
        checkpoint_enabled=cfg.nn.checkpoint_enabled,
    )

    num_indices = int(panel_t[cfg.data.id_cols[0]].nunique())
    num_features = len(feature_cols)
    model = build_nn_model(cfg.nn.model_name, nn_cfg, num_indices, num_features)
    device = get_torch_device(nn_cfg)
    trainer = TorchTrainer(model, nn_cfg, device)

    history = trainer.fit(samples["train"], samples["val"])
    metrics = trainer.evaluate(samples["test"])
    pred_df = trainer.predict(samples["test"])

    out_dir = Path(cfg.runner.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    model_path = out_dir / f"nn_{cfg.nn.model_name}_{run_id}.pt"
    torch.save(trainer.model.state_dict(), model_path)

    pred_path = out_dir / f"nn_{cfg.nn.model_name}_{run_id}_test_pred.parquet"
    pred_df.to_parquet(pred_path, index=False)

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
    }
    save_json_report(str(out_dir / f"nn_{cfg.nn.model_name}_{run_id}_meta.json"), report)
    return report


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    cfg = _load_base_config(args.family)
    _update_config_from_args(cfg, args)

    init_reproducibility(reproducible=True, seed=cfg.runner.seed)

    if cfg.model.family in {"lgbm", "xgb"}:
        report = _train_gbdt(cfg)
    else:
        report = _train_nn(cfg)

    print("[Done] train runner finished")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
