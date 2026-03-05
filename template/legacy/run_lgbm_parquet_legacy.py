# pyright: reportGeneralTypeIssues=false, reportAttributeAccessIssue=false, reportMissingImports=false, reportArgumentType=false
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, cast
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from eval import pearson_ic_by_date, rank_ic_by_date
from predata.parquet_adapter import LargeParquetConfig, iter_parquet_batches, prepare_large_parquet_train_frame
from utils import get_lgbm_seed_params, init_reproducibility


@dataclass
class LGBMParquetConfig:
    train_parquet_path: str = "data/train.parquet"
    test_parquet_path: str = "data/test.parquet"
    label_col: str = "LabelA"

    id_cols: tuple[str, ...] = ("stockid", "dateid", "timeid", "exchangeid")
    date_col: str = "dateid"
    feature_mode: Literal["all", "no_missing", "top_ic", "custom"] = "no_missing"
    custom_features: list[str] = field(default_factory=list)

    train_ratio: float = 0.70
    val_ratio: float = 0.15
    max_train_rows: int | None = 3_000_000
    max_test_rows: int | None = None
    batch_size: int = 200_000
    row_group_stride: int = 1
    row_group_offset: int = 0

    objective: Literal["regression", "lambdarank"] = "regression"
    learning_rate: float = 0.05
    n_estimators: int = 3000
    num_leaves: int = 127
    max_depth: int = -1
    min_child_samples: int = 50
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    early_stopping_rounds: int = 100
    device_type: Literal["cpu", "gpu"] = "cpu"

    predict_batch_size: int = 1_000_000
    output_dir: str = "res/lgbm_parquet"
    save_test_pred: bool = True


def _to_large_data_cfg(cfg: LGBMParquetConfig) -> LargeParquetConfig:
    return LargeParquetConfig(
        train_path=cfg.train_parquet_path,
        test_path=cfg.test_parquet_path,
        label_col=cfg.label_col,
        id_cols=cfg.id_cols,
        date_col=cfg.date_col,
        feature_mode=cfg.feature_mode,
        custom_features=cfg.custom_features,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        max_train_rows=cfg.max_train_rows,
        max_test_rows=cfg.max_test_rows,
        batch_size=cfg.batch_size,
        row_group_stride=cfg.row_group_stride,
        row_group_offset=cfg.row_group_offset,
    )


def _split_frame_by_date(
    frame: pd.DataFrame,
    date_col: str,
    train_end: object,
    val_end: object,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = frame.loc[frame[date_col] <= train_end]
    val_df = frame.loc[(frame[date_col] > train_end) & (frame[date_col] <= val_end)]
    test_df = frame.loc[frame[date_col] > val_end]
    return train_df, val_df, test_df


def _make_model(cfg: LGBMParquetConfig) -> lgb.LGBMModel:
    params: dict[str, Any] = {
        "learning_rate": cfg.learning_rate,
        "n_estimators": cfg.n_estimators,
        "num_leaves": cfg.num_leaves,
        "max_depth": cfg.max_depth,
        "min_child_samples": cfg.min_child_samples,
        "subsample": cfg.subsample,
        "colsample_bytree": cfg.colsample_bytree,
        "reg_alpha": cfg.reg_alpha,
        "reg_lambda": cfg.reg_lambda,
        "device_type": cfg.device_type,
        "n_jobs": -1,
        "verbose": -1,
    }
    params.update(get_lgbm_seed_params())

    if cfg.objective == "lambdarank":
        params.update({"objective": "lambdarank", "metric": "ndcg"})
        return lgb.LGBMRanker(**cast(Any, params))

    params.update({"objective": "regression", "metric": "l2"})
    return lgb.LGBMRegressor(**cast(Any, params))


def _to_relevance_by_date(y, date, n_levels: int = 10):
    frame = pd.DataFrame({"y": y, "date": date})

    def _bin(series):
        try:
            return pd.qcut(series, q=n_levels, labels=range(n_levels), duplicates="drop")
        except ValueError:
            ranked = series.rank(method="first")
            return pd.cut(ranked, bins=n_levels, labels=range(n_levels))

    rel = frame.groupby("date")["y"].transform(_bin)
    return np.asarray(rel.fillna(0).astype(np.int32))


def _stream_test_predictions(
    cfg: LGBMParquetConfig,
    feature_cols: list[str],
    model: lgb.LGBMModel,
    output_path: Path,
) -> None:
    data_cfg = _to_large_data_cfg(cfg)
    if not data_cfg.test_path:
        return

    pred_cols = list(cfg.id_cols)
    read_cols = pred_cols + feature_cols
    writer: pq.ParquetWriter | None = None

    try:
        for batch in iter_parquet_batches(
            parquet_path=data_cfg.test_path,
            columns=read_cols,
            batch_size=cfg.predict_batch_size,
            row_group_stride=data_cfg.row_group_stride,
            row_group_offset=data_cfg.row_group_offset,
            max_rows=data_cfg.max_test_rows,
        ):
            preds = model.predict(batch[feature_cols].to_numpy(dtype=np.float32, copy=False))
            out = batch[pred_cols].copy()
            out["pred"] = np.asarray(preds, dtype=np.float32)

            table = pa.Table.from_pandas(out, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(str(output_path), table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def run_lgbm_parquet(cfg: LGBMParquetConfig) -> None:
    init_reproducibility(reproducible=True, seed=42)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")

    data_cfg = _to_large_data_cfg(cfg)
    full_train, feature_cols, train_end, val_end = prepare_large_parquet_train_frame(data_cfg)
    train_df, val_df, holdout_df = _split_frame_by_date(full_train, cfg.date_col, train_end, val_end)
    del full_train

    print(f"[Data] train={len(train_df):,}, val={len(val_df):,}, holdout={len(holdout_df):,}")
    print(f"[Data] feature_count={len(feature_cols)}, feature_mode={cfg.feature_mode}")

    X_train = train_df[feature_cols].to_numpy(dtype=np.float32, copy=False)
    y_train = train_df[cfg.label_col].to_numpy(dtype=np.float32, copy=False)
    X_val = val_df[feature_cols].to_numpy(dtype=np.float32, copy=False)
    y_val = val_df[cfg.label_col].to_numpy(dtype=np.float32, copy=False)

    model = _make_model(cfg)
    callbacks = [
        lgb.early_stopping(stopping_rounds=cfg.early_stopping_rounds, verbose=True),
        lgb.log_evaluation(50),
    ]

    if cfg.objective == "lambdarank":
        if not train_df[cfg.date_col].is_monotonic_increasing:
            train_df = train_df.sort_values(cfg.date_col)
        if not val_df[cfg.date_col].is_monotonic_increasing:
            val_df = val_df.sort_values(cfg.date_col)
        train_groups = train_df.groupby(cfg.date_col).size().to_numpy()
        val_groups = val_df.groupby(cfg.date_col).size().to_numpy()
        y_train_rel = _to_relevance_by_date(train_df[cfg.label_col], train_df[cfg.date_col])
        y_val_rel = _to_relevance_by_date(val_df[cfg.label_col], val_df[cfg.date_col])
        assert isinstance(model, lgb.LGBMRanker)
        model.fit(
            X_train,
            y_train_rel,
            group=train_groups,
            eval_set=[(X_val, y_val_rel)],
            eval_group=[val_groups],
            eval_metric="ndcg",
            eval_at=[10, 20, 50],
            callbacks=callbacks,
        )
    else:
        assert isinstance(model, lgb.LGBMRegressor)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks,
        )

    holdout_pred = model.predict(holdout_df[feature_cols].to_numpy(dtype=np.float32, copy=False))
    holdout_eval = pd.DataFrame(
        {
            "date": holdout_df[cfg.date_col].to_numpy(),
            "pred": np.asarray(holdout_pred, dtype=np.float32),
            "y": holdout_df[cfg.label_col].to_numpy(dtype=np.float32, copy=False),
        }
    )
    holdout_ic = pearson_ic_by_date(holdout_eval, date_col="date", score_col="pred", target_col="y")
    holdout_rankic = rank_ic_by_date(holdout_eval, date_col="date", score_col="pred", target_col="y")
    holdout_mse = float(np.mean((holdout_eval["pred"] - holdout_eval["y"]) ** 2))

    print(f"[Holdout] mse={holdout_mse:.6f}, ic={holdout_ic:.5f}, rankic={holdout_rankic:.5f}")

    meta = {
        "run_id": run_id,
        "feature_cols": feature_cols,
        "train_end": train_end,
        "val_end": val_end,
        "holdout_mse": holdout_mse,
        "holdout_ic": float(holdout_ic),
        "holdout_rankic": float(holdout_rankic),
        "config": asdict(cfg),
    }
    with open(out_dir / f"run_{run_id}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)

    if cfg.save_test_pred and cfg.test_parquet_path:
        pred_path = out_dir / f"run_{run_id}_test_pred.parquet"
        _stream_test_predictions(cfg, feature_cols, model, pred_path)
        print(f"[Saved] {pred_path}")


if __name__ == "__main__":
    config = LGBMParquetConfig(
        train_parquet_path="data/train.parquet",
        test_parquet_path="data/test.parquet",
        label_col="LabelA",
        feature_mode="no_missing",
        max_train_rows=3_000_000,
        batch_size=200_000,
        row_group_stride=1,
        objective="regression",
        device_type="cpu",
    )
    run_lgbm_parquet(config)
