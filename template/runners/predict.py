"""Unified prediction runner for persisted GBDT models."""

# pyright: reportMissingImports=false, reportMissingTypeArgument=false, reportArgumentType=false
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal, cast
import os
import sys

_TEMPLATE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_TEMPLATE_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _TEMPLATE_DIR)

from template.data.io_parquet import iter_test_batches, stream_predictions_to_parquet
from template.data.preprocess import PreprocessorState, transform_features
from template.models.gbdt.lgbm_model import LGBMModelWrapper
from template.models.gbdt.xgb_model import XGBModelWrapper
from template.data.parquet_adapter import LargeParquetConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified prediction entrypoint for GBDT models")
    parser.add_argument("--family", choices=["lgbm", "xgb"], required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--meta-path", required=True)
    parser.add_argument("--output-path", default="res/pipeline/predict_output.parquet")
    parser.add_argument("--test-parquet", default="")
    return parser


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
    )


def main() -> None:
    args = _build_parser().parse_args()

    with open(args.meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    feature_cols = list(meta["feature_cols"])
    data_cfg = dict(meta["data_config"])
    preprocess_state = _state_from_dict(dict(meta["preprocess_state"]))

    if args.test_parquet:
        data_cfg["test_parquet_path"] = args.test_parquet
    test_parquet_path = str(data_cfg.get("test_parquet_path", ""))
    if not test_parquet_path:
        raise ValueError("test_parquet_path is empty, provide --test-parquet or valid meta data_config")
    if not os.path.exists(test_parquet_path):
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

    wrapper = _load_wrapper(args.family, args.model_path)

    def _iter_batches():
        for batch in iter_test_batches(large_cfg, feature_cols, batch_size=int(large_cfg.batch_size)):
            yield transform_features(batch, feature_cols, preprocess_state, copy_frame=False)

    out = stream_predictions_to_parquet(
        output_path=args.output_path,
        id_cols=large_cfg.id_cols,
        batch_iter=_iter_batches(),
        predict_fn=wrapper.predict,
        feature_cols=feature_cols,
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    print(f"[Done] prediction saved to {out}")


if __name__ == "__main__":
    main()
