"""Unified evaluation runner for prediction files."""

# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
import json
import os
import sys

_TEMPLATE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_TEMPLATE_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _TEMPLATE_DIR)

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from template.eval import pearson_ic_by_date, rank_ic_by_date, save_json_report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate prediction parquet file")
    parser.add_argument("--pred-path", required=True)
    parser.add_argument("--date-col", default="date")
    parser.add_argument("--score-col", default="pred")
    parser.add_argument("--target-col", default="y")
    parser.add_argument("--max-rows-for-ic", type=int, default=3000000)
    parser.add_argument("--output-json", default="")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    pf = pq.ParquetFile(args.pred_path)
    all_cols = list(pf.schema.names)

    if args.target_col not in all_cols:
        report = {
            "rows": int(pf.metadata.num_rows),
            "columns": all_cols,
            "note": (
                f"target column '{args.target_col}' not found; this is expected for test predictions. "
                "Use a holdout/validation prediction parquet containing both score and target columns for IC/MSE evaluation."
            ),
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    required_cols = [args.date_col, args.score_col, args.target_col]
    missing = [c for c in required_cols if c not in all_cols]
    if missing:
        raise ValueError(f"prediction file missing required columns: {missing}")

    total_rows = int(pf.metadata.num_rows)
    sum_abs = 0.0
    sum_sq = 0.0
    n = 0
    ic_frames: list[pd.DataFrame] = []
    can_collect_ic = total_rows <= int(args.max_rows_for_ic)

    for batch in pf.iter_batches(batch_size=200000, columns=required_cols, use_threads=True):
        df = batch.to_pandas()
        pred = df[args.score_col].to_numpy(dtype=np.float32, copy=False)
        y = df[args.target_col].to_numpy(dtype=np.float32, copy=False)
        diff = pred - y
        sum_abs += float(np.abs(diff).sum())
        sum_sq += float((diff ** 2).sum())
        n += len(df)
        if can_collect_ic:
            ic_frames.append(df[[args.date_col, args.score_col, args.target_col]].copy())

    mse = sum_sq / max(n, 1)
    mae = sum_abs / max(n, 1)

    ic = float("nan")
    rankic = float("nan")
    note = ""
    if can_collect_ic and ic_frames:
        ic_df = pd.concat(ic_frames, axis=0, ignore_index=True)
        ic = float(pearson_ic_by_date(ic_df, date_col=args.date_col, score_col=args.score_col, target_col=args.target_col))
        rankic = float(rank_ic_by_date(ic_df, date_col=args.date_col, score_col=args.score_col, target_col=args.target_col))
    else:
        note = f"ic/rankic skipped because rows({total_rows}) > max_rows_for_ic({args.max_rows_for_ic})"

    report = {
        "rows": n,
        "mse": float(mse),
        "mae": float(mae),
        "ic": ic,
        "rankic": rankic,
        "note": note,
    }

    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    if args.output_json:
        save_json_report(args.output_json, report)


if __name__ == "__main__":
    main()
