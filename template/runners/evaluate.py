"""Unified evaluation runner for prediction files."""

# pyright: reportMissingImports=false
from __future__ import annotations

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

# 直接运行本脚本时使用的默认参数，请按需修改 pred_path
DEFAULT_EVALUATE = {
    "pred_path": "",
    "date_col": "date",
    "score_col": "pred",
    "target_col": "y",
    "max_rows_for_ic": 3000000,
    "output_json": "",
}


def run_evaluate(
    pred_path: str,
    date_col: str = "date",
    score_col: str = "pred",
    target_col: str = "y",
    max_rows_for_ic: int = 3000000,
    output_json: str = "",
) -> dict:
    """Evaluate a prediction parquet file; returns the report dict."""
    pf = pq.ParquetFile(pred_path)
    all_cols = list(pf.schema.names)

    if target_col not in all_cols:
        report = {
            "rows": int(pf.metadata.num_rows),
            "columns": all_cols,
            "note": (
                f"target column '{target_col}' not found; this is expected for test predictions. "
                "Use a holdout/validation prediction parquet containing both score and target columns for IC/MSE evaluation."
            ),
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return report

    required_cols = [date_col, score_col, target_col]
    missing = [c for c in required_cols if c not in all_cols]
    if missing:
        raise ValueError(f"prediction file missing required columns: {missing}")

    total_rows = int(pf.metadata.num_rows)
    sum_abs = 0.0
    sum_sq = 0.0
    sum_y = 0.0
    sum_y_sq = 0.0
    n = 0
    ic_frames: list[pd.DataFrame] = []
    can_collect_ic = total_rows <= max_rows_for_ic

    for batch in pf.iter_batches(batch_size=200000, columns=required_cols, use_threads=True):
        df = batch.to_pandas()
        pred = df[score_col].to_numpy(dtype=np.float64, copy=False)
        y = df[target_col].to_numpy(dtype=np.float64, copy=False)
        diff = pred - y
        sum_abs += float(np.abs(diff).sum())
        sum_sq += float((diff ** 2).sum())
        sum_y += float(y.sum())
        sum_y_sq += float((y ** 2).sum())
        n += len(df)
        if can_collect_ic:
            ic_frames.append(df[[date_col, score_col, target_col]].copy())

    y_mean = sum_y / max(n, 1)
    sum_y_dev_sq = sum_y_sq - n * y_mean ** 2

    mse = sum_sq / max(n, 1)
    mae = sum_abs / max(n, 1)
    r2 = 1.0 - sum_sq / sum_y_dev_sq if sum_y_dev_sq > 1e-15 else float("nan")

    ic = float("nan")
    rankic = float("nan")
    note = ""
    if can_collect_ic and ic_frames:
        ic_df = pd.concat(ic_frames, axis=0, ignore_index=True)
        ic = float(pearson_ic_by_date(ic_df, date_col=date_col, score_col=score_col, target_col=target_col))
        rankic = float(rank_ic_by_date(ic_df, date_col=date_col, score_col=score_col, target_col=target_col))
    else:
        note = f"ic/rankic skipped because rows({total_rows}) > max_rows_for_ic({max_rows_for_ic})"

    report = {
        "rows": n,
        "mse": float(mse),
        "mae": float(mae),
        "r2": float(r2),
        "ic": ic,
        "rankic": rankic,
        "note": note,
    }

    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    if output_json:
        save_json_report(output_json, report)
    return report


def main() -> None:
    opts = DEFAULT_EVALUATE
    if not opts["pred_path"]:
        print("Set DEFAULT_EVALUATE['pred_path'] in evaluate.py (or call run_evaluate(pred_path=..., ...) with explicit args).")
        return
    run_evaluate(
        pred_path=opts["pred_path"],
        date_col=opts["date_col"],
        score_col=opts["score_col"],
        target_col=opts["target_col"],
        max_rows_for_ic=opts["max_rows_for_ic"],
        output_json=opts["output_json"],
    )


if __name__ == "__main__":
    main()
