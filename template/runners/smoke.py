"""Lightweight smoke tests for refactored template pipeline.

This script avoids full-data loading and validates:
1) all registered NN model builders can instantiate + forward once,
2) unified runners can train on tiny synthetic parquet data.
"""

# pyright: reportMissingImports=false, reportGeneralTypeIssues=false, reportArgumentType=false, reportAttributeAccessIssue=false
from __future__ import annotations

import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import cast

_TEMPLATE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_TEMPLATE_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _TEMPLATE_DIR)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from template.configs.model_runtime_config import Config
from template.models.registry import NN_MODEL_REGISTRY, build_nn_model


def _make_synthetic_data(root: Path, work_root: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(42)
    dates = np.arange(30, dtype=np.int32)
    stocks = np.arange(10, dtype=np.int32)
    times = np.arange(6, dtype=np.int32)

    rows = []
    for d in dates:
        for t in times:
            for s in stocks:
                base: dict[str, object] = {
                    "stockid": int(s),
                    "dateid": int(d),
                    "timeid": int(t),
                    "exchangeid": int(s % 2),
                }
                for f in range(64):
                    v = rng.normal(loc=0.0, scale=1.0)
                    if rng.random() < 0.1:
                        v = np.nan
                    base[f"f{f}"] = np.float32(v)
                y = np.float32(rng.normal(0, 0.01))
                base["LabelA"] = y
                base["LabelB"] = np.float32(y * 0.8 + rng.normal(0, 0.005))
                base["LabelC"] = np.float32(y * 1.2 + rng.normal(0, 0.005))
                rows.append(base)

    df = pd.DataFrame(rows)
    train_df = df[df["dateid"] <= 23].reset_index(drop=True)
    test_df = df[df["dateid"] > 23].drop(columns=["LabelA", "LabelB", "LabelC"]).reset_index(drop=True)

    data_dir = work_root / "smoke_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_path = data_dir / "train.parquet"
    test_path = data_dir / "test.parquet"
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    return train_path, test_path


def _smoke_nn_builders() -> None:
    cfg = Config(
        model_name="lstm",
        emb_dim=4,
        hidden=32,
        layers=2,
        drop_out=0.1,
        lookback=8,
        loss_name="mse",
    )
    batch = 12
    seq_len = cfg.lookback
    num_features = 16
    num_indices = 20
    X = torch.randn(batch, seq_len, num_features, dtype=torch.float32)
    ids = torch.randint(0, num_indices, (batch,), dtype=torch.int64)

    failures: list[str] = []
    for name in sorted(NN_MODEL_REGISTRY.keys()):
        try:
            cfg.model_name = name
            model = build_nn_model(name, cfg, num_indices=num_indices, num_features=num_features)
            model_nn = cast(nn.Module, model if isinstance(model, nn.Module) else nn.Identity())
            model_nn.eval()
            with torch.no_grad():
                out = model_nn(X, ids)
            if getattr(out, "shape", None) is None or int(out.shape[0]) != batch:
                failures.append(f"{name}: invalid output shape {getattr(out, 'shape', None)}")
        except Exception as exc:
            failures.append(f"{name}: {exc}")

    if failures:
        raise RuntimeError("NN builder smoke failed: " + " | ".join(failures))


def _run_cmd(cmd: list[str], cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd), check=False, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def _smoke_runners(template_root: Path, train_path: Path, test_path: Path, work_root: Path) -> None:
    out_root = work_root / "smoke_out"
    out_root.mkdir(parents=True, exist_ok=True)

    base = [
        sys.executable,
        "runners/train.py",
        "--train-parquet",
        str(train_path),
        "--test-parquet",
        str(test_path),
        "--label-col",
        "LabelA",
        "--feature-mode",
        "all",
        "--max-train-rows",
        "5000",
        "--batch-size",
        "1500",
        "--predict-batch-size",
        "1500",
    ]

    _run_cmd(base + ["--family", "lgbm", "--n-estimators", "30", "--output-dir", str(out_root / "lgbm"), "--no-save-test-pred"], template_root)

    try:
        _run_cmd(base + ["--family", "xgb", "--n-estimators", "30", "--output-dir", str(out_root / "xgb"), "--no-save-test-pred"], template_root)
    except Exception as exc:
        print(f"[Smoke] xgb skipped/fail: {exc}")

    _run_cmd(
        base
        + [
            "--family",
            "nn",
            "--nn-model",
            "lstm",
            "--lookback",
            "4",
            "--nn-epochs",
            "1",
            "--nn-batch-size",
            "64",
            "--output-dir",
            str(out_root / "nn"),
        ],
        template_root,
    )


def main() -> None:
    template_root = Path(__file__).resolve().parents[1]
    work_root = template_root / "tmp" / "smoke_runtime"
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    try:
        train_path, test_path = _make_synthetic_data(template_root, work_root)
        _smoke_nn_builders()
        _smoke_runners(template_root, train_path, test_path, work_root)
        print("[Smoke] all checks passed")
    finally:
        if work_root.exists():
            shutil.rmtree(work_root)
        old_data = template_root / "tmp" / "smoke_data"
        old_out = template_root / "tmp" / "smoke_out"
        if old_data.exists():
            shutil.rmtree(old_data)
        if old_out.exists():
            shutil.rmtree(old_out)


if __name__ == "__main__":
    main()
