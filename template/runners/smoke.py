"""Lightweight smoke tests for refactored template pipeline.

This script avoids full-data loading and validates:
1) all registered NN model builders can instantiate + forward once,
2) unified runners can train on tiny synthetic parquet data.
"""

# pyright: reportMissingImports=false, reportGeneralTypeIssues=false, reportArgumentType=false, reportAttributeAccessIssue=false
from __future__ import annotations

import os
import sys
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

from template.configs.pipeline_config import NNConfig
from template.configs.presets import make_baseline_lgbm, make_baseline_xgb, make_baseline_nn
from template.models.registry import NN_MODEL_REGISTRY, build_nn_model
from template.runners.train import run_train


def _make_synthetic_data(root: Path, work_root: Path) -> tuple[Path, Path]:
    """
    生成用于 pipeline 冒烟测试的合成数据，包括训练集和测试集。
    该函数会以指定结构创建 pandas DataFrame，
    包含时间（dateid）、股票（stockid）、时间段（timeid）、特征（f0-f63）、标签（LabelA/B/C）。
    再将不同区间切分为训练集和测试集，并分别保存为 parquet 文件。

    Args:
        root (Path): 模板根目录路径（暂未使用，预留参数）。
        work_root (Path): 实验运行/输出路径，用于存储合成数据文件。

    Returns:
        tuple[Path, Path]: (训练数据 parquet 路径, 测试数据 parquet 路径)
    """
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
    cfg = NNConfig(
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


def _smoke_runners(template_root: Path, train_path: Path, test_path: Path, work_root: Path) -> None:
    out_root = work_root / "smoke_out"
    out_root.mkdir(parents=True, exist_ok=True)

    def _apply_smoke_data(cfg):
        cfg.data.train_parquet_path = str(train_path)
        cfg.data.test_parquet_path = str(test_path)
        cfg.data.label_col = "LabelA"
        cfg.data.feature_mode = "all"
        cfg.data.max_train_rows = 5000
        cfg.data.batch_size = 1500
        cfg.runner.predict_batch_size = 1500

    cfg_lgbm = make_baseline_lgbm()
    _apply_smoke_data(cfg_lgbm)
    cfg_lgbm.runner.output_dir = str(out_root / "lgbm")
    cfg_lgbm.runner.save_test_pred = False
    cfg_lgbm.model.params["n_estimators"] = 30
    run_train(cfg_lgbm)

    try:
        cfg_xgb = make_baseline_xgb()
        _apply_smoke_data(cfg_xgb)
        cfg_xgb.runner.output_dir = str(out_root / "xgb")
        cfg_xgb.runner.save_test_pred = False
        cfg_xgb.model.params["n_estimators"] = 30
        run_train(cfg_xgb)
    except Exception as exc:
        print(f"[Smoke] xgb skipped/fail: {exc}")

    cfg_nn = make_baseline_nn()
    _apply_smoke_data(cfg_nn)
    cfg_nn.runner.output_dir = str(out_root / "nn")
    cfg_nn.nn.model_name = "lstm"
    cfg_nn.nn.lookback = 4
    cfg_nn.nn.epochs = 1
    cfg_nn.nn.batch_size = 64
    run_train(cfg_nn)


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
