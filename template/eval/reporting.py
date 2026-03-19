# pyright: reportMissingImports=false
from __future__ import annotations

import csv
import os
from pathlib import Path
import json
import time
from typing import Any, cast

from .experiment_log import ExperimentConfig, log_experiment_full, log_experiment_full_csv

"""
eval/reporting.py - 实验报告与日志的统一输出模块

功能说明:
- 提供实验评估结果的持久化、保存与日志功能
- 支持保存 JSON 报告、实验日志与 CSV 完整记录
- 集成 experiment_log 日志工具
- 与 EvalReport、log_metrics_bundle 等通用评估流程协同使用
- 提供 append_leaderboard 用于自动追加训练结果到排行榜 CSV

设计原则:
- 面向多模型实验结果输出
- 保持接口代码简洁、类型清晰、目录结构清楚
"""

def save_json_report(path: str, report: dict[str, Any]) -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    return str(out)


def log_metrics_bundle(
    cfg: Any,
    metrics: dict[str, float],
    *,
    log_dir: str,
    experiment_name: str,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, str]:
    typed_cfg = cast(ExperimentConfig, cfg)
    paths = log_experiment_full(
        typed_cfg,
        metrics=metrics,
        log_dir=log_dir,
        experiment_name=experiment_name,
        extra_info=extra_info,
    )
    csv_path = log_experiment_full_csv(
        typed_cfg,
        metrics=metrics,
        log_dir=log_dir,
        experiment_name=experiment_name,
        extra_info=extra_info,
    )
    paths["csv_full"] = csv_path
    return paths


# ============================================================
# Leaderboard — 一张 CSV 汇总所有训练结果
# ============================================================

# 列顺序：固定列头，保证 CSV 可读且稳定
LEADERBOARD_COLUMNS = [
    "run_id",
    "timestamp",
    "family",
    "model_name",
    # 关键超参
    "lookback",
    "hidden",
    "layers",
    "lr",
    "batch_size",
    "epochs",
    "loss_fn",
    "dropout",
    # 预处理
    "fillna",
    "standardize",
    "std_mode",
    "n_features",
    # 评价指标
    "ic",
    "ic_ir",
    "rankic",
    "rankic_ir",
    "r2",
    "mse",
    "mae",
    "loss",
    # 训练过程
    "final_train_loss",
    "final_val_loss",
    "final_val_r2",
    # 备注
    "note",
]


def append_leaderboard(
    report: dict[str, Any],
    cfg: Any = None,
    note: str = "",
    leaderboard_path: str = "res/leaderboard.csv",
) -> str:
    """从 run_train 返回的 report 字典提取关键信息，追加一行到排行榜 CSV。

    Args:
        report: run_train() 返回的完整 report 字典
        cfg: PipelineConfig 对象（可选，用于补充 report 中缺失的信息）
        note: 备注文字（如 "首次基线" "换用 transformer" 等）
        leaderboard_path: CSV 文件路径

    Returns:
        CSV 文件的绝对路径
    """
    metrics = report.get("metrics", {})
    history = report.get("history", {})
    nn_cfg = report.get("nn_config", {})
    prep = report.get("preprocess_state", {})
    feature_cols = report.get("feature_cols", [])

    # 如果 report 里缺字段，尝试从 cfg 补
    if cfg is not None:
        if not nn_cfg and hasattr(cfg, "nn"):
            from dataclasses import asdict
            nn_cfg = asdict(cfg.nn)
        if not prep and hasattr(cfg, "preprocess"):
            from dataclasses import asdict
            prep = asdict(cfg.preprocess)

    row = {
        "run_id": report.get("run_id", ""),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "family": report.get("family", ""),
        "model_name": nn_cfg.get("model_name", "") or report.get("model_name", ""),
        # 超参
        "lookback": nn_cfg.get("lookback", ""),
        "hidden": nn_cfg.get("hidden", ""),
        "layers": nn_cfg.get("layers", ""),
        "lr": nn_cfg.get("lr", ""),
        "batch_size": nn_cfg.get("batch_size", ""),
        "epochs": nn_cfg.get("epochs", ""),
        "loss_fn": nn_cfg.get("loss_name", ""),
        "dropout": nn_cfg.get("drop_out", ""),
        # 预处理
        "fillna": prep.get("fillna_strategy", ""),
        "standardize": prep.get("standardize", ""),
        "std_mode": prep.get("standardize_mode", ""),
        "n_features": len(feature_cols),
        # 评价指标
        "ic": _fmt(metrics.get("ic")),
        "ic_ir": _fmt(metrics.get("ic_ir")),
        "rankic": _fmt(metrics.get("rankic")),
        "rankic_ir": _fmt(metrics.get("rankic_ir")),
        "r2": _fmt(metrics.get("r2")),
        "mse": _fmt(metrics.get("mse")),
        "mae": _fmt(metrics.get("mae")),
        "loss": _fmt(metrics.get("loss")),
        # 训练过程
        "final_train_loss": _fmt(history.get("train_loss")),
        "final_val_loss": _fmt(history.get("val_loss")),
        "final_val_r2": _fmt(history.get("val_r2")),
        # 备注
        "note": note,
    }

    # GBDT 没有 nn_config，补充 model_config 信息
    if not nn_cfg and report.get("model_config"):
        mc = report["model_config"]
        row["model_name"] = mc.get("family", row["family"])
        # GBDT 的关键参数放 note
        params = mc.get("params", {})
        if params and not note:
            parts = []
            for k in ("n_estimators", "num_leaves", "max_depth", "learning_rate"):
                if k in params:
                    parts.append(f"{k}={params[k]}")
            row["note"] = "; ".join(parts)

    # 写 CSV
    path = Path(leaderboard_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0

    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=LEADERBOARD_COLUMNS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"[Leaderboard] 结果已追加到 {path}")
    return str(path.resolve())


def _fmt(v: Any, digits: int = 6) -> str:
    """格式化数值，None 返回空字符串。"""
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)
