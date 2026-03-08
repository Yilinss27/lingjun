# pyright: reportMissingImports=false
from __future__ import annotations

from pathlib import Path
import json
from typing import Any, cast

from .experiment_log import ExperimentConfig, log_experiment_full, log_experiment_full_csv

"""
eval/reporting.py - 实验报告与日志的统一输出模块

功能说明:
- 提供实验评估结果的持久化、保存与日志功能
- 支持保存 JSON 报告、实验日志与 CSV 完整记录
- 集成 experiment_log 日志工具
- 与 EvalReport、log_metrics_bundle 等通用评估流程协同使用

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
