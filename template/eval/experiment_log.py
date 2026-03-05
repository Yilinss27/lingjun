"""
eval/experiment_log.py - 跨模型通用的实验结果落盘模块

提供实验结果的记录和管理功能，包括：
- 文本格式日志记录
- CSV 格式结果表格
- JSON 格式完整配置保存
- 便于后续做对比表和画大图

设计原则:
- 所有函数为纯函数，不依赖全局状态
- 可用于任意模型的实验记录，不限于 LSTM
- 支持灵活的配置协议
- 自动创建必要的目录
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable, Optional, Dict, Any, List
import time
import json
import os
import pandas as pd

__all__ = [
    # 核心记录函数
    "log_experiment_results",
    # 高级功能
    "log_experiment_full",
    "log_experiment_full_csv",
    "load_experiment_history",
    "compare_experiments",
    # 配置协议
    "ExperimentConfig",
]


# ============================================================
# 配置协议定义
# ============================================================
@runtime_checkable
class ExperimentConfig(Protocol):
    """实验记录模块所需的配置协议，任何具有这些属性的对象都可使用"""
    loss_name: str
    lr: float
    epochs: int
    batch_size: int
    lookback: int
    feature_cols: List[str]
    
    # 可选属性（使用 getattr 获取）
    # emb_dim: int
    # lstm_hidden: int
    # lstm_layers: int
    # drop_out: float


# ============================================================
# 核心记录函数
# ============================================================
def log_experiment_results(
    cfg: ExperimentConfig,
    test_loss: float,
    test_mse: float,
    test_mae: float,
    test_ric: float,
    best_val_loss: float,
    log_file: str = "lstm_experiments.log",
    extra_log_paths: Optional[List[str]] = None,
    model_name: str = "LSTM"
) -> str:
    """
    记录实验结果到日志文件（文本格式）
    
    Args:
        cfg: 配置对象，需符合 ExperimentConfig 协议
        test_loss: 测试集损失
        test_mse: 测试集 MSE
        test_mae: 测试集 MAE
        test_ric: 测试集 RankIC
        best_val_loss: 最佳验证集损失
        log_file: 主日志文件路径
        extra_log_paths: 额外的日志路径列表（如网络路径）
        model_name: 模型名称（用于日志标题）
    
    Returns:
        str: 日志内容
    
    Example:
        >>> from eval.experiment_log import log_experiment_results
        >>> log_experiment_results(cfg, test_loss, test_mse, test_mae, test_ric, best_val_loss)
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # 获取模型配置（使用 getattr 处理可选属性）
    emb_dim = getattr(cfg, 'emb_dim', 0)
    hidden = getattr(cfg, 'hidden', 0)
    layers = getattr(cfg, 'layers', 0)
    drop_out = getattr(cfg, 'drop_out', 0.0)
    
    # 构建日志内容
    result = f"""[{timestamp}] {model_name} Experiment Results
Features: {cfg.feature_cols[:5]}{'...' if len(cfg.feature_cols) > 5 else ''} ({len(cfg.feature_cols)} total)
Config: emb_dim={emb_dim}, hidden={hidden}, layers={layers}, dropout={drop_out}
        lr={cfg.lr}, epochs={cfg.epochs}, batch_size={cfg.batch_size}, lookback={cfg.lookback}, loss_fn={cfg.loss_name}
Test Results: {cfg.loss_name}={test_loss:.6f}, MSE={test_mse:.6f}, MAE={test_mae:.6f}, RankIC={test_ric:.4f}
Best Val {cfg.loss_name}={best_val_loss:.6f}
{'-'*60}
"""
    
    # 写入主日志文件
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(result)
        print(f"实验结果已记录到: {log_file}")
    except OSError as e:
        print(f"警告: 无法写入主日志文件 {log_file}: {e}")
    
    # 写入额外日志路径
    if extra_log_paths:
        for path in extra_log_paths:
            try:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(result)
            except OSError:
                pass  # 静默忽略网络路径错误
    
    return result


def log_experiment_full(
    cfg: ExperimentConfig,
    metrics: Dict[str, float],
    log_dir: str = "log/experiments",
    experiment_name: Optional[str] = None,
    extra_info: Optional[Dict[str, Any]] = None
) -> Dict[str, str]:
    """
    完整记录实验结果（包括配置、指标、时间戳等）
    
    生成三种格式的记录：
    - .txt: 人类可读的文本摘要
    - .csv: 追加到汇总表格（方便对比）
    - .json: 完整的配置和结果快照
    
    Args:
        cfg: 配置对象
        metrics: 指标字典，如 {'test_loss': 0.1, 'test_mse': 0.2, 'test_rankic': 0.05}
        log_dir: 日志目录
        experiment_name: 实验名称（默认使用时间戳）
        extra_info: 额外信息（如 git commit hash, 备注等）
    
    Returns:
        Dict[str, str]: 生成的文件路径字典
    
    Example:
        >>> from eval.experiment_log import log_experiment_full
        >>> metrics = {'test_loss': 0.1, 'test_mse': 0.2, 'test_rankic': 0.05}
        >>> paths = log_experiment_full(cfg, metrics, experiment_name="baseline_v1")
    """
    # 创建目录
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_name = experiment_name or f"exp_{timestamp}"
    
    # 准备完整记录
    cfg_any: Any = cfg
    model_name = getattr(cfg_any, "model_name", None)
    if not model_name:
        model_func = getattr(cfg_any, "model", None)
        if callable(model_func):
            try:
                model_name = model_func()
            except Exception:
                model_name = None

    record = {
        "experiment_name": exp_name,
        "timestamp": timestamp,
        "model_name": model_name or "LSTM",
        "metrics": metrics,
        "extra_info": extra_info or {},
    }
    
    # 尝试序列化配置
    try:
        to_dict = getattr(cfg_any, "to_dict", None)
        if callable(to_dict):
            record["config"] = to_dict()
        else:
            record["config"] = dict(getattr(cfg_any, "__dict__", {}))
    except Exception:
        # 如果无法序列化，只保存关键属性
        record["config"] = {
            "loss_name": cfg.loss_name,
            "lr": cfg.lr,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "lookback": cfg.lookback,
            "feature_cols": cfg.feature_cols,
        }
    
    saved_paths = {}
    
    # 1. 保存 JSON 完整快照
    json_path = os.path.join(log_dir, f"{exp_name}.json")
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False, default=str)
        saved_paths["json"] = json_path
    except Exception as e:
        print(f"警告: 无法保存 JSON: {e}")
    
    # 2. 保存 TXT 摘要
    txt_path = os.path.join(log_dir, f"{exp_name}.txt")
    try:
        summary = _format_experiment_summary(record)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(summary)
        saved_paths["txt"] = txt_path
    except Exception as e:
        print(f"警告: 无法保存 TXT: {e}")
    
    # 3. 追加到 CSV 汇总表
    csv_path = os.path.join(log_dir, "experiments_summary.csv")
    try:
        _append_to_csv(csv_path, record)
        saved_paths["csv"] = csv_path
    except Exception as e:
        print(f"警告: 无法追加到 CSV: {e}")
    
    print(f"实验 '{exp_name}' 已完整记录到: {log_dir}")
    return saved_paths


def _stringify_csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict, tuple)):
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            return str(value)
    return value


def log_experiment_full_csv(
    cfg: ExperimentConfig,
    metrics: Dict[str, float],
    log_dir: str = "log/experiments",
    experiment_name: Optional[str] = None,
    extra_info: Optional[Dict[str, Any]] = None,
    csv_name: str = "experiments_summary_full.csv"
) -> str:
    """
    记录完整信息到 CSV（包含配置、指标、额外信息）
    """
    os.makedirs(log_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_name = experiment_name or f"exp_{timestamp}"

    cfg_any: Any = cfg
    model_name = getattr(cfg_any, "model_name", None)
    if not model_name:
        model_func = getattr(cfg_any, "model", None)
        if callable(model_func):
            try:
                model_name = model_func()
            except Exception:
                model_name = None

    try:
        to_dict = getattr(cfg_any, "to_dict", None)
        if callable(to_dict):
            config_obj = to_dict()
            if isinstance(config_obj, dict):
                config = config_obj
            else:
                config = {"config": str(config_obj)}
        else:
            cfg_dict = getattr(cfg_any, "__dict__", {})
            if isinstance(cfg_dict, dict):
                config = dict(cfg_dict)
            else:
                config = {}
    except Exception:
        config = {
            "loss_name": cfg.loss_name,
            "lr": cfg.lr,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "lookback": cfg.lookback,
            "feature_cols": cfg.feature_cols,
        }

    record: Dict[str, Any] = {
        "experiment_name": exp_name,
        "timestamp": timestamp,
    }
    if model_name:
        record["model_name"] = model_name

    for k, v in config.items():
        record[f"cfg_{k}"] = _stringify_csv_value(v)

    for k, v in metrics.items():
        record[f"metric_{k}"] = _stringify_csv_value(v)

    if extra_info:
        for k, v in extra_info.items():
            record[f"extra_{k}"] = _stringify_csv_value(v)

    csv_path = os.path.join(log_dir, csv_name)
    new_row = pd.DataFrame([record])

    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        for col in new_row.columns:
            if col not in existing.columns:
                existing[col] = None
        for col in existing.columns:
            if col not in new_row.columns:
                new_row[col] = None
        new_row = new_row[existing.columns]
        combined = pd.concat([existing, new_row], ignore_index=True)
        combined.to_csv(csv_path, index=False, encoding="utf-8-sig")
    else:
        new_row.to_csv(csv_path, index=False, encoding="utf-8-sig")

    return csv_path


def _format_experiment_summary(record: Dict[str, Any]) -> str:
    """格式化实验摘要为可读文本"""
    lines = [
        "=" * 60,
        f"Experiment: {record['experiment_name']}",
        f"Timestamp: {record['timestamp']}",
        "=" * 60,
        "",
        "## Metrics",
    ]
    
    for k, v in record.get("metrics", {}).items():
        if isinstance(v, float):
            lines.append(f"  {k}: {v:.6f}")
        else:
            lines.append(f"  {k}: {v}")
    
    lines.extend(["", "## Config"])
    for k, v in record.get("config", {}).items():
        if isinstance(v, list) and len(v) > 5:
            lines.append(f"  {k}: [{v[0]}, {v[1]}, ..., {v[-1]}] ({len(v)} items)")
        else:
            lines.append(f"  {k}: {v}")
    
    if record.get("extra_info"):
        lines.extend(["", "## Extra Info"])
        for k, v in record["extra_info"].items():
            lines.append(f"  {k}: {v}")
    
    lines.append("")
    return "\n".join(lines)


def _append_to_csv(csv_path: str, record: Dict[str, Any]) -> None:
    """追加记录到 CSV 汇总表"""
    # 扁平化记录
    flat_record = {
        "experiment_name": record["experiment_name"],
        "timestamp": record["timestamp"],
    }
    
    # 添加指标
    for k, v in record.get("metrics", {}).items():
        flat_record[f"metric_{k}"] = v
    
    # 添加关键配置
    config = record.get("config", {})
    for key in ["loss_name", "lr", "epochs", "batch_size", "lookback"]:
        if key in config:
            flat_record[f"cfg_{key}"] = config[key]
    
    # 读取现有 CSV 或创建新的
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # 添加新记录可能引入的新列
        for col in flat_record.keys():
            if col not in df.columns:
                df[col] = None
    else:
        df = pd.DataFrame()
    
    # 追加新记录
    new_row = pd.DataFrame([flat_record])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")


# ============================================================
# 实验历史加载与对比
# ============================================================
def load_experiment_history(
    log_dir: str = "log/experiments",
    filter_name: Optional[str] = None
) -> pd.DataFrame:
    """
    加载实验历史记录
    
    Args:
        log_dir: 日志目录
        filter_name: 按实验名称过滤（支持部分匹配）
    
    Returns:
        pd.DataFrame: 实验历史表格
    
    Example:
        >>> from eval.experiment_log import load_experiment_history
        >>> df = load_experiment_history()
        >>> print(df[['experiment_name', 'metric_test_rankic']].head())
    """
    csv_path = os.path.join(log_dir, "experiments_summary.csv")
    
    if not os.path.exists(csv_path):
        print(f"警告: 未找到实验历史文件 {csv_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(csv_path)
    
    if filter_name:
        df = df[df["experiment_name"].str.contains(filter_name, na=False)]
    
    return df


def compare_experiments(
    experiment_names: List[str],
    log_dir: str = "log/experiments",
    metrics: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    对比多个实验的结果
    
    Args:
        experiment_names: 要对比的实验名称列表
        log_dir: 日志目录
        metrics: 要对比的指标列表（默认所有 metric_ 开头的列）
    
    Returns:
        pd.DataFrame: 对比结果表格
    
    Example:
        >>> from eval.experiment_log import compare_experiments
        >>> df = compare_experiments(['baseline_v1', 'exp_v2'])
        >>> print(df)
    """
    df = load_experiment_history(log_dir)
    
    if len(df) == 0:
        return pd.DataFrame()
    
    # 过滤实验
    df = df[df["experiment_name"].isin(experiment_names)]
    
    if len(df) == 0:
        print(f"警告: 未找到指定的实验: {experiment_names}")
        return pd.DataFrame()
    
    # 选择列
    if metrics is None:
        metric_cols = [c for c in df.columns if c.startswith("metric_")]
    else:
        metric_cols = [f"metric_{m}" if not m.startswith("metric_") else m for m in metrics]
        metric_cols = [c for c in metric_cols if c in df.columns]
    
    result_cols = ["experiment_name", "timestamp"] + metric_cols
    result_cols = [c for c in result_cols if c in df.columns]
    
    return df[result_cols].set_index("experiment_name")


def get_best_experiment(
    log_dir: str = "log/experiments",
    metric: str = "test_rankic",
    higher_is_better: bool = True
) -> Optional[Dict[Any, Any]]:
    """
    获取最佳实验的详细信息
    
    Args:
        log_dir: 日志目录
        metric: 用于排名的指标
        higher_is_better: 指标是否越大越好
    
    Returns:
        Optional[Dict]: 最佳实验的完整记录（从 JSON 加载）
    """
    df = load_experiment_history(log_dir)
    
    metric_col = f"metric_{metric}" if not metric.startswith("metric_") else metric
    
    if metric_col not in df.columns:
        print(f"警告: 未找到指标 {metric_col}")
        return None
    
    # 找到最佳实验
    if higher_is_better:
        best_idx = df[metric_col].idxmax()
    else:
        best_idx = df[metric_col].idxmin()
    
    best_name = df.loc[best_idx, "experiment_name"]
    
    # 加载完整 JSON
    json_path = os.path.join(log_dir, f"{best_name}.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    # 如果 JSON 不存在，返回 CSV 行
    return df.loc[best_idx].to_dict()
