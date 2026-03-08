"""
PyTorch 通用训练器模块

包含：
    - 数据集类（NpDatasetTyped, DailyBatchDataset）
    - DataLoader 工厂函数
    - 训练/评估循环
    - TorchTrainer 类（统一封装）

适用于：LSTM, GRU, Transformer 等所有 PyTorch 序列模型

GPU 优化特性：
    - 混合精度训练 (AMP) - 支持 float16/bfloat16
    - 内存锁页 (pin_memory) - 加速 CPU->GPU 数据传输
    - cuDNN 自动调优 - 自动选择最优卷积算法
    - torch.compile() - PyTorch 2.0+ JIT 编译优化
"""
from __future__ import annotations
import json
import math
import os
import sys
import time
from typing import Dict, Tuple, Callable, Optional, TYPE_CHECKING, cast
from utils import seed_worker, get_generator, get_rng
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

from .checkpoint import CheckpointManager
from .losses import get_loss_fn
from eval import mse_np, mae_np, rank_ic_by_date, pearson_ic_by_date

if TYPE_CHECKING:
    from template.configs.model_runtime_config import Config


# ============================================================
# AMP 工具函数
# ============================================================

def get_amp_dtype(cfg: "Config") -> torch.dtype:
    """根据配置获取 AMP 数据类型
    
    Args:
        cfg: 配置对象，需包含 amp_dtype 属性
    
    Returns:
        torch.dtype: float16 或 bfloat16
    """
    if getattr(cfg, 'amp_dtype', 'float16') == 'bfloat16':
        # 检查设备是否支持 bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
    return torch.float16


def create_scaler(cfg: "Config", device: torch.device) -> Optional[GradScaler]:
    """创建 GradScaler（仅在 CUDA + AMP 时需要）
    
    Args:
        cfg: 配置对象
        device: PyTorch 设备
    
    Returns:
        GradScaler 或 None
    """
    use_amp = getattr(cfg, 'use_amp', False)
    if use_amp and device.type == 'cuda':
        # bfloat16 不需要 scaler（不会 overflow）
        if getattr(cfg, 'amp_dtype', 'float16') == 'bfloat16':
            return None
        return GradScaler()
    return None


# ============================================================
# 设备管理
# ============================================================

def get_torch_device(cfg: "Config") -> torch.device:
    """根据配置获取 PyTorch 设备
    
    Args:
        cfg: 配置对象，需包含 device 属性 ('auto', 'cpu', 'gpu')
    
    Returns:
        torch.device 对象
    """
    if cfg.device == "cpu":
        return torch.device("cpu")
    if cfg.device == "gpu":
        return torch.device("cuda")
    # auto: 优先使用 GPU
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 数据集类
# ============================================================

class NpDatasetTyped(Dataset):
    """NumPy数组数据集，支持多进程DataLoader
    
    用于常规 batch 训练（非按日划分）
    
    Args:
        X: 特征数组 (N, L, F) 或 (N, F)
        ids: 股票ID数组 (N,)
        y: 标签数组 (N,)
    """
    def __init__(self, X: np.ndarray, ids: np.ndarray, y: np.ndarray):
        self.X = X
        self.ids = ids
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.X[i], self.ids[i], self.y[i]


def collate_fn(batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """批处理函数：将 numpy 数组转换为 torch tensor"""
    Xb, idb, yb = zip(*batch)
    Xb = torch.tensor(np.stack(Xb, axis=0), dtype=torch.float32)
    idb = torch.tensor(np.array(idb), dtype=torch.long)
    yb = torch.tensor(np.stack(yb, axis=0), dtype=torch.float32).unsqueeze(-1)
    return Xb, idb, yb


class DailyBatchDataset:
    """按交易日组织的数据集，每个batch是同一天的所有股票
    
    用于IC/RankIC损失函数训练，确保每个batch内的排序是日内截面排序
    
    GPU 优化特性：
        - 预先转换为 Tensor 并使用 pin_memory 加速 CPU->GPU 传输
        - 避免每次 get_daily_batch 时重复创建 Tensor
    
    Args:
        X: 特征数组 (N, L, F)
        ids: 股票ID数组 (N,)
        y: 标签数组 (N,)
        dates: 日期数组 (N,)
        pin_memory: 是否使用内存锁页（需要 CUDA 支持）
    """
    def __init__(
        self, 
        X: np.ndarray, 
        ids: np.ndarray, 
        y: np.ndarray, 
        dates: np.ndarray,
        pin_memory: bool = True
    ):
        # GPU 优化：预先转换为 Tensor
        self.X = torch.from_numpy(X).float()
        self.ids = torch.from_numpy(ids).long()
        self.y = torch.from_numpy(y).float()
        self.dates = dates
        
        # 如果 CUDA 可用且启用 pin_memory，则锁页内存
        if pin_memory and torch.cuda.is_available():
            self.X = self.X.pin_memory()
            self.ids = self.ids.pin_memory()
            self.y = self.y.pin_memory()
        
        # 按日期分组，记录每个日期对应的样本索引
        unique_dates = np.unique(dates)
        self.date_to_indices: Dict[np.datetime64, np.ndarray] = {}
        for d in unique_dates:
            self.date_to_indices[d] = np.where(dates == d)[0]
        self.unique_dates = list(self.date_to_indices.keys())
    
    def __len__(self) -> int:
        return len(self.unique_dates)
    
    def get_daily_batch(self, day_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取某一天的所有样本作为一个batch
        
        GPU 优化：直接切片已转换的 Tensor，避免重复创建
        """
        date = self.unique_dates[day_idx]
        indices = self.date_to_indices[date]
        
        # 直接索引已转换的 Tensor（比每次创建新 Tensor 更快）
        X_day = self.X[indices]
        ids_day = self.ids[indices]
        y_day = self.y[indices].unsqueeze(-1)
        
        return X_day, ids_day, y_day
    
    def iter_daily_batches(self, shuffle: bool = True, extra_seed: int = 0):
        """迭代所有日期的batch
        
        随机性由 repro 模块统一管理，可复现模式下使用固定种子
        
        Args:
            shuffle: 是否打乱日期顺序（注意：日内样本顺序不打乱）
            extra_seed: 额外的种子偏移量（例如 epoch 编号），用于生成不同但可复现的序列
        
        Yields:
            (X_day, ids_day, y_day) 元组
        """
        day_indices = list(range(len(self.unique_dates)))
        if shuffle:
            rng = get_rng(extra_seed)
            rng.shuffle(day_indices)
        
        for day_idx in day_indices:
            yield self.get_daily_batch(day_idx)


# ============================================================
# DataLoader 工厂函数
# ============================================================

def make_dataloader(
    split_data: Dict[str, np.ndarray], 
    cfg: "Config", 
    shuffle: bool = True, 
) -> DataLoader:
    """创建优化的DataLoader，支持可复现的随机性控制
    
    GPU 优化特性：
        - pin_memory: 加速 CPU->GPU 数据传输（由 cfg.pin_memory 控制）
        - num_workers: 多进程数据加载（由 cfg.dataloader_workers 控制）
        - prefetch_factor: 预取批次数（由 cfg.prefetch_factor 控制）
    
    Args:
        split_data: 数据字典，需包含 "X", "id", "y" 键
        cfg: 配置对象，需包含 batch_size 及 GPU 优化属性
        shuffle: 是否打乱数据
    
    Returns:
        DataLoader 对象
    """
    ds = NpDatasetTyped(split_data["X"], split_data["id"], split_data["y"])

    # GPU 优化：使用配置中的参数
    num_workers = getattr(cfg, 'dataloader_workers', 4)
    num_workers = min(num_workers, os.cpu_count() or 1)
    # Windows 下多进程 DataLoader 易触发 pickle/管道错误，若显式设为 0 则保持单进程
    if sys.platform == 'win32' and num_workers > 0:
        num_workers = 2
    prefetch_factor = getattr(cfg, 'prefetch_factor', 4)
    pin_memory = getattr(cfg, 'pin_memory', True) and torch.cuda.is_available()
    
    # 创建随机数生成器，用于控制 shuffle 的可复现性
    g = get_generator()

    # num_workers=0 时不能传 prefetch_factor，否则会报错
    if num_workers > 0:
        return DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            drop_last=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=True,
            pin_memory=pin_memory,
            worker_init_fn=seed_worker,
            generator=g,
        )
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        persistent_workers=False,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=g,
    )


# ============================================================
# 训练/评估循环（函数式接口）
# ============================================================

def _group_grad_norms(model: nn.Module, group_level: int) -> Dict[str, float]:
    sums: Dict[str, float] = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad
        if grad.is_sparse:
            grad = grad.coalesce().values()
        grad_norm = float(grad.norm(2).item())
        if group_level > 0:
            parts = name.split(".")
            group = ".".join(parts[:group_level]) if parts else name
        else:
            group = name
        sums[group] = sums.get(group, 0.0) + grad_norm * grad_norm
    return {k: math.sqrt(v) for k, v in sums.items()}


def _write_grad_norm_record(log_path: str, record: Dict[str, object]) -> None:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def _log_grad_norms(
    model: nn.Module,
    grad_norm_cfg: Optional[Dict[str, object]],
    epoch: int,
    step: int,
    loss_value: float,
) -> None:
    if not grad_norm_cfg or not bool(grad_norm_cfg.get("enabled")):
        return
    raw_freq = grad_norm_cfg.get("freq", 1)
    if isinstance(raw_freq, (int, float)):
        freq = int(raw_freq)
    else:
        freq = 1
    if step % freq != 0:
        return
    raw_group_level = grad_norm_cfg.get("group_level", 3)
    if isinstance(raw_group_level, (int, float)):
        group_level = int(raw_group_level)
    else:
        group_level = 3
    raw_top_k = grad_norm_cfg.get("top_k", 200)
    if isinstance(raw_top_k, (int, float)):
        top_k = int(raw_top_k)
    else:
        top_k = 200
    groups = _group_grad_norms(model, group_level)
    items = sorted(groups.items(), key=lambda item: item[1], reverse=True)
    if top_k > 0 and len(items) > top_k:
        items = items[:top_k]
    record = {
        "ts": time.time(),
        "epoch": epoch,
        "step": step,
        "loss": float(loss_value),
        "groups": [{"name": name, "norm": norm} for name, norm in items],
    }
    log_path = grad_norm_cfg.get("log_path")
    if isinstance(log_path, str) and log_path:
        _write_grad_norm_record(log_path, record)

def train_one_epoch(
    model: nn.Module, 
    loader: DataLoader, 
    optimizer: torch.optim.Optimizer, 
    loss_fn: Callable, 
    device: torch.device,
    epoch: int = 0,
    scaler: Optional[GradScaler] = None,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    grad_norm_cfg: Optional[Dict[str, object]] = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """训练一个epoch（常规batch模式）
    
    GPU 优化特性：
        - 支持混合精度训练 (AMP)，使用 float16/bfloat16 加速
        - 自动处理 GradScaler（仅 float16 需要）
    
    Args:
        model: PyTorch 模型
        loader: DataLoader
        optimizer: 优化器
        loss_fn: 损失函数
        device: 设备
        scaler: GradScaler（AMP float16 模式需要，bfloat16 不需要）
        use_amp: 是否启用混合精度训练
        amp_dtype: AMP 数据类型（torch.float16 或 torch.bfloat16）
    
    Returns:
        (avg_loss, pred_all, y_all): 平均损失，所有预测值，所有真实值
    """
    model.train()
    total_loss = 0.0
    n = 0
    preds, ys = [], []
    step = 0
    
    for X, ids, y in loader:
        X, ids, y = X.to(device, non_blocking=True), ids.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)  # 性能优化：使用 set_to_none
        
        # GPU 优化：混合精度训练
        with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
            pred = model(X, ids)
            loss = loss_fn(pred, y)
        
        if scaler is not None:
            # float16 模式：使用 GradScaler 防止梯度下溢
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            _log_grad_norms(model, grad_norm_cfg, epoch, step, float(loss.item()))
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # bfloat16 或 fp32 模式：直接反向传播
            loss.backward()
            _log_grad_norms(model, grad_norm_cfg, epoch, step, float(loss.item()))
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        total_loss += float(loss.item()) * len(X)
        n += len(X)
        preds.append(pred.detach().float().cpu().numpy())  # 确保转回 float32
        ys.append(y.detach().cpu().numpy())
        step += 1
    
    pred_all = np.concatenate(preds, axis=0) if preds else np.zeros((0, 1), dtype=np.float32)
    y_all = np.concatenate(ys, axis=0) if ys else np.zeros((0, 1), dtype=np.float32)
    avg_loss = total_loss / max(n, 1)
    return avg_loss, pred_all, y_all


def train_one_epoch_daily(
    model: nn.Module, 
    daily_dataset: DailyBatchDataset, 
    optimizer: torch.optim.Optimizer, 
    loss_fn: Callable, 
    device: torch.device,
    epoch_seed: int = 0,
    epoch: int = 0,
    scaler: Optional[GradScaler] = None,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    grad_norm_cfg: Optional[Dict[str, object]] = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """按日batch训练一个epoch（用于IC/RankIC损失函数）
    
    每个batch是同一天的所有股票，确保日内截面排序的有效性
    
    GPU 优化特性：
        - 支持混合精度训练 (AMP)
    
    Args:
        model: PyTorch 模型
        daily_dataset: DailyBatchDataset 对象
        optimizer: 优化器
        loss_fn: 损失函数
        device: 设备
        epoch_seed: epoch种子，用于打乱日期顺序
        scaler: GradScaler（AMP float16 模式需要）
        use_amp: 是否启用混合精度训练
        amp_dtype: AMP 数据类型
    
    Returns:
        (avg_loss, pred_all, y_all): 按天平均的损失，所有预测值，所有真实值
    """
    model.train()
    total_loss = 0.0
    n_days = 0
    preds, ys = [], []
    step = 0
    
    for X, ids, y in daily_dataset.iter_daily_batches(shuffle=True, extra_seed=epoch_seed):
        if len(X) < 3:  # 样本太少跳过
            continue
        
        X, ids, y = X.to(device, non_blocking=True), ids.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        
        # GPU 优化：混合精度训练
        with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
            pred = model(X, ids)
            loss = loss_fn(pred, y)
        
        # 检查loss是否有效
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            _log_grad_norms(model, grad_norm_cfg, epoch, step, float(loss.item()))
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            _log_grad_norms(model, grad_norm_cfg, epoch, step, float(loss.item()))
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        total_loss += float(loss.item())
        n_days += 1
        preds.append(pred.detach().float().cpu().numpy())
        ys.append(y.detach().cpu().numpy())
        step += 1
    
    pred_all = np.concatenate(preds, axis=0) if preds else np.zeros((0, 1), dtype=np.float32)
    y_all = np.concatenate(ys, axis=0) if ys else np.zeros((0, 1), dtype=np.float32)
    avg_loss = total_loss / max(n_days, 1)
    return avg_loss, pred_all, y_all


def evaluate(
    model: nn.Module, 
    loader: DataLoader, 
    loss_fn: Callable, 
    device: torch.device
) -> Tuple[float, np.ndarray, np.ndarray]:
    """评估模型（常规batch模式）
    
    Args:
        model: PyTorch 模型
        loader: DataLoader
        loss_fn: 损失函数
        device: 设备
    
    Returns:
        (avg_loss, pred_all, y_all): 平均损失，所有预测值，所有真实值
    """
    model.eval()
    preds, ys = [], []
    total_loss = 0.0
    n = 0
    
    with torch.no_grad():
        for X, ids, y in loader:
            X, ids, y = X.to(device), ids.to(device), y.to(device)
            pred = model(X, ids)
            loss = loss_fn(pred, y)
            total_loss += float(loss.item()) * len(X)
            n += len(X)
            preds.append(pred.detach().cpu().numpy())
            ys.append(y.detach().cpu().numpy())
    
    pred_all = np.concatenate(preds, axis=0) if preds else np.zeros((0, 1), dtype=np.float32)
    y_all = np.concatenate(ys, axis=0) if ys else np.zeros((0, 1), dtype=np.float32)
    avg_loss = total_loss / max(n, 1)
    return avg_loss, pred_all, y_all


def evaluate_daily(
    model: nn.Module, 
    daily_dataset: DailyBatchDataset, 
    loss_fn: Callable, 
    device: torch.device
) -> Tuple[float, np.ndarray, np.ndarray]:
    """按日batch评估模型（用于IC/RankIC损失函数）
    
    Args:
        model: PyTorch 模型
        daily_dataset: DailyBatchDataset 对象
        loss_fn: 损失函数
        device: 设备
    
    Returns:
        (avg_loss, pred_all, y_all): 按天平均的损失，所有预测值，所有真实值
    """
    model.eval()
    total_loss = 0.0
    n_days = 0
    preds, ys = [], []
    
    with torch.no_grad():
        for X, ids, y in daily_dataset.iter_daily_batches(shuffle=False, extra_seed=0):
            if len(X) < 3:
                continue
            
            X, ids, y = X.to(device), ids.to(device), y.to(device)
            pred = model(X, ids)
            loss = loss_fn(pred, y)
            
            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += float(loss.item())
                n_days += 1
            
            preds.append(pred.detach().cpu().numpy())
            ys.append(y.detach().cpu().numpy())
    
    pred_all = np.concatenate(preds, axis=0) if preds else np.zeros((0, 1), dtype=np.float32)
    y_all = np.concatenate(ys, axis=0) if ys else np.zeros((0, 1), dtype=np.float32)
    avg_loss = total_loss / max(n_days, 1)
    return avg_loss, pred_all, y_all


# ============================================================
# 推理辅助函数
# ============================================================

def predict_to_df(
    model: nn.Module, 
    split_data: Dict[str, np.ndarray], 
    device: torch.device
) -> pd.DataFrame:
    """使用模型预测并返回 DataFrame
    
    Args:
        model: PyTorch 模型
        split_data: 数据字典，需包含 "X", "id", "y", "date", "index" 键
        device: 设备
    
    Returns:
        DataFrame，包含 date, index_id, pred, y 列，按 (date, pred DESC) 排序
    """
    model.eval()
    X_np = split_data["X"]
    ids_np = split_data["id"]
    n_samples = len(X_np)
    batch_size = 1024  # 分批预测，避免显存溢出
    
    preds = []
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            X_batch = torch.tensor(X_np[i:i+batch_size], dtype=torch.float32).to(device)
            ids_batch = torch.tensor(ids_np[i:i+batch_size], dtype=torch.long).to(device)
            pred_batch = model(X_batch, ids_batch).detach().cpu().numpy()
            preds.append(pred_batch)
    
    pred = np.concatenate(preds, axis=0).reshape(-1)
    
    df = pd.DataFrame({
        "date": pd.to_datetime(split_data["date"]),
        "index_id": split_data["index"],
        "pred": pred,
        "y": split_data["y"].reshape(-1)
    })
    return df.sort_values(["date", "pred"], ascending=[True, False]).reset_index(drop=True)


# ============================================================
# TorchTrainer 类（统一封装）
# ============================================================

class TorchTrainer:
    """PyTorch 模型通用训练器
    
    统一封装训练、评估、预测流程，自动处理：
    - 常规batch vs 按日batch（根据损失函数自动选择）
    - 检查点管理（早停、最佳模型保存）
    - 训练历史记录
    
    GPU 优化特性：
        - 混合精度训练 (AMP) - 由 cfg.use_amp 控制
        - torch.compile() - 由 cfg.compile_model 控制（PyTorch 2.0+）
    
    使用示例：
        trainer = TorchTrainer(model, cfg, device)
        history = trainer.fit(samples["train"], samples["val"])
        test_df = trainer.predict(samples["test"])
    """
    
    def __init__(self, model: nn.Module, cfg: "Config", device: torch.device):
        """
        Args:
            model: PyTorch 模型，需支持 forward(X, ids) 接口
            cfg: 配置对象
            device: PyTorch 设备
        """
        self.cfg = cfg
        self.device = device
        
        # GPU 优化：torch.compile() 加速（PyTorch 2.0+）
        compile_model = getattr(cfg, 'compile_model', False)
        if compile_model and hasattr(torch, 'compile') and device.type == 'cuda':
            compile_mode = getattr(cfg, 'compile_mode', 'reduce-overhead')
            print(f"[GPU优化] 启用 torch.compile(), mode='{compile_mode}'")
            try:
                model = cast(nn.Module, torch.compile(model, mode=compile_mode))
            except Exception as e:
                print(f"[GPU优化] torch.compile() 失败: {e}, 使用原始模型")
        
        self.model = model.to(device)
        self.loss_fn = get_loss_fn(cfg)
        self.ckpt_mgr = CheckpointManager(cfg) if cfg.checkpoint_enabled else None
        
        # GPU 优化：混合精度训练 (AMP)
        self.use_amp = getattr(cfg, 'use_amp', False) and device.type == 'cuda'
        self.amp_dtype = get_amp_dtype(cfg) if self.use_amp else torch.float32
        self.scaler = create_scaler(cfg, device)
        
        if self.use_amp:
            dtype_name = 'bfloat16' if self.amp_dtype == torch.bfloat16 else 'float16'
            scaler_status = 'enabled' if self.scaler else 'disabled'
            print(f"[GPU优化] 启用混合精度训练 (AMP), dtype={dtype_name}, scaler={scaler_status}")

        self.grad_norm_cfg: Optional[Dict[str, object]] = None
        if getattr(cfg, "log_grad_norm", False):
            log_dir = str(getattr(cfg, "grad_norm_log_dir", "log/grad_norms"))
            os.makedirs(log_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            model_name = str(getattr(cfg, "model_name", "model"))
            log_path = os.path.join(log_dir, f"{model_name}_gradnorm_{timestamp}.jsonl")
            self.grad_norm_cfg = {
                "enabled": True,
                "log_path": log_path,
                "freq": int(getattr(cfg, "grad_norm_log_freq", 50)),
                "group_level": int(getattr(cfg, "grad_norm_group_level", 3)),
                "top_k": int(getattr(cfg, "grad_norm_topk", 200)),
                "model_name": model_name,
            }
            print(f"[GradNorm] Logging grad norms to: {log_path}")
        
        # 自动判断训练模式：IC/RankIC 损失需要按日batch
        self.use_daily = cfg.use_daily_batch or cfg.loss_name in [
            "pearson_ic", "sigmoid_rankic", "neuralsort_rankic"
        ]
        
        # 训练历史
        self.history: Dict[str, list] = {
            'train_loss': [],
            'val_loss': [],
            'val_ic': [],
            'val_rankic': [],
        }
    
    def fit(
        self, 
        train_data: Dict[str, np.ndarray], 
        val_data: Dict[str, np.ndarray],
        verbose: bool = True
    ) -> Dict[str, list]:
        """训练模型
        
        Args:
            train_data: 训练数据字典，需包含 "X", "id", "y", "date" 键
            val_data: 验证数据字典
            verbose: 是否打印训练日志
        
        Returns:
            训练历史字典
        """
        import torch.optim as optim
        
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.cfg.lr, 
            weight_decay=self.cfg.weight_decay
        )
        
        # 根据训练模式准备数据
        if self.use_daily:
            if verbose:
                print(f"[Mode] 使用按日batch训练模式 (loss={self.cfg.loss_name})")
            train_ds = DailyBatchDataset(
                train_data["X"], train_data["id"], 
                train_data["y"], train_data["date"]
            )
            val_ds = DailyBatchDataset(
                val_data["X"], val_data["id"], 
                val_data["y"], val_data["date"]
            )
            if verbose:
                print(f"  训练集: {len(train_ds)} 天, 验证集: {len(val_ds)} 天")
        else:
            train_loader = make_dataloader(train_data, self.cfg, shuffle=True)
            val_loader = make_dataloader(val_data, self.cfg, shuffle=False)
        
        if verbose and self.ckpt_mgr:
            print(f"[CheckpointManager] Strategy: '{self.cfg.checkpoint_strategy}', "
                  f"metric: '{self.cfg.checkpoint_metric}'")
        
        # 训练循环
        for ep in range(1, self.cfg.epochs + 1):
            # 训练一个epoch（使用 AMP 优化）
            if self.use_daily:
                tr_loss, tr_pred, tr_y = train_one_epoch_daily(
                    self.model, train_ds, optimizer, self.loss_fn, self.device, 
                    epoch_seed=ep,
                    epoch=ep,
                    scaler=self.scaler,
                    use_amp=self.use_amp,
                    amp_dtype=self.amp_dtype,
                    grad_norm_cfg=self.grad_norm_cfg,
                )
                val_loss, val_pred, val_y = evaluate_daily(
                    self.model, val_ds, self.loss_fn, self.device
                )
            else:
                tr_loss, tr_pred, tr_y = train_one_epoch(
                    self.model, train_loader, optimizer, self.loss_fn, self.device,
                    epoch=ep,
                    scaler=self.scaler,
                    use_amp=self.use_amp,
                    amp_dtype=self.amp_dtype,
                    grad_norm_cfg=self.grad_norm_cfg,
                )
                val_loss, val_pred, val_y = evaluate(
                    self.model, val_loader, self.loss_fn, self.device
                )
            
            # 记录历史
            self.history['train_loss'].append(tr_loss)
            self.history['val_loss'].append(val_loss)
            
            # 计算 IC/RankIC
            val_df = predict_to_df(self.model, val_data, self.device)
            ric_val = rank_ic_by_date(val_df)
            ic_val = pearson_ic_by_date(val_df)
            self.history['val_ic'].append(ic_val)
            self.history['val_rankic'].append(ric_val)
            
            # 打印日志
            if verbose:
                print(f"[Epoch {ep:02d}] train_{self.cfg.loss_name}={tr_loss:.6f}, "
                      f"val_{self.cfg.loss_name}={val_loss:.6f}")
                print(f"           val_mse={mse_np(val_pred, val_y):.6f}, "
                      f"val_mae={mae_np(val_pred, val_y):.6f}")
                print(f"           val_ic={ic_val:.4f}, val_rankic={ric_val:.4f}")
            
            # 检查点管理
            should_stop = False
            if self.ckpt_mgr:
                epoch_metrics = {
                    'val_loss': val_loss,
                    'val_rankic': ric_val,
                    'val_ic': ic_val,
                    'train_loss': tr_loss,
                }
                should_save, should_stop = self.ckpt_mgr.step(ep, self.model, epoch_metrics)
                if verbose and should_save:
                    print(f"  -> New best model saved "
                          f"({self.cfg.checkpoint_metric}: {epoch_metrics[self.cfg.checkpoint_metric]:.6f})")
            
            if verbose:
                print()
            
            if should_stop:
                if verbose:
                    print(f"[Early Stop] No improvement for {self.cfg.checkpoint_patience} epochs.")
                break
        
        # 加载最佳模型
        if self.ckpt_mgr and self.ckpt_mgr.load_best(self.model):
            if verbose:
                self.ckpt_mgr.print_summary()
        
        return self.history
    
    def predict(self, data: Dict[str, np.ndarray]) -> pd.DataFrame:
        """使用模型预测
        
        Args:
            data: 数据字典，需包含 "X", "id", "y", "date", "index" 键
        
        Returns:
            预测结果 DataFrame
        """
        return predict_to_df(self.model, data, self.device)
    
    def evaluate(self, data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """评估模型
        
        Args:
            data: 数据字典
        
        Returns:
            评估指标字典
        """
        if self.use_daily:
            daily_ds = DailyBatchDataset(
                data["X"], data["id"], data["y"], data["date"]
            )
            loss, pred, y = evaluate_daily(self.model, daily_ds, self.loss_fn, self.device)
        else:
            loader = make_dataloader(data, self.cfg, shuffle=False)
            loss, pred, y = evaluate(self.model, loader, self.loss_fn, self.device)
        
        pred_df = self.predict(data)
        
        return {
            'loss': loss,
            'mse': mse_np(pred, y),
            'mae': mae_np(pred, y),
            'ic': pearson_ic_by_date(pred_df),
            'rankic': rank_ic_by_date(pred_df),
        }
