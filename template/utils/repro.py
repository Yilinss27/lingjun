'''
Author: Young yyyao75@163.com
Date: 2026-01-08 16:30:35
LastEditors: Young yyyao75@163.com
Description: 可复现性控制模块
'''

from __future__ import annotations
import os
import random
from typing import Dict
import numpy as np
import torch

# 模块级状态(记忆)
_reproducible: bool = True
_seed: int = 42


def init_reproducibility(
    reproducible: bool = True, 
    seed: int = 42,
    cudnn_benchmark: bool = True
) -> None:
    """
    统一的可复现性初始化入口，在程序入口处调用一次即可。
    
    Args:
        reproducible: True=可复现模式，False=随机模式（性能更优）
        seed: 随机种子（仅 reproducible=True 时生效）
        cudnn_benchmark: 启用 cuDNN 自动调优（仅 reproducible=False 时生效）
    """
    global _reproducible, _seed
    _reproducible = reproducible
    _seed = seed
    
    if reproducible:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                # Older PyTorch versions don't support warn_only
                torch.use_deterministic_algorithms(True)
        print(f"[Repro] 可复现模式, seed={seed}, cudnn_benchmark=False")
    else:
        torch.backends.cudnn.benchmark = cudnn_benchmark
        torch.backends.cudnn.deterministic = False
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(False)
        print(f"[Repro] 随机模式, cudnn_benchmark={cudnn_benchmark}")


def seed_worker(worker_id: int) -> None:
    """DataLoader worker 初始化函数，自动根据模式决定是否固定种子"""
    if _reproducible:
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)


def get_generator() -> torch.Generator | None:
    """获取 DataLoader 的 Generator，随机模式返回 None"""
    if _reproducible:
        g = torch.Generator()
        g.manual_seed(_seed)
        return g
    return None


def get_seed() -> int:
    """获取当前的随机种子值"""
    return _seed


def is_reproducible() -> bool:
    """获取当前是否为可复现模式"""
    return _reproducible


def get_rng(extra_seed: int = 0) -> np.random.RandomState:
    """
    获取一个 NumPy RandomState 对象，用于需要独立随机状态的场景
    
    Args:
        extra_seed: 额外的种子偏移量（例如 epoch 编号），用于生成不同但可复现的序列
    
    Returns:
        RandomState 对象；可复现模式下种子为 _seed + extra_seed，否则使用系统随机
    """
    if _reproducible:
        return np.random.RandomState(_seed + extra_seed)
    return np.random.RandomState()


def get_lgbm_seed_params() -> Dict[str, object]:
    """
    LightGBM 可复现性参数。
    可复现模式下返回固定 seed + deterministic 参数；随机模式返回空 dict。
    """
    if not _reproducible:
        return {}
    seed = _seed
    return {
        "seed": seed,
        "random_state": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "data_random_seed": seed,
        "drop_seed": seed,
        "deterministic": True,
    }
