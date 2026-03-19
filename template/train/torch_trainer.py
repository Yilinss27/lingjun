"""设备管理工具。"""
from __future__ import annotations

import torch


def get_torch_device(device_str: str = "auto") -> torch.device:
    """根据设备字符串返回 PyTorch 设备。

    Args:
        device_str: "cpu" | "gpu" | "auto" (default)
    """
    if device_str == "cpu":
        return torch.device("cpu")
    if device_str == "gpu":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
