"""
CheckpointManager: 模型检查点管理器

功能：
    - 模型保存/加载
    - 早停机制
    - 多种回退策略（simple, warmup, ema, patience, combined）

使用示例：
    ckpt_mgr = CheckpointManager(cfg)
    for ep in range(1, epochs+1):
        # ... 训练和验证 ...
        metrics = {'val_loss': val_loss, 'val_rankic': ric_val, 'val_ic': ic_val}
        should_save, should_stop = ckpt_mgr.step(ep, model, metrics)
        if should_stop:
            break
    # 加载最佳模型
    ckpt_mgr.load_best(model)
"""
from __future__ import annotations
import copy
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, cast

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from template.configs.model_configs._base import NNModelConfig


class CheckpointManager:
    """模块化的模型回退/早停管理器。

    Args:
        nn_cfg: NNModelConfig 实例，包含 checkpoint_* 系列字段。
    """

    def __init__(self, nn_cfg: "NNModelConfig"):
        self.strategy = nn_cfg.checkpoint_strategy
        self.metric_name = nn_cfg.checkpoint_metric
        self.warmup_epochs = nn_cfg.checkpoint_warmup_epochs
        self.patience = nn_cfg.checkpoint_patience
        self.min_delta = nn_cfg.checkpoint_min_delta
        self.ema_alpha = nn_cfg.checkpoint_ema_alpha
        self.save_to_disk = nn_cfg.checkpoint_save_to_disk
        self.save_path = nn_cfg.checkpoint_save_path
        
        # 判断指标方向：越大越好还是越小越好
        self.higher_is_better = self.metric_name in ['val_rankic', 'val_ic', 'val_r2']
        
        # 状态变量
        self.best_metric: float = -float("inf") if self.higher_is_better else float("inf")
        self.best_state: Optional[Dict] = None
        self.best_epoch: int = 0
        self.bad_epochs: int = 0  # 连续无改善的epoch计数
        self.metric_ema: Optional[float] = None  # EMA平滑后的指标
        self.history: List[Dict] = []  # 记录每个epoch的指标
        
    def _is_better(self, current: float, best: float) -> bool:
        """判断current是否比best更好（考虑min_delta）"""
        if self.higher_is_better:
            return current > best + self.min_delta
        else:
            return current < best - self.min_delta
    
    def _get_metric(self, metrics: Dict[str, float]) -> float:
        """从metrics字典中获取监控指标的值"""
        if self.metric_name not in metrics:
            raise ValueError(f"Metric '{self.metric_name}' not found in metrics dict. "
                           f"Available: {list(metrics.keys())}")
        return metrics[self.metric_name]
    
    def _update_ema(self, value: float) -> float:
        """更新并返回EMA平滑后的值"""
        if self.ema_alpha <= 0:
            return value
        if self.metric_ema is None:
            self.metric_ema = value
        else:
            self.metric_ema = self.ema_alpha * value + (1 - self.ema_alpha) * self.metric_ema
        return cast(float, self.metric_ema)
    
    def step(self, epoch: int, model: nn.Module, metrics: Dict[str, float]) -> Tuple[bool, bool]:
        """每个epoch结束时调用，判断是否保存模型和是否早停
        
        Args:
            epoch: 当前epoch编号（从1开始）
            model: 模型对象
            metrics: 包含各种指标的字典，如 {'val_loss': 0.1, 'val_rankic': 0.05}
        
        Returns:
            (should_save, should_stop): 是否保存了模型，是否应该早停
        """
        raw_metric = self._get_metric(metrics)
        
        # 应用EMA平滑
        metric = self._update_ema(raw_metric)
        
        # 记录历史
        self.history.append({
            'epoch': epoch,
            'raw_metric': raw_metric,
            'ema_metric': metric,
            **metrics
        })
        
        should_save = False
        should_stop = False
        
        # 根据策略判断是否保存
        if self.strategy == 'simple':
            # 简单策略：全局最优即保存
            should_save = self._simple_step(epoch, model, metric)
        elif self.strategy == 'warmup':
            # warmup策略：前N个epoch不参与竞争
            should_save = self._warmup_step(epoch, model, metric)
        elif self.strategy == 'ema':
            # ema策略：使用平滑指标判断
            should_save = self._simple_step(epoch, model, metric)  # EMA已在上面应用
        elif self.strategy == 'patience':
            # patience策略：支持早停
            should_save, should_stop = self._patience_step(epoch, model, metric)
        elif self.strategy == 'combined':
            # 组合策略：warmup + ema + patience
            should_save, should_stop = self._combined_step(epoch, model, metric)
        else:
            raise ValueError(f"Unknown checkpoint strategy: {self.strategy}")
        
        return should_save, should_stop
    
    def _simple_step(self, epoch: int, model: nn.Module, metric: float) -> bool:
        """简单策略：全局最优即保存"""
        if self._is_better(metric, self.best_metric):
            self.best_metric = metric
            self.best_state = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            self._save_to_disk_if_needed(model, epoch)
            return True
        return False
    
    def _warmup_step(self, epoch: int, model: nn.Module, metric: float) -> bool:
        """warmup策略：前N个epoch不参与best竞争"""
        if epoch < self.warmup_epochs:
            return False
        return self._simple_step(epoch, model, metric)
    
    def _patience_step(self, epoch: int, model: nn.Module, metric: float) -> Tuple[bool, bool]:
        """patience策略：连续P个epoch无改善则早停"""
        if self._is_better(metric, self.best_metric):
            self.best_metric = metric
            self.best_state = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            self.bad_epochs = 0
            self._save_to_disk_if_needed(model, epoch)
            return True, False
        else:
            self.bad_epochs += 1
            if self.patience > 0 and self.bad_epochs >= self.patience:
                return False, True  # 触发早停
            return False, False
    
    def _combined_step(self, epoch: int, model: nn.Module, metric: float) -> Tuple[bool, bool]:
        """组合策略：warmup + ema(已在外层应用) + patience"""
        # warmup期间不竞争也不计算bad_epochs
        if epoch < self.warmup_epochs:
            return False, False
        
        return self._patience_step(epoch, model, metric)
    
    def _save_to_disk_if_needed(self, model: nn.Module, epoch: int) -> None:
        """如果配置了保存到磁盘，则保存模型"""
        if not self.save_to_disk:
            return
        import os
        os.makedirs(self.save_path, exist_ok=True)
        path = os.path.join(self.save_path, f"best_model_epoch{epoch}.pt")
        torch.save(model.state_dict(), path)
        print(f"  -> Model saved to disk: {path}")
    
    def load_best(self, model: nn.Module) -> bool:
        """加载最佳模型状态
        
        Returns:
            是否成功加载（如果没有保存过best，返回False）
        """
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
            return True
        return False
    
    def get_best_info(self) -> Dict:
        """获取最佳模型的信息"""
        return {
            'best_epoch': self.best_epoch,
            'best_metric': self.best_metric,
            'metric_name': self.metric_name,
            'strategy': self.strategy,
            'has_state': self.best_state is not None
        }
    
    def print_summary(self) -> None:
        """打印回退机制的配置和结果摘要"""
        info = self.get_best_info()
        direction = "↑" if self.higher_is_better else "↓"
        print(f"[CheckpointManager] Strategy: {self.strategy}")
        print(f"  Metric: {self.metric_name} ({direction} is better)")
        if self.warmup_epochs > 0:
            print(f"  Warmup: {self.warmup_epochs} epochs")
        if self.patience > 0:
            print(f"  Patience: {self.patience} epochs")
        if self.ema_alpha > 0:
            print(f"  EMA alpha: {self.ema_alpha}")
        if self.min_delta > 0:
            print(f"  Min delta: {self.min_delta}")
        print(f"  Best epoch: {info['best_epoch']}, Best {self.metric_name}: {info['best_metric']:.6f}")
