# pyright: reportMissingImports=false, reportCallIssue=false, reportAttributeAccessIssue=false
"""Memmap-compatible NN trainer.

Lightweight trainer that works with memmap DataLoaders directly.
Works with standard DataLoaders returning (X, ids, y) tensors.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from template.configs.model_configs._base import NNModelConfig

# 每 N 步做一次 GPU→CPU loss 同步（用于 progress callback）
# 避免每 batch 都触发 .item() 隐式同步（~50μs/次 × 100 batches = 5ms/epoch）
_LOSS_SYNC_INTERVAL = 10


class NNTrainer:
    """Memmap-compatible NN trainer.

    Works with standard DataLoaders returning (X, ids, y) tensors.
    """

    def __init__(self, model: nn.Module, nn_cfg: "NNModelConfig", device: torch.device):
        from template.train.losses import get_loss_fn
        from template.train.checkpoint import CheckpointManager

        self.nn_cfg = nn_cfg
        self.device = device

        # Optional torch.compile
        compile_model = getattr(nn_cfg, "compile_model", False)
        if compile_model and hasattr(torch, "compile") and device.type == "cuda":
            compile_mode = getattr(nn_cfg, "compile_mode", "reduce-overhead")
            print(f"[GPU优化] 启用 torch.compile(), mode='{compile_mode}'")
            try:
                model = cast(nn.Module, torch.compile(model, mode=compile_mode))
            except Exception as e:
                print(f"[GPU优化] torch.compile() 失败: {e}, 使用原始模型")

        self.model = model.to(device)
        self.loss_fn = get_loss_fn(nn_cfg)

        # AMP
        self.use_amp = getattr(nn_cfg, "use_amp", False) and device.type == "cuda"
        self.amp_dtype = torch.float32
        if self.use_amp:
            if getattr(nn_cfg, "amp_dtype", "float16") == "bfloat16":
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                    self.amp_dtype = torch.bfloat16
                else:
                    self.amp_dtype = torch.float16
            else:
                self.amp_dtype = torch.float16

        # GradScaler — bfloat16 不需要 scaler
        self.scaler = None
        if self.use_amp and device.type == "cuda" and self.amp_dtype != torch.bfloat16:
            from torch.amp.grad_scaler import GradScaler
            self.scaler = GradScaler()

        if self.use_amp:
            dtype_name = "bfloat16" if self.amp_dtype == torch.bfloat16 else "float16"
            scaler_status = "enabled" if self.scaler else "disabled"
            print(f"[GPU优化] 启用混合精度训练 (AMP), dtype={dtype_name}, scaler={scaler_status}")

        # Checkpoint
        if nn_cfg.checkpoint_enabled:
            self.ckpt_mgr = CheckpointManager(nn_cfg)
        else:
            self.ckpt_mgr = None

        self.history: dict[str, list] = {
            "train_loss": [], "val_loss": [],
            "val_r2": [],
        }

    def fit(
        self,
        train_loader,
        val_loader,
        on_step_factory=None,
    ) -> dict[str, list]:
        import torch.optim as optim
        from torch.amp.autocast_mode import autocast
        from template.eval import r2_score_np

        nn_cfg = self.nn_cfg
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=nn_cfg.lr,
            weight_decay=nn_cfg.weight_decay,
        )

        for ep in range(1, nn_cfg.epochs + 1):
            step_cb = on_step_factory(ep) if on_step_factory else None

            # ── Train epoch ──
            self.model.train()
            # 在 GPU 上累加 loss，避免每 batch 触发 CPU-GPU 同步（.item()）
            total_loss_gpu = torch.tensor(0.0, device=self.device)
            n = 0
            step = 0
            total_steps = len(train_loader)

            for X, ids, y in train_loader:
                X = X.to(self.device, non_blocking=True)
                ids = ids.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                    pred = self.model(X, ids)
                    loss = self.loss_fn(pred, y)

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

                total_loss_gpu += loss.detach().float() * X.shape[0]
                n += X.shape[0]
                step += 1
                # 每 _LOSS_SYNC_INTERVAL 步或最后一步才做 GPU→CPU 同步
                # 避免每 batch 都触发 .item() 隐式同步（~50μs/次）
                if step_cb is not None and (step % _LOSS_SYNC_INTERVAL == 0 or step == total_steps):
                    step_cb(step, total_steps, total_loss_gpu.item() / max(n, 1))

            tr_loss = total_loss_gpu.item() / max(n, 1)

            # ── Validate ──
            val_loss, val_pred, val_y = self._evaluate_internal(val_loader)
            r2_val = r2_score_np(val_pred, val_y)

            self.history["train_loss"].append(tr_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_r2"].append(r2_val)

            print(f"[Epoch {ep:02d}] train_loss={tr_loss:.6f}, "
                  f"val_loss={val_loss:.6f}, val_r2={r2_val:.6f}")

            # Checkpoint
            should_stop = False
            if self.ckpt_mgr:
                epoch_metrics = {"val_loss": val_loss, "val_r2": r2_val, "train_loss": tr_loss}
                should_save, should_stop = self.ckpt_mgr.step(ep, self.model, epoch_metrics)
                if should_save:
                    print(f"  -> New best model saved")

            if should_stop:
                print(f"[Early Stop] No improvement for {nn_cfg.checkpoint_patience} epochs.")
                break

        if self.ckpt_mgr and self.ckpt_mgr.load_best(self.model):
            self.ckpt_mgr.print_summary()

        return self.history

    def _evaluate_internal(self, loader) -> tuple[float, np.ndarray, np.ndarray]:
        """Evaluate model on a DataLoader, return (avg_loss, preds, targets)."""
        self.model.eval()
        preds_list, ys_list = [], []
        total_loss = 0.0
        n = 0

        with torch.no_grad():
            for X, ids, y in loader:
                X = X.to(self.device, non_blocking=True)
                ids = ids.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                pred = self.model(X, ids)
                loss = self.loss_fn(pred, y)
                total_loss += float(loss.item()) * len(X)
                n += len(X)
                preds_list.append(pred.detach().float().cpu().numpy())
                ys_list.append(y.detach().cpu().numpy())

        pred_all = np.concatenate(preds_list, axis=0) if preds_list else np.zeros((0, 1))
        y_all = np.concatenate(ys_list, axis=0) if ys_list else np.zeros((0, 1))
        return total_loss / max(n, 1), pred_all, y_all

    def evaluate_loader(self, loader) -> dict[str, float]:
        """Evaluate and return metrics dict (loss/mse/mae/r2 only, no date info)."""
        from template.eval import mse_np, mae_np, r2_score_np
        loss, pred, y = self._evaluate_internal(loader)
        return {
            "loss": loss,
            "mse": mse_np(pred, y),
            "mae": mae_np(pred, y),
            "r2": r2_score_np(pred, y),
        }

    def predict_loader(self, loader) -> pd.DataFrame:
        """Generate predictions as a DataFrame."""
        self.model.eval()
        preds_list, ys_list = [], []

        with torch.no_grad():
            for X, ids, y in loader:
                X = X.to(self.device, non_blocking=True)
                ids = ids.to(self.device, non_blocking=True)
                pred = self.model(X, ids)
                preds_list.append(pred.detach().float().cpu().numpy())
                ys_list.append(y.detach().cpu().numpy())

        pred_all = np.concatenate(preds_list, axis=0).reshape(-1) if preds_list else np.array([])
        y_all = np.concatenate(ys_list, axis=0).reshape(-1) if ys_list else np.array([])

        return pd.DataFrame({
            "pred": pred_all,
            "y": y_all,
        })


def evaluate_v2(trainer: NNTrainer, derived_meta, lookback, n_features, batch_size,
                num_workers, pin_mem) -> dict:
    """V2 管线的评估（含 IC/RankIC）。"""
    from template.eval import (
        mse_np, mae_np, r2_score_np,
        ic_series_by_date, ic_summary,
    )
    from template.data.memmap_cache import make_v2_pred_dataloader

    pred_loader = make_v2_pred_dataloader(
        derived_meta, split="test", lookback=lookback,
        batch_size=batch_size, num_workers=num_workers, pin_memory=pin_mem,
    )

    trainer.model.eval()
    all_preds, all_ys, all_dates = [], [], []
    total_loss, n = 0.0, 0

    with torch.no_grad():
        for X, ids, y, dates, _ in pred_loader:
            X = X.to(trainer.device, non_blocking=True)
            ids = ids.to(trainer.device, non_blocking=True)
            y_gpu = y.to(trainer.device, non_blocking=True)
            pred = trainer.model(X, ids)
            loss = trainer.loss_fn(pred, y_gpu)
            total_loss += float(loss.item()) * len(X)
            n += len(X)
            all_preds.append(pred.detach().float().cpu().numpy())
            all_ys.append(y.numpy())
            all_dates.append(dates)

    pred_all = np.concatenate(all_preds).reshape(-1)
    y_all = np.concatenate(all_ys).reshape(-1)
    dates_all = np.concatenate(all_dates)

    pred_df = pd.DataFrame({"date": dates_all, "pred": pred_all, "y": y_all})
    avg_loss = total_loss / max(n, 1)

    ic_df = ic_series_by_date(pred_df, date_col="date", score_col="pred", target_col="y")
    summary = ic_summary(ic_df)

    return {
        "loss": avg_loss,
        "mse": mse_np(pred_all, y_all),
        "mae": mae_np(pred_all, y_all),
        "r2": r2_score_np(pred_all, y_all),
        "ic": float(summary.get("ic_mean", float("nan"))),
        "ic_ir": float(summary.get("ic_ir", float("nan"))),
        "rankic": float(summary.get("rank_ic_mean", float("nan"))),
        "rankic_ir": float(summary.get("rank_ic_ir", float("nan"))),
    }
