# pyright: reportGeneralTypeIssues=false, reportAttributeAccessIssue=false, reportMissingImports=false, reportArgumentType=false, reportMissingTypeArgument=false, reportConstantRedefinition=false
"""
统一模型训练入口

使用模块化架构：
- data_pipeline: 数据预处理
- models: 模型定义
- train.TorchTrainer: 训练器
- eval: 评估、回测、可视化、实验记录
"""

from __future__ import annotations
import json
import pandas as pd
import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from typing import Callable, Optional, cast
from config import Config
from utils import init_reproducibility
from predata.data_pipeline import prepare_nn_data
from models.lstm_model_compatible import build_lstm_model
from models.GRUModel import build_gru_model
from models.Attention_GRUModel import build_attention_gru_model
from models.TCNModel import build_tcn_model
from models.AutoformerModel import build_autoformer_model
from models.Autoformer_encoderModel import build_autoformer_encoder_model
from models.TransformerModel import build_transformer_model
from models.Transformer_Enc import build_transformer_enc_model
from train import TorchTrainer, get_torch_device
from eval import (
    # 回测
    backtest_decile_strategy,
    compute_backtest_summary,
    BacktestConfig,
    # 可视化
    plot_training_metrics,
    plot_backtest_results,
    plot_long_short_curve,
    plot_ic_series,
    PlotConfig,
    # 实验记录
    log_experiment_full,
    ExperimentConfig,
)

MODEL_BUILDERS: dict[str, Callable] = {
    "lstm": build_lstm_model,
    "gru": build_gru_model,
    "attention_gru": build_attention_gru_model,
    "tcn": build_tcn_model,
    "autoformer": build_autoformer_model,
    "autoformer_encoder": build_autoformer_encoder_model,
    "transformer": build_transformer_model,
    "transformer_encoder": build_transformer_enc_model,
}

_RUN_LOG_FILE = None


def setup_run_log(log_dir: str = "log/experiments", prefix: str = "run_model") -> str:
    """Redirect stdout/stderr to both console and a log file."""
    global _RUN_LOG_FILE
    os.makedirs(log_dir, exist_ok=True)
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{prefix}_{timestamp}.log")

    _RUN_LOG_FILE = open(log_path, "a", buffering=1, encoding="utf-8")

    class TeeStream:
        def __init__(self, stream, log_file):
            self._stream = stream
            self._log_file = log_file

        def write(self, data):
            if data:
                self._stream.write(data)
                self._log_file.write(data)

        def flush(self):
            self._stream.flush()
            self._log_file.flush()

        def isatty(self):
            return False

    sys.stdout = TeeStream(sys.stdout, _RUN_LOG_FILE)
    sys.stderr = TeeStream(sys.stderr, _RUN_LOG_FILE)
    print(f"[Log] Streaming console output to: {log_path}")
    return log_path


def _normalize_model_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def _resolve_model_builder(cfg: Config) -> tuple[str, Callable]:
    raw_name = str(getattr(cfg, "model_name", "lstm"))
    model_name = _normalize_model_name(raw_name)
    builder = MODEL_BUILDERS.get(model_name)
    if builder is None:
        supported = ", ".join(sorted(MODEL_BUILDERS.keys()))
        raise ValueError(f"Unsupported model_name='{raw_name}'. Supported: {supported}")
    return model_name, builder


def run_lstm(
    cfg: Config, 
    reproducible: bool = True, 
    use_parallel: bool = True,
    num_workers: Optional[int] = None,
    verbose: bool = True,
    save_test_pred: bool = False) -> None:
    """LSTM 模型训练全流程
    
    Args:
        cfg: 配置对象
        reproducible: 是否启用可复现模式（False 时使用 cudnn_benchmark 提升性能）
        save_test_pred: 是否保存测试集预测结果
    """
    # GPU 优化：非复现模式下启用 cudnn_benchmark
    cudnn_benchmark = getattr(cfg, 'cudnn_benchmark', True)
    init_reproducibility(reproducible=reproducible, seed=cfg.seed, cudnn_benchmark=cudnn_benchmark)

    model_name, build_model = _resolve_model_builder(cfg)
    cfg.model_name = model_name
    log_dir = os.path.join("log", "experiments", model_name)
    res_dir = os.path.join("res", model_name)
    
    # 1. 数据预处理
    daily_df = pd.read_feather(cfg.daily_data_path)
    samples, train_end, val_end, num_factors, num_concept = prepare_nn_data(
        daily_data_path=cfg.daily_data_path,
        json_path=cfg.json_path,
        factor_path=cfg.factor_path,
        target_period_start=cfg.target_period_start,
        target_period_end=cfg.target_period_end,
        start_date=cfg.start_date,
        end_date=cfg.end_date,
        lookback=cfg.lookback,
        use_parallel=use_parallel,
        num_workers=num_workers,
        verbose=verbose,
        date_col=cfg.date_col,
        id_col=cfg.id_col,
    )
    
    # 2. 构建模型 + 训练器（TorchTrainer 内部处理 torch.compile 和 AMP）
    device = get_torch_device(cfg)
    model = build_model(cfg, num_concept, num_factors)
    trainer = TorchTrainer(model, cfg, device)  # model.to(device) 在 TorchTrainer 内部完成
    
    # 3. 训练
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)
    history = trainer.fit(samples["train"], samples["val"])
    
    # 4. 测试集评估
    print("=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    
    test_metrics = trainer.evaluate(samples["test"])
    print(f"[Test] loss={test_metrics['loss']:.6f}, mse={test_metrics['mse']:.6f}, "
          f"mae={test_metrics['mae']:.6f}")
    print(f"[Test] IC={test_metrics['ic']:.4f}, RankIC={test_metrics['rankic']:.4f}")
    
    # 5. 可视化 - 训练曲线（多指标）
    plot_training_metrics(history, cast(PlotConfig, cfg), save_dir=res_dir, model_name=model_name)
    
    # 6. 实验记录 - 完整记录（JSON + CSV + TXT）
    if trainer.ckpt_mgr is not None:
        best_val = trainer.ckpt_mgr.get_best_info()['best_metric']
        best_epoch = trainer.ckpt_mgr.get_best_info()['best_epoch']
    else:
        best_val = min(history['val_loss']) if history['val_loss'] else float('nan')
        best_epoch = history['val_loss'].index(min(history['val_loss'])) + 1 if history['val_loss'] else 0
    
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_name = f"{model_name}_{cfg.loss_name}_{cfg.hidden}h_{cfg.layers}L_{timestamp}"
    
    log_experiment_full(
        cast(ExperimentConfig, cfg),
        metrics={
            'test_loss': test_metrics['loss'],
            'test_mse': test_metrics['mse'],
            'test_mae': test_metrics['mae'],
            'test_ic': test_metrics['ic'],
            'test_rankic': test_metrics['rankic'],
            'best_val': best_val,
            'best_epoch': best_epoch,
        },
        log_dir=log_dir,
        experiment_name=exp_name,
        extra_info={
            'train_end': str(train_end.date()),
            'val_end': str(val_end.date()),
            'n_features': num_factors,
            'n_stocks': num_concept,
        }
    )
    
    # 7. 策略回测
    print("\n" + "=" * 60)
    print("STRATEGY BACKTEST")
    print("=" * 60)
    
    train_df = trainer.predict(samples["train"])
    val_df = trainer.predict(samples["val"])
    test_df = trainer.predict(samples["test"])
    
    if save_test_pred:
        os.makedirs(log_dir, exist_ok=True)
        test_pred_path = os.path.join(log_dir, f"{exp_name}_test_pred.feather")
        test_df.to_feather(test_pred_path)
        print(f"[Saved] test predictions -> {test_pred_path}")
    
    total_df = pd.concat([train_df, val_df, test_df])
    
    # 全量回测
    backtest_total = backtest_decile_strategy(total_df, daily_df, cast(BacktestConfig, cfg))
    plot_backtest_results(backtest_total, train_end, val_end, "total", save_dir=res_dir, model_name=model_name)
    
    # 测试集回测
    backtest_test = backtest_decile_strategy(test_df, daily_df, cast(BacktestConfig, cfg))
    plot_backtest_results(backtest_test, type="test", save_dir=res_dir, model_name=model_name)
    
    # 8. 增强分析
    print("\n" + "=" * 60)
    print("ENHANCED ANALYSIS")
    print("=" * 60)
    
    # 多空净值曲线
    plot_long_short_curve(backtest_test, save_dir=res_dir, model_name=model_name)
    
    # IC 时序图
    plot_ic_series(test_df, save_dir=res_dir, model_name=model_name)
    
    # 回测汇总统计
    summary = compute_backtest_summary(backtest_test)
    print("\n[Backtest Summary - Test Set]")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    # 消融实验：在相同数据与训练超参下，依次跑不同模型
    for model_name in ["autoformer_encoder"]:
        print("\n" + "=" * 80)
        print(f"[Ablation] Running model: {model_name}")
        print("=" * 80)

        cfg = Config(
            # 模型选择
            model_name=model_name,

            # 数据与时间区间（与已复现实验保持一致）
            standardized_methods="ts_zscore",
            filter_method="多均累计收益",
            is_all_factors=False,
            lookback=60,

            # 训练超参（与已复现实验保持一致）
            hidden=256,
            layers=2,
            drop_out=0.3,
            epochs=48,
            batch_size=128,
            lr=1e-3,
            loss_name="huber",

            # 任务类型
            emb_dim=0,
            task_type="regression",
        )

        if not os.path.exists(cfg.factor_path):
            fallback_factor_path = r"C:\Users\Administrator\Desktop\dixi_thsgn\data\daily_features_时序zscore.f"
            if os.path.exists(fallback_factor_path):
                cfg.factor_path = fallback_factor_path
                print(f"[Config] factor_path not found, fallback to: {fallback_factor_path}")

        cfg.dataloader_workers = 8

        norm_name, _ = _resolve_model_builder(cfg)
        setup_run_log(log_dir=os.path.join("log", "experiments", norm_name), prefix=f"run_{norm_name}")
        run_lstm(cfg)


