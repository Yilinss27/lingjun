# pyright: reportGeneralTypeIssues=false, reportAttributeAccessIssue=false, reportMissingImports=false, reportArgumentType=false, reportMissingTypeArgument=false, reportPossiblyUnboundVariable=false
"""
超参数搜索框架

策略：
1. 两条线分开搜索（线A：常规回归batch，线B：按日batch）
2. 每条线分三阶段：先搜lr，再搜结构，最后搜lookback
3. 使用val_rankic作为统一选型指标（量纲一致，越大越好）

使用方法：
    python hyperparam_search.py --line A --stage 1  # 线A阶段1
    python hyperparam_search.py --line B --stage 2  # 线B阶段2
    python hyperparam_search.py --line A --stage all  # 线A全部阶段
"""

from __future__ import annotations
# 解决 OpenMP 运行时库重复加载问题（PyTorch + NumPy/MKL 冲突）
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import itertools
import json
import time
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from config import Config
from repro import init_reproducibility
from data_pipeline import prepare_lstm_data
from models.lstm import build_lstm_model
from train import TorchTrainer, get_torch_device
from eval.plots import plot_training_metrics, plot_backtest_results
from eval.backtest import backtest_decile_strategy


# ============================================================
# 搜索空间定义
# ============================================================

# 线A：常规回归batch（mse, huber, quantile_loss）
LINE_A_LOSSES = ["mse", "huber", "quantile_loss"]

# 线B：按日batch（IC/RankIC损失）
LINE_B_LOSSES = ["pearson_ic", "sigmoid_rankic", "neuralsort_rankic"]

# 通用结构参数
HIDDEN_CHOICES = [32, 64, 128, 256]
LAYERS_CHOICES = [1, 2, 3]
DROPOUT_CHOICES = [0.1, 0.2, 0.3]

# lookback（后面间隔放宽）
LOOKBACK_CHOICES = [5, 10, 20, 40, 60, 100, 200]

# 学习率（对数尺度）
LR_CHOICES = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]

# batch_size（仅线A使用）
BATCH_SIZE_CHOICES = [64, 128, 256, 512]

# softrank_tau（仅线B使用）
SOFTRANK_TAU_CHOICES = [0.1, 0.5, 1.0, 2.0, 5.0]

# 训练epochs（搜索阶段用较少epoch加速）
SEARCH_EPOCHS = 16  # 搜索时用16轮快速筛选
FINAL_EPOCHS = 32   # 最终验证用32轮


# ============================================================
# 默认基线配置
# ============================================================

def get_baseline_config() -> Dict[str, Any]:
    """获取基线配置（搜索时的固定值）"""
    return {
        "hidden": 64,
        "layers": 2,
        "drop_out": 0.3,
        "lookback": 60,
        "lr": 1e-3,
        "batch_size": 128,
        "softrank_tau": 1.0,
        "epochs": SEARCH_EPOCHS,
        # 以下固定不变
        "checkpoint_enabled": True,
        "checkpoint_strategy": "combined",
        "checkpoint_metric": "val_rankic",  # 统一用rankic作为选型指标
        "checkpoint_warmup_epochs": 4,
        "checkpoint_patience": 8,
    }


# ============================================================
# 搜索阶段定义
# ============================================================

def get_search_grid_line_a(stage: int, best_params: Optional[Dict] = None) -> List[Dict[str, Any]]:
    """
    获取线A（常规回归）的搜索网格
    
    阶段1：搜索 lr + batch_size + loss_name
    阶段2：搜索 hidden + layers + dropout（使用阶段1最优lr/batch_size）
    阶段3：搜索 lookback（使用阶段2最优结构）
    """
    baseline = get_baseline_config()
    if best_params:
        baseline.update(best_params)
    
    grids = []
    
    if stage == 1:
        # 阶段1：搜索lr + batch_size，每个loss分别测试
        for loss_name in LINE_A_LOSSES:
            for lr, batch_size in itertools.product(LR_CHOICES, BATCH_SIZE_CHOICES):
                cfg = baseline.copy()
                cfg.update({
                    "loss_name": loss_name,
                    "lr": lr,
                    "batch_size": batch_size,
                })
                grids.append(cfg)
        print(f"[线A-阶段1] lr×batch_size×loss = {len(LR_CHOICES)}×{len(BATCH_SIZE_CHOICES)}×{len(LINE_A_LOSSES)} = {len(grids)} 组")
    
    elif stage == 2:
        # 阶段2：搜索结构（使用阶段1最优的lr/batch_size/loss_name）
        assert best_params and "lr" in best_params, "阶段2需要阶段1的最优参数"
        for hidden, layers, dropout in itertools.product(HIDDEN_CHOICES, LAYERS_CHOICES, DROPOUT_CHOICES):
            cfg = baseline.copy()
            cfg.update({
                "hidden": hidden,
                "layers": layers,
                "drop_out": dropout,
            })
            grids.append(cfg)
        print(f"[线A-阶段2] hidden×layers×dropout = {len(HIDDEN_CHOICES)}×{len(LAYERS_CHOICES)}×{len(DROPOUT_CHOICES)} = {len(grids)} 组")
    
    elif stage == 3:
        # 阶段3：搜索lookback（使用阶段2最优结构）
        assert best_params and "hidden" in best_params, "阶段3需要阶段2的最优参数"
        for lookback in LOOKBACK_CHOICES:
            cfg = baseline.copy()
            cfg.update({
                "lookback": lookback,
                "epochs": FINAL_EPOCHS,  # 最后阶段用更多epoch
            })
            grids.append(cfg)
        print(f"[线A-阶段3] lookback = {len(LOOKBACK_CHOICES)} 组")
    
    return grids


def get_search_grid_line_b(stage: int, best_params: Optional[Dict] = None) -> List[Dict[str, Any]]:
    """
    获取线B（按日batch）的搜索网格
    
    阶段1：搜索 lr + softrank_tau + loss_name
    阶段2：搜索 hidden + layers + dropout（使用阶段1最优lr/tau）
    阶段3：搜索 lookback（使用阶段2最优结构）
    """
    baseline = get_baseline_config()
    if best_params:
        baseline.update(best_params)
    
    grids = []
    
    if stage == 1:
        # 阶段1：搜索lr + softrank_tau，每个loss分别测试
        for loss_name in LINE_B_LOSSES:
            for lr, tau in itertools.product(LR_CHOICES, SOFTRANK_TAU_CHOICES):
                cfg = baseline.copy()
                cfg.update({
                    "loss_name": loss_name,
                    "lr": lr,
                    "softrank_tau": tau,
                })
                grids.append(cfg)
        print(f"[线B-阶段1] lr×tau×loss = {len(LR_CHOICES)}×{len(SOFTRANK_TAU_CHOICES)}×{len(LINE_B_LOSSES)} = {len(grids)} 组")
    
    elif stage == 2:
        # 阶段2：搜索结构
        assert best_params and "lr" in best_params, "阶段2需要阶段1的最优参数"
        for hidden, layers, dropout in itertools.product(HIDDEN_CHOICES, LAYERS_CHOICES, DROPOUT_CHOICES):
            cfg = baseline.copy()
            cfg.update({
                "hidden": hidden,
                "layers": layers,
                "drop_out": dropout,
            })
            grids.append(cfg)
        print(f"[线B-阶段2] hidden×layers×dropout = {len(HIDDEN_CHOICES)}×{len(LAYERS_CHOICES)}×{len(DROPOUT_CHOICES)} = {len(grids)} 组")
    
    elif stage == 3:
        # 阶段3：搜索lookback
        assert best_params and "hidden" in best_params, "阶段3需要阶段2的最优参数"
        for lookback in LOOKBACK_CHOICES:
            cfg = baseline.copy()
            cfg.update({
                "lookback": lookback,
                "epochs": FINAL_EPOCHS,
            })
            grids.append(cfg)
        print(f"[线B-阶段3] lookback = {len(LOOKBACK_CHOICES)} 组")
    
    return grids


# ============================================================
# 单次实验运行
# ============================================================

def run_single_experiment(
    df_panel: pd.DataFrame,
    params: Dict[str, Any],
    exp_id: int,
    total: int,
    verbose: bool = False,
    precomputed_data: Optional[Tuple] = None,
    plot: bool = True,
    plot_dir: str = "log/hyperparam_search/plots"
) -> Dict[str, Any]:
    """
    运行单次实验
    
    Args:
        df_panel: 原始面板数据
        params: 实验参数
        exp_id: 当前实验编号
        total: 总实验数
        verbose: 是否打印详细日志
        precomputed_data: 预处理好的数据 (samples, train_end, val_end, id_map)，
                          如果提供则直接使用，避免重复预处理
        plot: 是否绘制训练曲线和回测图
        plot_dir: 图片保存目录
    
    Returns:
        包含参数和结果的字典
    """
    # 构建Config
    cfg = Config(**{k: v for k, v in params.items() if hasattr(Config, k)})
    
    print(f"\n{'='*60}")
    print(f"[实验 {exp_id}/{total}]")
    print(f"  loss={cfg.loss_name}, lr={cfg.lr:.0e}, hidden={cfg.hidden}, "
          f"layers={cfg.layers}, dropout={cfg.drop_out}")
    if cfg.loss_name in LINE_B_LOSSES:
        print(f"  softrank_tau={cfg.softrank_tau}")
    else:
        print(f"  batch_size={cfg.batch_size}")
    print(f"  lookback={cfg.lookback}, epochs={cfg.epochs}")
    print("="*60)
    
    start_time = time.time()
    
    try:
        # 初始化可复现性
        init_reproducibility(reproducible=True, seed=42, cudnn_benchmark=False)
        
        # 数据预处理（如果已提供预处理数据则直接使用）
        if precomputed_data is not None:
            samples, train_end, val_end, id_map = precomputed_data
        else:
            samples, train_end, val_end, id_map = prepare_lstm_data(df_panel, cfg)
        
        # 构建模型
        device = get_torch_device(cfg)
        model = build_lstm_model(cfg, len(id_map), len(cfg.feature_cols))
        trainer = TorchTrainer(model, cfg, device)
        
        # 训练
        history = trainer.fit(samples["train"], samples["val"], verbose=verbose)
        
        # 测试集评估
        test_metrics = trainer.evaluate(samples["test"])
        
        # 获取最优epoch信息
        if trainer.ckpt_mgr is not None:
            best_info = trainer.ckpt_mgr.get_best_info()
            best_val_rankic = best_info['best_metric']
            best_epoch = best_info['best_epoch']
        else:
            best_val_rankic = max(history['val_rankic']) if history['val_rankic'] else 0
            best_epoch = history['val_rankic'].index(max(history['val_rankic'])) + 1 if history['val_rankic'] else 0
        
        elapsed = time.time() - start_time
        
        result = {
            # 参数
            **params,
            # 验证集结果
            "best_val_rankic": best_val_rankic,
            "best_epoch": best_epoch,
            "final_val_loss": history['val_loss'][-1] if history['val_loss'] else None,
            "final_val_ic": history['val_ic'][-1] if history['val_ic'] else None,
            "final_val_rankic": history['val_rankic'][-1] if history['val_rankic'] else None,
            # 测试集结果
            "test_loss": test_metrics['loss'],
            "test_mse": test_metrics['mse'],
            "test_mae": test_metrics['mae'],
            "test_ic": test_metrics['ic'],
            "test_rankic": test_metrics['rankic'],
            # 元信息
            "elapsed_seconds": elapsed,
            "status": "success",
        }
        
        print(f"\n[结果] best_val_rankic={best_val_rankic:.4f} (epoch {best_epoch}), "
              f"test_rankic={test_metrics['rankic']:.4f}, 耗时={elapsed:.1f}s")
        
        # 绘图（如果启用）
        if plot:
            os.makedirs(plot_dir, exist_ok=True)
            exp_prefix = f"exp{exp_id:03d}_{cfg.loss_name}_lr{cfg.lr:.0e}"
            
            # 1. 训练曲线图
            plot_training_metrics(history, cfg, save_dir=plot_dir, show=False)
            
            # 2. 测试集回测图
            test_df = trainer.predict(samples["test"])
            backtest_test = backtest_decile_strategy(test_df, df_panel, cfg)
            plot_backtest_results(backtest_test, type=f"test_{exp_prefix}", save_dir=plot_dir, show=False)
        
    except Exception as e:
        elapsed = time.time() - start_time
        result = {
            **params,
            "best_val_rankic": float('-inf'),
            "status": "failed",
            "error": str(e),
            "elapsed_seconds": elapsed,
        }
        print(f"\n[失败] {e}")
    
    return result


# ============================================================
# 搜索阶段运行器
# ============================================================

def run_search_stage(
    df_panel: pd.DataFrame,
    line: str,
    stage: int,
    best_params: Optional[Dict] = None,
    output_dir: str = "log/hyperparam_search",
    verbose: bool = False,
    plot: bool = True
) -> Tuple[List[Dict], Dict[str, Any]]:
    """
    运行一个搜索阶段
    
    Args:
        df_panel: 原始面板数据
        line: "A" 或 "B"
        stage: 1, 2, 或 3
        best_params: 前一阶段的最优参数
        output_dir: 结果输出目录
        verbose: 是否打印详细训练日志
        plot: 是否为每个实验绘制训练曲线和回测图
    
    Returns:
        (所有结果列表, 最优参数字典)
    """
    os.makedirs(output_dir, exist_ok=True)
    plot_dir = os.path.join(output_dir, f"plots_line{line}_stage{stage}")
    
    # 获取搜索网格
    if line == "A":
        grids = get_search_grid_line_a(stage, best_params)
    else:
        grids = get_search_grid_line_b(stage, best_params)
    
    print(f"\n{'#'*60}")
    print(f"# 线{line} - 阶段{stage}: 共 {len(grids)} 组实验")
    print(f"{'#'*60}")
    
    # 阶段1/2：lookback固定，只需预处理一次数据
    # 阶段3：搜索lookback，每个lookback需要单独预处理
    precomputed_data = None
    if stage in [1, 2] and grids:
        print(f"\n[数据预处理] 阶段{stage} lookback固定，只预处理一次...")
        sample_cfg = Config(**{k: v for k, v in grids[0].items() if hasattr(Config, k)})
        precomputed_data = prepare_lstm_data(df_panel, sample_cfg)
        print(f"[数据预处理] 完成")
    
    # 运行所有实验
    results = []
    for i, params in enumerate(grids, 1):
        # 阶段3每个lookback不同，需要单独预处理（传None让函数内部处理）
        data_for_exp = precomputed_data if stage in [1, 2] else None
        result = run_single_experiment(
            df_panel, params, i, len(grids), verbose, data_for_exp,
            plot=plot, plot_dir=plot_dir
        )
        results.append(result)
        
        # 每次实验后保存中间结果（防止中断丢失）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        interim_path = os.path.join(output_dir, f"line{line}_stage{stage}_interim.csv")
        pd.DataFrame(results).to_csv(interim_path, index=False)
    
    # 保存最终结果
    results_df = pd.DataFrame(results)
    final_path = os.path.join(output_dir, f"line{line}_stage{stage}_{timestamp}.csv")
    results_df.to_csv(final_path, index=False)
    print(f"\n[保存] 结果已保存到: {final_path}")
    
    # 找出最优参数（按 best_val_rankic 排序）
    successful = [r for r in results if r.get("status") == "success"]
    if not successful:
        print("[警告] 所有实验都失败了！")
        return results, best_params or {}
    
    best_result = max(successful, key=lambda x: x.get("best_val_rankic", float('-inf')))
    
    # 提取最优参数
    param_keys = ["loss_name", "lr", "batch_size", "softrank_tau", 
                  "hidden", "layers", "drop_out", "lookback"]
    best_new_params = {k: best_result[k] for k in param_keys if k in best_result}
    
    # 合并之前的最优参数
    if best_params:
        merged_params = best_params.copy()
        merged_params.update(best_new_params)
    else:
        merged_params = best_new_params
    
    print(f"\n{'='*60}")
    print(f"[阶段{stage}最优] val_rankic={best_result['best_val_rankic']:.4f}")
    print(f"  参数: {merged_params}")
    print("="*60)
    
    # 保存最优参数
    best_params_path = os.path.join(output_dir, f"line{line}_stage{stage}_best.json")
    with open(best_params_path, 'w', encoding='utf-8') as f:
        json.dump(merged_params, f, indent=2, ensure_ascii=False)
    
    return results, merged_params


def run_full_search(
    df_panel: pd.DataFrame,
    line: str,
    output_dir: str = "log/hyperparam_search",
    verbose: bool = False,
    plot: bool = True
) -> Dict[str, Any]:
    """
    运行完整的三阶段搜索
    
    Args:
        df_panel: 原始面板数据
        line: "A" 或 "B"
        output_dir: 结果输出目录
        verbose: 是否打印详细训练日志
        plot: 是否为每个实验绘制训练曲线和回测图
    
    Returns:
        最终最优参数
    """
    print(f"\n{'#'*60}")
    print(f"# 开始线{line}完整搜索（三阶段）")
    print(f"{'#'*60}")
    
    best_params = None
    all_results = {}
    
    for stage in [1, 2, 3]:
        results, best_params = run_search_stage(
            df_panel, line, stage, best_params, output_dir, verbose, plot
        )
        all_results[f"stage{stage}"] = results
    
    # 保存完整搜索结果
    final_path = os.path.join(output_dir, f"line{line}_final_best.json")
    with open(final_path, 'w', encoding='utf-8') as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'#'*60}")
    print(f"# 线{line}搜索完成！最终最优参数:")
    print(f"# {best_params}")
    print(f"# 保存到: {final_path}")
    print(f"{'#'*60}")
    
    if best_params is None:
        raise RuntimeError(f"搜索失败：线{line}未能找到有效参数")
    
    return best_params


# ============================================================
# 结果分析工具
# ============================================================

def analyze_results(csv_path: str, top_n: int = 10) -> pd.DataFrame:
    """
    分析搜索结果
    
    Args:
        csv_path: 结果CSV文件路径
        top_n: 显示前N个结果
    
    Returns:
        排序后的结果DataFrame
    """
    df = pd.read_csv(csv_path)
    
    # 按 best_val_rankic 降序排序
    df_sorted = df.sort_values("best_val_rankic", ascending=False)
    
    # 显示关键列
    display_cols = [
        "loss_name", "lr", "batch_size", "softrank_tau",
        "hidden", "layers", "drop_out", "lookback",
        "best_val_rankic", "test_rankic", "best_epoch", "elapsed_seconds"
    ]
    display_cols = [c for c in display_cols if c in df_sorted.columns]
    
    print(f"\n=== Top {top_n} 结果 ===")
    print(df_sorted[display_cols].head(top_n).to_string(index=False))
    
    return df_sorted


def load_best_params(line: str, stage: int, output_dir: str = "log/hyperparam_search") -> Dict:
    """加载之前保存的最优参数"""
    path = os.path.join(output_dir, f"line{line}_stage{stage}_best.json")
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="超参数搜索")
    parser.add_argument("--line", type=str, choices=["A", "B"], required=True,
                        help="搜索线路：A=常规回归batch, B=按日batch")
    parser.add_argument("--stage", type=str, default="all",
                        help="搜索阶段：1/2/3/all")
    parser.add_argument("--output", type=str, default="log/hyperparam_search",
                        help="输出目录")
    parser.add_argument("--verbose", action="store_true",
                        help="是否打印详细训练日志")
    parser.add_argument("--plot", action="store_true",
                        help="是否为每个实验绘制训练曲线和回测图（会生成大量图片）")
    parser.add_argument("--data", type=str, default=r"data\ths_concept_daily.feather",
                        help="数据文件路径")
    args = parser.parse_args()
    
    # 加载数据
    print(f"[加载数据] {args.data}")
    df_panel = pd.read_feather(args.data)
    
    if args.stage == "all":
        # 运行完整三阶段搜索
        run_full_search(df_panel, args.line, args.output, args.verbose, args.plot)
    else:
        # 运行单个阶段
        stage = int(args.stage)
        
        # 如果是阶段2或3，尝试加载之前的最优参数
        best_params = None
        if stage >= 2:
            best_params = load_best_params(args.line, stage - 1, args.output)
            if not best_params:
                print(f"[警告] 未找到阶段{stage-1}的最优参数，使用基线配置")
        
        run_search_stage(df_panel, args.line, stage, best_params, args.output, args.verbose, args.plot)


if __name__ == "__main__":
    main()
