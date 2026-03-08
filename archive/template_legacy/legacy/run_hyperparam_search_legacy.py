# pyright: reportGeneralTypeIssues=false, reportAttributeAccessIssue=false, reportMissingImports=false, reportArgumentType=false
"""
超参数搜索快速启动脚本

使用方法：
    # 方式1：交互式运行（推荐）
    python run_hyperparam_search.py
    
    # 方式2：命令行指定
    python run_hyperparam_search.py --line A --stage 1
    python run_hyperparam_search.py --line B --stage all
    
    # 方式3：只分析已有结果
    python run_hyperparam_search.py --analyze log/hyperparam_search/lineA_stage1_xxx.csv

搜索策略说明：
    线A（常规回归）: mse, huber, quantile_loss
        - 阶段1: 搜索 lr × batch_size × loss_name = 5×4×3 = 60 组
        - 阶段2: 搜索 hidden × layers × dropout = 4×4×5 = 80 组
        - 阶段3: 搜索 lookback = 9 组
        
    线B（按日batch）: pearson_ic, sigmoid_rankic, neuralsort_rankic
        - 阶段1: 搜索 lr × softrank_tau × loss_name = 5×5×3 = 75 组
        - 阶段2: 搜索 hidden × layers × dropout = 4×4×5 = 80 组
        - 阶段3: 搜索 lookback = 9 组

预计时间（参考）：
    - 阶段1每组约2-5分钟，60组约2-5小时
    - 阶段2每组约2-5分钟，80组约3-7小时
    - 阶段3每组时间随lookback变化，9组约0.5-2小时
"""

from __future__ import annotations

import os
import sys
import argparse

# 确保能导入项目模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_menu():
    """打印交互式菜单"""
    print("\n" + "="*60)
    print("超参数搜索工具")
    print("="*60)
    print("\n请选择操作：")
    print("  [1] 线A-阶段1: 搜索 lr × batch_size × loss (60组)")
    print("  [2] 线A-阶段2: 搜索 hidden × layers × dropout (80组)")
    print("  [3] 线A-阶段3: 搜索 lookback (9组)")
    print("  [4] 线A-全部: 依次运行阶段1→2→3")
    print()
    print("  [5] 线B-阶段1: 搜索 lr × softrank_tau × loss (75组)")
    print("  [6] 线B-阶段2: 搜索 hidden × layers × dropout (80组)")
    print("  [7] 线B-阶段3: 搜索 lookback (9组)")
    print("  [8] 线B-全部: 依次运行阶段1→2→3")
    print()
    print("  [9] 分析结果: 查看搜索结果排名")
    print("  [0] 退出")
    print()


def run_search(line: str, stage: str, verbose: bool = False):
    """运行搜索"""
    import pandas as pd
    from hyperparam_search import (
        run_search_stage, run_full_search, load_best_params
    )
    
    # 加载数据
    data_path = r"data\ths_concept_daily.feather"
    print(f"\n[加载数据] {data_path}")
    df_panel = pd.read_feather(data_path)
    
    output_dir = "log/hyperparam_search_lstm"
    os.makedirs(output_dir, exist_ok=True)
    
    if stage == "all":
        run_full_search(df_panel, line, output_dir, verbose)
    else:
        stage_num = int(stage)
        best_params = None
        if stage_num >= 2:
            best_params = load_best_params(line, stage_num - 1, output_dir)
            if best_params:
                print(f"[加载] 使用阶段{stage_num-1}的最优参数: {best_params}")
            else:
                print(f"[警告] 未找到阶段{stage_num-1}的最优参数，将使用基线配置")
                confirm = input("是否继续？(y/n): ").strip().lower()
                if confirm != 'y':
                    print("已取消")
                    return
        run_search_stage(df_panel, line, stage_num, best_params, output_dir, verbose)


def analyze_results_interactive():
    """交互式分析结果"""
    import glob
    from hyperparam_search import analyze_results
    
    output_dir = "log/hyperparam_search_lstm"
    csv_files = glob.glob(os.path.join(output_dir, "*.csv"))
    
    if not csv_files:
        print(f"\n[错误] 未找到结果文件，请先运行搜索")
        return
    
    print("\n可用的结果文件：")
    for i, f in enumerate(csv_files, 1):
        print(f"  [{i}] {os.path.basename(f)}")
    
    choice = input("\n请选择文件编号: ").strip()
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(csv_files):
            analyze_results(csv_files[idx], top_n=15)
        else:
            print("无效选择")
    except ValueError:
        print("无效输入")


def interactive_mode():
    """交互式模式"""
    while True:
        print_menu()
        choice = input("请输入选项: ").strip()
        
        if choice == '0':
            print("再见！")
            break
        elif choice == '1':
            run_search("A", "1")
        elif choice == '2':
            run_search("A", "2")
        elif choice == '3':
            run_search("A", "3")
        elif choice == '4':
            run_search("A", "all")
        elif choice == '5':
            run_search("B", "1")
        elif choice == '6':
            run_search("B", "2")
        elif choice == '7':
            run_search("B", "3")
        elif choice == '8':
            run_search("B", "all")
        elif choice == '9':
            analyze_results_interactive()
        else:
            print("无效选项，请重新输入")


def main():
    parser = argparse.ArgumentParser(description="超参数搜索快速启动")
    parser.add_argument("--line", type=str, choices=["A", "B"],
                        help="搜索线路")
    parser.add_argument("--stage", type=str,
                        help="搜索阶段：1/2/3/all")
    parser.add_argument("--analyze", type=str,
                        help="分析指定的结果CSV文件")
    parser.add_argument("--verbose", action="store_true",
                        help="打印详细训练日志")
    args = parser.parse_args()
    
    if args.analyze:
        from hyperparam_search import analyze_results
        analyze_results(args.analyze, top_n=20)
    elif args.line and args.stage:
        run_search(args.line, args.stage, args.verbose)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
