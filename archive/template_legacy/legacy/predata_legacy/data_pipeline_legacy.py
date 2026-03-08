# pyright: reportGeneralTypeIssues=false, reportAttributeAccessIssue=false, reportMissingImports=false, reportArgumentType=false, reportMissingTypeArgument=false
'''
Author: Young yyyao75@163.com
Date: 2026-01-09
Description: 数据预处理全流程

层次结构：
  1. 原始面板 (Raw Panel) - 构建干净的(date, id)面板
  2. 因子工程 (Feature Engineering) - 内部因子计算 + 外部因子合并
  3. 目标列   (Target) - 监督学习标签计算
  4. 标准化   (Normalization) - 时序zscore + 截面rank
  5. 样本构建 (Sample Building) - 时序切分 + 滑窗样本

调用入口：prepare_lstm_data(df_panel, cfg) -> (samples, train_end, val_end)
'''

from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import os
try:
    from template.utils import check_data, build_id_map
except ImportError:
    _utils = __import__("utils", fromlist=["check_data", "build_id_map"])
    check_data = getattr(_utils, "check_data")
    build_id_map = getattr(_utils, "build_id_map")

# ============================================================
# 第1层：加载指定数据
# ============================================================
def load_selected_factors(
    json_path: str,
    factor_path: str,
    date_col: str = "date",
    id_col: str = "thscode",
    verbose: bool = True
) -> Tuple[pd.DataFrame, List[str]]:
    """
    解析 json 中的指定列，并从完整因子.f(Feather) 文件中读取对应列。
    返回：df(date, thscode, 读取到的因子列...), missing_factors(list), df中包含了数据是否被标准化的信息
    
    参数：
        json_path: JSON配置文件路径，需包含selected_factors（因子列表）和is_standardized（是否标准化）
        factor_path: Feather格式因子文件路径
        date_col: 日期列名，默认date
        id_col: 代码列名，默认thscode
        verbose: 是否输出日志，默认True
    
    返回：
        df: 筛选后的因子DataFrame，绑定is_standardized属性
        missing_factors: 缺失的因子列表
    """
    # 1. 读取JSON配置
    with open(json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    factors = cfg.get("selected_factors", None)
    
    if not isinstance(factors, list) or len(factors) == 0:
        raise ValueError(f"JSON文件{json_path}中未找到非空list类型的selected_factors字段")
    
    factors = [str(x) for x in factors]

    df_all: Optional[pd.DataFrame] = None
    available_cols = set()
    
    # 2. 读取Feather文件，优先用pyarrow，失败则降级pandas
    try:
        import pyarrow.ipc as ipc
        with ipc.open_file(factor_path, 'r') as reader:
            available_cols = set(reader.schema.names)
    except Exception as e:
        if verbose:
            print(f"[警告] pyarrow读取Feather文件失败({e})，降级使用pandas读取")
        df_all = pd.read_feather(factor_path)
        available_cols = set(df_all.columns)
    
    # 3. 校验基础列是否存在
    for base_col in (date_col, id_col):
        if base_col not in available_cols:
            raise KeyError(f"因子文件{factor_path}缺少基础列:{base_col}")
    
    # 4. 筛选存在/缺失的因子
    found_factors = [c for c in factors if c in available_cols]
    missing_factors = [c for c in factors if c not in available_cols]
    if not found_factors:
        found_factors = sorted([c for c in available_cols if c not in {date_col, id_col}])

    # 5. 输出日志
    if verbose:
        print(
            f"[selected_factors] JSON中指定因子数={len(factors)}, "
            f"因子文件中存在={len(found_factors)}, 缺失={len(missing_factors)}"
        )
        if missing_factors:
            print(f"[缺失因子] {', '.join(missing_factors)}")

    # 6. 读取指定列并返回
    selected_cols = [date_col, id_col] + found_factors
    result_df: pd.DataFrame
    
    if df_all is not None:
        result_df = df_all[selected_cols].copy()
    else:
        result_df = pd.read_feather(factor_path, columns=selected_cols)
    
    return result_df, missing_factors


# 第2层：目标列计算
# ============================================================
def add_target_column(
    df: pd.DataFrame, 
    data_path: str,
    *,
    time_col: str = "date",
    id_col: str = "thscode",
    target_period_start: int = 1,
    target_period_end: int = 2,
    ) -> pd.DataFrame:
    """
    添加 target 列（未来收益率）
    
    (基准)计算公式：
        target = T+1open-T+2close的收益率
        在指数中,只是想利用这一指标来衡量指数的表现.
    
    参数：
        df: 需要被添加target列的面板数据
        data_path: 包含了目标列的数据文件路径
        time_col: 时间列名，默认date
        id_col: 代码列名，默认thscode
        target_period_start: 目标周期开始时间，默认1
        target_period_end: 目标周期结束时间，默认2
    
    返回：
        添加了 target 列的 DataFrame
    """
    df_r = pd.read_feather(data_path)
    df_r = df_r.sort_values([time_col, id_col])
    df_r[time_col] = pd.to_datetime(df_r[time_col])
    df[time_col] = pd.to_datetime(df[time_col])

    grouped_open = df_r.groupby(id_col)['open']
    grouped_close = df_r.groupby(id_col)['close']
    
    # 未来收益率：T+1 open 买入，T+target_period+1 close 卖出
    df_r["target"] = grouped_close.shift(-target_period_end) / grouped_open.shift(-target_period_start) - 1
    df_target = df_r[[time_col, id_col, "target"]]

    df = df.merge(df_target, on=[time_col, id_col], how="left")
    return df

# ============================================================
# 第3层：样本nn构建
# ============================================================
def time_split_dates(df: pd.DataFrame, date_col: str = "date", train_ratio: float = 0.70, val_ratio: float = 0.15) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    根据配置的比例计算 train/val/test 时间切分点
    
    参数：
        df: 面板数据
        train_ratio: 训练集比例
        val_ratio: 验证集比例
    
    返回：
        (train_end_date, val_end_date)
    """
    dates = sorted(df[date_col].unique())
    n = len(dates)
    train_end = dates[int(n * train_ratio) - 1]
    val_end = dates[int(n * (train_ratio + val_ratio)) - 1]
    return pd.to_datetime(train_end), pd.to_datetime(val_end)


def _process_stock(args) -> Dict[str, Dict[str, List]]:
    """
    并行处理单个股票的样本构建（可被并行调用）
    
    参数：
        args: (idx, Xmat, yvec, dates, lookback, train_end, val_end)
    
    返回：
        return:
    {
        "train": {
            "X": np.ndarray,  # 形状 (N_train, lookback, F)，float32，时序特征样本
            "y": np.ndarray,  # 形状 (N_train,)，float32，标签值
            "date": np.ndarray,  # 形状 (N_train,)，datetime64[ns]，样本对应的日期
            "index": np.ndarray  # 形状 (N_train,)，object，样本对应的股票ID（thscode）
        },
        "val": {
            "X": np.ndarray,  # 形状 (N_val, lookback, F)
            "y": np.ndarray,  # 形状 (N_val,)
            "date": np.ndarray, 
            "index": np.ndarray
        },
        "test": {
            "X": np.ndarray,  # 形状 (N_test, lookback, F)
            "y": np.ndarray,  # 形状 (N_test,)
            "date": np.ndarray,
            "index": np.ndarray
        }
    }
    """
    idx, Xmat, yvec, dates, id, lookback, train_end, val_end = args
    
    Xmat = np.asarray(Xmat, dtype=np.float32)
    yvec = np.asarray(yvec, dtype=np.float32)
    dates = pd.to_datetime(dates)

    out = {
        "train": {"X":[], "y":[], "date":[], "index":[], "id":[]},
        "val": {"X":[], "y":[], "date":[], "index":[], "id":[]},
        "test": {"X":[], "y":[], "date":[], "index":[], "id":[]}
    }
    
    if len(dates) < lookback:
        return out
    
    for t in range(lookback - 1, len(dates)):
        x_win = Xmat[t - lookback + 1:t + 1].copy()
        y = float(yvec[t])
        
        # 同时过滤NaN和inf
        if np.any(np.isnan(x_win)) or np.any(np.isinf(x_win)) or np.isnan(y) or np.isinf(y):
            continue

        d = pd.Timestamp(dates[t])
        split = "train" if d <= train_end else ("val" if d <= val_end else "test")

        out[split]["X"].append(x_win.astype(np.float32, copy=False))
        out[split]["y"].append(y)
        out[split]["date"].append(d)
        out[split]["index"].append(idx)
        out[split]["id"].append(id)
    
    return out


def build_lstm_samples(
    df: pd.DataFrame,
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
    lookback: int,
    id_col: str = "thscode",
    date_col: str = "date",
    use_parallel: bool = True, #是否使用并行处理
    num_workers: Optional[int] = None,
    verbose: bool = True #是否输出日志
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    根据输入的面板数据和时间切分点构建 LSTM 序列样本
    
    返回：
        {
            "train": {"X": [N,L,F], "id": [N], "y": [N], "date": [N], "index": [N]},
            "val": {...},
            "test": {...}
        }
    """
    if df.empty:
        raise ValueError("df 为空，无法构建样本。")
    
    feats = [col for col in df.columns if col not in [date_col, id_col, "target"]]

    # 基础列检查: 可选
    need_cols = {id_col, date_col, "target", *feats}
    miss = [c for c in need_cols if c not in df.columns]
    if miss:
        raise ValueError(f"df 缺少必要列: {miss}")

    # 统一类型与排序（不改动外部 df）: 可选
    work = df[[id_col, date_col, "target", *feats]].copy()
    work[date_col] = pd.to_datetime(work[date_col])

    # 计算理论样本统计(无缺失数据假设)
    work = work.sort_values([id_col, date_col])
    id_map = build_id_map(work, id_col)
    F = len(feats)
    pos_in_id = work.groupby(id_col).cumcount()
    eligible = pos_in_id >= lookback - 1

    d = work[date_col].to_numpy()
    split = np.where(d <= train_end, "train", np.where(d <= val_end, "val", "test"))

    theo_counts = (
        pd.Series(split[eligible])
        .value_counts()
        .reindex(["train", "val", "test"], fill_value=0)
        .to_dict()
    )

    if verbose:
        print("\n[样本统计-理论(无缺失假设)]")
        for s in ["train", "val", "test"]:
            n = int(theo_counts.get(s, 0))
            print(f" {s}: X=({n},{lookback},{F}), y=({n},)")
    # 构建并行/串行参数
    stock_groups = []
    for idx, g in work.groupby(id_col, sort=False):
        id = id_map.get(str(idx), -1) #TODO
        if id == -1:
            raise ValueError(f"股票 {idx} 在 id_map 中不存在")
        g = g.sort_values(date_col)
        Xmat = g[feats].to_numpy(dtype=np.float32)
        yvec = g["target"].to_numpy(dtype=np.float32)
        dates = g[date_col].to_numpy()
        stock_groups.append((idx, Xmat, yvec, dates, id, lookback, train_end, val_end))
    
    # 输出容器
    buckets = {
        "train": {"X": [], "y": [], "date": [], "index": [], "id": []},
        "val":   {"X": [], "y": [], "date": [], "index": [], "id": []},
        "test":  {"X": [], "y": [], "date": [], "index": [], "id": []},
    }

    def _merge_one(res: Dict[str, Dict[str, List]]):
        for split in ["train", "val", "test"]:
            for k in buckets[split].keys():
                buckets[split][k].extend(res[split][k])
    # -------- 执行构建 --------
    if use_parallel and len(stock_groups) > 1:
        max_workers = num_workers or min(os.cpu_count() or 1, len(stock_groups))
        if verbose:
            print(f"\n使用 {max_workers} 个进程并行构建样本...")

        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_process_stock, args) for args in stock_groups]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="构建样本", disable=not verbose):
                _merge_one(fut.result())
    else:
        if verbose:
            print("\n使用单进程构建样本...")
        for args in tqdm(stock_groups, desc="构建样本", disable=not verbose):
            _merge_one(_process_stock(args))
    
    def pack(b: Dict[str, list]) -> Dict[str, np.ndarray]:
        n = len(b["X"])
        X = np.stack(b["X"], axis=0).astype(np.float32, copy=False) if n else np.zeros((0, lookback, F), np.float32)
        y = np.asarray(b["y"], dtype=np.float32) if n else np.zeros((0,), np.float32)
        dates = np.asarray(b["date"], dtype="datetime64[ns]") if n else np.asarray([], dtype="datetime64[ns]")
        idx = np.asarray(b["index"], dtype=object) if n else np.asarray([], dtype=object)
        ids = np.asarray(b["id"], dtype=np.int64) if n else np.asarray([], dtype=np.int64)
        return {"X": X, "y": y, "date": dates, "index": idx, "id": ids}

    samples = {k: pack(v) for k, v in buckets.items()}

    if verbose:
        print("\n[样本统计-实际(已过滤NaN)]")
        for split in ["train", "val", "test"]:
            print(f"  {split}: X={samples[split]['X'].shape}, y={samples[split]['y'].shape}")
    return samples


# ============================================================
# 统一入口函数
# ============================================================
def prepare_nn_data(
    daily_data_path: str,
    json_path: str,
    factor_path: str,
    start_date,
    end_date,
    lookback: int,
    target_period_start: int = 1,
    target_period_end: int = 2,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    date_col: str = "date",
    id_col: str = "thscode",
    use_parallel: bool = True,
    num_workers: Optional[int] = None,
    verbose: bool = True
) -> Tuple[Dict[str, Dict[str, np.ndarray]], pd.Timestamp, pd.Timestamp, int, int]:
    """
    基于面板数据(['date', 'thscode', 'factor1', 'factor2', ..., 'target'])构建神经网络可适配的数据结构. 
    注意: 因子选择是独立于模型构建,训练和预测步骤的.
    
    流程：
        1. 利用load_selected_data读取目标数据(由于一般情况下因子筛选完后是一个不会改变的参数, 所以此处直接指定路径, 在路径下会设置一个默认文件, 可以通过调整默认文件指定不同的因子)
        2. 计算目标列target(index为默认T+1open-T+2close, stock为默认T+1open-T+2open)
        4. 构建序列样本
    
    参数：
        df_panel: 原始面板数据
        cfg: 配置对象
    
    返回：
        samples: {"train": {...}, "val": {...}, "test": {...}}
        train_end: 训练集截止日期
        val_end: 验证集截止日期
    """
    print("=" * 60)
    print("NN DATA PIPELINE - 开始")
    print("=" * 60)
    
    # 第1层：原始面板排序
    print("\n[第1层] 读取选中因子")
    df, missing_factors = load_selected_factors(json_path, factor_path, date_col, id_col)
    # df = check_data(df, date_col, id_col)
    
    print(f"  面板规模: {len(df)} 行, {df[date_col].nunique()} 天(约{df[date_col].nunique()/244}年), {df[id_col].nunique()} 只股票(日均{df.groupby(date_col).count()[id_col].mean():.2f}/天)")
    
    
    # 第2层：目标列计算
    print("\n[第2层] 计算目标列...")
    df = add_target_column(df, daily_data_path, target_period_start=target_period_start, target_period_end=target_period_end)
    print(f"  目标周期: T+{target_period_start}open-T+{target_period_end}close (open买入, close卖出)")
    
    # 过滤起始日期
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    df = df[(df[date_col] >= start) & (df[date_col] <= end)]
    num_factors = len([col for col in df.columns if col not in {date_col, id_col, 'target'}])
    num_concept = df[id_col].nunique()
    print(f"  过滤后: {len(df)} 行 {start_date}<=date<={end_date}")
    
    # 第3层：样本构建
    print("\n[第3层] 构建序列样本...")
    train_end, val_end = time_split_dates(df, date_col, train_ratio, val_ratio)
    print(f"  时间切分: train_end={train_end.date()}, val_end={val_end.date()}")

    samples = build_lstm_samples(df, train_end, val_end, lookback, id_col, date_col, use_parallel, num_workers, verbose)
    
    print("\n" + "=" * 60)
    print("NN DATA PIPELINE - 完成")
    print("=" * 60 + "\n")
    
    
    return samples, train_end, val_end, num_factors, num_concept
