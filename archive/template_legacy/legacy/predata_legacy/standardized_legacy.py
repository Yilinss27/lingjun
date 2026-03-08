# pyright: reportGeneralTypeIssues=false, reportAttributeAccessIssue=false, reportMissingImports=false, reportArgumentType=false, reportMissingTypeArgument=false
r'''
Author: Young yyyao75@163.com
Date: 2026-01-27 14:23:56
LastEditors: Young yyyao75@163.com
LastEditTime: 2026-01-28 15:48:41
FilePath: \dixi_thsgn\predata\standardized.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from __future__ import annotations
import pandas as pd
import numpy as np
from scipy.stats import norm
from typing import Sequence, Optional, Dict, Callable, Union, Tuple, List

# ---------------------------
# 1) 标准化方法
# ---------------------------
# 每个函数必须包含id_col和date_col这两个参数
def std_ts_zscore(
    work: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    id_col: str = "thscode",
    date_col: str = "date",
    window: int = 60,
    min_periods: Optional[int] = None,
    ddof: int = 0,
    suffix: str = "_tsz",
) -> pd.DataFrame:
    """
    时序滚动 zscore：对每个 id 做滚动均值/方差；使用 shift(1) 避免当天信息泄露
    返回：与 work 同行数同索引的 DataFrame，列名加 suffix
    """
    if min_periods is None:
        min_periods = window
    
    shifted = work.groupby(id_col, sort=False)[list(feature_cols)].shift(0)
    roll = shifted.groupby(work[id_col], sort=False).rolling(window, min_periods=min_periods)

    mu = roll.mean().reset_index(level=0, drop=True)
    sd = roll.std(ddof=ddof).reset_index(level=0, drop=True)

    z = (work[list(feature_cols)] - mu) / sd
    z = z.add_suffix(suffix)
    
    return z

def std_cs_rank(
    work: pd.DataFrame, 
    feature_cols: Sequence[str],
    *,
    id_col: str = "thscode",
    date_col: str = "date",
    rank_method: str = "average",
    suffix: str = "_csr",
) -> pd.DataFrame:
    """
    截面 rank 标准化：对每个 date 做 rank -> 映射到 [-1, 1]
    返回：与 work 同行数同索引的 DataFrame，列名加 suffix
    """
    r = work.groupby(date_col, sort=False)[list(feature_cols)].rank(
        method=rank_method, na_option="keep", pct=False
    )
    n = work.groupby(date_col, sort=False)[list(feature_cols)].transform("count")
    pct = (r - 1.0) / (n - 1.0)
    pct = pct.where(n > 1, 0.5)
    cs = pct * 2.0 - 1.0
    cs = cs.add_suffix(suffix)
    return cs

def std_cs_gauss(
    work: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    id_col: str = "thscode",
    date_col: str = "date",
    rank_method: str = "average",
    eps: Optional[float] = None,
    suffix: str = "_csg",
) -> pd.DataFrame:
    """
    截面计算出分位数后做一个Gauss逆运算
    返回：与 work 同行数同索引的 DataFrame，列名加 suffix
    """
    r = work.groupby(date_col, sort=False)[list(feature_cols)].rank(
        method=rank_method, na_option="keep", pct=False
    )

    n = work.groupby(date_col, sort=False)[list(feature_cols)].transform("count")

    u = (r - 0.5) / n
    u = u.where(n > 1, 0.5)

    if eps is None:
        eps_df = 1.0 / (n + 1.0)
        u = np.clip(u, eps_df, 1.0 - eps_df)
    else:
        u = np.clip(u, float(eps), 1.0 - float(eps))

    z = pd.DataFrame(
        norm.ppf(np.asarray(u, dtype=np.float64)),
        index=work.index,
        columns=list(feature_cols),
    )

    z = z.add_suffix(suffix).astype(np.float32)
    return z

def std_cs_zscore(
    work: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    id_col: str = "thscode",
    date_col: str = "date",
    ddof: int = 0,
    suffix: str = "_csz",
) -> pd.DataFrame:
    """
    截面zscore
    返回：与 work 同行数同索引的 DataFrame，列名加 suffix
    """
    grp = work.groupby(date_col, sort=False)[feature_cols]

    mu = grp.transform("mean")
    sd = grp.transform(lambda x: x.std(ddof=ddof))
    z = (work[feature_cols] - mu) / sd

    z = z.add_suffix(suffix).astype(np.float32)
    return z

DEFAULT_STANDARDIZERS: Dict[str, Callable[..., pd.DataFrame]] = {
    "ts_zscore": std_ts_zscore,
    "cs_rank": std_cs_rank,
    "cs_gauss": std_cs_gauss,
    "cs_zscore": std_cs_zscore,
}

def standardize_features(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    id_col: str = "thscode",
    target_col: Optional[str] = "target",
    methods: Union[str, Sequence[str]] = "combined",
    method_params: Optional[Dict[str, dict]] = None,
    drop_raw: bool = True,
    skip_existing_suffixes: Sequence[str] = ("_ts", "_cs"),
    standardizers: Optional[Dict[str, Callable[..., pd.DataFrame]]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    统一标准化接口：传 df + date_col/id_col/target_col，剩余列自动识别为特征列并按 methods 标准化。

    methods:
      - "combined" / "all"
      - "ts_zscore": 只做时序滚动 zscore
      - "cs_rank": 只做截面 rank -> [-1, 1]
      - "cs_gauss": 对截面分位数做一个Gauss逆运算
      - "cs_zscore": 只做截面 zscore
      - 或者传 list/tuple：如 ["cs_rank", "ts_zscore"]

    method_params:
      - dict，按方法名传参，例如：
        {
          "ts_zscore": {"window": 120, "min_periods": 60},
          "cs_rank": {"rank_method": "average"}
        }

    standardizers:
      - 方法注册表（可扩展）。不传则用 DEFAULT_STANDARDIZERS。
    """
    if standardizers is None:
        standardizers = DEFAULT_STANDARDIZERS
    
    if method_params is None:
        method_params = {}

    if isinstance(methods, str):
        m = methods.lower()
        if m in ("combined", "all"):
            methods_list = list(standardizers.keys())
        else: 
            methods_list = [methods]
    else:
        methods_list = list(methods)

    # 基础列校验
    for col in (id_col, date_col):
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")
    
    # 找出需要标准化的特征列：排除 id/date/target + 已经带后缀的列
    exclude_cols = {id_col, date_col}
    if target_col and target_col in df.columns:
        exclude_cols.add(target_col)

    suffix_exclude = {
        c for c in df.columns
        if any(str(c).endswith(suf) for suf in skip_existing_suffixes)
    }
    exclude_cols |= suffix_exclude

    feature_cols = [c for c in df.columns if c not in exclude_cols]

    out = df.copy()

    if not feature_cols:
        # 没有可标准化的列，直接返回（或只做 drop_raw 也无意义）
        return out, []
    

    work = out[[id_col, date_col] + feature_cols].reset_index(names="_orig_idx")
    work.sort_values([id_col, date_col, "_orig_idx"], kind="mergesort", na_position="last", inplace=True)

    new_cols_data = {}
    for name in methods_list:
        if name not in standardizers:
            raise KeyError(f"Unknown standardize method: {name}. Available: {list(standardizers.keys())}")

        fn = standardizers[name]
        params = method_params.get(name, {}).copy()

        # 统一把必要信息传进去
        params.setdefault("id_col", id_col)
        params.setdefault("date_col", date_col)

        std_df = fn(work, feature_cols, **params)
        # 逐列收集数据，避免多次concat
        for col in std_df.columns:
            new_cols_data[col] = std_df[col].values

    # 一次性构建新列DataFrame，减少内存拷贝
    new_cols = pd.DataFrame(new_cols_data, index=work.index)
    # 恢复原始索引并对齐out的索引
    new_cols["_orig_idx"] = work["_orig_idx"].values
    new_cols.sort_values("_orig_idx", kind="mergesort", inplace=True)
    new_cols.drop(columns=["_orig_idx"], inplace=True)
    new_cols.index = out.index

    # 写回结果（一次性赋值，优化性能）
    for c in new_cols.columns:
        out[c] = new_cols[c].astype(np.float32, copy=False)

    # 是否删除原始特征列（保留 id/date/target 以及已有 _ts/_cs 列）
    if drop_raw:
        out.drop(columns=feature_cols, inplace=True)

    std_col_names = [c for c in out.columns if c not in exclude_cols]
    remain_cols = std_col_names + list(suffix_exclude)
    return out, remain_cols
