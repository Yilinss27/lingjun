'''
Author: Young yyyao75@163.com
Date: 2026-01-09 09:43:20
Description: 通用支持函数（跨模块复用）
'''

from __future__ import annotations

from typing import Dict, TYPE_CHECKING
import pandas as pd

if TYPE_CHECKING:
    from template.configs.model_runtime_config import Config

# ============================================================
# 面板数据工具
# ============================================================

def check_data(
    df: pd.DataFrame,
    id_col: str = "thscode",
    date_col: str = "date",
    *,
    coerce_date: bool = True,
) -> pd.DataFrame:
    """
    检查 df 是否包含 (id_col, date_col) 两个基础列；若具备则按 (id_col, date_col) 排序并重置索引。

    Args:
        df: 面板数据
        id_col: 个体ID列名（默认 thscode）
        date_col: 日期列名（默认 date）
        coerce_date: 是否将 date_col 强制转为 datetime（默认 True）

    Returns:
        排序后的 DataFrame

    Raises:
        KeyError: 缺少必要列
        ValueError: 日期无法解析（coerce_date=True 且出现无法解析的值）
    """
    required = {id_col, date_col}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"check_data: df 缺少必要列 {missing}. 当前列: {list(df.columns)}")

    out = df  
    if coerce_date and not pd.api.types.is_datetime64_any_dtype(out[date_col]):
        parsed = pd.to_datetime(out[date_col], errors="coerce")
        if parsed.isna().any():
            bad = out.loc[parsed.isna(), date_col].head(10).tolist()
            raise ValueError(
                f"check_data: {date_col} 存在无法解析为 datetime 的值（示例前10个）：{bad}"
            )
        out[date_col] = parsed

    return out.sort_values([id_col, date_col]).reset_index(drop=True)


def build_id_map(df: pd.DataFrame, id_col: str = "thscode") -> Dict[str, int]:
    """
    构建 thscode -> 连续整数 ID 的映射
    
    用途：
        - Embedding 层需要连续整数索引
        - 分组统计时的快速查找
    
    参数：
        df: 面板数据
        cfg: 配置对象
    
    返回：
        {thscode: int} 映射字典
    """
    ids = sorted(df[id_col].astype(str).unique().tolist())
    return {k: i for i, k in enumerate(ids)}

def make_daily_topk(
    pred_df: pd.DataFrame,
    sort_col: str = "pred",
    k: int = 20,
    ascending: bool = False
) -> pd.DataFrame:
    """
    每日按指定列排序后选取 Top-K 股票
    
    参数：
        pred_df: 包含预测结果的 DataFrame，需包含 'date' 列和排序列
        sort_col: 排序依据的列名，默认为 'pred'
        k: 每日选取的股票数量，默认为 20
        ascending: 是否升序排序，默认 False（降序，即选最大的 k 个）
    
    返回：
        每日 Top-K 股票的 DataFrame
    """
    return (
        pred_df
        .sort_values(["date", sort_col], ascending=[True, ascending])
        .groupby("date", group_keys=False)
        .head(k)
    )

# ============================================================
# 时间工具
# ============================================================

def get_trading_dates(df: pd.DataFrame, cfg: Config) -> pd.DatetimeIndex:
    """
    获取面板中的所有交易日期（已排序）
    
    参数：
        df: 面板数据
        cfg: 配置对象
    
    返回：
        排序后的交易日期 DatetimeIndex
    """
    dates = pd.to_datetime(df[cfg.date_col].unique())
    return dates.sort_values()


# ============================================================
# 数据验证工具
# ============================================================

def check_panel_integrity(df: pd.DataFrame, cfg: Config) -> Dict:
    """
    检查面板数据完整性
    
    参数：
        df: 面板数据
        cfg: 配置对象
    
    返回：
        包含各项检查结果的字典
    """
    result = {
        'n_rows': len(df),
        'n_dates': df[cfg.date_col].nunique(),
        'n_stocks': df[cfg.id_col].nunique(),
        'date_range': (df[cfg.date_col].min(), df[cfg.date_col].max()),
        'duplicates': df.duplicated(subset=[cfg.date_col, cfg.id_col]).sum(),
        'null_counts': df.isnull().sum().to_dict()
    }
    return result


def print_panel_info(df: pd.DataFrame, cfg: Config, name: str = "Panel") -> None:
    """
    打印面板数据的基本信息
    
    参数：
        df: 面板数据
        cfg: 配置对象
        name: 数据集名称
    """
    info = check_panel_integrity(df, cfg)
    print(f"[{name}] {info['n_rows']} 行, {info['n_dates']} 天, {info['n_stocks']} 只股票")
    print(f"  日期范围: {info['date_range'][0]} ~ {info['date_range'][1]}")
    if info['duplicates'] > 0:
        print(f"  ⚠️ 发现 {info['duplicates']} 条主键重复")