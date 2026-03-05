r'''
Author: Young yyyao75@163.com
Date: 2026-01-27 11:27:46
LastEditors: Young yyyao75@163.com
LastEditTime: 2026-01-29 10:12:12
FilePath: \dixi_thsgn\predata\__init__.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
"""
数据预处理模块

层次结构：
  1. 原始面板 (Raw Panel) - 构建干净的(date, id)面板
  2. 因子工程 (Feature Engineering) - 内部因子计算 + 外部因子合并
  3. 目标列   (Target) - 监督学习标签计算
  4. 标准化   (Normalization) - 时序zscore + 截面rank
  5. 样本构建 (Sample Building) - 时序切分 + 滑窗样本
"""

from .data_pipeline import (
    # 统一入口函数
    prepare_nn_data,
    
    load_selected_factors,
    # 第3层：目标列计算
    add_target_column,
    
    # 第5层：样本构建
    time_split_dates,
    build_lstm_samples,
)
from .standardized import (
  std_cs_gauss,
  std_cs_rank,
  std_cs_zscore,
  std_ts_zscore,
  standardize_features,
  DEFAULT_STANDARDIZERS,
)
from .parquet_adapter import (
    LargeParquetConfig,
    NO_MISSING_FEATURES,
    TOP_IC_FEATURES,
    discover_factor_columns,
    select_feature_columns,
    iter_parquet_batches,
    load_parquet_frame,
    prepare_large_parquet_train_frame,
    prepare_large_parquet_test_frame,
)

__all__ = [
    # 统一入口
    "prepare_nn_data",
    
    "load_selected_factors",
    # 第3层：目标列
    "add_target_column",
    
    # 第5层：样本构建
    "time_split_dates",
    "build_lstm_samples",
    "std_cs_gauss",
    "std_cs_rank",
    "std_cs_zscore",
    "std_ts_zscore",
    "standardize_features",
    "DEFAULT_STANDARDIZERS",
    "LargeParquetConfig",
    "NO_MISSING_FEATURES",
    "TOP_IC_FEATURES",
    "discover_factor_columns",
    "select_feature_columns",
    "iter_parquet_batches",
    "load_parquet_frame",
    "prepare_large_parquet_train_frame",
    "prepare_large_parquet_test_frame",
]
