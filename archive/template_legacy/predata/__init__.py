"""Compatibility namespace for historical `predata.*` imports.

New data workflow lives under `data/`.
"""

from .data_pipeline import (
    prepare_nn_data,
    load_selected_factors,
    add_target_column,
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
    "prepare_nn_data",
    "load_selected_factors",
    "add_target_column",
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
