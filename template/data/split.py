from __future__ import annotations

import pandas as pd


def split_frame_by_date(
    frame: pd.DataFrame,
    date_col: str,
    train_end: object,
    val_end: object,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = frame.loc[frame[date_col] <= train_end]
    val_df = frame.loc[(frame[date_col] > train_end) & (frame[date_col] <= val_end)]
    holdout_df = frame.loc[frame[date_col] > val_end]
    return train_df, val_df, holdout_df
