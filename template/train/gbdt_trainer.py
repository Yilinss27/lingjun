# pyright: reportMissingImports=false, reportMissingTypeArgument=false
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

try:
    from template.eval import pearson_ic_by_date, rank_ic_by_date
    from template.models.gbdt import LGBMModelWrapper, LGBMParams, XGBModelWrapper, XGBParams
except ImportError:
    from importlib import import_module

    _eval_mod = import_module("eval")
    pearson_ic_by_date = getattr(_eval_mod, "pearson_ic_by_date")
    rank_ic_by_date = getattr(_eval_mod, "rank_ic_by_date")

    _gbdt_mod = import_module("models.gbdt")
    LGBMModelWrapper = getattr(_gbdt_mod, "LGBMModelWrapper")
    LGBMParams = getattr(_gbdt_mod, "LGBMParams")
    XGBModelWrapper = getattr(_gbdt_mod, "XGBModelWrapper")
    XGBParams = getattr(_gbdt_mod, "XGBParams")


@dataclass
class GBDTTrainResult:
    metrics: dict[str, float]
    feature_cols: list[str]


def _to_relevance_by_date(y: pd.Series, date: pd.Series, n_levels: int = 10) -> np.ndarray:
    frame = pd.DataFrame({"y": y, "date": date})

    def _bin(series: pd.Series) -> pd.Series:
        try:
            return pd.qcut(series, q=n_levels, labels=range(n_levels), duplicates="drop")
        except ValueError:
            ranked = series.rank(method="first")
            return pd.cut(ranked, bins=n_levels, labels=range(n_levels))

    rel = frame.groupby("date")["y"].transform(_bin)
    return np.asarray(rel.fillna(0).astype(np.int32))


class GBDTTrainer:
    def __init__(
        self,
        family: Literal["lgbm", "xgb"],
        objective: Literal["regression", "ranking"],
        model_params: dict,
        *,
        date_col: str,
        label_col: str,
    ):
        self.family = family
        self.objective = objective
        self.date_col = date_col
        self.label_col = label_col
        params_copy = dict(model_params)
        self.max_matrix_gb = float(params_copy.pop("max_matrix_gb", 6.0))
        if family == "lgbm":
            params = LGBMParams(objective=objective, **params_copy)
            self.model = LGBMModelWrapper(params)
        else:
            params = XGBParams(objective=objective, **params_copy)
            self.model = XGBModelWrapper(params)

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_cols: list[str],
    ) -> None:
        rows_total = len(train_df) + len(val_df)
        est_matrix_gb = rows_total * len(feature_cols) * 4 * 2 / (1024 ** 3)
        if est_matrix_gb > self.max_matrix_gb:
            raise ValueError(
                f"Estimated dense matrix memory {est_matrix_gb:.2f} GB exceeds limit {self.max_matrix_gb:.2f} GB. "
                "Reduce max_train_rows / feature count, or increase max_matrix_gb in model params."
            )

        if self.objective == "ranking":
            train_df = train_df.sort_values(self.date_col)
            val_df = val_df.sort_values(self.date_col)

        X_train = train_df[feature_cols].to_numpy(dtype=np.float32, copy=False)
        X_val = val_df[feature_cols].to_numpy(dtype=np.float32, copy=False)

        if self.objective == "ranking":
            y_train = _to_relevance_by_date(train_df[self.label_col], train_df[self.date_col])
            y_val = _to_relevance_by_date(val_df[self.label_col], val_df[self.date_col])
            train_group = train_df.groupby(self.date_col).size().to_numpy()
            val_group = val_df.groupby(self.date_col).size().to_numpy()
            self.model.fit(
                X_train,
                y_train,
                X_val,
                y_val,
                train_group=train_group,
                val_group=val_group,
            )
            return

        y_train = train_df[self.label_col].to_numpy(dtype=np.float32, copy=False)
        y_val = val_df[self.label_col].to_numpy(dtype=np.float32, copy=False)
        self.model.fit(X_train, y_train, X_val, y_val)

    def evaluate(self, frame: pd.DataFrame, feature_cols: list[str]) -> dict[str, float]:
        pred = self.model.predict(frame[feature_cols].to_numpy(dtype=np.float32, copy=False))
        truth = frame[self.label_col].to_numpy(dtype=np.float32, copy=False)
        pred_df = pd.DataFrame(
            {
                "date": frame[self.date_col].to_numpy(),
                "pred": pred,
                "y": truth,
            }
        )
        mse = float(np.mean((pred - truth) ** 2))
        mae = float(np.mean(np.abs(pred - truth)))
        ss_res = float(np.sum((truth - pred) ** 2))
        ss_tot = float(np.sum((truth - np.mean(truth)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else float("nan")
        ic = pearson_ic_by_date(pred_df, date_col="date", score_col="pred", target_col="y")
        rankic = rank_ic_by_date(pred_df, date_col="date", score_col="pred", target_col="y")
        return {
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "ic": float(ic),
            "rankic": float(rankic),
        }

    def save_model(self, path: str) -> str:
        return self.model.save(path)
