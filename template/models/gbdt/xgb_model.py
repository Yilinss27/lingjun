# pyright: reportMissingTypeArgument=false
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np


@dataclass
class XGBParams:
    objective: str = "regression"
    learning_rate: float = 0.05
    n_estimators: int = 1000
    max_depth: int = 8
    min_child_weight: float = 1.0
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    early_stopping_rounds: int = 80
    device_type: str = "cpu"
    random_state: int = 42


class XGBModelWrapper:
    def __init__(self, params: XGBParams):
        self.params = params
        self.model = None

    def _import_xgb(self):
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError("xgboost is required for XGBModelWrapper") from exc
        return xgb

    def _build_model(self):
        xgb = self._import_xgb()
        tree_method = "gpu_hist" if self.params.device_type == "gpu" else "hist"
        base = {
            "learning_rate": self.params.learning_rate,
            "n_estimators": self.params.n_estimators,
            "max_depth": self.params.max_depth,
            "min_child_weight": self.params.min_child_weight,
            "subsample": self.params.subsample,
            "colsample_bytree": self.params.colsample_bytree,
            "reg_alpha": self.params.reg_alpha,
            "reg_lambda": self.params.reg_lambda,
            "tree_method": tree_method,
            "early_stopping_rounds": self.params.early_stopping_rounds,
            "n_jobs": -1,
            "random_state": self.params.random_state,
        }

        if self.params.objective == "ranking":
            base.update({"objective": "rank:ndcg", "eval_metric": "ndcg@10"})
            return xgb.XGBRanker(**base)

        base.update({"objective": "reg:squarederror", "eval_metric": "rmse"})
        return xgb.XGBRegressor(**base)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        *,
        train_group: np.ndarray | None = None,
        val_group: np.ndarray | None = None,
    ) -> None:
        self.model = self._build_model()

        if self.params.objective == "ranking":
            if train_group is None or val_group is None:
                raise ValueError("Ranking mode requires train_group and val_group")
            self.model.fit(
                X_train,
                y_train,
                group=train_group,
                eval_set=[(X_val, y_val)],
                eval_group=[val_group],
                verbose=50,
            )
            return

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted")
        pred = self.model.predict(X)
        return np.asarray(pred, dtype=np.float32)

    def save(self, path: str) -> str:
        if self.model is None:
            raise RuntimeError("Model is not fitted")
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            pickle.dump(self, f)
        return str(out)

    @staticmethod
    def load(path: str) -> "XGBModelWrapper":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, XGBModelWrapper):
            raise TypeError("Loaded object is not XGBModelWrapper")
        return obj
