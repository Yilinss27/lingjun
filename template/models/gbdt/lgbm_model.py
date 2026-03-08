# pyright: reportMissingImports=false, reportArgumentType=false, reportMissingTypeArgument=false, reportCallIssue=false
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
import lightgbm as lgb

from template.utils import get_lgbm_seed_params


@dataclass
class LGBMParams:
    objective: str = "regression"
    learning_rate: float = 0.05
    n_estimators: int = 1200
    num_leaves: int = 127
    max_depth: int = -1
    min_child_samples: int = 50
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    early_stopping_rounds: int = 80
    device_type: str = "cpu"


class LGBMModelWrapper:
    def __init__(self, params: LGBMParams):
        self.params = params
        self.model: lgb.LGBMModel | None = None

    def _build_model(self) -> lgb.LGBMModel:
        base = {
            "learning_rate": self.params.learning_rate,
            "n_estimators": self.params.n_estimators,
            "num_leaves": self.params.num_leaves,
            "max_depth": self.params.max_depth,
            "min_child_samples": self.params.min_child_samples,
            "subsample": self.params.subsample,
            "colsample_bytree": self.params.colsample_bytree,
            "reg_alpha": self.params.reg_alpha,
            "reg_lambda": self.params.reg_lambda,
            "device_type": self.params.device_type,
            "n_jobs": -1,
            "verbose": -1,
        }
        base.update(get_lgbm_seed_params())

        if self.params.objective == "ranking":
            base.update({"objective": "lambdarank", "metric": "ndcg"})
            return lgb.LGBMRanker(**base)

        base.update({"objective": "regression", "metric": "l2"})
        return lgb.LGBMRegressor(**base)

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
        callbacks = [
            lgb.early_stopping(self.params.early_stopping_rounds, verbose=True),
            lgb.log_evaluation(50),
        ]

        if isinstance(self.model, lgb.LGBMRanker):
            if train_group is None or val_group is None:
                raise ValueError("Ranking mode requires train_group and val_group")
            self.model.fit(
                X_train,
                y_train,
                group=train_group,
                eval_set=[(X_val, y_val)],
                eval_group=[val_group],
                eval_metric="ndcg",
                eval_at=[10, 20, 50],
                callbacks=callbacks,
            )
            return

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks,
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
    def load(path: str) -> "LGBMModelWrapper":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, LGBMModelWrapper):
            raise TypeError("Loaded object is not LGBMModelWrapper")
        return obj
