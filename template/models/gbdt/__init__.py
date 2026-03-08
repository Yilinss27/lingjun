from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .lgbm_model import LGBMParams, LGBMModelWrapper
    from .xgb_model import XGBParams, XGBModelWrapper

__all__ = [
    "LGBMParams",
    "LGBMModelWrapper",
    "XGBParams",
    "XGBModelWrapper",
]


def __getattr__(name: str) -> Any:
    if name in {"LGBMParams", "LGBMModelWrapper"}:
        from .lgbm_model import LGBMModelWrapper, LGBMParams

        return {"LGBMParams": LGBMParams, "LGBMModelWrapper": LGBMModelWrapper}[name]
    if name in {"XGBParams", "XGBModelWrapper"}:
        from .xgb_model import XGBModelWrapper, XGBParams

        return {"XGBParams": XGBParams, "XGBModelWrapper": XGBModelWrapper}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
