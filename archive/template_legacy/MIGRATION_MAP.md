# Migration Map

This file records legacy path migration from `template/` to `archive/template_legacy/`.

## Root entrypoints migrated

- `template/run_lgbm.py` -> `archive/template_legacy/entrypoints/run_lgbm.py` -> use `template/runners/train.py --family lgbm`
- `template/run_xgboost.py` -> `archive/template_legacy/entrypoints/run_xgboost.py` -> use `template/runners/train.py --family xgb`
- `template/run_nn_model.py` -> `archive/template_legacy/entrypoints/run_nn_model.py` -> use `template/runners/train.py --family nn`
- `template/run_lgbm_parquet.py` -> `archive/template_legacy/entrypoints/run_lgbm_parquet.py` -> use `template/runners/train.py --family lgbm`
- `template/hyperparam_search.py` -> `archive/template_legacy/entrypoints/hyperparam_search.py` -> keep for historical reference
- `template/run_hyperparam_search.py` -> `archive/template_legacy/entrypoints/run_hyperparam_search.py` -> keep for historical reference
- `template/config.py` -> `archive/template_legacy/entrypoints/config.py` -> runtime-compatible config now provided at `template/configs/model_runtime_config.py`

## Legacy workflow migrated

- `template/legacy/*` -> `archive/template_legacy/legacy/*`
- `template/legacy/models_archive/*` -> `archive/template_legacy/legacy/models_archive/*`
- `template/legacy/predata_legacy/*` -> `archive/template_legacy/legacy/predata_legacy/*`

## Predata compatibility layer migrated

- `template/predata/*` -> `archive/template_legacy/predata/*`

## Current mainline retained in template/

- `template/configs/`
- `template/data/`
- `template/models/`
- `template/train/`
- `template/eval/`
- `template/runners/`
- `template/utils/`
- `template/README.md`
