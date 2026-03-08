# template_legacy Archive

This folder stores all historical entrypoints and legacy workflows migrated out of `template/`.

## Why this archive exists

- Keep `template/` focused on the active mainline modules only.
- Preserve historical scripts for traceability and rollback reference.
- Provide clear migration mapping from old entrypoints to current runners.

## Structure

- `entrypoints/`: old root-level entry scripts moved from `template/`
- `legacy/`: historical scripts and archived model implementations
- `predata/`: old predata compatibility layer and shims
- `MIGRATION_MAP.md`: path-by-path migration and replacement mapping

## Active workflow (current)

Run from project root:

```bash
python template/runners/train.py --family lgbm --label-col LabelA
python template/runners/predict.py --family lgbm --model-path <model> --meta-path <meta>
python template/runners/evaluate.py --pred-path <pred_parquet>
python template/runners/smoke.py
```
