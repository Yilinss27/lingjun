# Template 重构说明

本目录已按“数据适配 + 模型可插拔 + 流程统一”的目标完成第一轮重构。

## 当前主流程

- 训练入口：`runners/train.py`
- 预测入口：`runners/predict.py`
- 评估入口：`runners/evaluate.py`

兼容入口仍可用（会转发到新 runner）：

- `run_lgbm.py`
- `run_xgboost.py`
- `run_nn_model.py`
- `run_lgbm_parquet.py`

## 新目录结构

- `configs/`：统一配置与基线 preset
- `data/`：Parquet 读取、预处理、切分、NN 样本构建
- `models/`：NN + GBDT 模型封装与 registry
- `train/`：`TorchTrainer` + `GBDTTrainer`
- `eval/`：指标、回测、日志与 reporting
- `runners/`：统一 CLI 工作流
- `legacy/`：历史脚本归档

## 与当前数据的适配点

- 数据源：`data/train.parquet` / `data/test.parquet`
- 列结构：`stockid/dateid/timeid/exchangeid + f0~f383 + LabelA/B/C(train only)`
- 大文件策略：按列读取 + row-group 分块 + 流式预测写出
- 默认切分：按 `dateid` 时间顺序切分 train/val/holdout，避免泄露

## 快速开始

在 `template/` 目录执行：

```bash
python runners/train.py --family lgbm --label-col LabelA --feature-mode no_missing
```

XGBoost：

```bash
python runners/train.py --family xgb --label-col LabelA --feature-mode no_missing
```

神经网络（LSTM）：

```bash
python runners/train.py --family nn --nn-model lstm --label-col LabelA --lookback 32
```

## 说明

- `legacy/` 下脚本保留用于参考，不再作为主流程。
- 当前重构目标是先统一主干与质量门槛；后续你可以主要修改：
  - `data/preprocess.py`
  - `models/nn/*` 与 `models/registry.py`
