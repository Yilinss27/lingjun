# Template 重构说明

本目录已按“数据适配 + 模型可插拔 + 流程统一”的目标完成瘦身收敛。

## 当前主流程

- 训练入口：`runners/train.py`
- 预测入口：`runners/predict.py`
- 评估入口：`runners/evaluate.py`
- 冒烟测试：`runners/smoke.py`

旧入口与历史流程已迁移到 `archive/template_legacy/`。

## 新目录结构

- `configs/`：统一配置与基线 preset
- `data/`：Parquet 读取、预处理、切分、NN 样本构建
- `models/`：NN + GBDT 模型封装与 registry
- `train/`：`TorchTrainer` + `GBDTTrainer`
- `eval/`：指标、回测、日志与 reporting
- `runners/`：统一 CLI 工作流
- `utils/`：复现与辅助工具

## 与当前数据的适配点

- 数据源：项目根目录下的 `data/train.parquet` / `data/test.parquet`
- 列结构：`stockid/dateid/timeid/exchangeid + f0~f383 + LabelA/B/C(train only)`
- 大文件策略：按列读取 + row-group 分块 + 流式预测写出
- 默认切分：按 `dateid` 时间顺序切分 train/val/holdout，避免泄露

注意：`template/data/` 是代码目录，不是数据目录。

## 快速开始

在项目根目录执行：

```bash
python template/runners/train.py --family lgbm --label-col LabelA --feature-mode no_missing
```

XGBoost：

```bash
python template/runners/train.py --family xgb --label-col LabelA --feature-mode no_missing
```

神经网络（LSTM）：

```bash
python template/runners/train.py --family nn --nn-model lstm --label-col LabelA --lookback 32
```

## 说明

- 旧入口、旧流程兼容层、历史脚本已统一迁移到 `archive/template_legacy/`。
- 当前重构目标是先统一主干与质量门槛；后续你可以主要修改：
  - `data/preprocess.py`
  - `models/nn/*` 与 `models/registry.py`

## 冒烟测试

不加载全量数据，使用脚本自动生成小样本 parquet 并验证：

- 所有已注册 NN 模型可实例化并前向一次
- `runners/train.py` 在 `lgbm/xgb/nn` 三条链路可跑通（xgb 若环境缺失会跳过并提示）

```bash
python template/runners/smoke.py
```

## 评估提示

- `runners/evaluate.py` 需要预测文件包含真实标签列（默认 `y`）才能计算 `mse/mae/ic/rankic`。
- 仅包含测试集预测分数（无标签）的 `*_test_pred.parquet` 会输出提示信息，不会报错。
