# China A-Share Market Microstructure Prediction

面向超大规模 Parquet（train 约 42.6GB / test 约 11.7GB）的时序截面预测项目。

## 当前推荐入口

- 主流程文档：`template/README.md`
- 数据探索结论：`数据探索报告.md`
- 统一训练入口：`template/runners/train.py`

## 目标与约束

- 主目标：预测 `LabelA`（并保留 `LabelB/LabelC` 扩展能力）
- 数据特征：`stockid/dateid/timeid/exchangeid + f0~f383`
- 核心约束：数据过大，禁止一次性全量读入 + 全量特征暴力训练

## 已完成的框架调整方向

- 增加面向大文件的数据适配层：按列选择、按行组/批次读取、类型压缩
- 保留原有老模板脚本用于历史兼容，但默认建议使用 parquet 基线路径
- 文档统一为“先跑通高效基线，再扩展复杂模型”的工作流

## 最小可运行示例

```bash
python template/runners/train.py --family lgbm --label-col LabelA --feature-mode no_missing
```

默认读取：

- `data/train.parquet`
- `data/test.parquet`

并输出：

- `res/pipeline_lgbm/lgbm_*_meta.json`
- `res/pipeline_lgbm/lgbm_*_test_pred.parquet`

