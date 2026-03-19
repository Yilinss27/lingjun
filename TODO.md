# 项目待办与开发记录

> 持续更新的结构化文档，记录待办事项、已完成工作、技术决策和代码变更。

**项目**：中国 A 股微观结构预测（Parquet 42GB + Transformer/LSTM）  
**最后更新**：2026-03-19

---

## 一、待办事项

### 紧急

- [ ] 全量数据验证 V2 管线（用户自行进行）
- [ ] 实现 `_generate_submission_v2()`：对 test.parquet 构建 raw+derived cache 后推理生成提交 CSV

### 计划中

- [ ] **P1：可学习 [MISS] Embedding** — 在 `DataEmbedding` 前插入 `MissingValueEmbedding` 层，端到端学习缺失表示
- [ ] **P2：Attention Mask 排除缺失** — 修改 `FullAttention`，缺失位置完全不参与注意力计算
- [ ] **P3：SAITS 式自监督预训练** — 利用已有 `imputation` task 分支做预训练，再微调到 LabelA

### 低优先级

- [ ] derived cache 支持 float16 存储（节省磁盘）
- [ ] V2 管线的 GBDT 适配（当前仅 NN 走 V2）

---

## 二、已完成

### 2026-03-19：V2 数据管线重构

**动机**：旧管线将标准化 bake 进 memmap，改标准化方法需全量重建（20 分钟+）；且截面标准化因按 stock 存储而退化为 global。

**新架构**：

```
Pass 1: scan_train_metadata → feature_cols, train_end, val_end
Pass 2: fit_medians_streaming → 中位数（用于缺失填充）
Pass 3: build_raw_cache → X_raw.dat [n_stock, n_step, n_feature]
         只做填充，不做标准化，只建一次
Pass 4: build_derived_cache → X_derived.dat
         从 raw 做截面 RankGauss，改方法时分钟级重建
Training: DataLoader 从 derived cache 读 [lookback, F] 窗口
```

**新增文件**：

| 文件 | 职责 |
|------|------|
| `template/data/preprocess_v2.py` | 标准化策略（RankGauss / GlobalZScore / NoOp）+ 流式中位数 |
| `template/data/memmap_v2.py` | 双层缓存构建 + Dataset + DataLoader |

**修改文件**：

| 文件 | 变更 |
|------|------|
| `template/configs/pipeline_config.py` | 新增 `standardize_method`、`raw_cache_dir`、`derived_cache_dir`、`use_v2_pipeline`；`feature_mode` 默认改为 `"all"` |
| `template/runners/train.py` | 新增 `_train_nn_v2` + `_evaluate_v2`，通过 `use_v2_pipeline` 开关分叉 |

**备份**：旧代码 → `archive/data_pipeline_v1/`

**验证**：合成数据冒烟测试通过（8s x 20d x 6t x 32f，RankGauss 输出正态范围 [-1.5, 1.5]）

**回退方式**：
```python
cfg.nn.use_v2_pipeline = False
cfg.data.feature_mode = "no_missing"
```

---

## 三、缺失值处理路线图

### 数据现状

- 384 个因子中 297 个有缺失（6%~20% 缺失率）
- 旧方案只用 87 个无缺失特征，丢弃了大量有信息的因子

### 分阶段方案

| 阶段 | 方案 | 核心思想 | 状态 |
|------|------|---------|------|
| **P0** | 全特征 + 中位数填充 + 截面 RankGauss | V2 管线已实现，启用 384 个特征 | **已完成** |
| **P1** | 可学习 [MISS] Embedding | 每个特征学一个缺失向量，替代固定填充值 | 待实施 |
| **P2** | Attention Mask 排除 | 修改 attention，缺失位置不参与计算（NAIM/IFIAL） | 待实施 |
| **P3** | 自监督预训练 | BERT 式 mask→reconstruct 预训练，再微调预测 | 待实施 |

### 关键参考

| 论文 | 核心贡献 |
|------|---------|
| NAIM (arXiv:2407.11540) | Attention mask 完全排除缺失，无需填充 |
| IFIAL (arXiv:2504.14610) | 免填充增量注意力学习 |
| SAITS / PyPOTS | 对角遮蔽自注意力，时序缺失处理 SOTA |
| TabTransformer (arXiv:2012.06678) | 每个特征独立 embedding + [MISS] token |

---

## 四、技术决策记录

### 决策 1：memmap 4D 布局选择

**问题**：训练窗口需要 stock-major（同一 stock 连续时间步连续内存），截面标准化需要 cross-section（同一 dateid+timeid 的所有 stock）。

**决策**：stock-major `[n_stock, n_step, n_feature]`。

**理由**：训练窗口读取是每 epoch 的热路径，截面标准化只在构建时执行一次。Oracle 建议一致。

### 决策 2：双层缓存（raw + derived）

**问题**：改标准化方法是否需要从 Parquet 重新读取？

**决策**：raw cache 存填充后原始值（不含标准化），derived cache 存标准化后的值。改标准化只需从 raw 重建 derived。

**理由**：Parquet 读取耗时 20+ 分钟，raw→derived 只需 3-5 分钟。磁盘空间充足（>150GB）。

### 决策 3：RankGauss 中的并列处理

**问题**：中位数填充会产生大量并列值，朴素 rank 会产生偏差。

**决策**：使用 `scipy.stats.rankdata(method='average')` 处理并列。

**理由**：Oracle 明确指出 median fill 创建重复值，不应使用 ordinal rank 或朴素双重 argsort。

---

## 五、代码结构索引

```
template/
├── configs/
│   ├── pipeline_config.py        # 总配置（DataConfig, PreprocessConfig, NNConfig 等）
│   └── presets/                   # 预设配置（baseline_nn, baseline_lgbm 等）
├── data/
│   ├── preprocess_v2.py           # [V2] 标准化策略（RankGauss, GlobalZScore, NoOp）
│   ├── memmap_v2.py               # [V2] 双层缓存 + Dataset + DataLoader
│   ├── preprocess.py              # [V1 旧] 预处理（fillna + standardize）
│   ├── memmap_dataset.py          # [V1 旧] 单层 memmap
│   ├── io_parquet.py              # Parquet 读取工具（V1/V2 共用）
│   ├── parquet_adapter.py         # 特征发现、批次迭代（V1/V2 共用）
│   └── split.py                   # 日期分割（V1/V2 共用）
├── models/                        # 模型定义（Transformer, LSTM, GRU 等）
├── runners/
│   └── train.py                   # 训练入口（_train_nn_v2 / _train_nn_impl）
├── train/                         # 训练器、损失函数、checkpoint
└── eval/                          # 评估指标（IC, RankIC, MSE 等）

archive/
└── data_pipeline_v1/              # V1 旧代码备份
```

---

## 六、配置速查

### 当前默认值（V2 管线）

```python
# 数据
feature_mode = "all"                    # 全部 384 个特征
fillna_strategy = "median"              # 中位数填充

# 标准化
standardize_method = "rank_gauss"       # 截面 RankGauss

# 缓存
raw_cache_dir = "cache/v2_raw"          # raw cache 目录
derived_cache_dir = "cache/v2_derived"  # derived cache 目录
use_v2_pipeline = True                  # 启用 V2 管线

# 训练
lookback = 32                           # 滑动窗口长度
cross_date_windows = False              # 窗口不跨日期
```

### 切换标准化方法

```python
# 改为 global zscore
cfg.preprocess.standardize_method = "global_zscore"
# 只需重建 derived cache（~3-5 分钟），raw cache 不动

# 关闭标准化
cfg.preprocess.standardize_method = "none"
```

### 回退到 V1 管线

```python
cfg.nn.use_v2_pipeline = False
cfg.data.feature_mode = "no_missing"
cfg.preprocess.standardize_mode = "global"
```
