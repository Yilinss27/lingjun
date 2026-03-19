# 🔍 template/ 目录全面代码审计报告

## 一、死代码扫描

### 🔴 P0 — 整个文件疑似死代码

| 文件 | 行数 | 问题 | 修复难度 |
|------|------|------|----------|
| `data/rank_gauss.py` | 272 | **无任何地方 import**。V1 时期的 RankGauss 实现，V2 已改用 `preprocess_nn.RankGaussStandardizer`（完全不同的实现）。| ✅ 可直接删除 |
| `models/layers/ETSformer_EncDec.py` | ~100+ | 无任何模型文件引用 ETSformer 层 | ✅ 可直接删除 |
| `models/layers/Pyraformer_EncDec.py` | — | 同上 | ✅ 可直接删除 |
| `models/layers/MultiWaveletCorrelation.py` | — | 同上 | ✅ 可直接删除 |
| `models/layers/FourierCorrelation.py` | — | 同上 | ✅ 可直接删除 |
| `models/layers/DWT_Decomposition.py` | — | 同上 | ✅ 可直接删除 |
| `models/layers/Crossformer_EncDec.py` | — | 同上 | ✅ 可直接删除 |
| `models/layers/Conv_Blocks.py` | — | 同上 | ✅ 可直接删除 |
| `models/layers/StandardNorm.py` | — | 同上 | ✅ 可直接删除 |

**影响**：这些 layers/ 文件是 TSLib (Time Series Library) 的遗留组件，当前只有 `AutoCorrelation.py`、`Autoformer_EncDec.py`、`Embed.py`、`SelfAttention_Family.py`、`Transformer_EncDec.py` 被实际模型使用。

### 🔴 P0 — `torch_trainer.py` 中的 V1 遗留代码（整个 TorchTrainer 类）

`template/train/torch_trainer.py`（858 行）中的 **`TorchTrainer` 类**（L608-858）是 V1 管线的遗留物：

- **`TorchTrainer`** 接受 `cfg: PipelineConfig`，通过 `cfg.nn` 访问配置
- V2 管线使用 `_MemmapTorchTrainer`（在 `runners/train.py` L565-853 中），接受独立的 `nn_cfg`
- `TorchTrainer` 被 `template/train/__init__.py` 导出，但 `runners/train.py` **虽然 import 了它（L50），却从未实际调用**
- V2 管线直接使用 `_MemmapTorchTrainer`

**具体死代码**：
| 函数/类 | 位置 | V1/V2 | 状态 |
|---------|------|-------|------|
| `get_amp_dtype(cfg)` | torch_trainer.py:46 | V1 | ❌ 读 `cfg.nn.amp_dtype`，V2 不调用 |
| `create_scaler(cfg, device)` | torch_trainer.py:55 | V1 | ❌ 读 `cfg.nn`，V2 不调用 |
| `NpDatasetTyped` | torch_trainer.py:83 | V1 | ❌ V2 用 `StockWindowDataset` |
| `collate_fn` (in torch_trainer) | torch_trainer.py:105 | V1 | ❌ V2 有自己的 `collate_fn` in memmap_cache |
| `DailyBatchDataset` | torch_trainer.py:114 | V1 | ❌ V2 不使用按日 batch 模式 |
| `make_dataloader(split_data, cfg)` | torch_trainer.py:200 | V1 | ❌ V2 用 `make_v2_dataloader` |
| `train_one_epoch()` | torch_trainer.py:314 | V1 | ❌ V2 内联训练循环 |
| `train_one_epoch_daily()` | torch_trainer.py:392 | V1 | ❌ 同上 |
| `evaluate()` | torch_trainer.py:478 | V1 | ❌ V2 用 `_evaluate_internal` |
| `evaluate_daily()` | torch_trainer.py:516 | V1 | ❌ 同上 |
| `predict_to_df()` | torch_trainer.py:564 | V1 | ❌ V2 用 `predict_loader` |
| `TorchTrainer` 整个类 | torch_trainer.py:608 | V1 | ❌ V2 用 `_MemmapTorchTrainer` |

**但注意**：`get_torch_device()` (L69) 仍被 V2 管线使用（`runners/train.py` L50 import）。

### 🟡 P1 — `runners/train.py` 中引用不存在的类

`_MemmapTorchTrainer.evaluate_loader_full()` (L752-832) 引用了 `MemmapPredictionDataset`：
```python
from template.data.memmap_cache import MemmapPredictionDataset, pred_collate_fn
```
但 **`MemmapPredictionDataset` 不存在于 `memmap_cache.py`**！该类在 `archive/data_pipeline_v1/memmap_dataset.py` 中。`memmap_cache.py` 中叫 `StockWindowPredictionDataset`。

→ `evaluate_loader_full()` 方法如果被调用会立即 crash。幸运的是它目前没被调用（V2 管线用 `_evaluate_v2` 替代），但这是一个**隐藏的破损路径**。

### 🟡 P1 — `runners/train.py` 未使用的 import

```python
from template.train import GBDTTrainer, TorchTrainer, get_torch_device  # TorchTrainer 未使用
```
`TorchTrainer` 被 import 但从未调用。

### 🟢 P2 — `utils/masking.py`

需要检查是否有任何模型引用。如果只是 TSLib 遗留，可删除。

### 🟢 P2 — V1 预处理函数仅 GBDT 管线使用

`data/preprocess.py` 中的 `fit_preprocessor_streaming()` (L167-300) 只被 `data/__init__.py` 导出，但**没有任何调用方**。V2 NN 管线用 `preprocess_nn.fit_medians_streaming()`，GBDT 管线用 `fit_transform_splits()`。

---

## 二、职责越界检查

### 🔴 P0 — `runners/train.py` 是"上帝文件"（906 行）

这是最严重的结构问题。该文件承担了**至少 5 个不同职责**：

| 职责 | 行范围 | 应该在哪 |
|------|--------|----------|
| 进度跟踪 `_ProgressTracker` | L64-168 | `template/utils/progress.py` |
| **`_MemmapTorchTrainer` 类**（290 行） | L565-853 | `template/train/memmap_trainer.py` |
| V2 评估逻辑 `_evaluate_v2` | L510-559 | `template/eval/` 或 `template/train/memmap_trainer.py` |
| 配置加载 `_load_base_config` | L171-176 | 可内联或移入 configs |
| 入口/编排逻辑 | 其余 | 正确位置 |

**`_MemmapTorchTrainer` 是最应该搬出去的**——它是一个完整的 trainer，放在 runner 文件里是因为重构时为了"先让它跑起来"。

### 🔴 P0 — 适配器类暴露底层接口缺陷

`runners/train.py` 中有两个内联适配器类：

**`_LossCfgAdapter`** (L591-594):
```python
class _LossCfgAdapter:
    class model:
        objective = "regression"
    nn = nn_cfg
```

**`_CkptCfgAdapter`** (L622-623):
```python
class _CkptCfgAdapter:
    nn = nn_cfg
```

**根因分析**：
- `get_loss_fn(cfg)` (losses.py:193) 期望 `cfg.model.objective` + `cfg.nn.*`
- `CheckpointManager(cfg)` (checkpoint.py:34) 期望 `cfg.nn`
- V2 管线没有 `cfg.nn`（直接传 `nn_cfg`），所以被迫构造假 adapter

**修复方向**：`get_loss_fn` 和 `CheckpointManager` 应改为直接接受 `nn_cfg` 参数，而非从 `PipelineConfig` 中挖取。

### 🟡 P1 — `torch_trainer.py`（858 行）

同样超过 500 行，但大部分是 V1 死代码（见上）。如果清理掉 V1 代码，只保留 `get_torch_device()`，该文件可缩减到 ~20 行或合并到 `__init__.py`。

### 🟡 P1 — `memmap_cache.py`（854 行）

合理但偏大，承担了：缓存元数据定义 + 原始缓存构建 + 派生缓存构建 + Dataset + Collate + DataLoader 工厂。建议将 Dataset 类和 DataLoader 工厂拆分到 `data/memmap_dataset.py`。

### 🟡 P1 — `eval/reporting.py` 中的 V1 遗留

L130-132:
```python
if not nn_cfg and hasattr(cfg, "nn"):
    from dataclasses import asdict
    nn_cfg = asdict(cfg.nn)
```
`PipelineConfig` **已经没有 `.nn` 属性**（V2 重构后移除了）。这段代码永远不会触发，但说明 `append_leaderboard` 还带着 V1 的痕迹。

---

## 三、性能瓶颈识别

### 当前系统指标
- GPU: RTX 5080, **利用率 11%**
- CPU: 75%
- RAM: **99% (31.4/31.8GB)**

### 🔴 P0 — RAM 压力导致 page cache thrashing（首要瓶颈）

**memmap 文件大小估算**：

假设 384 个特征，500 只股票，240 天 × 48 个时间步：
- `X_raw.dat` = 500 × 11520 × 384 × 4 bytes ≈ **8.2 GB**
- `X_derived.dat` = 同上 ≈ **8.2 GB**
- `labels.dat` = 500 × 11520 × 4 ≈ **22 MB**
- 总 memmap 文件 ≈ **~16.4 GB**

系统仅有 31.8GB RAM，已使用 99%。memmap 依赖 OS page cache 缓存热页面，但 **可用 page cache ≈ 0**。

每次 DataLoader worker 访问 memmap 页面 → page fault → 从磁盘读取 → 挤出其他页面 → **thrashing**。

这解释了：
- CPU 75%（大量时间在内核态处理 page faults）
- GPU 11%（等待数据）

### 🔴 P0 — DataLoader 的 `.copy()` 在 worker 进程中导致额外内存压力

`StockWindowDataset.__getitem__()` (memmap_cache.py:664):
```python
X = self._X[stock_idx, start_step:end_step + 1, :].copy()  # [lookback, F]
```

每次访问都做 `.copy()`。对于 memmap，这是必要的（防止跨 worker 竞争），但在 4 个 worker × prefetch_factor=4 的情况下，同时有 **4×4=16 个 batch** 在内存中。

如果 batch_size=2048, lookback=32, features=384:
- 每个样本 = 32 × 384 × 4 = 48KB
- 16 个 batch = 16 × 2048 × 48KB ≈ **1.5 GB** prefetch 缓冲区

### 🟡 P1 — `loss.item()` 同步点（影响较小）

训练循环中 (L682):
```python
total_loss += float(loss.item()) * len(X)
```

`loss.item()` 会触发 **CPU-GPU 隐式同步**。但这在每个 batch 结束时只调用一次，影响远小于数据加载瓶颈。

### 🟡 P1 — DataLoader 配置

当前默认配置 (`RunnerConfig`):
```python
dataloader_workers: int = 4
prefetch_factor: int = 4
pin_memory: bool = True
```

Windows 下被限制为 `min(num_workers, 4)` = 4 workers。在 RAM 99% 的情况下，**减少 workers 和 prefetch_factor 可以缓解内存压力**：

建议：`num_workers=2, prefetch_factor=2`。这会减少 ~750MB 的 prefetch 内存占用。

### 🟡 P1 — `build_raw_cache` 中的内存峰值

构建 raw cache 时 (memmap_cache.py:308):
```python
stock_x = np.array(X_raw[s_idx])   # copy to RAM (~N_step * N_feat * 4 bytes per stock)
```

如果 N_step=11520, N_feat=384：每只股票 copy ≈ 17MB。这本身不大，但在 RAM 99% 的环境下任何额外分配都可能触发 swap。

### 🟢 P2 — 缺少 `persistent_workers` + worker affinity 优化

`make_v2_dataloader` 已正确使用 `persistent_workers=True`（避免每个 epoch 重新 fork workers），这是正确的。

### 性能修复优先级建议

1. **降低 RAM 消耗**：减少 `num_workers=2, prefetch_factor=2`
2. **考虑 SSD vs HDD**：如果 memmap 在 HDD 上，随机访问会极慢
3. **考虑 `mmap_mode` 优化**：将 `mode="r"` 改为显式 mmap hint（madvise SEQUENTIAL）
4. **长期方案**：将 memmap 替换为 chunked tensor dataset（`torch.save` + `torch.load` mmap），或使用 WebDataset/TFRecord 格式

---

## 四、接口不一致

### 🔴 P0 — `cfg.nn` vs `nn_cfg` 双轨并存

| 文件 | 模式 | 详情 |
|------|------|------|
| `train/torch_trainer.py` | V1: `cfg.nn` | 所有函数通过 `cfg.nn` 访问。`PipelineConfig` 已不含 `.nn` |
| `train/losses.py:193` | V1: `cfg.model.objective` + `cfg.nn` | `get_loss_fn` 需要 `cfg.model.objective`，但 `PipelineConfig` 无 `.model` 属性 |
| `train/checkpoint.py:34` | V1: `cfg.nn` | `CheckpointManager.__init__` 读 `cfg.nn.*` |
| `runners/train.py` | V2: 直接传 `nn_cfg` | 使用 adapter 类桥接 |
| `eval/reporting.py:130` | V1: `cfg.nn` | `append_leaderboard` 中 `hasattr(cfg, "nn")` |

**核心矛盾**：`PipelineConfig` 已移除 `.nn` 属性（V2 将其提取为独立 `NNModelConfig`），但下游的 `get_loss_fn`、`CheckpointManager`、`TorchTrainer` 仍期望 `cfg.nn` 接口。V2 通过内联 adapter 类绕过了这个问题，但根因未修。

### 🟡 P1 — 大量 `getattr` 防御性访问（93 处）

代码中有 **93 处 `getattr(xxx, "yyy", default)`**。分为两类：

**合理使用**（模型 builder 中读取可选超参）：
```python
# models/AutoformerModel.py — 合理：不同模型有不同超参，getattr 做 fallback
emb_dim=int(getattr(cfg, "emb_dim"))
```

**接口不稳定信号**（runner 和 trainer 中）：
```python
# runners/train.py:289 — 不合理：smoke_test 已定义在 RunnerConfig dataclass 中
if getattr(cfg.runner, "smoke_test", False):
# runners/train.py:354 — 不合理：standardize_method 已定义在 PreprocessConfig 中
std_method = getattr(cfg.preprocess, "standardize_method", "rank_gauss")
```

这些 `getattr` 说明：开发者不信任 config dataclass 的字段一定存在，可能是因为旧缓存的 config 序列化对象缺少新增字段。

### 🟡 P1 — `experiment_log.py` 中的鸭子类型

`experiment_log.py` (L168-191, L262-280) 对 cfg 做了极度防御性的鸭子类型访问：
```python
model_name = getattr(cfg_any, "model_name", None)
model_func = getattr(cfg_any, "model", None)
to_dict = getattr(cfg_any, "to_dict", None)
cfg_dict = getattr(cfg_any, "__dict__", {})
```

这是因为它需要同时兼容 V1 PipelineConfig（有 `.model`, `.nn`）和 V2 report dict。

---

## 五、建议的文件结构目标

| 文件路径 | 应承担的职责（一句话） |
|---------|----------------------|
| `configs/pipeline_config.py` | 模型无关的管线配置（数据路径/预处理/运行参数） |
| `configs/model_configs/_base.py` | NN 和 GBDT 模型配置 dataclass 定义 |
| `configs/model_configs/*.py` | 各模型的具体参数实例 |
| `data/parquet_adapter.py` | Parquet 文件的批量读取和特征选择 |
| `data/io_parquet.py` | 训练面板加载和预测流式写出 |
| `data/preprocess.py` | V1 GBDT 管线的特征预处理（fit/transform） |
| `data/preprocess_nn.py` | V2 NN 管线的标准化策略（RankGauss 等） |
| `data/memmap_cache.py` | memmap 双层缓存的构建逻辑 |
| `data/memmap_dataset.py` | **新建**：PyTorch Dataset + DataLoader 工厂（从 memmap_cache.py 拆出） |
| ~~`data/rank_gauss.py`~~ | **删除**：V1 遗留，已被 preprocess_nn.py 替代 |
| `data/split.py` | 按日期分割 DataFrame |
| `train/losses.py` | 损失函数定义（**改为接受 nn_cfg 而非 cfg**） |
| `train/checkpoint.py` | 检查点管理器（**改为接受 nn_cfg 而非 cfg**） |
| `train/memmap_trainer.py` | **新建**：从 runners/train.py 搬出 `_MemmapTorchTrainer` |
| ~~`train/torch_trainer.py`~~ | **清理**：只保留 `get_torch_device()`，其余删除 |
| `train/gbdt_trainer.py` | GBDT 训练器 |
| `runners/train.py` | **瘦编排层**：加载 config → 调用 trainer → 保存结果 |
| `eval/*.py` | 保持现状（职责划分合理） |
| `models/*.py` | 各模型实现 + registry |
| `models/layers/*.py` | **清理**：删除 8 个未使用的 TSLib 遗留层文件 |
| `utils/repro.py` | 可复现性工具 |
| `utils/utils.py` | 通用工具函数 |

---

## 六、安全性分级总结

### ✅ Round 2 可直接删除（零风险）

| 项目 | 理由 |
|------|------|
| `data/rank_gauss.py` | 无 import，V2 有完全替代 |
| `models/layers/{ETSformer,Pyraformer,MultiWavelet,Fourier,DWT,Crossformer,Conv_Blocks,StandardNorm}*.py` | 无模型引用 |
| `data/preprocess.py` 中的 `fit_preprocessor_streaming()` | 无调用方 |
| `train/__init__.py` 中对 V1 函数的导出 | 去掉导出不影响任何调用方 |
| `runners/train.py` 中 `import TorchTrainer` | 未使用 |

### ⚠️ 需要重构（低风险但需测试）

| 项目 | 改动 | 测试方式 |
|------|------|----------|
| `get_loss_fn(cfg)` → `get_loss_fn(nn_cfg, objective="regression")` | 改签名 + 删除 `_LossCfgAdapter` | 跑一次完整训练 |
| `CheckpointManager(cfg)` → `CheckpointManager(nn_cfg)` | 同上 + 删除 `_CkptCfgAdapter` | 同上 |
| `_MemmapTorchTrainer` → 搬入 `train/memmap_trainer.py` | 纯移动 + import 调整 | import 检查 |
| `torch_trainer.py` 清理 | 删除所有 V1 代码，只保留 `get_torch_device` | 确认无其他调用方 |
| `evaluate_loader_full()` 中引用不存在的 `MemmapPredictionDataset` | 要么删除该方法，要么改为引用 `StockWindowPredictionDataset` | — |

### 🔧 需要性能调优（需基准测试）

| 项目 | 建议 |
|------|------|
| DataLoader workers/prefetch | 降至 `num_workers=2, prefetch_factor=2` |
| memmap + RAM | 考虑换用 SSD / 减小 cache 范围 / 增加 RAM |
| `loss.item()` 同步 | 改为 `loss.detach()` + 累加后一次性 `.item()` |
