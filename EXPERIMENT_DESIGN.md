# 实验设计文档

本文档规划论文所需的全部实验，供协作者对照执行。

---

## 论文结构回顾

**三个 Gap：**
1. 尚无人评测 LLM 理解时序质量维度 + 定量比较的能力
2. LLM 无法准确识别应比较的质量维度（维度选择不准）
3. LLM 无法对特定维度进行定量比较（只能定性推理）

**对应解决方案：**
1. 构建 Benchmark，暴露 LLM 的两方面能力缺陷
2. 以 GRPO 训练 Perceiver，instill 最优维度选择的 thinking 能力
3. 引入 Agent 工具调用，实现各维度的定量比较

---

## 实验总览

| 编号 | 实验 | 对应 Gap/RQ | 位置 | 数据集 |
|------|------|------------|------|--------|
| Exp1 | 主实验：Data Selection | 整体方法有效性 | 正文 Table 1 | 全部 11 个 |
| Exp2 | GRPO Perceiver 消融 | Gap2 / RQ2 | 正文 Figure | Long-term 4 个 |
| Exp3 | 工具调用消融 + 多 LLM | Gap3 / RQ3 | 正文 Table 3 | Long-term 4 个 |
| Exp4 | 维度设计消融 | 维度必要性 | 正文 Table 4 | Long-term 4 个 |
| Exp5 | Case Study | 定性分析 | 正文 Figure | 典型样本 |
| Exp6 | 选择比例曲线 | 评分区分度 | Appendix | Long-term 4 个 |
| Exp7 | 更多下游模型 | 泛化性 | Appendix | 全部 11 个 |

---

## Exp1：主实验 — Data Selection（正文 Table 1）

### 目的
验证 TSqualityAgent 的质量评分用于数据选择时，下游模型性能优于现有 baseline。

### 设置

| 项目 | 内容 |
|------|------|
| 任务 | 长期预测 / 短期预测 / 分类 |
| 数据集 | 长期：Electricity, ExchangeRate, Traffic, Weather |
|        | 短期：M4-Yearly, M4-Monthly, M4-Daily |
|        | 分类：MedicalImages, CBF, BME, Handwriting |
| 下游模型（Table 1） | Linear, CNN, PatchTST |
| 下游模型（Figure） | Time-MoE, Time-LLM（TSFM，仅长期预测，见下方） |
| 选择比例 | top-50% |
| 重复次数 | 多次实验取最优 |
| 指标 | 长期→RMSE↓, 短期→MAPE↓, 分类→Accuracy↑ |

### Baselines

| 方法 | 说明 |
|------|------|
| Random | 随机选 50%（引用上一工作） |
| DataOob | Data-OOB 估值（引用上一工作） |
| DataShapley | Data Shapley（引用上一工作） |
| KNNShapley | KNN-Shapley（引用上一工作） |
| TimeInf | 时序 Influence Function（引用上一工作） |
| TSRating | 上一工作方法（引用上一工作） |
| **Ours** | TSqualityAgent quality_score top-50% |

### 结果表格样式

Long-term (RMSE↓) | Short-term (MAPE↓) | Classification (Accuracy↑)

| Model | Method | Elec. | ExRate | Traffic | Wea. | M4Y | M4M | M4D | MImg | CBF | BME | HW |
|-------|--------|------:|-------:|--------:|-----:|----:|----:|----:|----:|----:|----:|---:|
| Linear | Random | | | | | | | | | | | |
| | DataOob | | | | | | | | | | | |
| | DataShapley | | | | | | | | | | | |
| | KNNShapley | | | | | | | | | | | |
| | TimeInf | | | | | | | | | | | |
| | TSRating | | | | | | | | | | | |
| | **Ours** | | | | | | | | | | | |
| CNN | Random | | | | | | | | | | | |
| | ... | | | | | | | | | | | |
| | **Ours** | | | | | | | | | | | |
| PatchTST | Random | | | | | | | | | | | |
| | ... | | | | | | | | | | | |
| | **Ours** | | | | | | | | | | | |

**Figure（TSFM 对比，仅长期预测 4 个数据集）**：Time-MoE 和 Time-LLM 作为下游模型，样式参考 TSRating 论文 Fig 3。

### 运行

```bash
# 长期预测
CUDA_VISIBLE_DEVICES=3 python -m evaluation.run_eval --task long_term_forecast

# 短期预测
CUDA_VISIBLE_DEVICES=3 python -m evaluation.run_eval --task short_term_forecast

# 分类
CUDA_VISIBLE_DEVICES=3 python -m evaluation.run_eval --task classification
```

> **注意**：Time-MoE 和 Time-LLM 需要额外集成到 evaluation 框架中（待实现）。
> Random/DataOob/DataShapley/KNNShapley/TimeInf/TSRating 数值直接引用上一工作论文。

### 预期效果
Ours 在多数数据集和模型上优于所有 baseline，尤其在预测任务上相比 TSRating 有明显提升；TSFM 图中 Time-MoE/Time-LLM 在 Ours 数据选择下top50% 优于bottom 50% 并接近full，体现 agent 标注对基础模型同样有效。

---

## Exp2：GRPO Perceiver 消融（正文 Figure）

### 目的
验证 GRPO 训练对 Perceiver 维度选择能力的提升，体现在：
1. 下游 data selection 性能提升
2. Agent 效率提升（选择更精准的维度，减少冗余调用）

> 仅两组对比，最终可能转换为分组柱状图呈现（画图由我负责，你们只需负责跑出数据填表即可）。

### 对比组

| 配置 | Perceiver | Inspector/Adjudicator |
|------|-----------|----------------------|
| w/ GRPO | perceiver-grpo-v2（LoRA） | 不变 |
| w/o GRPO | Qwen3-4B 原始模型 | 不变 |

### 数据集
Long-term 4 个：Electricity, ExchangeRate, Traffic, Weather

### 指标
| 类别 | 指标 |
|------|------|
| 下游性能 | RMSE（data selection top-50%，下游模型使用 PatchTST，也可替换为其他模型） |
| 效率 | 平均选择维度数、平均 token 消耗、平均推理时间 |

### 结果表格格式

| 配置 | Electricity | ExchangeRate | Traffic | Weather | Avg维度数 | Avg Token | Avg时间(s) |
|------|-------------|--------------|---------|---------|----------|-----------|-----------|
| w/ GRPO | | | | | | | |
| w/o GRPO | | | | | | | |

### 运行步骤

1. **w/o GRPO 标注**：用原始 Qwen3-4B 作为 Perceiver（不加载 LoRA），对 4 个数据集重新跑标注，每个数据集标满 500 对
2. **w/o GRPO 打分**：用 w/o GRPO 的标注结果训练打分模型 → 打分 → data selection
3. **对比**：与 w/ GRPO 的主实验结果对比

> **注意**：下游 data selection 实验依赖标注结果，每个数据集需要标满 500 对才能保证打分模型质量。
> 此外，打分模型不一定要沿用 meta-learning（MAML）框架——可以直接使用现成的 per-dataset Single Rater 方案（`meta_learning_rater/train_single.py`），
> 对每个数据集独立训练一个 Bradley-Terry rater，无需跨数据集元学习。具体运行方式参见 REPRODUCE.md 模块五（分支）。
> **Exp3 和 Exp4 同理**——每组消融配置均需重新标注 + 训练打分模型，也可采用同样的简化 rater 方案。

```bash
# w/o GRPO 标注（Perceiver 不加载 LoRA，直接用基础模型）
python -m annotation.run_annotation \
    --batch_config annotation/dataset_configs.json \
    --model Qwen/Qwen3-4B \
    --base_url http://localhost:8000/v1 \
    --dataset_filter electricity,exchange_rate,traffic,weather
    # 注：--dataset_filter 尚未实现，需自行在 run_annotation.py 中添加，
    # 作用是只对指定数据集运行标注，跳过其余数据集。

# 效率统计：annotation 目前可能没有直接输出维度数和 token 数的接口，
# 需要自行在标注流程中埋点（如在 run_annotation.py 中记录每次调用的
# planned_dimensions 长度和 LLM usage.total_tokens），汇总后计算均值。
# 注：效率统计不需要跑满 500 对，抽取少量样本（如 50-100 对）统计均值即可。
```

> **Appendix 补充**：在合成 Benchmark 上的维度 Precision/Recall 对比（不涉及 data selection），这个不需要你们做。

### 预期效果
左图：w/ GRPO 在 4 个数据集上 RMSE 均低于 w/o GRPO，说明维度选择更准带来了更好的标注质量。右图：w/ GRPO 平均维度数更少、token 消耗更低，体现出 GRPO 训练使 Perceiver 更聚焦、更高效。

---

## Exp3：工具调用消融 + 多 LLM（正文）

### 目的
验证 Inspector 工具调用（定量比较）相比纯文本推理的必要性，并在多个 LLM 上验证泛化性。

### 对比组

| 配置 | Inspector 模式 | 说明 |
|------|---------------|------|
| w/ Tools | 正常工具调用 | Inspector 调用统计工具做定量比较 |
| w/o Tools | 纯文本推理 | Inspector 仅凭 LLM 推理判断维度差异 |

### LLM 选择

| LLM | 参数量 | 说明 |
|-----|--------|------|
| Qwen3-4B | 4B | 主模型 |
| Gemma-3-4B | 4B | 同规模开源对比 |
| GPT-4o-mini | — | 闭源 API |
| （待定） | — | 可再加一个闭源模型（如 Claude Haiku / Gemini Flash） |

### 数据集
Long-term 4 个：Electricity, ExchangeRate, Traffic, Weather

### 指标
下游 RMSE（data selection top-50%，下游模型使用 PatchTST，也可替换为其他模型）

### 运行步骤

对每个 LLM × {w/ Tools, w/o Tools}：
1. 重新跑标注（切换 Inspector 工具调用开关）
2. 训练 TSRater → 打分 → data selection

```bash
# w/o Tools：需在 Inspector 中禁用工具调用（TODO: 加一个 --no_tools 开关）
# 注意：禁用工具后 Inspector 需要改为纯文本推理模式，
# 可能需要重新组织提示词（去掉工具调用指令，改为让 LLM 直接
# 基于序列统计信息进行定性推理并给出维度判断），需自行实现。

# 不同 LLM 示例
# Qwen3-4B
python -m annotation.run_annotation \
    --batch_config annotation/dataset_configs.json \
    --model Qwen/Qwen3-4B --base_url http://localhost:8000/v1 \
    --dataset_filter electricity,exchange_rate,traffic,weather

# GPT-4o-mini
python -m annotation.run_annotation \
    --batch_config annotation/dataset_configs.json \
    --model gpt-4o-mini --api_key <KEY> \
    --dataset_filter electricity,exchange_rate,traffic,weather
```

### 结果表格格式

先填表，最终可能转换为分组柱状图呈现（画图由我负责，你们只需负责跑出数据填表即可）。

| LLM | Tools | Electricity | ExchangeRate | Traffic | Weather | Avg |
|-----|-------|-------------|--------------|---------|---------|-----|
| Qwen3-4B | ✓ | | | | | |
| Qwen3-4B | ✗ | | | | | |
| Gemma-3-4B | ✓ | | | | | |
| Gemma-3-4B | ✗ | | | | | |
| GPT-4o-mini | ✓ | | | | | |
| GPT-4o-mini | ✗ | | | | | |
| （待定） | ✓ | | | | | |
| （待定） | ✗ | | | | | |

### 预期效果
对所有 LLM，w/ Tools 均优于 w/o Tools，说明工具调用带来的定量信息是必要的；不同 LLM 下 w/ Tools 之间差距较小，说明工具调用能弥补模型推理能力的差异，框架具有较好的 LLM 无关性。

---

## Exp4：维度设计消融（正文）

### 目的
验证三组维度对质量评估均不可或缺，缺少任一组都会导致下游性能下降。

### 维度分组

| 组 | 维度 | 含义 |
|----|------|------|
| Data Quality | missing_value, noise_level | 数据本身的完整性和清洁度 |
| Rare Pattern | rare_pattern | 异常值与有意义的稀有模式区分 |
| Pattern Structure | trend, frequency, amplitude, pattern_consistency | 时序的结构丰富性和一致性 |

### 对比组

| 配置 | 可用维度 |
|------|---------|
| Full（完整） | 全部 7 个维度 |
| w/o Data Quality | 去掉 missing_value, noise_level |
| w/o Rare Pattern | 去掉 rare_pattern |
| w/o Pattern Structure | 去掉 trend, frequency, amplitude, pattern_consistency |

### 数据集
Long-term 4 个：Electricity, ExchangeRate, Traffic, Weather

### 指标
下游 RMSE（data selection top-50%，下游模型使用 PatchTST，也可替换为其他模型）

### 实现方式
在 TSqualityAgent 的可选维度列表中屏蔽对应组的维度，重新跑标注 → 打分 → data selection。

```bash
# 需在 Agent 中加一个 --exclude_dimensions 参数，需要你们自行实现：
# 在 run_annotation.py 中接收该参数，传入 agent，
# Agent 在可选维度列表中过滤掉指定维度后再进行规划。
# 例如去掉 Data Quality 组：
python -m annotation.run_annotation \
    --batch_config annotation/dataset_configs.json \
    --model Qwen/Qwen3-4B --base_url http://localhost:8000/v1 \
    --perceiver_model perceiver-grpo-v2 \
    --perceiver_base_url http://localhost:8001/v1 \
    --exclude_dimensions missing_value,noise_level \
    --dataset_filter electricity,exchange_rate,traffic,weather
```

### 结果表格格式

| 配置 | Electricity | ExchangeRate | Traffic | Weather | Avg |
|------|-------------|--------------|---------|---------|-----|
| Full | | | | | |
| w/o Data Quality | | | | | |
| w/o Rare Pattern | | | | | |
| w/o Pattern Structure | | | | | |

### 预期效果
去掉任意一组维度后下游 RMSE 均有所上升，说明三组维度各自捕捉了不可替代的质量信号；其中去掉 Pattern Structure 预计影响最大（包含 4 个维度，覆盖最广），去掉单一维度组影响相对较小但仍可见。

---

## Exp5：Case Study（正文）

### 目的
通过两组典型案例，直观展示工具调用和 GRPO 训练各自带来的质量提升。

---

### 第一组：有工具 vs 无工具

**核心问题**：某些质量缺陷肉眼无法定量判断，只有通过工具调用的数值证据才能得出可靠结论。

**案例设计原则**：选择 `light` severity 的注入缺陷——这类缺陷在统计摘要和折线图预览中不明显（肉眼看差不多），但工具输出的频谱分析、趋势回归或变点检测能给出清晰的数值差值。

**推荐案例类型**（选择一个案例，下表仅供参考）：

| 案例 | 缺陷类型 | 为何需要工具 | 工具给出的关键证据 |
|------|---------|------------|-----------------|
| Case A | `frequency / jitter`（相位抖动） | 周期仍存在，折线图视觉上仅略显"不整齐" | 频谱熵：clean≈1.2，degraded≈2.8；dominant peak ratio 下降 |
| Case B | `amplitude / clip`（振幅截断） | 振幅被压低但均值不变，肉眼难判 | 峰-谷差、oscillation std 对比；clean≈3.2×degraded |
| Case C | `trend / flatten`（局部趋势压平） | 整体趋势统计相近，问题藏在中间一段 | 分段线性拟合斜率；工具定位 flatten 段，slope 接近 0 vs 原斜率 |

**数据来源**：用 `training/synthesis/defect_injector.py` 合成。现有合成样本（`test.jsonl`）的基础序列较简单（纯正弦），建议额外构造更"真实"的基础序列：

```python
# 推荐基础序列：含趋势 + 周期 + 轻噪声，与真实电力/气温数据形态接近
import numpy as np
from training.synthesis.defect_injector import inject_defect

t = np.arange(512)
base = 2.0 * np.sin(2 * np.pi * t / 48) + 0.003 * t + 0.3 * np.random.randn(512)

# Case A：注入轻度频率抖动
degraded, meta = inject_defect(base, "frequency", severity="light", seed=42)
# 注意 base_period=48 可传给 inject_defect 以获得更准确的周期估计
```

**展示内容**（每个 case 一张 2×2 subplot 图）：

```
┌─────────────────────────────┬─────────────────────────────┐
│  Clean 序列折线图             │  Degraded 序列折线图          │
├─────────────────────────────┴─────────────────────────────┤
│  无工具推理：Inspector 直接依赖 preview+stats 给出结论       │
│  → 结论模糊 / 置信度低 / 判断错误                           │
├─────────────────────────────────────────────────────────-─┤
│  有工具推理：Inspector 调用频谱/趋势工具，输出数值对比        │
│  → 结论明确，置信度高，附工具返回的关键数值                  │
└───────────────────────────────────────────────────────────┘
```

**运行方式**：构造好 `(clean, degraded)` 对后直接调用 `main.py`，分别以 `--no_tools`（Exp3 的开关）和正常模式各跑一次，完整推理链路见输出的 HTML 文件。各 Agent 的思维链（`<think>` 块）通常较长，写入论文时需人工从中提取关键信息。

---

### 第二组：有 GRPO vs 无 GRPO

**核心问题**：未经 GRPO 训练的 Perceiver 倾向于选取全部或大多数维度（保守策略），GRPO 训练后聚焦到真正关键的 1-3 个维度，既节省 token 又减少 Inspector 噪声。

**案例设计原则**：构造一条仅含单一维度缺陷的 degraded 序列——理想的 Perceiver 应只选该维度，baseline Perceiver 会多选。

**推荐案例**（选择一个案例，下表仅供参考）：

| 案例 | 注入缺陷 | 期望 Perceiver 选择 | baseline 典型多选 |
|------|---------|-------------------|-----------------|
| Case D | `amplitude / random_scale`（周期振幅随机波动） | `["amplitude", "pattern_consistency"]` | `["amplitude", "noise_level", "pattern_consistency", "trend", ...]` |
| Case E | `frequency / competing`（竞争频率成分）| `["frequency", "amplitude"]` | `["frequency", "amplitude", "noise_level", "pattern_consistency", ...]` |

**展示内容**（每个 case 一张 2×3 subplot 图）：

```
┌─────────────────┬──────────────────────────┬──────────────────────────┐
│  序列对比图      │  w/o GRPO：Perceiver 输出  │  w/ GRPO：Perceiver 输出  │
│                 │  planned_dimensions:      │  planned_dimensions:     │
│                 │  ["amplitude","noise",    │  ["amplitude",           │
│                 │   "pattern","trend",...]  │   "pattern_consistency"] │
│                 ├──────────────────────────┼──────────────────────────┤
│                 │  Inspector 调用 5 次工具   │  Inspector 调用 2 次工具  │
│                 │  Adjudicator: 结论分散     │  Adjudicator: 结论聚焦   │
└─────────────────┴──────────────────────────┴──────────────────────────┘
```

**运行方式**：

```bash
# w/o GRPO：直接使用基础模型作为 Perceiver
python main.py \
    --model Qwen/Qwen3-4B \
    --base_url http://localhost:8000/v1 \
    --api_key EMPTY

# w/ GRPO：使用 LoRA adapter
python main.py \
    --model Qwen/Qwen3-4B \
    --base_url http://localhost:8000/v1 \
    --api_key EMPTY \
    --perceiver_model perceiver-grpo-v2 \
    --perceiver_base_url http://localhost:8001/v1
```

完整输出见 HTML 文件；Perceiver 的 thinking 块通常较长，写入论文时需人工从中提取关键信息。

---

### 注意事项

- 合成序列要足够"真实"：基础序列建议混合趋势 + 周期 + 轻噪声，避免用纯正弦（太容易，Inspector 不需要工具也能判断）
- Case Study 的序列选定后固定 seed，保证论文图表可复现
- 可视化建议用 `matplotlib` 生成 PDF 矢量图，颜色方案：clean=蓝色实线，degraded=橙色实线，工具标注用灰色虚线/阴影区域

### 附录扩展（低优先级，有时间再做）

可用 `training/synthesis/build_dataset.py` 批量合成更多案例（覆盖不同维度组合，目前都是合成单个维度的），只展示主方法（有工具 + GRPO）的完整输出，无需与 baseline 对比，用于撑附录篇幅。
每个案例展示：序列可视化 + Agent 推理摘要 + 最终质量判断。

---

## Exp6：选择比例曲线（Appendix）

### 目的
验证评分的区分度——好的评分在低选择比例时应有更大优势。

### 设置
完全对齐引用论文 Table 18 的实验设定：Traffic 数据集 + iTransformer，选择比例 0.2 / 0.4 / 0.6 / 0.8 / 1.0，在已有 baseline 行基础上新增 Ours 一行。

### 结果表格格式

参照论文 Table 18（Forecasting performance (RMSE) under different selection ratios on the Traffic dataset with iTransformer），在原表末尾追加 Ours：

| Method | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 |
|--------|-----|-----|-----|-----|-----|
| Random | 0.382 | 0.366 | 0.356 | 0.352 | 0.339 |
| DataOob | **0.371** | 0.364 | 0.354 | 0.346 | 0.339 |
| DataShapley | 0.381 | **0.356** | **0.353** | **0.343** | 0.339 |
| KNNShapley | 0.378 | 0.363 | 0.355 | 0.346 | 0.339 |
| TimeInf | 0.377 | 0.360 | 0.357 | 0.348 | 0.339 |
| TSRating | 0.372 | **0.356** | **0.349** | **0.342** | 0.339 |
| **Ours** | | | | | |

### 预期效果
在低选择比例（0.2-0.4）时 Ours 的 RMSE 低于 Random 和大多数 baseline，随比例增大各方法趋近于 1.0 处的相同值（0.339）。

```bash
# 需修改 run_eval.py 支持 --select_ratio 参数
CUDA_VISIBLE_DEVICES=3 python -m evaluation.run_eval \
    --task long_term_forecast --dataset traffic \
    --models iTransformer --select_ratio 0.2
```

---

## Exp7：更多下游模型（Appendix）

### 目的
验证评分对不同下游模型的泛化性。

### 设置
对齐论文 Table 12-14，在各任务原有模型行基础上新增以下模型，各加一行 Ours：

| 任务 | 新增模型 | 参照论文表格 |
|------|---------|------------|
| 长期预测 | iTransformer, TimeMixer | Table 12 |
| 短期预测 | Nonstationary Transformer, DLinear | Table 13 |
| 分类 | Informer, Nonstationary Transformer | Table 14 |

数据集与 Exp1 保持一致（长期预测 4 个、短期预测 3 个、分类 4 个）。

### 预期效果
Ours 在不同架构的模型下均能带来一致的性能提升，说明质量评分的有效性不依赖特定下游模型结构。

---

## 待实现的代码改动

| 改动 | 涉及文件 | 用于实验 |
|------|---------|---------|
| Time-MoE 集成到 evaluation | `evaluation/exp.py`, `evaluation/models/` | Exp1 |
| Inspector `--no_tools` 开关 | `agents/inspector.py`, `annotation/run_annotation.py` | Exp3 |
| Perceiver `--exclude_dimensions` 参数 | `agents/perceiver.py`, `annotation/run_annotation.py` | Exp4 |
| `--select_ratio` 参数 | `evaluation/run_eval.py` | Exp6 |
| `--dataset_filter` 参数 | `annotation/run_annotation.py` | Exp2-4 |
| 效率统计（维度数、token 数） | `annotation/run_annotation.py` | Exp2 |

---

## 执行顺序

```
1. 完成主实验标注（23 个数据集） ← 最优先
2. 训练 TSRater + 打分
3. 跑 Exp1 主实验
4. 跑 Exp2（需要 w/o GRPO 的重新标注，仅 4 个数据集）
5. 实现 --no_tools 开关 → 跑 Exp3（多 LLM × 有无工具，仅 4 个数据集）
6. 实现 --exclude_dimensions → 跑 Exp4（3 组消融，仅 4 个数据集）
7. 挑选 Case Study 样本 → Exp5
8. 实现 --select_ratio → 跑 Exp6（Appendix）
9. 集成更多模型 → 跑 Exp7（Appendix）
```