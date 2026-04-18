# Perceiver 训练数据集设计文档

## 1. 背景与目标

TSqualityAgent 的 Perceiver 负责两个关键决策：
1. **维度选择**（`planned_dimensions`）：从 7 个质量维度中选出需要比较的子集
2. **工具决策**（`tool_required`）：每个选中维度是否需要调用工具，还是纯推理即可

当前 Perceiver 依赖 prompt 引导 LLM 完成这两个决策，效果不稳定——容易选择冗余维度、滥用或忽略工具。目标是构建结构化训练数据集，为后续强化学习优化 Perceiver 提供基础。

---

## 2. 核心思路

在合成数据基础设施（base series 生成器 + 缺陷注入器）上，自动构造大量 Perceiver 的输入-标签对：

```
每个训练样本 = 一个 Perceiver 应该做出的决策

输入：与推理时完全一致的 series preview + 统计量（不含场景描述）
标签：应选哪些维度 + 哪些需要工具
```

---

## 3. 数据生成流程

### 3.1 生成 base series

`base_generator.py` 的 `generate_random_base()` 支持三种组合方式：

| 组合方式 | 概率 | 说明 |
|:--|:-:|:--|
| additive | 55% | `y = trend + seasonal + noise`（最常见） |
| multiplicative | 30% | `y = trend_pos × (1 + α·seasonal_norm) + noise`，趋势放大/缩小周期振幅 |
| sequential | 15% | `y = [seg₁ \| seg₂ \| ...]  + noise`，2-3 段独立模式拼接，段间 level-align |

各组件独立采样：

- **趋势**：6 种类型（flat / linear_up / linear_down / piecewise / exponential / log）
- **周期**：6 种类型（none / sine / square / triangle / sawtooth / mixed），其中 none 概率较低（15%）
- **噪声结构**：4 种（white 45% / ar1 30% / heteroscedastic 15% / random_walk 10%），幅度保持在信号标准差的 3-8%（低水平基线）
- **尺度**：整体随机缩放（0.1 ~ 100，对数均匀）+ 随机偏移

A、B 两侧在共享同一 base 的基础上各自独立叠加微量噪声（base noise std 的 40-60%），使非缺陷区段也有细微差异。

> **注**：训练样本不包含场景描述（`dataset_description` 置为空字符串）。随机分配的描述与合成序列特征无关，会引入误导信息；推理时用户提供的真实描述作为额外信号。

### 3.2 维度组合采样

从 7 个质量维度中选 N 个进行缺陷注入，N 的分布偏向少维度（符合现实）：

| N（注入维度数） | 概率  | 说明 |
|:-:|:-:|:--|
| 0 | 5% | tie case，双方无缺陷 |
| 1 | 35% | 最常见：单维度比较 |
| 2 | 30% | 常见：两维度 |
| 3 | 18% | 中等频率 |
| 4 | 8% | 较少 |
| 5 | 3% | 罕见 |
| 6+ | 1% | 极罕见 |

7 个维度：`missing_value`, `noise_level`, `rare_pattern`, `trend`, `frequency`, `amplitude`, `pattern_consistency`

### 3.3 缺陷注入

对每个选中的维度：
- **随机选 severity**：`light` 或 `heavy`，各 50%
- **随机选降质侧**：A 或 B，各 50%（另一侧保持不变）
- **随机选注入方式**：每个维度有 3-4 种注入方式，随机选一种
- **区间采样参数**：每个参数从 `(lo, hi)` 区间均匀采样，而非固定值

每个维度的注入方式及其对应的检测工具：

| 维度 | 注入方式 | 主要检测工具 |
|:--|:--|:--|
| missing_value | random_scatter（随机散点）/ burst（连续块）/ periodic（周期性缺失） | `missing_ratio` |
| noise_level | gaussian（全局噪声）/ heteroscedastic（局部噪声）/ impulsive（脉冲噪声） | `noise_profile`, `volatility` |
| rare_pattern | point_outlier（尖峰）/ contextual（情境异常）/ level_shift（水平突变） | `mad_residual_outlier`, `contextual_rare_pattern` |
| trend | flatten（段平坦化）/ drift（渐变漂移）/ reversal（局部反转） | `trend_classifier` |
| frequency | competing（竞争频率）/ jitter（相位噪声）/ period_shift（周期突变） | `seasonality_detector` |
| amplitude | random_scale（随机缩放）/ decay（渐变衰减）/ clip（削峰限幅） | `cycle_amplitude`, `rolling_amplitude` |
| pattern_consistency | variance_switching（方差切换）/ structural_break（永久断裂）/ flat_spots（平坦段）/ mean_drift（均值漂移） | `pattern_consistency_indicators`, `stationarity_test`, `change_point_detector` |

### 3.4 标签生成

每个样本有两个标签：

**`target_dimensions`**：注入了缺陷的维度列表 = Perceiver 应该选的维度

**`tool_required`**：需要工具的维度子集，由两级规则决定：

**维度级覆盖**（`label_schema.py` 的 `DIMENSION_TOOL_OVERRIDE`）：calibration 显示某些维度无论 severity 如何，LLM 凭 preview/stats 即可可靠判断，固定为 heavy（无需工具）：

| 维度 | 固定策略 | Calibration 依据 |
|:--|:--|:--|
| `missing_value` | 始终 heavy（不需工具） | 所有参数值下准确率 = 100%，`missing_ratio` 字段直接可读 |

**方法级覆盖**（`defect_injector.py` 的 `always_heavy` 标记）：rare_pattern 内部分方法 calibration 显示全程 heavy，单独标记：

| 方法 | 标记 | Calibration 依据 |
|:--|:--|:--|
| `point_outlier` | `always_heavy=True` | sigma=2, count=1 时准确率已达 0.967 |
| `level_shift` | `always_heavy=True` | sigma=1.5 时准确率已达 0.90，全程无 light 区间 |

**Severity 默认规则**（其余方法）：

| severity | 是否需要工具 | 理由 |
|:-:|:-:|:--|
| light | 是 | 差异细微，肉眼难以判断，需要工具精确测量才能得出结论 |
| heavy | 否 | 差异明显，模型凭 preview/统计量 + 自身推理即可判断 |

### 3.5 输入特征构造

与 Perceiver 推理时看到的完全一致：
- **series preview**：序列长度 ≤ 200 返回完整序列，否则采样（头 20 + 中间等距采样 + 尾 20，共 60 点）
- **basic stats**：`length`, `missing_ratio`, `mean`, `std`, `min`, `max`, `p25`, `p75`, `slope`
- **dataset_description**：训练时为空字符串，推理时为用户提供的真实场景描述

---

## 4. 文件结构

```
training/
├── __init__.py
├── DESIGN_ZH.md
├── data/                        # 生成的数据集（JSONL + 可视化 HTML）
├── synthesis/                   # 数据合成模块
│   ├── __init__.py
│   ├── base_generator.py        # 基础序列生成
│   ├── defect_injector.py       # 缺陷注入（7 维度 × 3-4 方法）
│   ├── label_schema.py          # 标签定义：severity 档位、N 分布权重、tool_required 判定
│   ├── sample_generator.py      # 核心生成逻辑：单样本生成
│   ├── build_dataset.py         # CLI 入口：批量生成 → 输出 JSONL
│   ├── calibration.py           # Severity 参数标定实验
│   └── visualize.py             # 可视化：JSONL → 折叠式 HTML
└── rl/                          # 强化学习训练模块
    ├── __init__.py
    ├── reward.py                # 奖励函数（format + dim F1 + tool acc）
    ├── data_loader.py           # JSONL → HuggingFace Dataset
    └── train_grpo.py            # GRPO 训练主脚本（LoRA + TRL）
```

### 模块依赖关系

```
synthesis/build_dataset.py
  └── synthesis/sample_generator.py
        ├── synthesis/base_generator.py
        ├── synthesis/defect_injector.py
        └── synthesis/label_schema.py

synthesis/calibration.py   (独立实验，依赖 synthesis/base_generator + defect_injector + models/llm)
synthesis/visualize.py     (独立，供 build_dataset --visualize 和单独调用)

rl/train_grpo.py
  ├── rl/data_loader.py    (依赖 agents/perceiver.py 的 SYSTEM_PROMPT)
  └── rl/reward.py
```

---

## 5. 输出格式

JSONL 文件，每行一个样本：

```json
{
  "sample_id": "seed42_noise_level_trend",
  "input": {
    "dataset_description": "",
    "preview_A": [1.23, 2.34, ...],
    "preview_B": [1.25, 2.31, ...],
    "stats_A": {"length": 150, "missing_ratio": 0.0, "mean": 5.12, "std": 2.34, "slope": 0.012, ...},
    "stats_B": {"length": 150, "missing_ratio": 0.0, "mean": 5.08, "std": 3.67, "slope": 0.008, ...}
  },
  "labels": {
    "target_dimensions": ["noise_level", "trend"],
    "tool_required": ["noise_level"]
  },
  "meta": {
    "defect_details": [
      {"dimension": "noise_level", "severity": "heavy", "side": "B", "metadata": {"method": "gaussian", "multiplier": 3.2}},
      {"dimension": "trend", "severity": "light", "side": "A", "metadata": {"method": "drift", "drift_strength": 0.45}}
    ],
    "base_attributes": {"composition": {"type": "additive"}, "noise": {"type": "ar1"}, ...}
  }
}
```

字段说明：
- `input`：Perceiver 的输入特征，与推理时格式一致（`dataset_description` 训练时为空）
- `labels`：训练标签
  - `target_dimensions`：应选维度（ground truth）
  - `tool_required`：需要工具的维度子集
- `meta`：元信息，用于分析和调试，不参与训练

---

## 6. 使用方式

### 6.1 生成训练数据集

```bash
# 生成训练集（默认 1500 样本）
python -m training.synthesis.build_dataset --output training/data/perceiver_train.jsonl --stats

# 自定义样本数
python -m training.synthesis.build_dataset --n_samples 2000 --output training/data/train.jsonl --stats

# 生成验证集（不同 seed 范围避免与训练集重叠）
python -m training.synthesis.build_dataset --n_samples 500 --seed_offset 1000000 --output training/data/perceiver_val.jsonl --stats

# 少量样本检查格式
python -m training.synthesis.build_dataset --n_samples 20 --output training/data/test.jsonl --visualize

# 生成并可视化（折叠式 HTML）
python -m training.synthesis.build_dataset --n_samples 20 --output training/data/test.jsonl --visualize
```

`--stats` 参数会在生成完成后打印数据集统计信息（维度数量分布、各维度出现频率、severity 分布、工具使用占比），并自动保存为同名 `.stats.json` 文件。

### 6.2 Severity 参数标定实验（`calibration.py`）

标定实验的目标：对每个维度的每种注入方式，系统性地扫描关键参数，通过让 LLM 在**仅凭 preview + 统计量**（无工具）的条件下判断哪侧更差，确定 light / heavy 的合理参数区间边界。

**实验逻辑**：
- 为每个 `（维度, 注入方式, 参数, 参数值）` 组合生成 N 对干净/降质序列
- 随机将降质侧分配给 A 或 B
- 询问 LLM 哪侧更差（无工具），统计准确率 + 95% Wilson 置信区间
- **准确率 ≥ 85% → heavy**（LLM 无需工具即可可靠判断）
- **准确率 < 85% → light**（差异细微，需要工具辅助）
- 95% CI 作为参考，辅助判断结果是否稳定（CI 较宽时可增加 n_pairs 重跑）

**覆盖范围**：7 个维度 × 共 34 个（方法, 参数）配置——每个多参数方法的每个参数都单独扫描，其余参数固定在中间值，每条配置扫描 4-8 个参数值。

**输出文件**：每次运行自动保存两个文件（同名不同后缀）：
- `.json`：各参数值的准确率、CI、correct/total 统计
- `.html`：可视化报告，含颜色编码（绿色 heavy / 灰色 light）和 95% CI 参考列

```bash
# 全量运行（本地 Qwen3-4B，需先启动 vLLM）
python -m training.synthesis.calibration \
    --model Qwen/Qwen3-4B \
    --base_url http://localhost:8000/v1 \
    --api_key EMPTY

# 全量运行（云端模型）
python -m training.synthesis.calibration --model gpt-4o-mini

# 只扫描指定维度
python -m training.synthesis.calibration --dim noise_level rare_pattern

# 自定义保存路径
python -m training.synthesis.calibration \
    --model Qwen/Qwen3-4B \
    --base_url http://localhost:8000/v1 \
    --api_key EMPTY \
    --out logs/calibration_v2.json
```

**参数说明**：

| 参数 | 默认值 | 说明 |
|:--|:-:|:--|
| `--model` | `gpt-4o-mini` | LLM 模型名称 |
| `--base_url` | chatanywhere | API 地址，本地 vLLM 填 `http://localhost:8000/v1` |
| `--api_key` | 读 `OPENAI_API_KEY` 环境变量 | 本地模式填 `EMPTY` |
| `--enable_thinking` | 关闭 | 是否启用 Qwen3 思考模式 |
| `--dim` | 全部 7 个维度 | 指定要扫描的维度（空格分隔） |
| `--n_pairs` | `30` | 每个参数值生成的样本对数（CI 较宽时可加到 50） |
| `--seed` | `42` | 随机种子 |
| `--out` | 自动生成 | 自定义保存路径（同时生成同名 `.html`） |

---

## 6.3 RL 训练（`rl/train_grpo.py`）

用 GRPO 对 Qwen/Qwen3-4B 做 LoRA 微调，训练 Perceiver 准确选维度 + 判断是否需要工具。

**依赖安装**（服务器上首次运行前）：

```bash
pip install torch transformers peft trl datasets accelerate
```

**奖励函数**（`rl/reward.py`）：

| 组件 | 权重  | 计算方式 |
|:--|:---:|:--|
| format | 0.1 | JSON 可解析且 schema 正确 |
| dim F1 | 0.6 | predicted vs target_dimensions 的 F1 score |
| tool acc | 0.3 | 仅在命中维度交集上算 tool_required 准确率 |

**运行命令**：

```bash
# 调试运行（用 test.jsonl 跑通框架，20 条，1 epoch）
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 -m training.rl.train_grpo \
    --data training/data/test.jsonl \
    --val_data training/data/val.jsonl \
    --eval_steps 1 \
    --model Qwen/Qwen3-4B \
    --output training/checkpoints/perceiver-debug

# 正式训练 — 单卡
CUDA_VISIBLE_DEVICES=3 python -m training.rl.train_grpo \
    --data training/data/perceiver_train.jsonl \
    --val_data training/data/val.jsonl \
    --model Qwen/Qwen3-4B \
    --output training/checkpoints/perceiver-grpo


# 正式训练 — 双卡 DDP（有效 batch 翻倍，速度约 2×）
CUDA_VISIBLE_DEVICES=2,3 PYTORCH_ALLOC_CONF=expandable_segments:True \
    torchrun --nproc_per_node=2 -m training.rl.train_grpo \
        --data training/data/perceiver_train.jsonl \
        --val_data training/data/perceiver_val.jsonl \
        --model Qwen/Qwen3-4B \
        --output training/checkpoints/perceiver-rl

# 部署训练后的 LoRA adapter
CUDA_VISIBLE_DEVICES=3 vllm serve Qwen/Qwen3-4B \
    --enable-lora \
    --lora-modules perceiver=training/checkpoints/perceiver-grpo \
    --port 8000 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --max-model-len 32768
```

**主要参数说明**：

| 参数 |                  默认值                  | 说明 |
|:--|:-------------------------------------:|:--|
| `--data` |                  必填                   | JSONL 训练数据路径 |
| `--model` |            `Qwen/Qwen3-4B`            | 基础模型名或本地路径 |
| `--output` | `training/checkpoints/perceiver-grpo` | LoRA adapter 保存路径 |
| `--lora_rank` |                 `16`                  | LoRA rank |
| `--lora_alpha` |                 `32`                  | LoRA alpha |
| `--epochs` |                  `1`                  | 训练轮数 |
| `--batch_size` |                  `4`                  | 每步唯一 prompt 数 |
| `--num_generations` |                  `4`                  | 每个 prompt 采样次数（GRPO 组内比较） |
| `--gradient_accumulation_steps` |                  `4`                  | 梯度累积步数 |
| `--lr` |                `5e-5`                 | 学习率 |
| `--beta` |                `0.04`                 | KL 惩罚系数 |

**显存估算（GPU 3，A5880 48GB）**：base model ~8GB + reference model ~8GB + 生成/梯度 ~8GB ≈ 24-28GB，余量充足。

---

## 7. 后续迭代方向

- **继续 calibration**：对 trend / frequency / amplitude / pattern_consistency 维度运行标定实验，根据输出边界更新 `defect_injector.py` 中的 light/heavy 区间；若某方法全程 heavy 则加入 `always_heavy` 标记或 `DIMENSION_TOOL_OVERRIDE`
- **训练监控**：在 `GRPOConfig` 中设置 `report_to="wandb"` 并分别记录 format / dim / tool 三项奖励分量，便于诊断优化瓶颈
- **验证集评估**：在 `rl/train_grpo.py` 中加入 TRL Callback，每隔 N 步在 `val.jsonl` 上计算 dim F1 和 tool accuracy，追踪泛化效果