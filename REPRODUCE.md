# TSqualityAgent 复现指南

本文档面向需要继续推进实验的协作者，按模块介绍系统的功能与运行方法。文档中的运行指令基本涵盖复现所需的常用用法；
如需了解更多参数和高级选项，可查阅对应脚本顶部的 Usage 注释。除非清楚了解某个参数的具体含义并认为有必要修改，否则请遵循文档中的默认值。

---

## 环境配置

```bash
conda activate tsagent
pip install -r requirements.txt
```

> **注意**：`meta_learning_rater` 模块依赖 `momentfm`，与主环境的 transformers 版本冲突（但是一般来说可以忽略，不要动transformers版本）：

---

## 系统总览

```
主推理流程：  main.py  →  workflow.py  →  agents/
训练流程：
  Step 1  training/synthesis/   合成 Perceiver 训练数据
  Step 2  training/rl/          Perceiver GRPO 强化学习
  Step 3  annotation/           用训练好的 Agent 对 23 个数据集做成对标注
  Step 4  meta_learning_rater/  MAML 元学习训练 TSRater 打分模型
  Step 5  evaluation/           数据选择实验，验证评分效果
```

---

## 模块一：主推理框架（`main.py` + `agents/`）

### 功能
调用三个 Agent（Perceiver → Inspector → Adjudicator）对一条时间序列进行质量评估，输出质量分数和分析报告。三个 Agent 均基于 Qwen3-4B，通过 vLLM 服务提供 OpenAI 兼容接口。

- **Perceiver**（`agents/perceiver.py`）：感知时间序列的关键维度（趋势、频率、幅度、模式）
- **Inspector**（`agents/inspector.py`）：对每个维度调用工具或直接推理进行深度检验
- **Adjudicator**（`agents/adjudicator.py`）：汇总各维度结论，输出最终质量判断

### 运行

```bash
# 使用云端 API
python main.py --model gpt-4o-mini --api_key <YOUR_KEY>

# 使用本地 vLLM（需先启动服务，见下方）
python main.py \
    --model Qwen/Qwen3-4B \
    --base_url http://localhost:8000/v1 \
    --api_key EMPTY \
    --perceiver_model perceiver-grpo-v2 \
    --perceiver_base_url http://localhost:8001/v1

# 启动 vLLM 服务（基础模型，端口 8000；Inspector 需要工具调用支持）
vllm serve Qwen/Qwen3-4B \
    --port 8000 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --max-model-len 32768

# 启动带 LoRA 的 Perceiver 服务（端口 8001）
# perceiver-grpo-v2 是已微调好的 Perceiver LoRA adapter，压缩包已发群里
# 解压后放到 training/checkpoints/perceiver-grpo-v2/
vllm serve Qwen/Qwen3-4B \
    --enable-lora \
    --lora-modules perceiver-grpo-v2=training/checkpoints/perceiver-grpo-v2 \
    --port 8001 \
    --max-model-len 32768
```

---

## 模块二：合成训练数据（`training/synthesis/`）

### 功能
通过规则注入缺陷（趋势异常、频率异常、幅度异常、模式异常）自动合成带标签的时间序列样本，用于 Perceiver 的 GRPO 训练。主文件为 `build_dataset.py`。

### 运行

```bash
# 生成 Perceiver 训练集
python -m training.synthesis.build_dataset \
    --n_samples 4000 \
    --output training/data/perceiver_train_filtered.jsonl \
    --filter_by_hints --stats

# 生成 Perceiver 验证集
python -m training.synthesis.build_dataset \
    --n_samples 500 --seed_offset 1000000 \
    --output training/data/perceiver_val.jsonl --stats
```

> **当前状态**：`perceiver_train_filtered.jsonl` 和 `perceiver_val.jsonl` 已生成，在training/data文件夹下。

---

## 模块三：Perceiver GRPO 强化学习（`training/rl/`）

### 功能
以合成数据的标签为奖励信号，对 Perceiver 进行 GRPO（Group Relative Policy Optimization）强化学习训练。奖励由维度 Precision（权重 0.9）+ 格式合规（0.1）组成。

### 运行

```bash
# 单卡
PYTORCH_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=3 python -m training.rl.train_grpo \
    --data training/data/perceiver_train_filtered.jsonl \
    --val_data training/data/perceiver_val.jsonl \
    --model Qwen/Qwen3-4B \
    --output training/checkpoints/perceiver-grpo-v2 \
    --epochs 1 --batch_size 1 --num_generations 8 \
    --gradient_accumulation_steps 4

# 双卡
PYTORCH_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    -m training.rl.train_grpo \
        --data training/data/perceiver_train_filtered.jsonl \
        --val_data training/data/perceiver_val.jsonl \
        --model Qwen/Qwen3-4B \
        --output training/checkpoints/perceiver-grpo-v2 \
        --epochs 1 --batch_size 1 --num_generations 8 \
        --gradient_accumulation_steps 4
```

> **当前状态**：已有 `training/checkpoints/perceiver-grpo-v2`，可直接部署用于标注。

---

## 模块四：成对标注（`annotation/`）

### 功能
对 23 个时间序列数据集中的 `blocks.jsonl` 进行成对比较标注。每对样本调用完整 Agent 流程（Perceiver + Inspector + Adjudicator），输出胜者与置信度。每个数据集目标收集 **500 个高置信度有效对**（`|2p-1| ≥ 0.5`），小数据集以 `C(N,2)` 为上限。标注结果保存为 `datasets/<name>/annotation.jsonl`。

数据集配置文件：`annotation/dataset_configs.json`（23 个数据集的路径均已配置）。

### 运行

```bash
# 需先启动 vLLM 服务（基础模型 + Perceiver LoRA）

# 单个数据集
python -m annotation.run_annotation \
    --dataset datasets/electricity/blocks.jsonl \
    --output datasets/electricity/annotation.jsonl \
    --dataset_description "Electricity consumption time series" \
    --model Qwen/Qwen3-4B \
    --base_url http://localhost:8000/v1 \
    --perceiver_model perceiver-grpo-v2 \
    --perceiver_base_url http://localhost:8001/v1

# 批量标注所有 23 个数据集（支持断点续标，--resume 默认开启）
python -m annotation.run_annotation \
    --batch_config annotation/dataset_configs.json \
    --model Qwen/Qwen3-4B \
    --base_url http://localhost:8000/v1 \
    --perceiver_model perceiver-grpo-v2 \
    --perceiver_base_url http://localhost:8001/v1
```

> **当前状态**：标注尚未完成，预计耗时约 2 天。可多进程并行跑不同数据集以加速。

---

## 模块五：元学习打分模型（`meta_learning_rater/`）

### 功能
基于 MAML（Model-Agnostic Meta-Learning）训练跨数据集的质量打分模型 TSRater。
输入为 MOMENT-1-base 提取的时间序列嵌入（dim=768），通过 Bradley-Terry 损失学习成对偏好。

**前置条件**：23 个数据集的 `annotation.jsonl` 均已生成。

### 运行

```bash
# 标准训练（论文最优超参）
python -m meta_learning_rater.run_meta_train \
    --config annotation/dataset_configs.json \
    --output meta_learning_rater/checkpoints/tsrater.pth

# 超参搜索（Optuna，50 trials）
python -m meta_learning_rater.run_meta_train \
    --config annotation/dataset_configs.json \
    --output meta_learning_rater/checkpoints/tsrater.pth \
    --tune --n_trials 50

# 对单个数据集打分（scores.jsonl 默认输出到 blocks.jsonl 同目录）
python -m meta_learning_rater.score \
    --blocks   datasets/electricity/blocks.jsonl \
    --model    meta_learning_rater/checkpoints/tsrater.pth \
    --annotation datasets/electricity/annotation.jsonl

# 批量对所有数据集打分（一般不需要，只对需要evaluation的几个数据集打分即可）
python -m meta_learning_rater.score \
    --config annotation/dataset_configs.json \
    --model  meta_learning_rater/checkpoints/tsrater.pth
```

> **当前状态**：标注完成后方可运行。打分结果存储为各数据集目录下的 `scores.jsonl`。

---

## 模块五（分支）：Per-dataset Single Rater（`meta_learning_rater/train_single.py`）

### 功能
不使用元学习，直接对每个数据集独立训练一个 Bradley-Terry 打分模型。
每个数据集对应一个 checkpoint（`rater_<name>.pth`），打分时同样调用 `score.py`，通过 `--model_dir` 加载。

相比 MAML 分支，该方案更简单稳定，在单数据集上准确率更高，推荐作为主要实验方案。

**前置条件**：各数据集的 `annotation.jsonl` 已生成。

### 运行

```bash
# 训练单个数据集的 rater
python -m meta_learning_rater.train_single \
    --blocks     datasets/weather/blocks.jsonl \
    --annotation datasets/weather/annotation.jsonl \
    --output     meta_learning_rater/checkpoints/rater_weather.pth

# 批量训练所有数据集（checkpoint 命名为 rater_<dataset_name>.pth）
python -m meta_learning_rater.train_single \
    --config     annotation/dataset_configs.json \
    --output_dir meta_learning_rater/checkpoints/

# 对单个数据集打分（使用对应 checkpoint）
python -m meta_learning_rater.score \
    --blocks     datasets/weather/blocks.jsonl \
    --model      meta_learning_rater/checkpoints/rater_weather.pth \
    --annotation datasets/weather/annotation.jsonl

# 批量打分（自动匹配 rater_<name>.pth，scores.jsonl 输出到各数据集目录）
python -m meta_learning_rater.score \
    --config     annotation/dataset_configs.json \
    --model_dir  meta_learning_rater/checkpoints/
```

> **当前状态**：标注完成后方可运行。打分结果存储为各数据集目录下的 `scores.jsonl`。

---

## 模块六：数据选择评估（`evaluation/`）

### 功能
通过"用质量分数选 top-50% 训练样本，训练下游预测/分类模型"来验证 TSRater 评分的有效性。支持三类任务：

| 任务 | 数据集 | 指标 |
|------|--------|------|
| 长期预测 | Electricity, ExchangeRate, Traffic, Weather | RMSE（越低越好） |
| 短期预测 | M4-Yearly, M4-Monthly, M4-Daily | MAPE（越低越好） |
| 分类 | MedicalImages, CBF, BME, Handwriting | Accuracy（越高越好） |

**前置条件**：各数据集的 `scores.jsonl` 已生成，各数据集的原始 CSV/ts 文件已放入对应目录（见下方）。

### 数据文件放置要求

```
datasets/
  electricity/electricity.csv       ✅ 已就绪
  exchange_rate/exchange_rate.csv   ✅ 已就绪
  traffic/traffic.csv               ✅ 已就绪
  weather/weather.csv               ✅ 已就绪
  m4_yearly/m4_yearly.csv           ✅ 已就绪
  m4_monthly/m4_monthly.csv         ✅ 已就绪
  m4_daily/m4_daily.csv             ✅ 已就绪
  MedicalImages/                    ✅ .ts 文件已就绪
  CBF/                              ✅ .ts 文件已就绪
  BME/                              ✅ .ts 文件已就绪
  Handwriting/                      ✅ .ts 文件已就绪
```

### 运行

```bash
cd /data/home/shunyu/TSqualityAgent

# 长期预测（全部数据集，默认5次重复实验，3个模型）
CUDA_VISIBLE_DEVICES=3 python -m evaluation.run_eval --task long_term_forecast

# 短期预测
CUDA_VISIBLE_DEVICES=3 python -m evaluation.run_eval --task short_term_forecast

# 分类
CUDA_VISIBLE_DEVICES=3 python -m evaluation.run_eval --task classification

# 单数据集快速验证
CUDA_VISIBLE_DEVICES=3 python -m evaluation.run_eval \
    --task long_term_forecast --dataset electricity \
    --models Linear --train_epochs 1 --itr 1
```

> **当前状态**：流程已调通（用 mock scores 测试通过）。等 `scores.jsonl` 生成后直接运行即可。

---

## 待完成事项清单

| 优先级 | 任务 | 状态 |
|--------|------|------|
| 1 | 对 23 个数据集运行成对标注 | ⬜ 待完成（Perceiver-grpo-v2 ✅） |
| 2 | 运行 meta_learning_rater 训练 TSRater | ⬜ 依赖标注完成 |
| 3 | 对评估数据集打分（生成 scores.jsonl） | ⬜ 依赖 TSRater 训练完成 |
| 4 | 运行 evaluation 三组实验 | ⬜ 依赖 scores.jsonl（数据文件 ✅） |

---

## 硬件建议

- 标注（模块四）：GPU 显存 ≥ 24GB，建议 vLLM 服务运行在 GPU 3（A5880 48GB）
- GRPO 训练：单卡 48GB 或双卡 24GB×2
- meta_learning_rater：CPU 或单卡均可（MOMENT 推理 + MLP 训练，显存需求低）
- evaluation：单卡即可（Linear/CNN/PatchTST 小模型）