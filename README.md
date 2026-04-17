# TSqualityAgent

基于多 Agent 的时序数据质量评估系统，通过成对比较判断两段时序样本的质量优劣。

## 系统架构

```
输入 (series_A, series_B, dataset_description, ...)
        │
        ▼
  Perceiver（感知 + 规划）
  └─ 分析两段序列的基础特征，选择需要评估的质量维度
        │
        ▼
  Inspector（检测 + 推理）
  └─ 对每个维度执行 Thought → Tool Call → Observation 循环
        │
        ▼
  Adjudicator（汇总 + 反思）
  └─ 综合判断 A/B 优劣，必要时触发反思循环
        │
   ┌────┴────┐
   │         │
needs_recheck  needs_replan
   │         │
Inspector  Perceiver  （有次数上限）
        │
        ▼
  最终输出：{ winner, confidence, explanation }
```

## 质量评估维度

| 类别 | 维度 | 主要工具 | 辅助工具 |
|------|------|---------|---------|
| Bad quality | missing_value | `missing_ratio` | — |
| Bad quality | noise_level | `noise_profile`, `volatility` | `range_stats` |
| Rare pattern (Cat-1 评分) | rare_pattern | `mad_residual_outlier`, `zscore_outlier`, `outlier_density` | — |
| Rare pattern (Cat-2 标记) | rare_pattern | `contextual_rare_pattern` | — |
| Pattern structure | trend | `trend_classifier` | `change_point_detector`, `range_stats`, `stationarity_test` |
| Pattern structure | frequency | `seasonality_detector` | `autocorr` |
| Pattern structure | amplitude | `cycle_amplitude`, `rolling_amplitude` | `change_point_detector` |
| Pattern structure | pattern_consistency | `pattern_consistency_indicators` | `stationarity_test`, `change_point_detector` |

> 完整工具文档见 [TOOLS.md](TOOLS.md)（英文）/ [TOOLS_ZH.md](TOOLS_ZH.md)（中文）

## 环境安装

```bash
conda create -n tsquality python=3.11 -y
conda activate tsquality
pip install -r requirements.txt
```

## 配置 LLM

支持两种后端，通过 `--base_url` 切换，接口完全兼容。

### 云端模式（默认）

通过 [chatanywhere](https://api.chatanywhere.tech) 调用闭源模型：

```bash
export OPENAI_API_KEY="sk-..."
python main.py --model gpt-4o-mini
# 可选：gpt-4o / claude-haiku-20240307 / gemini-2.5-flash 等
```

### 本地模式（vLLM + Qwen3-4B）

启动 vLLM 服务（需 `vllm>=0.8.5`，首次运行自动从 HuggingFace 下载模型）：

```bash
CUDA_VISIBLE_DEVICES=3 vllm serve Qwen/Qwen3-4B \
    --port 8001 \
    --tensor-parallel-size 1 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --max-model-len 32768
    
CUDA_VISIBLE_DEVICES=2 vllm serve Qwen/Qwen3-4B \
    --enable-lora \
    --lora-modules perceiver-grpo-v1=training/checkpoints/perceiver-grpo-v1 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --port 8000 \
    --max-model-len 32768

```

然后运行 agent：

```bash
python main.py \
    --model Qwen/Qwen3-4B \
    --base_url http://localhost:8000/v1 \
    --api_key EMPTY
```

## 运行

```bash
# 运行全部内置测试用例（云端模式）
python main.py

# 运行单个 case
python main.py --case rare_point

# 运行多个 case
python main.py --case rare_point rare_contextual

# 指定云端模型
python main.py --model gpt-4o --case trend frequency

# 使用本地 Qwen3-4B
python main.py --model Qwen/Qwen3-4B --base_url http://localhost:8001/v1 --api_key EMPTY
python main.py --model perceiver-grpo-v1 --base_url http://localhost:8000/v1 --api_key EMPTY

python main.py \
      --model Qwen/Qwen3-4B \
      --base_url http://localhost:8001/v1 \
      --perceiver_model perceiver-grpo-v2 \
      --perceiver_base_url http://localhost:8000/v1 \
      --api_key EMPTY

# 调整 Inspector 最大推理步数和反思次数
python main.py --max_steps 8 --max_recheck 3 --max_replan 2
```

可选 case 名称：`missing` / `noise` / `rare_point` / `rare_contextual` / `trend` / `frequency` / `amplitude` / `pattern` / `all`

## 合成测试用例可视化

```bash
# 绘制全部 8 个 case，保存到 plots/synthetic_cases/
python synthetic_cases.py

# 只绘制指定 case（可选：missing / noise / rare_point / rare_contextual / trend / frequency / amplitude / pattern）
python synthetic_cases.py --case trend

# 保存同时弹窗显示
python synthetic_cases.py --show

# 指定输出目录
python synthetic_cases.py --out my_plots
```

## 自定义输入

```python
from config import Config, build_llm
from workflow import run_pipeline

# 云端
cfg = Config(model="gpt-4o-mini", base_url="https://api.chatanywhere.tech/v1",
             api_key="", max_steps_per_dimension=6, max_recheck=2, max_replan=1)

# 本地 vLLM
# cfg = Config(model="Qwen/Qwen3-4B", base_url="http://localhost:8000/v1",
#              api_key="EMPTY", max_steps_per_dimension=6, max_recheck=2, max_replan=1)

llm = build_llm(cfg)

result = run_pipeline(
    input_data={
        "dataset_description": "工业温度传感器，1分钟采样，共120个时间步",
        "series_A": [...],          # list[float]，支持 NaN
        "series_B": [...],
        "timestamps": [...],        # 可选
        "external_variables": {},   # 可选
    },
    llm=llm,
    config=cfg,
)

print(result["winner"])       # "A" | "B" | "tie"
print(result["confidence"])   # 0.0 ~ 1.0
print(result["explanation"])  # 详细推理说明
```

## 项目结构

```
TSqualityAgent/
├── main.py              # 入口（argparse 参数控制）
├── synthetic_cases.py   # 合成测试用例（bad/rare/pattern 三类）
├── workflow.py          # LangGraph 主工作流
├── config.py            # 全局配置
├── requirements.txt     # 依赖列表
├── models/
│   ├── state.py         # AgentState 数据结构
│   └── llm.py           # 抽象 LLM 接口
├── agents/
│   ├── perceiver.py     # Agent 1：感知 + 规划
│   ├── inspector.py     # Agent 2：ReAct 工具调用
│   └── adjudicator.py   # Agent 3：汇总 + 反思
└── tools/
    ├── bad_quality.py        # 缺失值、噪声检测
    ├── rare_pattern.py       # 异常检测
    └── pattern_structure.py  # 趋势、频率、幅值、一致性检测
```