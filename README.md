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

| 类别 | 维度 | 工具 |
|------|------|------|
| Bad quality | missing_value | `missing_ratio` |
| Bad quality | noise_level | `noise_profile`, `signal_to_noise_ratio` |
| Rare pattern | anomaly | `anomaly_detection`, `outlier_density` |
| Pattern structure | trend | `trend_classifier` |
| Pattern structure | frequency | `seasonality_detector` |
| Pattern structure | amplitude | `spike_detector` |
| Pattern structure | pattern_consistency | `change_point_detector`, `pattern_consistency_indicators` |

## 环境安装

```bash
conda create -n tsquality python=3.11 -y
conda activate tsquality
pip install -r requirements.txt
```

## 配置 LLM

通过 [chatanywhere](https://api.chatanywhere.tech) 统一调用多种模型，在 `config.py` 中指定模型名：

```python
# 可选模型：gpt-4o-mini / gpt-4o / claude-haiku-20240307 / gemini-2.5-flash 等
cfg = Config(model="gpt-4o-mini")
```

设置 API Key 环境变量：

```bash
export OPENAI_API_KEY="sk-..."
```

## 运行

```bash
# 运行内置测试用例
python main.py
```

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

cfg = Config(model="gpt-4o-mini", max_steps_per_dimension=6, max_recheck=2, max_replan=1)
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