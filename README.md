# TSqualityAgent

A multi-agent framework for pairwise time series quality assessment. Given two time series segments, the system determines which one is of higher quality and explains why.

## System Architecture

```
Input (series_A, series_B, dataset_description)
        │
        ▼
  Perceiver  (Perception & Planning)
  └─ Analyzes basic statistics of both series, selects relevant quality dimensions
        │
        ▼
  Inspector  (Detection & Reasoning)
  └─ Runs Thought → Tool Call → Observation loops per dimension
        │
        ▼
  Adjudicator  (Aggregation & Reflection)
  └─ Produces final verdict; triggers re-check or re-plan if uncertain
        │
   ┌────┴────┐
   │         │
needs_recheck  needs_replan
   │         │
Inspector  Perceiver  (capped iterations)
        │
        ▼
  Output: { winner, confidence, explanation }
```

## Quality Dimensions

| Category | Dimension | Primary Tools | Auxiliary Tools |
|----------|-----------|--------------|-----------------|
| Bad quality | `missing_value` | `missing_ratio` | — |
| Bad quality | `noise_level` | `noise_profile`, `volatility` | `range_stats` |
| Rare pattern (Cat-1) | `rare_pattern` | `mad_residual_outlier`, `zscore_outlier`, `outlier_density` | — |
| Rare pattern (Cat-2) | `rare_pattern` | `contextual_rare_pattern` | — |
| Pattern structure | `trend` | `trend_classifier` | `change_point_detector`, `range_stats`, `stationarity_test` |
| Pattern structure | `frequency` | `seasonality_detector` | `autocorr` |
| Pattern structure | `amplitude` | `cycle_amplitude`, `rolling_amplitude` | `change_point_detector` |
| Pattern structure | `pattern_consistency` | `pattern_consistency_indicators` | `stationarity_test`, `change_point_detector` |

> Full tool documentation: [TOOLS.md](TOOLS.md)

## Installation

```bash
conda create -n tsquality python=3.11 -y
conda activate tsquality
pip install -r requirements.txt
```

## LLM Configuration

### Cloud API

```bash
export OPENAI_API_KEY="sk-..."
python main.py --model gpt-4o-mini
# Other supported models: gpt-4o, claude-haiku-20240307, gemini-2.5-flash, etc.
```

### Local vLLM (Qwen3-4B)

Launch the base model server (Inspector requires tool-call support):

```bash
CUDA_VISIBLE_DEVICES=3 vllm serve Qwen/Qwen3-4B \
    --port 8000 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --max-model-len 32768
```

Launch the fine-tuned Perceiver with LoRA adapter (port 8001):

```bash
CUDA_VISIBLE_DEVICES=3 vllm serve Qwen/Qwen3-4B \
    --enable-lora \
    --lora-modules perceiver-grpo-v2=training/checkpoints/perceiver-grpo-v2 \
    --port 8001 \
    --max-model-len 32768
```

## Running Inference

```bash
# Run all built-in test cases (cloud mode)
python main.py

# Run a specific case
python main.py --case rare_point

# Run multiple cases
python main.py --case rare_point rare_contextual

# Local Qwen3-4B with fine-tuned Perceiver
python main.py \
    --model Qwen/Qwen3-4B \
    --base_url http://localhost:8000/v1 \
    --api_key EMPTY \
    --perceiver_model perceiver-grpo-v2 \
    --perceiver_base_url http://localhost:8001/v1

# Adjust Inspector reasoning budget
python main.py --max_steps 8 --max_recheck 3 --max_replan 2
```

Available case names: `missing` / `noise` / `rare_point` / `rare_contextual` / `trend` / `frequency` / `amplitude` / `pattern` / `all`

Each run produces an HTML report with the full reasoning chain.

## Custom Input

```python
from config import Config, build_llm
from workflow import run_pipeline

cfg = Config(
    model="gpt-4o-mini",
    base_url="https://api.openai.com/v1",
    api_key="sk-...",
    max_steps_per_dimension=6,
    max_recheck=2,
    max_replan=1,
)
llm = build_llm(cfg)

result = run_pipeline(
    input_data={
        "dataset_description": "Industrial temperature sensor, 1-min sampling, 120 steps",
        "series_A": [...],       # list[float], NaN supported
        "series_B": [...],
        "timestamps": [...],     # optional
        "external_variables": {},  # optional
    },
    llm=llm,
    config=cfg,
)

print(result["winner"])       # "A" | "B" | "tie"
print(result["confidence"])   # 0.0 – 1.0
print(result["explanation"])  # full reasoning summary
```

## Training Pipeline

The Perceiver agent is fine-tuned with GRPO reinforcement learning. The full pipeline:

```
Step 1  training/synthesis/    Synthesize labeled training data (defect injection)
Step 2  training/rl/           GRPO fine-tuning of Perceiver (LoRA on Qwen3-4B)
Step 3  annotation/            Pairwise annotation of 23 datasets using the trained agent
Step 4  meta_learning_rater/   Train TSRater scoring model (MAML or per-dataset)
Step 5  evaluation/            Data selection experiments to validate scoring quality
```

See [REPRODUCE.md](REPRODUCE.md) for detailed commands for each step.

## Project Structure

```
TSqualityAgent/
├── main.py                     # Entry point
├── workflow.py                 # LangGraph pipeline
├── config.py                   # Global configuration
├── synthetic_cases.py          # Built-in test case generator & visualizer
├── requirements.txt
├── agents/
│   ├── perceiver.py            # Agent 1: perception & dimension selection
│   ├── inspector.py            # Agent 2: ReAct tool-calling loops
│   └── adjudicator.py          # Agent 3: aggregation & reflection
├── models/
│   ├── state.py                # AgentState definition
│   └── llm.py                  # LLM abstraction layer
├── tools/
│   ├── bad_quality.py          # Missing value & noise tools
│   ├── rare_pattern.py         # Outlier detection tools
│   └── pattern_structure.py    # Trend, frequency, amplitude, consistency tools
├── training/
│   ├── synthesis/              # Defect injection & dataset generation
│   └── rl/                     # GRPO training script & reward functions
├── annotation/                 # Pairwise annotation pipeline (23 datasets)
├── meta_learning_rater/        # TSRater: MAML meta-learning & per-dataset rater
└── evaluation/                 # Downstream forecasting & classification experiments
```