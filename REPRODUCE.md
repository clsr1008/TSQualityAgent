# TSqualityAgent Reproduction Guide

This document introduces the system modules and their common commands for collaborators who continue the experiments. For additional parameters and advanced options, see the `Usage` section at the top of the relevant script. Use the documented defaults unless a change is necessary and understood.

## Environment setup

```bash
conda activate tsagent
pip install -r requirements.txt
```

> `meta_learning_rater` depends on `momentfm`, which can report a `transformers` version conflict with the main environment. This warning can normally be ignored; do not change the `transformers` version solely because of it.

## System overview

```
Inference pipeline: main.py -> workflow.py -> agents/
Training pipeline:
  Step 1  training/synthesis/   Generate synthetic Perceiver training data
  Step 2  training/rl/          Train the Perceiver with GRPO
  Step 3  annotation/           Produce pairwise annotations for 23 datasets
  Step 4  meta_learning_rater/  Train the TSRater score model with MAML
  Step 5  evaluation/           Run data-selection experiments
```

## 1. Main inference framework (`main.py` and `agents/`)

The framework uses a Perceiver, Inspector, and Adjudicator to assess a time series and return a quality score and analysis report. The agents use Qwen3-4B through OpenAI-compatible vLLM services.

- `agents/perceiver.py`: identifies the important quality dimensions, such as trend, frequency, amplitude, and pattern consistency.
- `agents/inspector.py`: inspects each selected dimension with tools or direct reasoning.
- `agents/adjudicator.py`: combines the dimension-level conclusions into a final quality judgment.

```bash
# Cloud API
python main.py --model gpt-4o-mini --api_key <YOUR_KEY>

# Local vLLM (start the services below first)
python main.py \
    --model Qwen/Qwen3-4B \
    --base_url http://localhost:8000/v1 \
    --api_key EMPTY \
    --perceiver_model perceiver-grpo-v2 \
    --perceiver_base_url http://localhost:8001/v1

# Base model service on port 8000; the Inspector requires tool-calling support.
vllm serve Qwen/Qwen3-4B \
    --port 8000 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --max-model-len 32768

# Perceiver service with the LoRA adapter on port 8001.
# Extract the provided perceiver-grpo-v2 adapter to training/checkpoints/perceiver-grpo-v2/.
vllm serve Qwen/Qwen3-4B \
    --enable-lora \
    --lora-modules perceiver-grpo-v2=training/checkpoints/perceiver-grpo-v2 \
    --port 8001 \
    --max-model-len 32768
```

## 2. Synthetic training data (`training/synthesis/`)

`build_dataset.py` creates labeled synthetic samples by injecting rule-based defects, including trend, frequency, amplitude, and pattern defects. The data is used for Perceiver GRPO training.

```bash
# Perceiver training set
python -m training.synthesis.build_dataset \
    --n_samples 4000 \
    --output training/data/perceiver_train_filtered.jsonl \
    --filter_by_hints --stats

# Perceiver validation set
python -m training.synthesis.build_dataset \
    --n_samples 500 --seed_offset 1000000 \
    --output training/data/perceiver_val.jsonl --stats
```

## 3. Perceiver GRPO training (`training/rl/`)

GRPO (Group Relative Policy Optimization) trains the Perceiver on synthetic labels. The reward combines dimension precision (weight 0.9) and format compliance (weight 0.1).

```bash
# One GPU
PYTORCH_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=3 python -m training.rl.train_grpo \
    --data training/data/perceiver_train_filtered.jsonl \
    --val_data training/data/perceiver_val.jsonl \
    --model Qwen/Qwen3-4B \
    --output training/checkpoints/perceiver-grpo-v2 \
    --epochs 1 --batch_size 1 --num_generations 8 \
    --gradient_accumulation_steps 4

# Two GPUs
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

## 4. Pairwise annotation (`annotation/`)

The annotation workflow compares samples in each dataset's `blocks.jsonl` with the complete agent pipeline. It produces a winner and confidence score for every pair. The target is 500 valid, high-confidence pairs per dataset (`|2p - 1| >= 0.5`), capped by `C(N, 2)` for small datasets. Results are saved as `datasets/<name>/annotation.jsonl`.

Dataset paths and descriptions are defined in `annotation/dataset_configs.json`.

```bash
# Start the base-model and Perceiver vLLM services first.

# One dataset
python -m annotation.run_annotation \
    --dataset datasets/electricity/blocks.jsonl \
    --output datasets/electricity/annotation.jsonl \
    --dataset_description "Electricity consumption time series" \
    --model Qwen/Qwen3-4B \
    --base_url http://localhost:8000/v1 \
    --perceiver_model perceiver-grpo-v2 \
    --perceiver_base_url http://localhost:8001/v1

# All configured datasets; --resume is enabled by default.
python -m annotation.run_annotation \
    --batch_config annotation/dataset_configs.json \
    --model Qwen/Qwen3-4B \
    --base_url http://localhost:8000/v1 \
    --perceiver_model perceiver-grpo-v2 \
    --perceiver_base_url http://localhost:8001/v1
```

## 5. Meta-learning rater (`meta_learning_rater/`)

TSRater is a cross-dataset quality rater trained with MAML (Model-Agnostic Meta-Learning). It takes MOMENT-1-base embeddings (dimension 768) and learns pairwise preferences with the Bradley-Terry loss.

Prerequisite: the datasets used for training have completed `annotation.jsonl` files.

```bash
# Standard training
python -m meta_learning_rater.run_meta_train \
    --config annotation/dataset_configs.json \
    --output meta_learning_rater/checkpoints/tsrater.pth

# Hyperparameter search (Optuna, 50 trials)
python -m meta_learning_rater.run_meta_train \
    --config annotation/dataset_configs.json \
    --output meta_learning_rater/checkpoints/tsrater.pth \
    --tune --n_trials 50

# Score one dataset; scores.jsonl is written next to blocks.jsonl by default.
python -m meta_learning_rater.score \
    --blocks datasets/electricity/blocks.jsonl \
    --model meta_learning_rater/checkpoints/tsrater.pth \
    --annotation datasets/electricity/annotation.jsonl

# Score all configured datasets.
python -m meta_learning_rater.score \
    --config annotation/dataset_configs.json \
    --model meta_learning_rater/checkpoints/tsrater.pth
```

### Per-dataset single rater (`meta_learning_rater/train_single.py`)

This alternative does not use meta-learning. It trains an independent Bradley-Terry rater for every dataset and writes `rater_<name>.pth`. It is simpler and can achieve better within-dataset accuracy.

```bash
# Train one rater
python -m meta_learning_rater.train_single \
    --blocks datasets/weather/blocks.jsonl \
    --annotation datasets/weather/annotation.jsonl \
    --output meta_learning_rater/checkpoints/rater_weather.pth

# Train all raters
python -m meta_learning_rater.train_single \
    --config annotation/dataset_configs.json \
    --output_dir meta_learning_rater/checkpoints/

# Score one dataset with its checkpoint
python -m meta_learning_rater.score \
    --blocks datasets/weather/blocks.jsonl \
    --model meta_learning_rater/checkpoints/rater_weather.pth \
    --annotation datasets/weather/annotation.jsonl

# Score all datasets; checkpoints are matched as rater_<dataset_name>.pth.
python -m meta_learning_rater.score \
    --config annotation/dataset_configs.json \
    --model_dir meta_learning_rater/checkpoints/
```

## 6. Data-selection evaluation (`evaluation/`)

This module evaluates quality scores by selecting the top 50% of training samples, then training downstream forecasting or classification models.

| Task | Datasets | Metric |
| --- | --- | --- |
| Long-term forecasting | Electricity, ExchangeRate, Traffic, Weather | RMSE (lower is better) |
| Short-term forecasting | M4-Yearly, M4-Monthly, M4-Daily | MAPE (lower is better) |
| Classification | MedicalImages, CBF, BME, Handwriting | Accuracy (higher is better) |

Prerequisites: `scores.jsonl` is available for each evaluated dataset, and its source CSV or `.ts` data is in the corresponding `datasets/` directory.

```bash
# Long-term forecasting: all datasets, five repetitions, and three models by default.
CUDA_VISIBLE_DEVICES=3 python -m evaluation.run_eval --task long_term_forecast

# Short-term forecasting
CUDA_VISIBLE_DEVICES=3 python -m evaluation.run_eval --task short_term_forecast

# Classification
CUDA_VISIBLE_DEVICES=3 python -m evaluation.run_eval --task classification

# Quick one-dataset check
CUDA_VISIBLE_DEVICES=3 python -m evaluation.run_eval \
    --task long_term_forecast --dataset electricity \
    --models Linear --train_epochs 1 --itr 1
```

## Hardware guidance

- Pairwise annotation: at least 24 GB of GPU memory; a 48 GB GPU is recommended for the vLLM services.
- GRPO training: one 48 GB GPU or two 24 GB GPUs.
- `meta_learning_rater`: CPU or one GPU is sufficient; MOMENT inference and MLP training have low memory requirements.
- `evaluation`: one GPU is sufficient for the small Linear, CNN, and PatchTST models.
