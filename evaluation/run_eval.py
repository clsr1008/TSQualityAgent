"""
Data selection evaluation for TSqualityAgent.

Evaluates quality scoring methods by selecting the top-50% training samples
based on each method's scores, then measuring downstream model performance.

Tasks
-----
  Long-term forecasting  : Electricity, ExchangeRate, Traffic, Weather
                           Metric: RMSE (lower is better)
  Short-term forecasting : M4-Yearly, M4-Monthly, M4-Daily
                           Metric: MAPE (lower is better)
  Classification         : MedicalImages, CBF, BME, Handwriting
                           Metric: Accuracy (higher is better)

Score keys
----------
  quality_score  : TSRater (our method) — from datasets/<name>/scores.jsonl
  random         : random baseline
  DataOob        : DataOob baseline       — placeholder, results cited from prior paper
  DataShapley    : DataShapley baseline   — placeholder, results cited from prior paper
  KNNShapley     : KNNShapley baseline    — placeholder, results cited from prior paper
  TimeInf        : TimeInf baseline       — placeholder, results cited from prior paper

Note: baseline methods (DataOob / DataShapley / KNNShapley / TimeInf) are skipped at
runtime when no baseline_file is provided in the dataset config.  Their numbers are
taken directly from the previous paper and do not need to be re-run.

Usage
-----
# Long-term forecasting
python -m evaluation.run_eval --task long_term_forecast

# Short-term forecasting
python -m evaluation.run_eval --task short_term_forecast

# Classification
python -m evaluation.run_eval --task classification

# Single dataset
python -m evaluation.run_eval --task long_term_forecast --dataset electricity
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

# evaluation/ 内部的 import 都是相对路径 (e.g. `from dataset import ...`)
# 把 evaluation/ 加入 sys.path 以保持兼容
sys.path.insert(0, str(Path(__file__).parent))

from configure_params import configure_model_params, print_experiment_results
from exp import Exp_Forecast, Exp_Classification


# ── Dataset configurations ────────────────────────────────────────────────────
# Each entry:
#   file_path   : path to the raw CSV file
#   data        : dataset type tag ('custom' or 'm4')
#   target      : target column name in CSV
#   freq        : time frequency for time encoding
#   start_idx   : start row in CSV (select 4000 consecutive points)
#   end_idx     : end row in CSV  (start_idx + 4000)
#   seq_len     : input window length
#   label_len   : decoder start token length
#   pred_len    : forecast horizon
#   block_len   : block length used when building blocks.jsonl (for score alignment)

LONG_TERM_CONFIGS = {
    "electricity": dict(
        file_path="datasets/electricity/electricity.csv",
        data="custom",
        target="OT",
        freq="h",
        start_idx=4000,
        end_idx=8000,
        seq_len=96,
        label_len=32,
        pred_len=32,
        score_file="datasets/electricity/scores.jsonl",
        baseline_file=None,  # baselines cited from prior paper, not re-run
    ),
    "exchange_rate": dict(
        file_path="datasets/exchange_rate/exchange_rate.csv",
        data="custom",
        target="OT",
        freq="d",
        start_idx=2000,
        end_idx=6000,
        seq_len=96,
        label_len=32,
        pred_len=32,
        score_file="datasets/exchange_rate/scores.jsonl",
        baseline_file=None,  # baselines cited from prior paper, not re-run
    ),
    "traffic": dict(
        file_path="datasets/traffic/traffic.csv",
        data="custom",
        target="OT",
        freq="h",
        start_idx=4000,
        end_idx=8000,
        seq_len=96,
        label_len=32,
        pred_len=32,
        score_file="datasets/traffic/scores.jsonl",
        baseline_file=None,  # baselines cited from prior paper, not re-run
    ),
    "weather": dict(
        file_path="datasets/weather/weather.csv",
        data="custom",
        target="OT",
        freq="h",
        start_idx=4000,
        end_idx=8000,
        seq_len=96,
        label_len=32,
        pred_len=32,
        score_file="datasets/weather/scores.jsonl",
        baseline_file=None,  # baselines cited from prior paper, not re-run
    ),
}

SHORT_TERM_CONFIGS = {
    "m4_yearly": dict(
        file_path="datasets/m4_yearly/m4_yearly.csv",
        data="m4",
        target="V2",
        freq="a",
        start_idx=0,
        end_idx=4000,
        seq_len=12,    # block_len=18 = seq_len(12) + pred_len(6)
        label_len=6,
        pred_len=6,
        score_file="datasets/m4_yearly/scores.jsonl",
        baseline_file=None,  # baselines cited from prior paper, not re-run
    ),
    "m4_monthly": dict(
        file_path="datasets/m4_monthly/m4_monthly.csv",
        data="m4",
        target="V2",
        freq="m",
        start_idx=0,
        end_idx=4000,
        seq_len=36,
        label_len=18,
        pred_len=18,
        score_file="datasets/m4_monthly/scores.jsonl",
        baseline_file=None,  # baselines cited from prior paper, not re-run
    ),
    "m4_daily": dict(
        file_path="datasets/m4_daily/m4_daily.csv",
        data="m4",
        target="V2",
        freq="d",
        start_idx=0,
        end_idx=4000,
        seq_len=28,
        label_len=14,
        pred_len=14,
        score_file="datasets/m4_daily/scores.jsonl",
        baseline_file=None,  # baselines cited from prior paper, not re-run
    ),
}

CLASSIFICATION_CONFIGS = {
    "MedicalImages": dict(
        file_path="datasets/MedicalImages",
        data="UEA",
        score_file="datasets/MedicalImages/scores.jsonl",
        baseline_file=None,  # baselines cited from prior paper, not re-run
        seq_len=99,
    ),
    "CBF": dict(
        file_path="datasets/CBF",
        data="UEA",
        score_file="datasets/CBF/scores.jsonl",
        baseline_file=None,  # baselines cited from prior paper, not re-run
        seq_len=128,
    ),
    "BME": dict(
        file_path="datasets/BME",
        data="UEA",
        score_file="datasets/BME/scores.jsonl",
        baseline_file=None,  # baselines cited from prior paper, not re-run
        seq_len=128,
    ),
    "Handwriting": dict(
        file_path="datasets/Handwriting",
        data="UEA",
        score_file="datasets/Handwriting/scores.jsonl",
        baseline_file=None,  # baselines cited from prior paper, not re-run
        seq_len=152,
    ),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def set_random_seed(seed: int = 2021) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


_BASELINE_KEYS = {"DataOob", "DataShapley", "KNNShapley", "TimeInf"}


def _score_file_for_key(cfg: dict, score_key: str) -> str | None:
    """Return the correct score file path for the given score_key.

    Returns None for random (random baseline needs no file) and for
    baseline methods when no baseline_file is configured (results are
    cited directly from the prior paper).
    """
    if score_key == "quality_score":
        return cfg["score_file"]
    elif score_key in _BASELINE_KEYS:
        return cfg.get("baseline_file")  # None if not provided
    else:  # random
        return None


def run_task(
    task_name: str,
    dataset_configs: dict,
    score_keys: list[str],
    models: list[str],
    args,
    itr: int = 1,
) -> dict:
    """Run all (score_key, model, dataset) combinations for one task."""
    if task_name in ("long_term_forecast", "short_term_forecast"):
        Exp = Exp_Forecast
    else:
        Exp = Exp_Classification

    results = {
        key: {model: {ds: [] for ds in dataset_configs} for model in models}
        for key in score_keys
    }

    for ds_name, cfg in dataset_configs.items():
        print(f"\n{'='*60}")
        print(f"  Dataset: {ds_name}")
        print(f"{'='*60}")

        # Apply dataset-specific args
        args.task_name   = task_name
        args.data        = cfg.get("data", "custom")
        args.file_path   = cfg["file_path"]
        args.target      = cfg.get("target", "OT")
        args.freq        = cfg.get("freq", "h")
        args.start_idx   = cfg.get("start_idx", 0)
        args.end_idx     = cfg.get("end_idx", 4000)
        args.seq_len     = cfg.get("seq_len", 96)
        args.label_len   = cfg.get("label_len", 48)
        args.pred_len    = cfg.get("pred_len", 96)
        args.features    = "S"
        args.model_id    = f"{ds_name}_{args.seq_len}_{args.pred_len}"
        # M4 has no real timestamps — use fixed positional embedding to avoid freq mismatch
        if args.data == "m4":
            args.embed   = "fixed"
            args.timeenc = 0

        for ii in range(itr):
            for score_key in score_keys:
                score_file = _score_file_for_key(cfg, score_key)

                # Skip baseline methods that have no file provided (cited from paper)
                if score_key in _BASELINE_KEYS and score_file is None:
                    print(f"\n[SKIP] {score_key} — no baseline_file configured, "
                          "results cited from prior paper")
                    continue

                args.score_key  = score_key
                args.score_file = score_file

                for model in models:
                    args.model = model
                    configure_model_params(args)

                    setting = "{}_{}_{}_ft{}_{}_{}_{}_{}".format(
                        task_name, args.model_id, model,
                        args.features, score_key,
                        args.proportion, args.des, ii,
                    )
                    print(f"\n>>> {setting}")

                    exp = Exp(args)
                    exp.train(setting)
                    result = exp.test(setting)
                    results[score_key][model][ds_name].append(result)

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    return results


def print_results_table(task_name: str, dataset_configs: dict,
                        score_keys: list[str], models: list[str],
                        results: dict) -> None:
    ds_names = list(dataset_configs.keys())
    metric = "RMSE" if "forecast" in task_name else "Accuracy"
    print(f"\n{'='*60}")
    print(f"  Task: {task_name}  |  Metric: {metric}")
    print(f"{'='*60}")
    header = f"  {'Method':<15}" + "".join(f"{m:<12}" for m in models)

    for ds in ds_names:
        print(f"\n  [{ds}]")
        print(header)
        for key in score_keys:
            # Per-iteration rows
            n_itr = max((len(results[key][model][ds]) for model in models), default=0)
            for ii in range(n_itr):
                tag = f"{key}[{ii}]" if n_itr > 1 else key
                row = f"  {tag:<15}"
                for model in models:
                    vals = results[key][model][ds]
                    val = f"{vals[ii]:.4f}" if ii < len(vals) else "N/A"
                    row += f"{val:<12}"
                print(row)
            # Mean row (only when multiple iterations)
            if n_itr > 1:
                row = f"  {key+'(mean)':<15}"
                for model in models:
                    vals = results[key][model][ds]
                    val = f"{np.mean(vals):.4f}" if vals else "N/A"
                    row += f"{val:<12}"
                print(row)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="TSRater data selection evaluation")

    parser.add_argument("--task", type=str, default="long_term_forecast",
                        choices=["long_term_forecast", "short_term_forecast", "classification"],
                        help="Evaluation task")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Run only this dataset (default: run all for the task)")
    parser.add_argument("--models", type=str, nargs="+",
                        default=["Linear", "CNN", "PatchTST"],
                        help="Models to evaluate")
    parser.add_argument("--score_keys", type=str, nargs="+",
                        default=["quality_score"],
                        help="Score methods to compare. Available: quality_score, random, "
                             "DataOob, DataShapley, KNNShapley, TimeInf")
    parser.add_argument("--proportion", type=float, default=0.5,
                        help="Fraction of training data to select (default: 0.5)")
    parser.add_argument("--itr", type=int, default=5, help="Repeat experiments N times")
    parser.add_argument("--seed", type=int, default=2021)

    # Training hyperparams
    parser.add_argument("--train_epochs",  type=int,   default=10)
    parser.add_argument("--batch_size",    type=int,   default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--patience",      type=int,   default=3)
    parser.add_argument("--num_workers",   type=int,   default=0)

    # Model architecture defaults (overridden by configure_model_params)
    parser.add_argument("--enc_in",    type=int,   default=1)
    parser.add_argument("--dec_in",    type=int,   default=1)
    parser.add_argument("--c_out",     type=int,   default=1)
    parser.add_argument("--d_model",   type=int,   default=256)
    parser.add_argument("--d_ff",      type=int,   default=512)
    parser.add_argument("--n_heads",   type=int,   default=8)
    parser.add_argument("--e_layers",  type=int,   default=2)
    parser.add_argument("--d_layers",  type=int,   default=1)
    parser.add_argument("--factor",    type=int,   default=3)
    parser.add_argument("--dropout",   type=float, default=0.1)
    parser.add_argument("--embed",     type=str,   default="timeF")
    parser.add_argument("--activation",type=str,   default="gelu")
    parser.add_argument("--top_k",     type=int,   default=5)
    parser.add_argument("--num_kernels",type=int,  default=6)
    parser.add_argument("--moving_avg", type=int,  default=25)
    parser.add_argument("--patch_len",  type=int,  default=16)
    parser.add_argument("--use_norm",   type=int,  default=1)
    parser.add_argument("--channel_independence", type=int, default=1)
    parser.add_argument("--decomp_method", type=str, default="moving_avg")
    parser.add_argument("--down_sampling_layers",  type=int, default=0)
    parser.add_argument("--down_sampling_window",  type=int, default=1)
    parser.add_argument("--down_sampling_method",  type=str, default=None)
    parser.add_argument("--seg_len",    type=int,  default=96)
    parser.add_argument("--expand",     type=int,  default=2)
    parser.add_argument("--d_conv",     type=int,  default=4)
    parser.add_argument("--p_hidden_dims", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--p_hidden_layers", type=int, default=2)
    parser.add_argument("--inverse",    action="store_true", default=False)
    parser.add_argument("--scale",      type=bool, default=True)
    parser.add_argument("--timeenc",    type=int,  default=1)
    parser.add_argument("--temperature",type=float,default=0.0)
    parser.add_argument("--checkpoints",type=str,  default="./checkpoints/")
    parser.add_argument("--des",        type=str,  default="Exp")
    parser.add_argument("--loss",       type=str,  default="MSE")
    parser.add_argument("--lradj",      type=str,  default="type1")
    parser.add_argument("--use_amp",    action="store_true", default=False)
    parser.add_argument("--use_gpu",    type=bool, default=True)
    parser.add_argument("--gpu",        type=int,  default=0)
    parser.add_argument("--gpu_type",   type=str,  default="cuda")
    parser.add_argument("--use_multi_gpu", action="store_true", default=False)
    parser.add_argument("--devices",    type=str,  default="0,1,2,3")
    parser.add_argument("--anomaly_ratio", type=float, default=0.25)
    parser.add_argument("--distil",     action="store_false", default=True)

    args = parser.parse_args()
    args.proportion = args.proportion

    set_random_seed(args.seed)

    # Select dataset config for this task
    if args.task == "long_term_forecast":
        all_configs = LONG_TERM_CONFIGS
    elif args.task == "short_term_forecast":
        all_configs = SHORT_TERM_CONFIGS
    else:
        all_configs = CLASSIFICATION_CONFIGS

    # Filter to single dataset if specified
    if args.dataset:
        if args.dataset not in all_configs:
            raise ValueError(f"Unknown dataset '{args.dataset}' for task '{args.task}'")
        dataset_configs = {args.dataset: all_configs[args.dataset]}
    else:
        dataset_configs = all_configs

    # Run
    results = run_task(
        task_name=args.task,
        dataset_configs=dataset_configs,
        score_keys=args.score_keys,
        models=args.models,
        args=args,
        itr=args.itr,
    )

    print_results_table(args.task, dataset_configs, args.score_keys, args.models, results)

    # Cleanup checkpoints
    if os.path.exists("./checkpoints/"):
        shutil.rmtree("./checkpoints/")


if __name__ == "__main__":
    main()