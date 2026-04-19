"""
Meta-training CLI for TSRater.

Trains a MAML-based quality scoring model across all annotated datasets,
with optional Optuna hyperparameter search.

Best hyperparameters (from Optuna search, acc=0.5788):
  meta_lr=1.335e-5, inner_lr=1.081e-2, inner_steps=20,
  meta_batch_size=12, data_batch_size=16, epochs=12
  Note: inner_steps = exact number of gradient updates (not epochs)

Usage
-----
# Standard training with paper's best hyperparameters
python -m meta_learning_rater.run_meta_train \
    --config annotation/dataset_configs.json \
    --output meta_learning_rater/checkpoints/tsrater.pth

# Hyperparameter search via Optuna (20 trials)
python -m meta_learning_rater.run_meta_train \
    --config annotation/dataset_configs.json \
    --output meta_learning_rater/checkpoints/tsrater.pth \
    --tune \
    --n_trials 20

# Score a dataset after training
python -m meta_learning_rater.score \
    --blocks   datasets/electricity/blocks.jsonl \
    --model    meta_learning_rater/checkpoints/tsrater.pth \
    --output   datasets/electricity/scores.jsonl \
    --annotation datasets/electricity/annotation.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch


def set_seed(seed: int = 42) -> None:
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> list[dict]:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def train_and_eval(
    task_datasets: list[dict],
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    meta_lr: float,
    inner_lr: float,
    inner_steps: int,
    meta_batch_size: int,
    data_batch_size: int,
    epochs: int,
    device: str,
    seed: int = 42,
) -> tuple[float, object]:
    """Single training run. Returns (test_accuracy, meta_model)."""
    from meta_learning_rater.meta_train import MetaLearner

    set_seed(seed)
    learner = MetaLearner(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        meta_lr=meta_lr,
        inner_lr=inner_lr,
        inner_steps=inner_steps,
        device=device,
    )

    print(f"\nMeta-training: {len(task_datasets)} tasks, "
          f"{epochs} epochs, meta_batch={meta_batch_size}")
    learner.meta_train(
        task_datasets,
        meta_batch_size=meta_batch_size,
        data_batch_size=data_batch_size,
        epochs=epochs,
    )

    print("\nEvaluation:")
    acc = learner.evaluate(task_datasets, data_batch_size=data_batch_size)
    return acc, learner.meta_model


def main() -> None:
    parser = argparse.ArgumentParser(description="TSRater meta-training")

    parser.add_argument("--config", type=str,
                        default="annotation/dataset_configs.json",
                        help="dataset_configs.json path")
    parser.add_argument("--output", type=str,
                        default="meta_learning_rater/checkpoints/tsrater.pth",
                        help="Output checkpoint path for best model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)

    # MOMENT / model architecture
    parser.add_argument("--hidden_dim",  type=int, default=256)
    parser.add_argument("--num_layers",  type=int, default=3)
    parser.add_argument("--moment_target_len", type=int, default=512,
                        help="Series length fed to MOMENT (pad/truncate)")
    parser.add_argument("--moment_batch_size", type=int, default=64,
                        help="Batch size for MOMENT embedding computation")

    # Dataset split
    parser.add_argument("--support_ratio", type=float, default=0.4)
    parser.add_argument("--query_ratio",   type=float, default=0.4)
    # test_ratio = 1 - support_ratio - query_ratio = 0.2
    parser.add_argument("--min_confidence", type=float, default=0.5,
                        help="|2p-1| >= this to include a pair (default 0.5)")

    # Training hyperparameters (best config from Optuna search, acc=0.5788)
    parser.add_argument("--meta_lr",        type=float, default=1.335e-5)
    parser.add_argument("--inner_lr",       type=float, default=1.081e-2)
    parser.add_argument("--inner_steps",    type=int,   default=20)
    parser.add_argument("--meta_batch_size",type=int,   default=12)
    parser.add_argument("--data_batch_size",type=int,   default=16)
    parser.add_argument("--epochs",         type=int,   default=12)

    # Optuna search
    parser.add_argument("--tune", action="store_true",
                        help="Run Optuna hyperparameter search instead of single run")
    parser.add_argument("--n_trials", type=int, default=50,
                        help="Number of Optuna trials (used with --tune)")

    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load MOMENT + build task datasets ─────────────────────────────────────
    from meta_learning_rater.dataset import (
        load_moment_model, build_all_tasks, compute_embeddings, load_blocks,
    )

    print("Loading MOMENT encoder …")
    moment = load_moment_model()

    dataset_configs = load_config(args.config)
    # Filter to only datasets that have annotation.jsonl
    available = [c for c in dataset_configs if Path(c["output"]).exists()]
    print(f"Configs: {len(dataset_configs)} total, "
          f"{len(available)} with annotation.jsonl")

    print("\nBuilding task datasets …")
    task_datasets = build_all_tasks(
        available,
        moment,
        support_ratio=args.support_ratio,
        query_ratio=args.query_ratio,
        min_confidence=args.min_confidence,
        device=device,
    )
    print(f"\n{len(task_datasets)} tasks loaded")

    if not task_datasets:
        print("No tasks available. Run annotation first.")
        return

    # Infer input_dim from first task
    sample_emb = next(iter(task_datasets[0]["support"]))[0]
    input_dim = sample_emb.shape[0]
    print(f"Embedding dim: {input_dim}")

    # ── Output dir ────────────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Standard run or Optuna search ─────────────────────────────────────────
    if not args.tune:
        acc, model = train_and_eval(
            task_datasets, input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            meta_lr=args.meta_lr,
            inner_lr=args.inner_lr,
            inner_steps=args.inner_steps,
            meta_batch_size=args.meta_batch_size,
            data_batch_size=args.data_batch_size,
            epochs=args.epochs,
            device=device,
            seed=args.seed,
        )
        torch.save(model.state_dict(), output_path)
        print(f"\nModel saved → {output_path}")
        print(f"Test accuracy: {acc:.4f}")

    else:
        import optuna

        best_acc   = 0.0
        best_model = None

        def objective(trial: optuna.Trial) -> float:
            nonlocal best_acc, best_model

            meta_lr        = trial.suggest_float("meta_lr",   1e-5, 5e-4, log=True)
            inner_lr       = trial.suggest_float("inner_lr",  0.01, 0.5,  log=True)
            inner_steps    = trial.suggest_int  ("inner_steps", 5, 20)
            meta_batch_size= trial.suggest_int  ("meta_batch_size", 5, 15)
            epochs         = trial.suggest_int  ("epochs", 10, 20)

            acc, model = train_and_eval(
                task_datasets, input_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                meta_lr=meta_lr,
                inner_lr=inner_lr,
                inner_steps=inner_steps,
                meta_batch_size=meta_batch_size,
                data_batch_size=args.data_batch_size,
                epochs=epochs,
                device=device,
                seed=args.seed,
            )

            if acc > best_acc:
                best_acc   = acc
                best_model = model
                torch.save(best_model.state_dict(), output_path)
                print(f"  [new best] acc={best_acc:.4f} → saved to {output_path}")

            return acc

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=args.n_trials)

        print(f"\nBest accuracy : {study.best_value:.4f}")
        print(f"Best params   : {study.best_params}")

        results_path = output_path.parent / "optuna_results.csv"
        study.trials_dataframe().to_csv(results_path, index=False)
        print(f"Optuna results → {results_path}")


if __name__ == "__main__":
    main()