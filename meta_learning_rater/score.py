"""
Inference: compute scalar quality scores for all series in a dataset.

Given a trained ScoreModel checkpoint and a blocks.jsonl, this module:
  1. Computes MOMENT embeddings for all series
  2. (Optionally) few-shot fine-tunes the model on a small annotation subset
  3. Runs forward pass to produce a quality score for each series index
  4. Saves scores to a JSONL file: {"index": int, "quality_score": float}

Usage
-----
# Single dataset
python -m meta_learning_rater.score \
    --blocks datasets/electricity/blocks.jsonl \
    --model  meta_learning_rater/checkpoints/tsrater.pth \
    --annotation datasets/electricity/annotation.jsonl

# Batch: score all datasets in dataset_configs.json
python -m meta_learning_rater.score \
    --config annotation/dataset_configs.json \
    --model  meta_learning_rater/checkpoints/tsrater.pth
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from meta_learning_rater.dataset import (
    load_blocks,
    load_annotation,
    load_moment_model,
    compute_embeddings,
    PairwiseDataset,
)
from meta_learning_rater.model import ScoreModel, BradleyTerryLoss


def load_score_model(
    checkpoint_path: str | Path,
    input_dim: int,
    hidden_dim: int = 256,
    num_layers: int = 3,
    device: str = "cpu",
) -> ScoreModel:
    model = ScoreModel(input_dim, hidden_dim, num_layers)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    return model.to(device)


def few_shot_adapt(
    model: ScoreModel,
    pairs: list[dict],
    embeddings: dict[int, torch.Tensor],
    adaptation_steps: int = 10,
    adaptation_lr: float = 5e-3,
    few_shot_n: int = 20,
    device: str = "cpu",
) -> ScoreModel:
    """Fine-tune model on a few labeled pairs from this dataset."""
    import copy
    from torch.utils.data import DataLoader

    model = copy.deepcopy(model).to(device)
    dataset = PairwiseDataset(embeddings, pairs[:few_shot_n])
    if len(dataset) == 0:
        return model

    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=adaptation_lr)
    loss_fn = BradleyTerryLoss()
    model.train()

    for _ in range(adaptation_steps):
        for emb_a, emb_b, p in loader:
            emb_a = emb_a.to(device)
            emb_b = emb_b.to(device)
            p     = p.to(device)
            loss = loss_fn(model(emb_a), model(emb_b), p)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def score_dataset(
    blocks_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    annotation_path: str | Path | None = None,
    hidden_dim: int = 256,
    num_layers: int = 3,
    adaptation_steps: int = 10,
    adaptation_lr: float = 5e-3,
    few_shot_n: int = 20,
    device: str | None = None,
) -> None:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading MOMENT encoder …")
    moment = load_moment_model()

    print(f"Computing embeddings for {blocks_path} …")
    blocks = load_blocks(blocks_path)
    embeddings = compute_embeddings(moment, blocks, device=device)

    # Infer input_dim from first embedding
    input_dim = next(iter(embeddings.values())).shape[0]

    print(f"Loading ScoreModel from {checkpoint_path} …")
    model = load_score_model(checkpoint_path, input_dim, hidden_dim, num_layers, device)

    # Optional: few-shot adapt on this dataset's annotation
    if annotation_path and Path(annotation_path).exists():
        pairs = load_annotation(annotation_path)
        if pairs:
            print(f"Few-shot adapting on {min(few_shot_n, len(pairs))} pairs …")
            model = few_shot_adapt(
                model, pairs, embeddings,
                adaptation_steps=adaptation_steps,
                adaptation_lr=adaptation_lr,
                few_shot_n=few_shot_n,
                device=device,
            )

    # Score all series
    model.eval()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_indices = [r["index"] for r in blocks]
    all_embs = torch.stack([embeddings[i] for i in all_indices]).to(device)

    with torch.no_grad():
        scores = model(all_embs).cpu().tolist()

    with open(output_path, "w", encoding="utf-8") as f:
        for idx, score in zip(all_indices, scores):
            f.write(json.dumps({"index": idx, "quality_score": score}) + "\n")

    print(f"Scores saved → {output_path}  ({len(scores)} records)")


def score_all_datasets(
    config_path: str | Path,
    checkpoint_path: str | Path,
    hidden_dim: int = 256,
    num_layers: int = 3,
    adaptation_steps: int = 10,
    adaptation_lr: float = 5e-3,
    few_shot_n: int = 20,
    device: str | None = None,
) -> None:
    """Score all datasets listed in a dataset_configs.json.

    Output is written to datasets/<name>/scores.jsonl alongside blocks.jsonl.
    Uses annotation.jsonl from the same directory for few-shot adaptation if present.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        configs = json.load(f)

    print(f"Scoring {len(configs)} datasets from {config_path}\n")
    for cfg in configs:
        blocks_path     = Path(cfg["dataset"])
        annotation_path = Path(cfg["output"])
        output_path     = blocks_path.parent / "scores.jsonl"
        name            = blocks_path.parent.name

        if not blocks_path.exists():
            print(f"  [skip] {name}: blocks.jsonl not found")
            continue

        print(f"  [{name}]")
        score_dataset(
            blocks_path=blocks_path,
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            annotation_path=annotation_path if annotation_path.exists() else None,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            adaptation_steps=adaptation_steps,
            adaptation_lr=adaptation_lr,
            few_shot_n=few_shot_n,
            device=device,
        )

    print("\nAll datasets scored.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score time-series quality")

    # Mode: single dataset or batch
    parser.add_argument("--blocks",  default=None, help="blocks.jsonl path (single dataset mode)")
    parser.add_argument("--output",  default=None, help="Output scores.jsonl (single dataset mode)")
    parser.add_argument("--config",  default=None,
                        help="dataset_configs.json path (batch mode — scores all datasets)")

    parser.add_argument("--model",       required=True, help="ScoreModel .pth checkpoint")
    parser.add_argument("--annotation",  default=None,  help="annotation.jsonl for few-shot adapt (single mode)")
    parser.add_argument("--hidden_dim",  type=int, default=256)
    parser.add_argument("--num_layers",  type=int, default=3)
    parser.add_argument("--adapt_steps", type=int, default=10)
    parser.add_argument("--adapt_lr",    type=float, default=5e-3)
    parser.add_argument("--few_shot_n",  type=int, default=20)
    parser.add_argument("--device",      default=None)
    args = parser.parse_args()

    if args.config:
        score_all_datasets(
            config_path=args.config,
            checkpoint_path=args.model,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            adaptation_steps=args.adapt_steps,
            adaptation_lr=args.adapt_lr,
            few_shot_n=args.few_shot_n,
            device=args.device,
        )
    elif args.blocks:
        if not args.output:
            args.output = str(Path(args.blocks).parent / "scores.jsonl")
        score_dataset(
            blocks_path=args.blocks,
            checkpoint_path=args.model,
            output_path=args.output,
            annotation_path=args.annotation,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            adaptation_steps=args.adapt_steps,
            adaptation_lr=args.adapt_lr,
            few_shot_n=args.few_shot_n,
            device=args.device,
        )
    else:
        parser.error("Provide either --blocks (single dataset) or --config (batch mode)")


if __name__ == "__main__":
    main()