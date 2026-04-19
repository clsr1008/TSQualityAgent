"""
Single-dataset Bradley-Terry rater training (no meta-learning).

Trains one ScoreModel per dataset directly using annotation.jsonl + blocks.jsonl.
Uses the same model architecture as the meta-learning rater, so the trained
checkpoint is compatible with score.py for inference.

Usage
-----
# Train on a single dataset
python -m meta_learning_rater.train_single \
    --blocks   datasets/weather/blocks.jsonl \
    --annotation datasets/weather/annotation.jsonl \
    --output   meta_learning_rater/checkpoints/rater_weather.pth

# Batch: train a separate rater for each dataset in dataset_configs.json
# Checkpoints saved as meta_learning_rater/checkpoints/rater_<dataset_name>.pth
python -m meta_learning_rater.train_single \
    --config annotation/dataset_configs.json \
    --output_dir meta_learning_rater/checkpoints/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from meta_learning_rater.dataset import (
    load_blocks,
    load_annotation,
    load_moment_model,
    compute_embeddings,
    PairwiseDataset,
)
from meta_learning_rater.model import ScoreModel, BradleyTerryLoss


def train_rater(
    blocks_path: str | Path,
    annotation_path: str | Path,
    moment_model,
    output_path: str | Path,
    hidden_dim: int = 256,
    num_layers: int = 3,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    train_ratio: float = 0.8,
    min_confidence: float = 0.0,
    device: str = "cpu",
) -> float:
    """
    Train a single Bradley-Terry rater on one dataset.
    Returns test accuracy.
    """
    # ── Load data ──────────────────────────────────────────────────────────────
    pairs = load_annotation(annotation_path, min_confidence)
    if len(pairs) < 10:
        print(f"  [skip] {annotation_path} — only {len(pairs)} valid pairs after filtering")
        return 0.0

    # Pair augmentation: add symmetric pairs (swap A↔B, flip p) before split
    augmented = pairs + [
        {**r, "block_a": r["block_b"], "block_b": r["block_a"],
         "comparisons_avg": 1.0 - r["comparisons_avg"]}
        for r in pairs
    ]

    blocks = load_blocks(blocks_path)
    embeddings = compute_embeddings(moment_model, blocks, device=device)

    # Normalize embeddings (z-score per dimension)
    all_embs = torch.stack(list(embeddings.values()))
    mean = all_embs.mean(0)
    std  = all_embs.std(0).clamp(min=1e-6)
    embeddings = {k: (v - mean) / std for k, v in embeddings.items()}

    dataset = PairwiseDataset(embeddings, augmented)
    if len(dataset) < 10:
        print(f"  [skip] only {len(dataset)} usable pairs")
        return 0.0
    print(f"  {len(pairs)} pairs → {len(dataset)} after augmentation")

    # ── Train / test split ────────────────────────────────────────────────────
    n_train = int(len(dataset) * train_ratio)
    n_test  = len(dataset) - n_train
    if n_test < 1:
        n_test = 1
        n_train = len(dataset) - 1

    train_set, test_set = random_split(dataset, [n_train, n_test])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────────
    input_dim = next(iter(dataset))[0].shape[0]
    model     = ScoreModel(input_dim, hidden_dim, num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn   = BradleyTerryLoss()

    # ── Training ──────────────────────────────────────────────────────────────
    best_loss  = float("inf")
    best_state = None
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for emb_a, emb_b, p in train_loader:
            emb_a = emb_a.to(device); emb_b = emb_b.to(device); p = p.to(device)
            loss  = loss_fn(model(emb_a), model(emb_b), p)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        scheduler.step()
        if avg_loss < best_loss:
            best_loss  = avg_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 10 == 0:
            print(f"  [Epoch {epoch+1:3d}/{epochs}] loss={avg_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # ── Evaluation ────────────────────────────────────────────────────────────
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for emb_a, emb_b, p in test_loader:
            emb_a = emb_a.to(device); emb_b = emb_b.to(device); p = p.to(device)
            pred  = (model(emb_b) > model(emb_a)).float()
            correct += (pred == (p > 0.5).float()).sum().item()
            total   += len(p)

    acc = correct / total if total else 0.0
    print(f"  Test accuracy: {acc:.4f}  ({correct}/{total})")

    # ── Save ──────────────────────────────────────────────────────────────────
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"  Saved → {output_path}")
    return acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-dataset rater training")

    # Single dataset mode
    parser.add_argument("--blocks",      type=str, default=None)
    parser.add_argument("--annotation",  type=str, default=None)
    parser.add_argument("--output",      type=str, default=None)

    # Batch mode
    parser.add_argument("--config",      type=str, default=None,
                        help="dataset_configs.json for batch training")
    parser.add_argument("--output_dir",  type=str,
                        default="meta_learning_rater/checkpoints/",
                        help="Directory for batch mode checkpoints")

    # Model / training
    parser.add_argument("--hidden_dim",     type=int,   default=256)
    parser.add_argument("--num_layers",     type=int,   default=3)
    parser.add_argument("--epochs",         type=int,   default=100)
    parser.add_argument("--batch_size",     type=int,   default=32)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--weight_decay",   type=float, default=1e-4)
    parser.add_argument("--train_ratio",    type=float, default=0.8)
    parser.add_argument("--min_confidence", type=float, default=0.5)
    parser.add_argument("--device",         type=str,   default=None)

    args = parser.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading MOMENT encoder …")
    moment = load_moment_model()

    if args.config:
        # ── Batch mode ────────────────────────────────────────────────────────
        with open(args.config, "r", encoding="utf-8") as f:
            dataset_configs = json.load(f)

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}
        for cfg in dataset_configs:
            blocks_path     = Path(cfg["dataset"])
            annotation_path = Path(cfg["output"])
            name = blocks_path.parent.name

            if not blocks_path.exists() or not annotation_path.exists():
                print(f"[skip] {name} — missing files")
                continue

            print(f"\n── {name} ──")
            checkpoint = output_dir / f"rater_{name}.pth"
            acc = train_rater(
                blocks_path, annotation_path, moment,
                output_path=checkpoint,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                train_ratio=args.train_ratio,
                min_confidence=args.min_confidence,
                device=device,
            )
            results[name] = acc

        print("\n── Summary ──")
        for name, acc in results.items():
            print(f"  {name:<25} acc={acc:.4f}")
        if results:
            print(f"  Mean accuracy: {sum(results.values())/len(results):.4f}")

    elif args.blocks and args.annotation:
        # ── Single dataset mode ───────────────────────────────────────────────
        output = args.output or str(
            Path(args.annotation).parent / "rater.pth"
        )
        train_rater(
            args.blocks, args.annotation, moment,
            output_path=output,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            train_ratio=args.train_ratio,
            min_confidence=args.min_confidence,
            device=device,
        )
    else:
        parser.error("Provide either --config (batch) or both --blocks and --annotation (single)")


if __name__ == "__main__":
    main()