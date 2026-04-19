"""
Dataset utilities for meta-learning rater.

Loads pairwise annotation results (annotation.jsonl) alongside raw series
(blocks.jsonl), computes MOMENT embeddings, and returns a PairwiseDataset
compatible with BradleyTerryLoss training.

Pairwise label format (from annotation/run_annotation.py output):
  {"block_a": int, "block_b": int, "comparisons_avg": float, ...}

Confidence filter applied at load time:
  |2 * comparisons_avg - 1| >= min_confidence  (default 0.5, per paper §C.2)
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


# ── Raw data loading ──────────────────────────────────────────────────────────

def load_blocks(jsonl_path: str | Path) -> list[dict]:
    """Load records from blocks.jsonl. Returns list of {index, input_arr, ...}."""
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_annotation(
    annotation_path: str | Path,
    min_confidence: float = 0.5,
) -> list[dict]:
    """
    Load pairwise annotation records, filtering by confidence.

    Keeps only records where:
      - error is None
      - |2 * comparisons_avg - 1| >= min_confidence
    """
    records = []
    with open(annotation_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("error") is not None:
                continue
            avg = rec.get("comparisons_avg", 0.5)
            if abs(2.0 * avg - 1.0) >= min_confidence:
                records.append(rec)
    return records


# ── MOMENT embeddings ─────────────────────────────────────────────────────────

def load_moment_model():
    """Load MOMENT-1-base in embedding mode."""
    from momentfm import MOMENTPipeline
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-base",
        model_kwargs={"task_name": "embedding"},
    )
    model.init()
    model.eval()
    return model


def compute_embeddings(
    moment_model,
    records: list[dict],
    target_len: int = 512,
    batch_size: int = 64,
    device: str = "cpu",
) -> dict[int, torch.Tensor]:
    """
    Compute MOMENT embeddings for all records.

    Series shorter than target_len are zero-padded; longer ones are truncated.
    MOMENT's default context window is 512 — we use that as target_len.

    Returns dict: index → embedding tensor (on CPU).
    """
    moment_model = moment_model.to(device)

    indices = [r["index"] for r in records]
    arrays  = [r["input_arr"] for r in records]

    embeddings_dict: dict[int, torch.Tensor] = {}

    for start in range(0, len(arrays), batch_size):
        batch_indices = indices[start:start + batch_size]
        batch_arrays  = arrays[start:start + batch_size]

        # Pad / truncate to target_len
        padded = []
        for arr in batch_arrays:
            if len(arr) < target_len:
                arr = arr + [0.0] * (target_len - len(arr))
            else:
                arr = arr[:target_len]
            padded.append(arr)

        # [B, 1, target_len]
        x = torch.tensor(padded, dtype=torch.float32).unsqueeze(1).to(device)

        with torch.no_grad():
            output = moment_model(x_enc=x)
        embs = output.embeddings.cpu()  # [B, D]

        for idx, emb in zip(batch_indices, embs):
            embeddings_dict[idx] = emb

    return embeddings_dict


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class PairwiseDataset(Dataset):
    """
    Dataset of (embedding_a, embedding_b, p_b_greater_a) triples.

    Parameters
    ----------
    embeddings_dict : dict[int, Tensor]
        Map from series index to its MOMENT embedding.
    pairs : list[dict]
        Each dict has keys: block_a (int), block_b (int), comparisons_avg (float).
    """

    def __init__(
        self,
        embeddings_dict: dict[int, torch.Tensor],
        pairs: list[dict],
    ):
        # Filter out pairs where either embedding is missing
        self.data = [
            p for p in pairs
            if p["block_a"] in embeddings_dict and p["block_b"] in embeddings_dict
        ]
        self.embeddings = embeddings_dict

        n_dropped = len(pairs) - len(self.data)
        if n_dropped:
            print(f"  [PairwiseDataset] dropped {n_dropped} pairs "
                  f"(embedding missing)")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data[idx]
        emb_a = self.embeddings[row["block_a"]]
        emb_b = self.embeddings[row["block_b"]]
        p     = torch.tensor(row["comparisons_avg"], dtype=torch.float32)
        return emb_a, emb_b, p


# ── Task builder ──────────────────────────────────────────────────────────────

def build_task(
    blocks_path: str | Path,
    annotation_path: str | Path,
    moment_model,
    support_ratio: float = 0.4,
    query_ratio: float = 0.4,
    # test_ratio = 1 - support_ratio - query_ratio
    min_confidence: float = 0.5,
    device: str = "cpu",
) -> dict | None:
    """
    Build one meta-learning task from a (blocks, annotation) pair.

    Returns dict with keys: support, query, test (torch Subsets),
    or None if the dataset has too few valid pairs.
    """
    from torch.utils.data import random_split

    pairs = load_annotation(annotation_path, min_confidence)
    if len(pairs) < 10:
        print(f"  [skip] {annotation_path} — only {len(pairs)} valid pairs")
        return None

    # Pair augmentation: add symmetric (B,A,1-p) before splitting
    augmented = pairs + [
        {**r, "block_a": r["block_b"], "block_b": r["block_a"],
         "comparisons_avg": 1.0 - r["comparisons_avg"]}
        for r in pairs
    ]

    blocks = load_blocks(blocks_path)
    embeddings = compute_embeddings(moment_model, blocks, device=device)

    dataset = PairwiseDataset(embeddings, augmented)
    if len(dataset) < 10:
        print(f"  [skip] {annotation_path} — only {len(dataset)} usable pairs")
        return None

    total = len(dataset)
    n_support = int(total * support_ratio)
    n_query   = int(total * query_ratio)
    n_test    = total - n_support - n_query
    if n_test < 1:
        n_test = 1
        n_query = total - n_support - n_test

    support_set, query_set, test_set = random_split(
        dataset, [n_support, n_query, n_test]
    )
    return {"support": support_set, "query": query_set, "test": test_set}


def build_all_tasks(
    dataset_configs: list[dict],
    moment_model,
    support_ratio: float = 0.4,
    query_ratio: float = 0.4,
    min_confidence: float = 0.5,
    device: str = "cpu",
) -> list[dict]:
    """
    Build task list from dataset_configs (list of {dataset, output, description}).

    `dataset` → blocks.jsonl path
    `output`  → annotation.jsonl path
    """
    tasks = []
    for cfg in dataset_configs:
        blocks_path     = Path(cfg["dataset"])
        annotation_path = Path(cfg["output"])

        if not blocks_path.exists():
            print(f"  [skip] blocks not found: {blocks_path}")
            continue
        if not annotation_path.exists():
            print(f"  [skip] annotation not found: {annotation_path}")
            continue

        name = blocks_path.parent.name
        print(f"  Loading task: {name} ...", end=" ", flush=True)
        task = build_task(
            blocks_path, annotation_path, moment_model,
            support_ratio=support_ratio,
            query_ratio=query_ratio,
            min_confidence=min_confidence,
            device=device,
        )
        if task is not None:
            task["name"] = name
            tasks.append(task)
            print(f"support={len(task['support'])}  "
                  f"query={len(task['query'])}  "
                  f"test={len(task['test'])}")

    return tasks