"""
Meta-learning rater for time-series quality scoring.

Pipeline:
  1. MOMENT encoder  → fixed-size embeddings per series
  2. ScoreModel(MLP) → scalar quality score per series
  3. BradleyTerryLoss trained on pairwise (block_a, block_b, comparisons_avg) labels
  4. MAML meta-learning across 23 datasets → fast-adaptable quality rater

Modules
-------
  dataset.py       — PairwiseDataset, data loading from blocks.jsonl + annotation.jsonl
  model.py         — ScoreModel, BradleyTerryLoss
  meta_train.py    — MetaLearner (MAML inner/outer loop), train + eval
  score.py         — Inference: produce scalar quality scores for a full dataset
  run_meta_train.py— CLI entry point with Optuna hyperparameter search
"""