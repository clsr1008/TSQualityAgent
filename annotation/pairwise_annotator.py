"""
Single-pair annotation using the Perceiver+Inspector+Adjudicator pipeline.

Maps agent output to a Bradley-Terry compatible `comparisons_avg` score:
  winner=A, conf=c  →  comparisons_avg = (1 - c) / 2   ∈ [0, 0.5)
  winner=B, conf=c  →  comparisons_avg = (1 + c) / 2   ∈ (0.5, 1]
  tie               →  comparisons_avg = 0.5

Confidence filter (paper §C.2): |2 * comparisons_avg - 1| ≥ 0.5
  ↔  winner ≠ tie  AND  confidence ≥ 0.5
"""
from __future__ import annotations

from workflow import run_pipeline
from config import Config, build_llm, build_perceiver_llm


def winner_to_comparisons_avg(winner: str, confidence: float) -> float:
    """Convert agent judgment to a Bradley-Terry probability score."""
    if winner == "A":
        return (1.0 - confidence) / 2.0
    elif winner == "B":
        return (1.0 + confidence) / 2.0
    else:  # tie
        return 0.5


def is_high_confidence(comparisons_avg: float, min_confidence: float = 0.5) -> bool:
    """Return True if the pair passes the confidence filter.

    Equivalent to: winner != tie AND agent confidence >= min_confidence.
    Translated to comparisons_avg: |2*avg - 1| >= min_confidence.
    """
    return abs(2.0 * comparisons_avg - 1.0) >= min_confidence


def annotate_pair(
    index_a: int,
    series_a: list[float],
    index_b: int,
    series_b: list[float],
    dataset_description: str,
    config: Config,
) -> dict:
    """
    Run the full agent pipeline on one pair and return a result record.

    Returns
    -------
    dict with keys:
      block_a, block_b        — original dataset indices
      winner                  — "A" | "B" | "tie"
      confidence              — float [0, 1]
      comparisons_avg         — float [0, 1], BT-compatible
      explanation             — str
      error                   — str | None  (set when pipeline raises)
    """
    input_data = {
        "dataset_description": dataset_description,
        "series_A": series_a,
        "series_B": series_b,
    }

    llm = build_llm(config)
    perceiver_llm = build_perceiver_llm(config)

    try:
        final_state = run_pipeline(input_data, llm, config, perceiver_llm=perceiver_llm)
        result = final_state.get("final_result") or {}
        winner = result.get("winner", "tie")
        confidence = float(result.get("confidence", 0.0))
        explanation = result.get("explanation", "")
    except Exception as exc:
        return {
            "block_a": index_a,
            "block_b": index_b,
            "winner": "tie",
            "confidence": 0.0,
            "comparisons_avg": 0.5,
            "explanation": "",
            "error": str(exc),
        }

    comparisons_avg = winner_to_comparisons_avg(winner, confidence)

    return {
        "block_a": index_a,
        "block_b": index_b,
        "winner": winner,
        "confidence": confidence,
        "comparisons_avg": comparisons_avg,
        "explanation": explanation,
        "error": None,
    }