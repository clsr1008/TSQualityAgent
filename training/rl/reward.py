"""
Reward function for Perceiver GRPO training.

Computes a scalar reward in [-1.0, 1.0] by comparing the model's JSON output
against ground-truth labels (target_dimensions, tool_required).

Three components:
  R_format (0.2) — JSON parsability and schema compliance
  R_dim    (0.5) — F1 score of dimension selection
  R_tool   (0.3) — tool decision accuracy on correctly predicted dims
"""
import json
import re

ALL_DIMENSIONS = [
    "missing_value", "noise_level", "rare_pattern",
    "trend", "frequency", "amplitude", "pattern_consistency",
]

W_FORMAT = 0.2
W_DIM = 0.5
W_TOOL = 0.3


# ── Parsing ──────────────────────────────────────────────────────────────────

def parse_perceiver_output(text: str) -> dict | None:
    """Extract JSON from model output, stripping <think> blocks and fences."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?", "", text).strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return None


def validate_schema(parsed: dict) -> bool:
    """Check required Perceiver output fields."""
    if not isinstance(parsed, dict):
        return False
    dims = parsed.get("planned_dimensions")
    if not isinstance(dims, list):
        return False
    # tool_required defaults to [] if absent — acceptable
    return True


# ── Reward components ────────────────────────────────────────────────────────

def compute_format_reward(text: str) -> tuple[float, dict | None]:
    """Return (format_reward, parsed_dict_or_None)."""
    parsed = parse_perceiver_output(text)
    if parsed is None:
        return -1.0, None
    if not validate_schema(parsed):
        return 0.0, None
    return 1.0, parsed


def compute_dimension_reward(predicted_dims: list[str],
                             target_dims: list[str]) -> float:
    """F1 score between predicted and target dimension sets."""
    pred_set = set(d for d in predicted_dims if d in ALL_DIMENSIONS)
    target_set = set(target_dims)
    if not target_set and not pred_set:
        return 1.0  # both empty = correct (tie case)
    if not target_set or not pred_set:
        return 0.0
    tp = len(pred_set & target_set)
    precision = tp / len(pred_set)
    recall = tp / len(target_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_tool_reward(predicted_dims: list[str],
                        predicted_tool: list[str],
                        target_dims: list[str],
                        target_tool: list[str]) -> float:
    """Tool decision accuracy, evaluated only on correctly predicted dims."""
    pred_set = set(d for d in predicted_dims if d in ALL_DIMENSIONS)
    target_set = set(target_dims)
    overlap = pred_set & target_set
    if not overlap:
        return 0.0
    target_tool_set = set(target_tool)
    pred_tool_set = set(predicted_tool)
    correct = sum(
        1 for d in overlap
        if (d in pred_tool_set) == (d in target_tool_set)
    )
    return correct / len(overlap)


# ── Main entry ───────────────────────────────────────────────────────────────

def compute_reward(completion: str,
                   target_dims: list[str],
                   target_tool: list[str]) -> float:
    """
    Compute scalar reward in [-1.0, 1.0].

    Parameters
    ----------
    completion : raw text output from the model
    target_dims : ground-truth target_dimensions
    target_tool : ground-truth tool_required
    """
    r_format, parsed = compute_format_reward(completion)
    if parsed is None:
        return -1.0

    predicted_dims = [d for d in parsed.get("planned_dimensions", [])
                      if d in ALL_DIMENSIONS]
    predicted_tool = [d for d in parsed.get("tool_required", [])
                      if d in predicted_dims]

    r_dim = compute_dimension_reward(predicted_dims, target_dims)
    r_tool = compute_tool_reward(
        predicted_dims, predicted_tool, target_dims, target_tool)

    return W_FORMAT * r_format + W_DIM * r_dim + W_TOOL * r_tool
