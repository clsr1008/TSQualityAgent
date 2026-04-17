"""
Reward function for Perceiver GRPO training.

Computes a scalar reward in [-1.0, 1.0] by comparing the model's JSON output
against ground-truth labels (target_dimensions).

Two components:
  R_format (0.15) — JSON parsability and schema compliance
  R_dim    (0.85) — Precision of dimension selection
"""
import json
import re

ALL_DIMENSIONS = [
    "missing_value", "noise_level", "rare_pattern",
    "trend", "frequency", "amplitude", "pattern_consistency",
]

W_FORMAT = 0.10
W_DIM = 0.90


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
    """Precision of predicted vs target dimension sets."""
    pred_set = set(d for d in predicted_dims if d in ALL_DIMENSIONS)
    target_set = set(target_dims)
    if not target_set and not pred_set:
        return 1.0  # both empty = correct (tie case)
    if not pred_set:
        return 0.0
    tp = len(pred_set & target_set)
    return tp / len(pred_set)


# ── Main entry ───────────────────────────────────────────────────────────────

def compute_reward(completion: str,
                   target_dims: list[str]) -> float:
    """
    Compute scalar reward in [-1.0, 1.0].

    Parameters
    ----------
    completion : raw text output from the model
    target_dims : ground-truth target_dimensions
    """
    r_format, parsed = compute_format_reward(completion)
    if parsed is None:
        return -1.0

    predicted_dims = [d for d in parsed.get("planned_dimensions", [])
                      if d in ALL_DIMENSIONS]

    r_dim = compute_dimension_reward(predicted_dims, target_dims)

    return W_FORMAT * r_format + W_DIM * r_dim


# ── TRL-compatible reward functions (split by component) ─────────────────────
# Each function parses completions only once per batch via a shared cache.
# TRL calls reward_funcs sequentially on the same batch; the last function
# clears the cache entry to avoid unbounded memory growth.

_parse_cache: dict = {}


def _extract_text(completion) -> str:
    if isinstance(completion, list):
        return completion[-1].get("content", "") if completion else ""
    if isinstance(completion, dict):
        return completion.get("content", "")
    return str(completion)


def _get_batch_components(completions, target_dimensions):
    """Parse all completions once; cache by batch object id."""
    cache_key = id(completions)
    if cache_key in _parse_cache:
        return _parse_cache[cache_key]

    r_fmt, r_dim = [], []
    for completion, tgt_dims in zip(completions, target_dimensions):
        text = _extract_text(completion)
        fmt_score, parsed = compute_format_reward(text)
        r_fmt.append(fmt_score)
        if parsed is None:
            r_dim.append(0.0)
        else:
            pred_dims = [d for d in parsed.get("planned_dimensions", []) if d in ALL_DIMENSIONS]
            r_dim.append(compute_dimension_reward(pred_dims, tgt_dims))

    result = (r_fmt, r_dim)
    _parse_cache[cache_key] = result
    return result


def grpo_reward_format(prompts, completions, target_dimensions, **kwargs):
    """R_format component for TRL reward_funcs."""
    r_fmt, _ = _get_batch_components(completions, target_dimensions)
    return [W_FORMAT * v for v in r_fmt]


def grpo_reward_dim(prompts, completions, target_dimensions, **kwargs):
    """R_dim component for TRL reward_funcs. Clears cache after last function."""
    _, r_dim = _get_batch_components(completions, target_dimensions)
    _parse_cache.pop(id(completions), None)
    return [W_DIM * v for v in r_dim]