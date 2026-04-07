"""
Agent 1 – Perceiver
Responsibility: Perception + Planning
  - Parse input, compute basic statistics for series A and B
  - Compare the two series at a high level to focus on their differences
  - Select which quality dimensions need to be assessed (avoid irrelevant checks)
  - Output: planned_dimensions: list[str]
"""
import json
import numpy as np
from models.state import AgentState
from models.llm import BaseLLM
from agents.dimensions import DIMENSION_GUIDE

# All supported dimensions
ALL_DIMENSIONS = [
    "missing_value",
    "noise_level",
    "rare_pattern",
    "trend",
    "frequency",
    "amplitude",
    "pattern_consistency",
]

SYSTEM_PROMPT = f"""You are the Perceiver agent in a time series quality assessment pipeline.

Your job is to:
1. Understand the task context and dataset description.
2. Examine the basic statistics and characteristics of time series A and B.
3. Identify the key differences between A and B that are relevant to quality.
4. Select which quality dimensions the Inspector should assess.
   Avoid dimensions that are clearly irrelevant to this particular dataset.
{DIMENSION_GUIDE}
Valid dimension names you may select from (choose by exact name):
  missing_value, noise_level, rare_pattern, trend, frequency, amplitude, pattern_consistency

You MUST respond with valid JSON in this exact format:
{{
  "perception_summary": "<brief description of what you observed about A and B>",
  "planned_dimensions": ["dim1", "dim2", ...],
  "tool_required": ["dim1", ...]
}}

- planned_dimensions: all dimensions worth assessing (at least 2), ordered by relevance.
- tool_required: subset of planned_dimensions that need tool-based measurement.
  Omit dimensions where the difference is already clear from stats/preview (e.g. obvious missing values, clear trend difference). Those will be assessed by reasoning only."""


def _basic_stats(series: list) -> dict:
    arr = np.array(series, dtype=float)
    valid = arr[~np.isnan(arr)]
    n = len(arr)

    # Rough linear slope (trend direction)
    if len(valid) >= 2:
        x = np.where(~np.isnan(arr))[0].astype(float)
        slope = float(np.polyfit(x, valid, 1)[0])
    else:
        slope = None

    return {
        "length": n,
        "missing_ratio": round(1 - len(valid) / n, 4) if n > 0 else None,
        "mean": round(float(np.mean(valid)), 4) if len(valid) else None,
        "std": round(float(np.std(valid)), 4) if len(valid) else None,
        "min": round(float(np.min(valid)), 4) if len(valid) else None,
        "max": round(float(np.max(valid)), 4) if len(valid) else None,
        "p25": round(float(np.percentile(valid, 25)), 4) if len(valid) else None,
        "p75": round(float(np.percentile(valid, 75)), 4) if len(valid) else None,
        "slope": round(slope, 6) if slope is not None else None,
    }


def _series_preview(series: list, max_full: int = 200, sample_size: int = 60) -> list:
    """
    Return the series as-is if short enough; otherwise return a representative sample:
    first 20 + evenly-spaced middle points + last 20, totalling ~sample_size values.
    NaN is preserved as null for JSON serialisation.
    """
    arr = [None if (v is None or (isinstance(v, float) and v != v)) else round(v, 4)
           for v in series]
    if len(arr) <= max_full:
        return arr
    head = arr[:20]
    tail = arr[-20:]
    middle_n = sample_size - 40
    step = max(1, (len(arr) - 40) // middle_n)
    middle = arr[20:-20:step][:middle_n]
    return head + middle + tail


def run_perceiver(state: AgentState, llm: BaseLLM) -> dict:
    """
    LangGraph node function.
    Reads state, calls LLM, returns partial state update.

    On replan: only proposes dimensions to ADD on top of what was already assessed.
    Already-completed dimension_results are preserved unchanged.
    """
    inp = state["input"]
    series_A = inp.get("series_A", [])
    series_B = inp.get("series_B", [])
    is_replan = state.get("reflection_type") == "needs_replan"

    stats_A = _basic_stats(series_A)
    stats_B = _basic_stats(series_B)
    preview_A = _series_preview(series_A)
    preview_B = _series_preview(series_B)

    user_content = f"""Dataset description: {inp.get('dataset_description', 'Not provided.')}

Series A — values ({len(series_A)} total{', sampled' if len(series_A) > 200 else ''}):
{json.dumps(preview_A)}

Series A statistics:
{json.dumps(stats_A, indent=2)}

Series B — values ({len(series_B)} total{', sampled' if len(series_B) > 200 else ''}):
{json.dumps(preview_B)}

Series B statistics:
{json.dumps(stats_B, indent=2)}

External variables: {json.dumps(inp.get('external_variables', {}), indent=2)}"""

    # On replan: show already-completed dimensions and constrain the output
    if is_replan:
        existing_results = state.get("dimension_results", [])
        done_summary = "\n".join(
            f"  - {r['dimension']}: winner={r['winner']}, confidence={r['confidence']:.0%} — {r['conclusion']}"
            for r in existing_results
        )
        done_names = [r["dimension"] for r in existing_results]
        remaining = [d for d in ALL_DIMENSIONS if d not in done_names]
        feedback = state.get("reflection_feedback", "")
        user_content += f"""

--- REPLAN REQUEST ---
The Adjudicator identified gaps in the current assessment and requests additional dimensions.
Feedback: {feedback}

Already assessed (DO NOT repeat these):
{done_summary or '  (none)'}

Remaining available dimensions: {remaining}

Select only the dimensions needed to address the feedback above.
Output ONLY the new dimensions to add (not the full list).
Remember: output ONLY valid JSON as specified."""
    else:
        user_content += """

Based on the above, decide which quality dimensions to assess.
Remember: output ONLY valid JSON as specified."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    response = llm.chat(messages)

    # Parse structured output
    try:
        parsed = json.loads(response.content)
        new_dimensions = [
            d for d in parsed.get("planned_dimensions", [])
            if d in ALL_DIMENSIONS
        ]
        tool_required = [
            d for d in parsed.get("tool_required", new_dimensions)
            if d in new_dimensions
        ]
        perception_summary = parsed.get("perception_summary", state.get("perception_summary", ""))
    except (json.JSONDecodeError, AttributeError):
        new_dimensions = ALL_DIMENSIONS if not is_replan else []
        tool_required = new_dimensions
        perception_summary = response.content if not is_replan else state.get("perception_summary", "")

    messages.append({"role": "assistant", "content": response.content})

    if is_replan:
        # Merge: keep existing planned dims + add new ones (deduplicated, order preserved)
        existing_planned = state.get("planned_dimensions", [])
        merged = existing_planned + [d for d in new_dimensions if d not in existing_planned]
        planned_dimensions = merged
        # Merge tool_required lists
        existing_tool_req = state.get("tool_required", [])
        merged_tool_req = existing_tool_req + [d for d in tool_required if d not in existing_tool_req]
        tool_required = merged_tool_req
        # Keep existing dimension_results intact — Inspector will only run new dims
        dimension_results = state.get("dimension_results", [])
    else:
        planned_dimensions = new_dimensions
        dimension_results = []   # fresh run

    return {
        "planned_dimensions": planned_dimensions,
        "tool_required": tool_required,
        "perception_summary": perception_summary,
        "perceiver_messages": messages,
        "dimension_results": dimension_results,
        "recheck_count": state.get("recheck_count", 0),
        "replan_count": state.get("replan_count", 0) + (1 if is_replan else 0),
        "reflection_type": None,
        "reflection_feedback": None,
        "recheck_dimensions": None,
    }
