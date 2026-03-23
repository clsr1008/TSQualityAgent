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

# All supported dimensions
ALL_DIMENSIONS = [
    "missing_value",
    "noise_level",
    "anomaly",
    "trend",
    "frequency",
    "amplitude",
    "pattern_consistency",
]

SYSTEM_PROMPT = """You are the Perceiver agent in a time series quality assessment pipeline.

Your job is to:
1. Understand the task context and dataset description.
2. Examine the basic statistics and characteristics of time series A and B.
3. Identify the key differences between A and B that are relevant to quality.
4. Decide which quality dimensions should be assessed by the Inspector agent.
   Avoid selecting dimensions that are clearly irrelevant to this particular dataset.

Available quality dimensions:
- missing_value       : fraction of NaN / missing values
- noise_level         : signal noise and SNR
- anomaly             : rare point anomalies / outliers
- trend               : overall direction (increasing / decreasing / flat)
- frequency           : periodicity / seasonality
- amplitude           : spike intensity and value range
- pattern_consistency : structural consistency (lumpiness, flat spots, change points)

You MUST respond with valid JSON in this exact format:
{
  "perception_summary": "<brief description of what you observed about A and B>",
  "planned_dimensions": ["dim1", "dim2", ...]
}

Only include dimensions from the list above. Include at least 2 dimensions."""


def _basic_stats(series: list) -> dict:
    arr = np.array(series, dtype=float)
    valid = arr[~np.isnan(arr)]
    return {
        "length": len(arr),
        "missing_count": int(np.sum(np.isnan(arr))),
        "mean": round(float(np.mean(valid)), 4) if len(valid) else None,
        "std": round(float(np.std(valid)), 4) if len(valid) else None,
        "min": round(float(np.min(valid)), 4) if len(valid) else None,
        "max": round(float(np.max(valid)), 4) if len(valid) else None,
    }


def run_perceiver(state: AgentState, llm: BaseLLM) -> dict:
    """
    LangGraph node function.
    Reads state, calls LLM, returns partial state update.
    """
    inp = state["input"]
    series_A = inp.get("series_A", [])
    series_B = inp.get("series_B", [])

    stats_A = _basic_stats(series_A)
    stats_B = _basic_stats(series_B)

    user_content = f"""Task prompt: {inp.get('task_prompt', 'No prompt provided.')}

Dataset description: {inp.get('dataset_description', 'Not provided.')}

Series A statistics:
{json.dumps(stats_A, indent=2)}

Series B statistics:
{json.dumps(stats_B, indent=2)}

External variables: {json.dumps(inp.get('external_variables', {}), indent=2)}

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
        planned_dimensions = [
            d for d in parsed.get("planned_dimensions", [])
            if d in ALL_DIMENSIONS
        ]
        perception_summary = parsed.get("perception_summary", "")
    except (json.JSONDecodeError, AttributeError):
        # Fallback: use all core dimensions
        planned_dimensions = ["missing_value", "noise_level", "anomaly", "trend", "amplitude"]
        perception_summary = response.content

    messages.append({"role": "assistant", "content": response.content})

    return {
        "planned_dimensions": planned_dimensions,
        "perceiver_messages": messages,
        "dimension_results": [],    # reset for fresh Inspector run
        "recheck_count": state.get("recheck_count", 0),
        "replan_count": state.get("replan_count", 0) + (1 if state.get("reflection_type") == "needs_replan" else 0),
        "reflection_type": None,
        "reflection_feedback": None,
        "recheck_dimensions": None,
    }
