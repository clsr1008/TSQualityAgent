"""
Agent 2 – Inspector
Responsibility: Action (ReAct loop)
  - For each planned dimension, run Thought → Tool Call → Observation until
    sufficient evidence is gathered (or max_steps reached).
  - Produces a DimensionResult for each dimension.
"""
import json
from models.state import AgentState, DimensionResult
from models.llm import BaseLLM, LLMResponse
from tools import NumpyEncoder

# ── Tool registry ─────────────────────────────────────────────────────────────

from tools.bad_quality import missing_ratio, noise_profile, signal_to_noise_ratio
from tools.rare_pattern import anomaly_detection, outlier_density
from tools.pattern_structure import (
    trend_classifier,
    seasonality_detector,
    spike_detector,
    change_point_detector,
    pattern_consistency_indicators,
)

TOOL_REGISTRY = {
    "missing_ratio": missing_ratio,
    "noise_profile": noise_profile,
    "signal_to_noise_ratio": signal_to_noise_ratio,
    "anomaly_detection": anomaly_detection,
    "outlier_density": outlier_density,
    "trend_classifier": trend_classifier,
    "seasonality_detector": seasonality_detector,
    "spike_detector": spike_detector,
    "change_point_detector": change_point_detector,
    "pattern_consistency_indicators": pattern_consistency_indicators,
}

# OpenAI-style function schemas for tool calling
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "missing_ratio",
            "description": "Compute fraction of missing (NaN) values in a series.",
            "parameters": {
                "type": "object",
                "properties": {
                    "series_name": {"type": "string", "enum": ["A", "B"]},
                },
                "required": ["series_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "noise_profile",
            "description": "Estimate noise level using rolling-window residuals.",
            "parameters": {
                "type": "object",
                "properties": {
                    "series_name": {"type": "string", "enum": ["A", "B"]},
                    "window": {"type": "integer", "default": 5},
                },
                "required": ["series_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "signal_to_noise_ratio",
            "description": "Compute signal-to-noise ratio (mean/std).",
            "parameters": {
                "type": "object",
                "properties": {
                    "series_name": {"type": "string", "enum": ["A", "B"]},
                },
                "required": ["series_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "anomaly_detection",
            "description": "Detect point anomalies using Z-score threshold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "series_name": {"type": "string", "enum": ["A", "B"]},
                    "anomaly_threshold": {"type": "number", "default": 3.0},
                },
                "required": ["series_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "outlier_density",
            "description": "Estimate outlier density using IQR fences.",
            "parameters": {
                "type": "object",
                "properties": {
                    "series_name": {"type": "string", "enum": ["A", "B"]},
                },
                "required": ["series_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "trend_classifier",
            "description": "Classify trend direction and strength via linear regression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "series_name": {"type": "string", "enum": ["A", "B"]},
                    "window": {"type": "integer"},
                },
                "required": ["series_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "seasonality_detector",
            "description": "Detect dominant seasonal period via autocorrelation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "series_name": {"type": "string", "enum": ["A", "B"]},
                    "max_period": {"type": "integer"},
                },
                "required": ["series_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spike_detector",
            "description": "Detect spikes (large amplitude excursions) by Z-score.",
            "parameters": {
                "type": "object",
                "properties": {
                    "series_name": {"type": "string", "enum": ["A", "B"]},
                    "threshold": {"type": "number", "default": 3.0},
                    "min_sep": {"type": "integer", "default": 1},
                },
                "required": ["series_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "change_point_detector",
            "description": "Detect structural change points using CUSUM.",
            "parameters": {
                "type": "object",
                "properties": {
                    "series_name": {"type": "string", "enum": ["A", "B"]},
                },
                "required": ["series_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pattern_consistency_indicators",
            "description": "Compute lumpiness, flat_spots, and crossing_points.",
            "parameters": {
                "type": "object",
                "properties": {
                    "series_name": {"type": "string", "enum": ["A", "B"]},
                },
                "required": ["series_name"],
            },
        },
    },
]

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the Inspector agent in a time series quality assessment pipeline.

You assess specific quality dimensions of two time series (A and B) by calling tools.
For each dimension, follow the ReAct loop:
  Thought: what you plan to do and why
  Action: call one or more tools
  Observation: interpret the tool results

Continue until you have sufficient evidence for the dimension, then output your conclusion
in the following JSON format (as the last message):
{
  "dimension": "<dimension_name>",
  "score_A": <0.0 to 1.0>,
  "score_B": <0.0 to 1.0>,
  # 改成confidence
  # 最终也是汇总结果也是得到成对比较结果（二元）加上confidence（0-1文章）
  "evidence": { ... },
  "conclusion": "<brief comparison sentence>"
}

Score interpretation: 1.0 = perfect quality, 0.0 = very poor quality.
Output ONLY valid JSON as your final response — no extra text."""


def _call_tool(tool_name: str, args: dict, series_A: list, series_B: list):
    """Execute a tool call with the correct series."""
    fn = TOOL_REGISTRY.get(tool_name)
    if fn is None:
        return {"error": f"Unknown tool: {tool_name}"}

    series_name = args.pop("series_name", "A")
    series = series_A if series_name == "A" else series_B

    try:
        result = fn(series, **args)
        result["series"] = series_name
        return result
    except Exception as e:
        return {"error": str(e), "series": series_name}


def _assess_dimension(
    dimension: str,
    series_A: list,
    series_B: list,
    llm: BaseLLM,
    max_steps: int,
    feedback: str = "",
) -> DimensionResult:
    """Run the ReAct loop for a single quality dimension."""

    user_msg = f"Assess the '{dimension}' quality dimension for series A and B."
    if feedback:
        user_msg += f"\n\nAdditional guidance from Adjudicator: {feedback}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    evidence = {}
    conclusion = ""
    score_A, score_B = 0.5, 0.5

    for step in range(max_steps):
        response: LLMResponse = llm.chat_with_tools(messages, TOOL_SCHEMAS)

        if response.has_tool_calls:
            # Append assistant message WITH tool_calls array (required by OpenAI API)
            messages.append({
                "role": "assistant",
                "content": response.content or None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in response.tool_calls
                ],
            })
            # Execute each tool and append result with matching tool_call_id
            for tc in response.tool_calls:
                args = dict(tc.arguments)
                obs = _call_tool(tc.name, args, series_A, series_B)
                evidence[f"{tc.name}_{obs.get('series', '?')}"] = obs
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(obs, cls=NumpyEncoder),
                })
        else:
            # LLM gave a final text response — try to parse as conclusion JSON
            try:
                parsed = json.loads(response.content)
                score_A = float(parsed.get("score_A") or 0.5)
                score_B = float(parsed.get("score_B") or 0.5)
                evidence.update(parsed.get("evidence", {}))
                conclusion = parsed.get("conclusion", "")
            except (json.JSONDecodeError, ValueError):
                conclusion = response.content
            break

    return DimensionResult(
        dimension=dimension,
        score_A=round(score_A, 4),
        score_B=round(score_B, 4),
        evidence=evidence,
        conclusion=conclusion,
    )


# ── LangGraph node ────────────────────────────────────────────────────────────

def run_inspector(state: AgentState, llm: BaseLLM, max_steps: int = 6) -> dict:
    inp = state["input"]
    series_A = inp.get("series_A", [])
    series_B = inp.get("series_B", [])

    reflection_type = state.get("reflection_type")
    feedback = state.get("reflection_feedback", "")

    # Determine which dimensions to process
    if reflection_type == "needs_recheck":
        dims_to_run = state.get("recheck_dimensions") or state.get("planned_dimensions", [])
    else:
        dims_to_run = state.get("planned_dimensions", [])

    # Merge results: keep old results for dims not being rechecked
    existing = {r["dimension"]: r for r in state.get("dimension_results", [])}

    new_results = []
    for dim in dims_to_run:
        result = _assess_dimension(dim, series_A, series_B, llm, max_steps, feedback)
        existing[dim] = result

    dimension_results = list(existing.values())

    return {
        "dimension_results": dimension_results,
        "recheck_count": state.get("recheck_count", 0) + (1 if reflection_type == "needs_recheck" else 0),
        "reflection_type": None,
        "reflection_feedback": None,
        "recheck_dimensions": None,
    }
