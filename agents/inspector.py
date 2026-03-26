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
from agents.dimensions import DIMENSION_GUIDE
from agents.perceiver import _series_preview

from tools.registry import TOOL_REGISTRY, TOOL_SCHEMAS

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""You are the Inspector agent in a time series quality assessment pipeline.

You assess a specific quality dimension of two time series (A and B) by calling tools.
For each dimension, follow the ReAct loop:
  Thought: what you plan to do and why
  Action: call one or more tools
  Observation: interpret the tool results
{DIMENSION_GUIDE}
Stop calling tools as soon as you have sufficient evidence to reach a confident conclusion.
Continue until you have sufficient evidence, then output your conclusion
in the following JSON format (as the last message):
{{
  "dimension": "<dimension_name>",
  "winner": "A" | "B" | "tie",
  "confidence": <0.0 to 1.0>,
  "evidence": {{ ... }},
  "conclusion": "<one sentence explaining why the winner is better on this dimension>"
}}

confidence interpretation: 1.0 = completely certain, 0.0 = no difference detected.
Output ONLY valid JSON as your final response — no extra text."""


def _annotate_react_roles(messages: list) -> list:
    """Add react_role to each message for display purposes (does not affect API calls)."""
    annotated = []
    for i, m in enumerate(messages):
        role = m.get("role", "")
        if role == "system":
            react_role = "system"
        elif role == "user":
            react_role = "query"
        elif role == "assistant":
            if m.get("tool_calls"):
                react_role = "action"
            elif i == len(messages) - 1:
                react_role = "final_answer"
            else:
                react_role = "thought"
        elif role == "tool":
            react_role = "observation"
        else:
            react_role = role
        annotated.append({**m, "react_role": react_role})
    return annotated


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
    perception_summary: str = "",
    feedback: str = "",
) -> DimensionResult:
    """Run the ReAct loop for a single quality dimension."""

    preview_A = _series_preview(series_A)
    preview_B = _series_preview(series_B)

    user_msg = (
        f"Assess the '{dimension}' quality dimension for series A and B.\n\n"
        f"Series A ({len(series_A)} points{', sampled' if len(series_A) > 200 else ''}):\n"
        f"{json.dumps(preview_A)}\n\n"
        f"Series B ({len(series_B)} points{', sampled' if len(series_B) > 200 else ''}):\n"
        f"{json.dumps(preview_B)}"
    )
    if perception_summary:
        user_msg += f"\n\nContext from Perceiver: {perception_summary}"
    if feedback:
        user_msg += f"\n\nAdditional guidance from Adjudicator: {feedback}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    evidence = {}
    conclusion = ""
    winner = "tie"
    confidence = 0.0

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
            # Text response: try to parse as Final Answer JSON.
            # If it parses with the required fields → this is F (Final Answer), exit.
            # Otherwise it is a Thought → append and continue the ReAct loop.
            try:
                parsed = json.loads(response.content)
                if "winner" in parsed and "conclusion" in parsed:
                    winner = parsed.get("winner") or "tie"
                    confidence = float(parsed.get("confidence") or 0.0)
                    evidence.update(parsed.get("evidence", {}))
                    conclusion = parsed.get("conclusion", "")
                    messages.append({"role": "assistant", "content": response.content})
                    break
            except (json.JSONDecodeError, ValueError):
                pass
            # Treat as Thought — append and let the loop continue
            messages.append({"role": "assistant", "content": response.content})

    return DimensionResult(
        dimension=dimension,
        winner=winner,
        confidence=round(confidence, 4),
        evidence=evidence,
        conclusion=conclusion,
        messages=_annotate_react_roles(messages),
    )


# ── LangGraph node ────────────────────────────────────────────────────────────

def run_inspector(state: AgentState, llm: BaseLLM, max_steps: int = 6) -> dict:
    inp = state["input"]
    series_A = inp.get("series_A", [])
    series_B = inp.get("series_B", [])

    reflection_type = state.get("reflection_type")
    feedback = state.get("reflection_feedback", "")
    perception_summary = state.get("perception_summary", "")

    # Determine which dimensions to process
    if reflection_type == "needs_recheck":
        dims_to_run = state.get("recheck_dimensions") or state.get("planned_dimensions", [])
    else:
        dims_to_run = state.get("planned_dimensions", [])

    # Merge results: keep old results for dims not being rechecked
    existing = {r["dimension"]: r for r in state.get("dimension_results", [])}

    for dim in dims_to_run:
        result = _assess_dimension(dim, series_A, series_B, llm, max_steps, perception_summary, feedback)
        existing[dim] = result

    dimension_results = list(existing.values())

    return {
        "dimension_results": dimension_results,
        "recheck_count": state.get("recheck_count", 0) + (1 if reflection_type == "needs_recheck" else 0),
        "reflection_type": None,
        "reflection_feedback": None,
        "recheck_dimensions": None,
    }
