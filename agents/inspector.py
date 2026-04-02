"""
Agent 2 – Inspector
Responsibility: Action (ReAct loop)
  - Assess all planned dimensions in a single unified ReAct session.
  - Tool results are cached and shared across dimensions.
  - Produces a DimensionResult for each dimension.
"""
import json
import re
from models.state import AgentState, DimensionResult
from models.llm import BaseLLM, LLMResponse
from tools import NumpyEncoder
from agents.dimensions import DIMENSION_GUIDE
from agents.perceiver import _series_preview

from tools.registry import TOOL_REGISTRY, TOOL_SCHEMAS


# ── Tool Cache ────────────────────────────────────────────────────────────────

class ToolCache:
    """Cache tool results keyed by (tool_name, series_name, frozen_args)."""

    def __init__(self):
        self._store: dict[tuple, dict] = {}

    def _key(self, tool_name: str, series_name: str, args: dict) -> tuple:
        other = tuple(sorted((k, v) for k, v in args.items()))
        return (tool_name, series_name, other)

    def get(self, tool_name: str, series_name: str, args: dict):
        return self._store.get(self._key(tool_name, series_name, args))

    def put(self, tool_name: str, series_name: str, args: dict, result: dict):
        self._store[self._key(tool_name, series_name, args)] = result


# ── Prompts ───────────────────────────────────────────────────────────────────

_JSON_EXAMPLE = """{
  "dimension": "<dimension_name>",
  "winner": "A" | "B" | "tie",
  "confidence": <0.0 to 1.0>,
  "evidence": {
    "A": { <only the decisive metrics for A> },
    "B": { <only the decisive metrics for B> }
  },
  "conclusion": "<one sentence explaining why the winner is better on this dimension>"
}"""



UNIFIED_SYSTEM_PROMPT = """You are the Inspector agent in a time series quality assessment pipeline.

You assess multiple quality dimensions of two time series (A and B) by calling tools.
Follow the ReAct loop: Thought → Action → Observation.

""" + DIMENSION_GUIDE + """

**For rare_pattern dimension:**
  - Score winner/confidence based ONLY on Category 1 outliers.
  - Summarise any Category 2 findings in the conclusion, but do not penalise either series for them.

## Workflow
You will be given a list of dimensions to assess. Work through them **STRICTLY ONE AT A TIME**.

**CRITICAL: You MUST output the DIMENSION_COMPLETE block for the current dimension BEFORE
calling any tools and reasoning for the next dimension. Do NOT batch all tool calls first and output all
conclusions at the end — that wastes steps and loses the benefit of earlier findings.**

For each dimension:
1. Think about what tools to call. **Reuse observations from earlier dimensions** — do NOT re-call
   a tool whose results you already have in the conversation history.
2. Call only the tools needed for THIS dimension.
3. Output the DIMENSION_COMPLETE block for THIS dimension immediately:

DIMENSION_COMPLETE
""" + _JSON_EXAMPLE + """
END_DIMENSION

4. Only THEN move to the next dimension.

After completing ALL dimensions, output:
ALL_DIMENSIONS_COMPLETE

confidence reflects how much better the winner is, not just certainty:
  1.0 = overwhelming advantage, 0.5 = moderate difference, ~0.0 = negligible gap (near tie).

## Parameter Selection
Tool window/threshold parameters auto-adapt when omitted, but you should override them
based on the series preview (length, scale, visible patterns) when appropriate.

Every dimension conclusion SHOULD be backed by tool observations — do NOT rely on perception_summary alone."""




# ── Helpers ───────────────────────────────────────────────────────────────────

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
            else:
                react_role = "thought"
        elif role == "tool":
            react_role = "observation"
        else:
            react_role = role
        annotated.append({**m, "react_role": react_role})
    return annotated


def _call_tool(tool_name: str, args: dict, series_A: list, series_B: list, cache: ToolCache = None):
    """Execute a tool call with the correct series, using cache if available."""
    fn = TOOL_REGISTRY.get(tool_name)
    if fn is None:
        return {"error": f"Unknown tool: {tool_name}"}

    series_name = args.pop("series_name", "A")
    series = series_A if series_name == "A" else series_B

    # Check cache
    if cache is not None:
        cached = cache.get(tool_name, series_name, args)
        if cached is not None:
            return {**cached, "_cached": True}

    try:
        result = fn(series, **args)
        result["series"] = series_name
        if cache is not None:
            cache.put(tool_name, series_name, args, result)
        return result
    except Exception as e:
        return {"error": str(e), "series": series_name}


def _extract_json_block(text: str, start: int) -> str | None:
    """Extract a balanced JSON object starting at position `start` (must be '{')."""
    if start >= len(text) or text[start] != '{':
        return None
    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _parse_dimension_results(text: str) -> list[dict]:
    """Extract DIMENSION_COMPLETE blocks from a text response.

    Handles both single and double-brace outputs (model may echo escaped template).
    Uses brace-depth-aware extraction instead of regex for nested JSON.
    """
    results = []

    # Try original text first, then double-brace-cleaned version
    variants = [text]
    if '{{' in text:
        variants.append(text.replace('{{', '{').replace('}}', '}'))

    for variant in variants:
        for match in re.finditer(r'DIMENSION_COMPLETE\s*', variant):
            json_str = _extract_json_block(variant, match.end())
            if json_str is None:
                continue
            try:
                parsed = json.loads(json_str)
                if "dimension" in parsed and "winner" in parsed:
                    if not any(r["dimension"] == parsed["dimension"] for r in results):
                        results.append(parsed)
            except json.JSONDecodeError:
                continue
        if results:
            break

    # Fallback: try parsing the entire text as a single JSON result
    if not results:
        for variant in variants:
            try:
                parsed = json.loads(variant.strip())
                if "dimension" in parsed and "winner" in parsed:
                    results.append(parsed)
                    break
            except json.JSONDecodeError:
                continue

    return results


# ── Unified multi-dimension assessment ────────────────────────────────────────

def _assess_all_dimensions(
    dimensions: list[str],
    series_A: list,
    series_B: list,
    llm: BaseLLM,
    max_steps: int,
    perception_summary: str = "",
    feedback: str = "",
    tool_required: list[str] | None = None,
) -> list[DimensionResult]:
    """Run a single unified ReAct session assessing all dimensions."""

    preview_A = _series_preview(series_A)
    preview_B = _series_preview(series_B)
    cache = ToolCache()

    if tool_required is None:
        tool_required = dimensions

    reasoning_only = [d for d in dimensions if d not in tool_required]

    dim_lines = []
    for i, d in enumerate(dimensions):
        tag = " [reasoning-only]" if d in reasoning_only else " [use tools]"
        dim_lines.append(f"  {i + 1}. {d}{tag}")
    dim_list = "\n".join(dim_lines)

    user_msg = (
        f"Assess the following quality dimensions for series A and B, in this order:\n{dim_list}\n\n"
        f"Dimensions marked [use tools]: call tools to collect evidence.\n"
        f"Dimensions marked [reasoning-only]: use your own reasoning based on the preview and domain knowledge — no tool calls needed.\n\n"
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
        {"role": "system", "content": UNIFIED_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    completed: dict[str, DimensionResult] = {}
    total_steps = max_steps * len(dimensions)

    for step in range(total_steps):
        response: LLMResponse = llm.chat_with_tools(messages, TOOL_SCHEMAS)

        if response.has_tool_calls:
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
            for tc in response.tool_calls:
                args = dict(tc.arguments)
                obs = _call_tool(tc.name, args, series_A, series_B, cache)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(obs, cls=NumpyEncoder),
                })

            # Also check if the assistant content contains dimension results
            if response.content:
                for parsed in _parse_dimension_results(response.content):
                    dim = parsed["dimension"]
                    if dim in dimensions and dim not in completed:
                        completed[dim] = DimensionResult(
                            dimension=dim,
                            winner=parsed.get("winner", "tie"),
                            confidence=round(float(parsed.get("confidence", 0.0)), 4),
                            evidence=parsed.get("evidence", {}),
                            conclusion=parsed.get("conclusion", ""),
                            messages=_annotate_react_roles(list(messages)),
                        )
        else:
            text = response.content or ""
            messages.append({"role": "assistant", "content": text})

            # Parse any completed dimensions from this response
            for parsed in _parse_dimension_results(text):
                dim = parsed["dimension"]
                if dim in dimensions and dim not in completed:
                    completed[dim] = DimensionResult(
                        dimension=dim,
                        winner=parsed.get("winner", "tie"),
                        confidence=round(float(parsed.get("confidence", 0.0)), 4),
                        evidence=parsed.get("evidence", {}),
                        conclusion=parsed.get("conclusion", ""),
                        messages=_annotate_react_roles(list(messages)),
                    )

            # Check if all dimensions are done
            if "ALL_DIMENSIONS_COMPLETE" in text or len(completed) == len(dimensions):
                break

    # Force conclusion for any incomplete dimensions
    if len(completed) < len(dimensions):
        remaining = [d for d in dimensions if d not in completed]
        force_msg = (
            f"You have reached the step limit. The following dimensions are still incomplete: "
            f"{', '.join(remaining)}. Based on all evidence collected so far, output the "
            f"DIMENSION_COMPLETE block for each remaining dimension NOW. Do NOT call any more tools."
        )
        messages.append({"role": "user", "content": force_msg})
        response = llm.chat(messages)
        text = response.content or ""
        messages.append({"role": "assistant", "content": text})

        for parsed in _parse_dimension_results(text):
            dim = parsed["dimension"]
            if dim in dimensions and dim not in completed:
                completed[dim] = DimensionResult(
                    dimension=dim,
                    winner=parsed.get("winner", "tie"),
                    confidence=round(float(parsed.get("confidence", 0.0)), 4),
                    evidence=parsed.get("evidence", {}),
                    conclusion=parsed.get("conclusion", ""),
                    messages=_annotate_react_roles(list(messages)),
                )

    # Last resort: fill any still-missing dimensions with defaults
    for dim in dimensions:
        if dim not in completed:
            completed[dim] = DimensionResult(
                dimension=dim,
                winner="tie",
                confidence=0.0,
                evidence={},
                conclusion=f"Insufficient evidence (budget exhausted before reaching {dim}).",
                messages=_annotate_react_roles(list(messages)),
            )

    return [completed[d] for d in dimensions]


# ── LangGraph node ────────────────────────────────────────────────────────────

def run_inspector(state: AgentState, llm: BaseLLM, max_steps: int = 6) -> dict:
    inp = state["input"]
    series_A = inp.get("series_A", [])
    series_B = inp.get("series_B", [])

    reflection_type = state.get("reflection_type")
    feedback = state.get("reflection_feedback", "")
    perception_summary = state.get("perception_summary", "")
    tool_required = state.get("tool_required", [])

    # Determine which dimensions to process
    if reflection_type == "needs_recheck":
        dims_to_run = state.get("recheck_dimensions") or state.get("planned_dimensions", [])
        # Recheck dimensions always use tools
        tool_required_for_run = dims_to_run
    else:
        dims_to_run = state.get("planned_dimensions", [])
        tool_required_for_run = [d for d in tool_required if d in dims_to_run]

    # Merge results: keep old results for dims not being rechecked
    existing = {r["dimension"]: r for r in state.get("dimension_results", [])}

    new_results = _assess_all_dimensions(
        dims_to_run, series_A, series_B, llm, max_steps,
        perception_summary, feedback,
        tool_required=tool_required_for_run,
    )
    for result in new_results:
        existing[result["dimension"]] = result

    dimension_results = list(existing.values())

    return {
        "dimension_results": dimension_results,
        "recheck_count": state.get("recheck_count", 0) + (1 if reflection_type == "needs_recheck" else 0),
        "reflection_type": None,
        "reflection_feedback": None,
        "recheck_dimensions": None,
    }