"""
Agent 3 – Adjudicator
Responsibility: Aggregation + Reflection
  - Summarize all dimension results into a final A vs B judgment
  - Reflect: if evidence is insufficient, trigger recheck (back to Inspector)
             if key dimensions were missed, trigger replan (back to Perceiver)
  - Output: final_result = { winner, confidence, explanation }
            OR reflection_type = "needs_recheck" / "needs_replan"
"""
import json
from models.state import AgentState
from models.llm import BaseLLM
from tools import NumpyEncoder

SYSTEM_PROMPT = """You are the Adjudicator agent in a time series quality assessment pipeline.

Your job is to:
1. Review all quality dimension results for series A and B.
2. Synthesize a final comparative judgment: which series has better overall quality?
3. Reflect critically: is the evidence sufficient and consistent?

You have three possible responses:

Option 1 – Final judgment (when evidence is sufficient):
{
  "decision": "done",
  "winner": "A" | "B" | "tie",
  "confidence": <0.0 to 1.0>,
  "explanation": "<detailed reasoning covering each dimension>"
}

Option 2 – Recheck (when a specific dimension has weak or conflicting evidence):
{
  "decision": "needs_recheck",
  "recheck_dimensions": ["dim1", "dim2"],
  "feedback": "<what specifically needs re-examination>"
}

Option 3 – Replan (when important quality aspects were not assessed at all):
{
  "decision": "needs_replan",
  "feedback": "<which dimensions are missing and why they matter>"
}

Output ONLY valid JSON — no extra text."""


def run_adjudicator(
    state: AgentState,
    llm: BaseLLM,
    max_recheck: int = 2,
    max_replan: int = 1,
) -> dict:
    dimension_results = state.get("dimension_results", [])
    recheck_count = state.get("recheck_count", 0)
    replan_count = state.get("replan_count", 0)

    results_text = json.dumps(dimension_results, indent=2, cls=NumpyEncoder)
    user_content = f"""Here are the quality assessment results for all evaluated dimensions:

{results_text}

Provide your final judgment or reflection."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    response = llm.chat(messages)
    messages.append({"role": "assistant", "content": response.content})

    try:
        parsed = json.loads(response.content)
    except json.JSONDecodeError:
        # Fallback: treat as done with uncertain result
        return {
            "reflection_type": "done",
            "final_result": {
                "winner": "tie",
                "confidence": 0.0,
                "explanation": response.content,
            },
            "adjudicator_messages": messages,
        }

    decision = parsed.get("decision", "done")

    # Respect loop limits
    if decision == "needs_recheck" and recheck_count >= max_recheck:
        decision = "done"
        parsed["winner"] = parsed.get("winner", "tie")
        parsed["confidence"] = parsed.get("confidence", 0.3)
        parsed["explanation"] = (
            f"[Max rechecks reached] {parsed.get('feedback', '')} "
            + parsed.get("explanation", "")
        )

    if decision == "needs_replan" and replan_count >= max_replan:
        decision = "done"
        parsed["winner"] = "tie"
        parsed["confidence"] = 0.3
        parsed["explanation"] = (
            f"[Max replans reached] {parsed.get('feedback', '')} "
            + "Unable to reach a confident conclusion."
        )

    if decision == "done":
        return {
            "reflection_type": "done",
            "final_result": {
                "winner": parsed.get("winner", "tie"),
                "confidence": float(parsed.get("confidence", 0.5)),
                "explanation": parsed.get("explanation", ""),
            },
            "adjudicator_messages": messages,
        }

    elif decision == "needs_recheck":
        return {
            "reflection_type": "needs_recheck",
            "recheck_dimensions": parsed.get("recheck_dimensions", []),
            "reflection_feedback": parsed.get("feedback", ""),
            "final_result": None,
            "adjudicator_messages": messages,
        }

    elif decision == "needs_replan":
        return {
            "reflection_type": "needs_replan",
            "reflection_feedback": parsed.get("feedback", ""),
            "final_result": None,
            "adjudicator_messages": messages,
        }

    # Unexpected decision value
    return {
        "reflection_type": "done",
        "final_result": {
            "winner": "tie",
            "confidence": 0.0,
            "explanation": f"Unexpected decision: {decision}",
        },
        "adjudicator_messages": messages,
    }
