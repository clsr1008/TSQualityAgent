from typing import TypedDict, Optional


class DimensionResult(TypedDict):
    dimension: str          # e.g. "missing_value", "trend", "anomaly"
    winner: str             # "A" | "B" | "tie"
    confidence: float       # 0~1, how certain the comparison is
    evidence: dict          # raw tool outputs
    conclusion: str         # brief text summary
    messages: list          # full ReAct message chain for this dimension


class AgentState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────
    input: dict             # original JSON input (task_prompt, series_A/B, etc.)

    # ── Perceiver output ───────────────────────────────────────────────────
    planned_dimensions: list[str]   # dimensions to assess, e.g. ["missing_value", "trend"]
    perception_summary: str         # Perceiver's natural-language observation of A vs B

    # ── Inspector output ───────────────────────────────────────────────────
    dimension_results: list[DimensionResult]

    # ── Adjudicator ────────────────────────────────────────────────────────
    reflection_type: Optional[str]          # "needs_recheck" | "needs_replan" | "done"
    reflection_feedback: Optional[str]      # natural-language feedback to the looped agent
    recheck_dimensions: Optional[list[str]] # specific dimensions to recheck (for needs_recheck)

    # ── Loop counters ──────────────────────────────────────────────────────
    recheck_count: int
    replan_count: int

    # ── Final output ───────────────────────────────────────────────────────
    final_result: Optional[dict]    # {winner, confidence, explanation}

    # ── Message history (per-agent conversation) ──────────────────────────
    perceiver_messages: list
    adjudicator_messages: list
