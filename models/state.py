from typing import TypedDict, Optional


class DimensionResult(TypedDict):
    dimension: str          # e.g. "missing_value", "trend", "anomaly"
    score_A: float          # 0~1, higher = better quality
    score_B: float
    evidence: dict          # raw tool outputs
    conclusion: str         # brief text summary


class AgentState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────
    input: dict             # original JSON input (task_prompt, series_A/B, etc.)

    # ── Perceiver output ───────────────────────────────────────────────────
    planned_dimensions: list[str]   # dimensions to assess, e.g. ["missing_value", "trend"]

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
    inspector_messages: list
    adjudicator_messages: list
