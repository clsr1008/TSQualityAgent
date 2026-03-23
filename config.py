"""
Global configuration for the TSqualityAgent pipeline.
"""
from dataclasses import dataclass
from models.llm import BaseLLM, ChatanywhereLLM


@dataclass
class Config:
    # ── LLM ───────────────────────────────────────────────────────────────────
    # Any model supported by chatanywhere, e.g.:
    #   "gpt-4o-mini", "gpt-4o", "claude-haiku-20240307", "gemini-2.5-flash"
    model: str = "gpt-4o-mini"

    # ── Inspector ReAct loop ──────────────────────────────────────────────────
    max_steps_per_dimension: int = 6

    # ── Adjudicator reflection limits ─────────────────────────────────────────
    max_recheck: int = 2
    max_replan: int = 1


def build_llm(config: Config) -> BaseLLM:
    return ChatanywhereLLM(model=config.model)