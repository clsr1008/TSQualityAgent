"""
Global configuration for the TSqualityAgent pipeline.
All defaults live in main.py's argparse — Config itself has no hardcoded values.
"""
from __future__ import annotations
from dataclasses import dataclass
from models.llm import BaseLLM, ChatanywhereLLM


@dataclass
class Config:
    model: str                      # e.g. "gpt-4o-mini", "claude-haiku-20240307"
    max_steps_per_dimension: int    # Inspector ReAct steps per dimension
    max_recheck: int                # Adjudicator → Inspector recheck limit
    max_replan: int                 # Adjudicator → Perceiver replan limit

    @classmethod
    def from_args(cls, args) -> "Config":
        """Build a Config directly from argparse.Namespace."""
        return cls(
            model=args.model,
            max_steps_per_dimension=args.max_steps,
            max_recheck=args.max_recheck,
            max_replan=args.max_replan,
        )


def build_llm(config: Config) -> BaseLLM:
    return ChatanywhereLLM(model=config.model)