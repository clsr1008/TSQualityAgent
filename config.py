"""
Global configuration for the TSqualityAgent pipeline.
All defaults live in main.py's argparse — Config itself has no hardcoded values.
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from models.llm import BaseLLM, OpenAICompatibleLLM, CHATANYWHERE_BASE_URL


@dataclass
class Config:
    model: str                      # e.g. "gpt-4o-mini", "Qwen/Qwen3-4B"
    base_url: str                   # OpenAI-compatible API base URL
    api_key: str                    # API key (empty → read from OPENAI_API_KEY env var)
    enable_thinking: bool           # Whether to enable Qwen3 thinking mode
    max_steps_per_dimension: int    # Inspector ReAct steps per dimension
    max_recheck: int                # Adjudicator → Inspector recheck limit
    max_replan: int                 # Adjudicator → Perceiver replan limit
    perceiver_model: str = ""       # Override model for Perceiver only (e.g. LoRA alias).
                                    # Falls back to `model` when empty.
    perceiver_base_url: str = ""    # Override base_url for Perceiver only.
                                    # Falls back to `base_url` when empty.

    @classmethod
    def from_args(cls, args) -> "Config":
        """Build a Config directly from argparse.Namespace."""
        return cls(
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            enable_thinking=args.enable_thinking,
            max_steps_per_dimension=args.max_steps,
            max_recheck=args.max_recheck,
            max_replan=args.max_replan,
            perceiver_model=getattr(args, "perceiver_model", "") or "",
            perceiver_base_url=getattr(args, "perceiver_base_url", "") or "",
        )


def build_llm(config: Config) -> BaseLLM:
    api_key = config.api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
    return OpenAICompatibleLLM(
        model=config.model,
        base_url=config.base_url,
        api_key=api_key,
        enable_thinking=config.enable_thinking,
    )


def build_perceiver_llm(config: Config) -> BaseLLM:
    """Return a dedicated LLM for the Perceiver agent.

    Uses perceiver_model / perceiver_base_url when specified,
    otherwise falls back to the default model / base_url.
    """
    api_key = config.api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
    return OpenAICompatibleLLM(
        model=config.perceiver_model or config.model,
        base_url=config.perceiver_base_url or config.base_url,
        api_key=api_key,
        enable_thinking=config.enable_thinking,
    )