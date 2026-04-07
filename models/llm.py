"""
LLM interface — supports any OpenAI-compatible API endpoint.

Default backend: chatanywhere (cloud, closed-source models).
Local backend:   vLLM serving Qwen/Qwen3-4B (or any other model) at localhost:8000.
"""
import json
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

CHATANYWHERE_BASE_URL = "https://api.chatanywhere.tech/v1"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class BaseLLM(ABC):
    @abstractmethod
    def chat(self, messages: list[dict]) -> LLMResponse: ...

    def chat_with_tools(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        return self.chat(messages)


# ── OpenAI-compatible backend (cloud or local vLLM) ───────────────────────────

class OpenAICompatibleLLM(BaseLLM):
    """
    Works with any OpenAI-compatible endpoint:
      - Cloud (chatanywhere): base_url=CHATANYWHERE_BASE_URL, api_key from OPENAI_API_KEY
      - Local vLLM:           base_url="http://localhost:8000/v1", api_key="EMPTY"

    Qwen3 thinking-mode output (<think>...</think>) is automatically stripped
    from response.content so all agents always receive clean text.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: str = CHATANYWHERE_BASE_URL,
        api_key: str = "",
        retries: int = 3,
        enable_thinking: bool = False,
        **kwargs,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package is required: pip install openai")

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.client = OpenAI(api_key=resolved_key, base_url=base_url)
        self.model = model
        self.retries = retries
        self.enable_thinking = enable_thinking
        self.kwargs = kwargs

    @staticmethod
    def _strip_thinking(content: str) -> str:
        """Remove <think>...</think> blocks produced by Qwen3 thinking mode."""
        return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    def _create(self, **kwargs) -> object:
        """Call completions.create with retry logic."""
        if not self.enable_thinking:
            extra = kwargs.setdefault("extra_body", {})
            extra.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
        retry_count = 0
        while True:
            try:
                return self.client.chat.completions.create(**kwargs)
            except Exception as error:
                err_str = str(error)
                if "Please retry after" in err_str:
                    wait = int(err_str.split("Please retry after ")[1].split(" second")[0]) + 1
                    print(f"[LLM] Rate limited, waiting {wait}s…")
                    time.sleep(wait)
                elif retry_count < self.retries:
                    retry_count += 1
                    print(f"[LLM] Retry {retry_count}/{self.retries} ({error})")
                    time.sleep(5)
                else:
                    raise

    def chat(self, messages: list[dict]) -> LLMResponse:
        resp = self._create(
            model=self.model,
            messages=messages,
            **self.kwargs,
        )
        content = self._strip_thinking(resp.choices[0].message.content or "")
        return LLMResponse(content=content)

    def chat_with_tools(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        resp = self._create(
            model=self.model,
            messages=messages,
            tools=tools,
            **self.kwargs,
        )
        msg = resp.choices[0].message
        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    )
                )
        content = self._strip_thinking(msg.content or "")
        return LLMResponse(content=content, tool_calls=tool_calls)


# Backward-compatible alias
ChatanywhereLLM = OpenAICompatibleLLM