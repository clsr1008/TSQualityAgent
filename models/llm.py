"""
LLM interface — unified via chatanywhere third-party OpenAI-compatible API.
All models (GPT / Claude / Gemini) are called through the same endpoint.
"""
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

BASE_URL = "https://api.chatanywhere.tech/v1"


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


# ── Chatanywhere unified backend ──────────────────────────────────────────────

class ChatanywhereLLM(BaseLLM):
    """
    Supports any model available on chatanywhere:
      gpt-4o-mini, gpt-4o, claude-haiku-20240307, gemini-2.5-flash, ...

    API key is read from the OPENAI_API_KEY environment variable.
    """

    def __init__(self, model: str = "gpt-4o-mini", retries: int = 3, **kwargs):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package is required: pip install openai")

        self.client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=BASE_URL,
        )
        self.model = model
        self.retries = retries
        self.kwargs = kwargs

    def _create(self, **kwargs) -> object:
        """Call completions.create with retry logic."""
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
        return LLMResponse(content=resp.choices[0].message.content or "")

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
        return LLMResponse(content=msg.content or "", tool_calls=tool_calls)