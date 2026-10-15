"""A thin chat-model interface, so every model in the benchmark is called the same way.

Any endpoint that speaks the OpenAI chat-completions format works here, which
includes Hugging Face Inference Providers (https://router.huggingface.co/v1).
"""

from __future__ import annotations

import json
import os
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResult:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0


class ChatModel(Protocol):
    name: str
    price_in: float  # US dollars per million prompt tokens
    price_out: float  # US dollars per million completion tokens

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> ChatResult: ...


class OpenAICompatModel:
    def __init__(
        self,
        name: str,
        model: str,
        base_url: str,
        api_key_env: str,
        price_in: float = 0.0,
        price_out: float = 0.0,
        temperature: float | None = 0.0,
        max_tokens: int = 600,
    ):
        from openai import OpenAI

        key = os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(f"set the {api_key_env} environment variable to use model {name!r}")
        self.name, self.model = name, model
        self.price_in, self.price_out = price_in, price_out
        self._temperature, self._max_tokens = temperature, max_tokens
        self._client = OpenAI(base_url=base_url, api_key=key, max_retries=3, timeout=120)

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> ChatResult:
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages, "max_tokens": self._max_tokens}
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        if tools:
            kwargs["tools"] = tools
        start = time.perf_counter()
        response = self._client.chat.completions.create(**kwargs)
        latency = time.perf_counter() - start
        message = response.choices[0].message
        calls = []
        for call in message.tool_calls or []:
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_invalid_json": call.function.arguments}
            calls.append(ToolCall(call.id, call.function.name, args if isinstance(args, dict) else {}))
        usage = response.usage
        return ChatResult(
            text=message.content or "",
            tool_calls=calls,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_s=latency,
        )


class ScriptedModel:
    """Replays canned responses. Used by the tests and for offline dry runs."""

    price_in = price_out = 0.0

    def __init__(self, responses: list[ChatResult], name: str = "scripted"):
        self.name = name
        self._responses = list(responses)
        self.calls: list[list[dict]] = []

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> ChatResult:
        self.calls.append([dict(m) for m in messages])
        if not self._responses:
            raise RuntimeError("ScriptedModel ran out of responses")
        return self._responses.pop(0)


def load_model(name: str, config_path: Path) -> OpenAICompatModel:
    with open(config_path, "rb") as f:
        models = tomllib.load(f).get("models", {})
    if name not in models:
        raise KeyError(f"model {name!r} is not in {config_path}; available: {', '.join(models)}")
    return OpenAICompatModel(name=name, **models[name])
