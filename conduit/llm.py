"""Chat model clients: any OpenAI-compatible endpoint, plus an offline extractive fallback."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Protocol

import httpx

from conduit.search.text import tokenize


class LLMError(RuntimeError):
    pass


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict
    raw_arguments: str = ""


@dataclass
class ChatResult:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)


class ChatModel(Protocol):
    model_id: str
    supports_tools: bool

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> ChatResult: ...


class OpenAICompatibleChat:
    """POST {base_url}/chat/completions. Works with OpenAI, Groq, Gemini's OpenAI endpoint, Ollama, vLLM..."""

    supports_tools = True

    def __init__(self, base_url: str, api_key: str, model: str, client: httpx.Client | None = None):
        self.model_id = model
        self.client = client or httpx.Client(base_url=base_url, timeout=60.0)
        self.client.headers["Authorization"] = f"Bearer {api_key}"

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> ChatResult:
        payload: dict = {"model": self.model_id, "messages": messages, "temperature": 0}
        if tools:
            payload["tools"] = tools
        message = self._post(payload)["choices"][0]["message"]
        calls = []
        for tc in message.get("tool_calls") or []:
            raw = tc["function"].get("arguments") or "{}"
            try:
                args = json.loads(raw)
            except json.JSONDecodeError:
                args = None  # surfaced to the model as a tool error, not a crash
            calls.append(ToolCall(id=tc["id"], name=tc["function"]["name"], arguments=args, raw_arguments=raw))
        return ChatResult(content=message.get("content"), tool_calls=calls)

    def _post(self, payload: dict) -> dict:
        for attempt in range(4):
            try:
                resp = self.client.post("/chat/completions", json=payload)
            except httpx.TransportError as exc:
                if attempt == 3:
                    raise LLMError(f"model provider unreachable: {exc}") from exc
            else:
                if resp.status_code == 200:
                    return resp.json()
                # Never include request headers in errors: they carry the API key.
                if resp.status_code != 429 and resp.status_code < 500:
                    raise LLMError(f"model provider returned {resp.status_code}")
            time.sleep(2**attempt)
        raise LLMError("model provider kept failing; giving up after 4 attempts")


_EXCERPT = re.compile(r'<excerpt id="(S\d+)"[^>]*>(.*?)</excerpt>', re.S)
_QUESTION = re.compile(r"<question>(.*?)</question>", re.S)


class OfflineChat:
    """No-network stand-in for the RAG step: quotes the excerpt sentence that best matches the question.

    It cannot call tools, so the agent falls back to plain RAG when this is active.
    """

    model_id = "offline-extractive"
    supports_tools = False

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> ChatResult:
        prompt = messages[-1]["content"]
        q = _QUESTION.search(prompt)
        q_terms = set(tokenize(q.group(1) if q else prompt))
        best: tuple[int, str, str] | None = None
        for sid, body in _EXCERPT.findall(prompt):
            for sentence in re.split(r"(?<=[.!?])\s+", body.strip()):
                overlap = len(q_terms & set(tokenize(sentence)))
                if overlap and (best is None or overlap > best[0]):
                    best = (overlap, sentence.strip(), sid)
        if best is None:
            return ChatResult(content="NOT_FOUND")
        return ChatResult(content=f"{best[1]} [{best[2]}]")
