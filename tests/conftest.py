from __future__ import annotations

import json
from collections.abc import Callable

import pytest

from conduit.config import PROJECT_ROOT, Settings
from conduit.llm import ChatResult, OfflineChat, ToolCall
from conduit.service import Conduit


class SpyLLM:
    """Records every prompt and replies with canned content."""

    model_id = "spy"
    supports_tools = False

    def __init__(self, reply: str | Callable[[list[dict]], str] = "NOT_FOUND"):
        self.reply = reply
        self.calls: list[list[dict]] = []

    def chat(self, messages, tools=None):
        self.calls.append(messages)
        content = self.reply(messages) if callable(self.reply) else self.reply
        return ChatResult(content=content)


class ScriptedLLM:
    """Plays back a fixed list of turns: each is either a list of (tool, args) calls or a final string."""

    model_id = "scripted"
    supports_tools = True

    def __init__(self, turns: list):
        self.turns = list(turns)
        self.seen: list[list[dict]] = []

    def chat(self, messages, tools=None):
        self.seen.append(list(messages))
        turn = self.turns.pop(0) if self.turns else "done"
        if isinstance(turn, str):
            return ChatResult(content=turn)
        return ChatResult(
            content=None,
            tool_calls=[
                ToolCall(id=f"call_{i}", name=name, arguments=args, raw_arguments=json.dumps(args))
                for i, (name, args) in enumerate(turn)
            ],
        )


@pytest.fixture
def settings(tmp_path):
    return Settings.from_env(
        seed_dir=PROJECT_ROOT / "data" / "acme",
        data_dir=tmp_path / "var",
        llm_api_key=None,
        embed_model="local",
        github_repo=None,
        slack_webhook_url=None,
    )


@pytest.fixture
def make_conduit(settings):
    def make(llm=None, sync=True) -> Conduit:
        c = Conduit(settings, llm=llm or OfflineChat())
        if sync:
            c.sync()
        return c

    return make
