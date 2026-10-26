"""Outbound Slack messages via an incoming webhook, or a local outbox file when none is configured."""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from conduit.store import now


class NotifyError(RuntimeError):
    pass


class SlackNotifier:
    def __init__(self, webhook_url: str | None, outbox: Path, client: httpx.Client | None = None):
        self.webhook_url = webhook_url
        self.outbox = outbox
        self.client = client or httpx.Client(timeout=10.0)

    def post(self, channel: str, text: str) -> dict:
        if self.webhook_url:
            resp = self.client.post(self.webhook_url, json={"channel": channel, "text": text})
            if resp.status_code != 200:
                raise NotifyError(f"Slack webhook returned {resp.status_code}")
            return {"delivered_to": "slack", "channel": channel}
        self.outbox.parent.mkdir(parents=True, exist_ok=True)
        with self.outbox.open("a") as f:
            f.write(json.dumps({"ts": now(), "channel": channel, "text": text}) + "\n")
        return {"delivered_to": "outbox", "channel": channel, "path": str(self.outbox)}
