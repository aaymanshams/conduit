"""Helpdesk connector (Zendesk-style tickets), backed by a JSON export.

Unlike the other connectors this one is two-way: the agent can create
tickets here once a human approves the action.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from conduit.connectors.base import Connector, JsonFileStore, changed_since
from conduit.models import Document

HELPDESK_GROUPS = ["support", "sales", "leadership"]


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class HelpdeskConnector(Connector):
    name = "helpdesk"

    def __init__(self, path: Path):
        self.store = JsonFileStore(path)

    def fetch(self, since: str | None) -> Iterator[Document]:
        for t in self.store.read():
            if changed_since(t["updated_at"], since):
                yield self._to_document(t)

    def list_ids(self) -> set[str]:
        return {t["id"] for t in self.store.read()}

    def list_tickets(self, account_id: str, status: str | None = "open") -> list[dict]:
        return [
            t
            for t in self.store.read()
            if t["account_id"] == account_id and (status is None or t["status"] == status)
        ]

    def create_ticket(self, account_id: str, subject: str, description: str, priority: str) -> dict:
        tickets = self.store.read()
        next_num = max((int(t["id"].split("-")[1]) for t in tickets), default=500) + 1
        ticket = {
            "id": f"TCK-{next_num}",
            "account_id": account_id,
            "subject": subject,
            "description": description,
            "status": "open",
            "priority": priority,
            "comments": [],
            "updated_at": utc_now(),
        }
        self.store.write([*tickets, ticket])
        return ticket

    @staticmethod
    def _to_document(t: dict) -> Document:
        comments = "\n".join(f"- {c}" for c in t["comments"]) or "- none"
        text = (
            f"Ticket {t['id']} for account {t['account_id']}. Status: {t['status']}. "
            f"Priority: {t['priority']}.\n\n{t['description']}\n\nComments:\n{comments}"
        )
        return Document(
            source="helpdesk",
            external_id=t["id"],
            title=f"Ticket {t['id']}: {t['subject']}",
            text=text,
            allowed_groups=HELPDESK_GROUPS,
            updated_at=t["updated_at"],
            metadata={"account_id": t["account_id"], "status": t["status"], "priority": t["priority"]},
        )
