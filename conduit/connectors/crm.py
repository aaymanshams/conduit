"""CRM connector (Salesforce-style accounts), backed by a JSON export."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from conduit.connectors.base import Connector, JsonFileStore, changed_since
from conduit.models import Document

CRM_GROUPS = ["sales", "leadership"]


class CRMConnector(Connector):
    name = "crm"

    def __init__(self, path: Path):
        self.store = JsonFileStore(path)

    def fetch(self, since: str | None) -> Iterator[Document]:
        for acc in self.store.read():
            if changed_since(acc["updated_at"], since):
                yield self._to_document(acc)

    def list_ids(self) -> set[str]:
        return {acc["id"] for acc in self.store.read()}

    def find_account(self, query: str) -> dict | None:
        """Exact id, else case-insensitive name match."""
        q = query.strip().lower()
        accounts = self.store.read()
        for acc in accounts:
            if acc["id"].lower() == q:
                return acc
        for acc in accounts:
            if q and q in acc["name"].lower():
                return acc
        return None

    def list_accounts(self) -> list[dict]:
        return self.store.read()

    @staticmethod
    def _to_document(acc: dict) -> Document:
        text = (
            f"Account {acc['name']} ({acc['id']}) in {acc['industry']}. "
            f"Annual recurring revenue: ${acc['arr_usd']:,}. Account owner: {acc['owner']}. "
            f"Renewal date: {acc['renewal_date']}. Health: {acc['health'].replace('_', ' ')}.\n\n"
            f"{acc['notes']}"
        )
        return Document(
            source="crm",
            external_id=acc["id"],
            title=f"CRM account: {acc['name']}",
            text=text,
            allowed_groups=CRM_GROUPS,
            updated_at=acc["updated_at"],
            metadata={"account_id": acc["id"], "health": acc["health"], "renewal_date": acc["renewal_date"]},
        )
