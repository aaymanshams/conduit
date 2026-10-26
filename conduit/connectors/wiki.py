"""Wiki connector: a folder of Markdown pages with a small front-matter header.

    ---
    title: Refund and Credit Policy
    groups: everyone            # comma-separated; who may read the page
    updated_at: 2026-08-01T00:00:00Z
    ---
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from conduit.connectors.base import Connector, changed_since
from conduit.models import Document


def parse_page(raw: str) -> tuple[dict[str, str], str]:
    if not raw.startswith("---\n"):
        raise ValueError("page has no front matter")
    header, _, body = raw[4:].partition("\n---\n")
    meta = {}
    for line in header.splitlines():
        key, sep, value = line.partition(":")
        if sep:
            meta[key.strip()] = value.strip()
    for required in ("title", "groups", "updated_at"):
        if not meta.get(required):
            raise ValueError(f"front matter is missing '{required}'")
    return meta, body.strip()


class WikiConnector(Connector):
    name = "wiki"

    def __init__(self, folder: Path):
        self.folder = folder

    def fetch(self, since: str | None) -> Iterator[Document]:
        for path in sorted(self.folder.glob("*.md")):
            meta, body = parse_page(path.read_text())
            if not changed_since(meta["updated_at"], since):
                continue
            yield Document(
                source="wiki",
                external_id=path.stem,
                title=meta["title"],
                text=body,
                allowed_groups=[g.strip() for g in meta["groups"].split(",") if g.strip()],
                updated_at=meta["updated_at"],
                url=f"wiki/{path.name}",
            )

    def list_ids(self) -> set[str]:
        return {p.stem for p in self.folder.glob("*.md")}
