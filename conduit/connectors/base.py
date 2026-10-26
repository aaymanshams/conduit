"""The connector contract every source system implements."""

from __future__ import annotations

import json
import os
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

from conduit.models import Document


class Connector(ABC):
    """A source system conduit can sync from.

    `fetch(since)` returns records changed at or after the cursor. It may
    return records that did not really change (cursors are inclusive so a
    record written in the same second as the last sync is never missed);
    the sync engine drops those by content hash. `list_ids()` returns every
    id that still exists, which is how deletions are detected.
    """

    name: str

    @abstractmethod
    def fetch(self, since: str | None) -> Iterator[Document]: ...

    @abstractmethod
    def list_ids(self) -> set[str]: ...


class JsonFileStore:
    """A JSON array on disk standing in for a SaaS system's database.

    Writes go to a temp file and are renamed into place, so a crash mid-write
    never leaves a half-written file behind.
    """

    def __init__(self, path: Path):
        self.path = path

    def read(self) -> list[dict]:
        return json.loads(self.path.read_text())

    def write(self, rows: list[dict]) -> None:
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(rows, f, indent=2)
        os.replace(tmp, self.path)


def changed_since(updated_at: str, since: str | None) -> bool:
    # ISO-8601 UTC timestamps in the same format compare correctly as strings.
    return since is None or updated_at >= since
