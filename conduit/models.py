"""Core data types shared by connectors, the store, search and the agent."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field


@dataclass(frozen=True)
class User:
    id: str
    name: str
    groups: frozenset[str]

    def can_see(self, allowed_groups: list[str] | frozenset[str]) -> bool:
        return bool(self.groups.intersection(allowed_groups))

    def in_any(self, *groups: str) -> bool:
        return bool(self.groups.intersection(groups))


@dataclass
class Document:
    """One record from a source system, normalised into a common shape.

    Every connector maps its own records (a CRM account, a ticket, a wiki page,
    a GitHub issue) into this. `allowed_groups` carries the source system's
    permissions into the index, so search can enforce them.
    """

    source: str
    external_id: str
    title: str
    text: str
    allowed_groups: list[str]
    updated_at: str
    url: str | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.source}:{self.external_id}"

    @property
    def content_hash(self) -> str:
        # Anything that changes what is indexed or who may see it must change the hash.
        payload = json.dumps(
            [self.title, self.text, sorted(self.allowed_groups), self.url, self.metadata],
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class Chunk:
    id: int
    doc_id: str
    ord: int
    text: str
    title: str
    source: str
    url: str | None
    allowed_groups: frozenset[str]


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    keyword_rank: int | None
    vector_rank: int | None
    cosine: float
