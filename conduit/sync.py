"""Incremental sync: pull changed records from a connector, re-embed only what changed, detect deletions."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from conduit.connectors.base import Connector
from conduit.embeddings import Embedder
from conduit.search.text import chunk_text
from conduit.store import Store


@dataclass
class SyncStats:
    source: str
    fetched: int = 0
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    deleted: int = 0
    chunks_embedded: int = 0


def sync_source(store: Store, connector: Connector, embedder: Embedder) -> SyncStats:
    stats = SyncStats(source=connector.name)
    cursor = store.get_cursor(connector.name)
    # The new cursor is the newest `updated_at` the source reported, not our
    # wall clock, so clock drift between systems cannot make us skip records.
    newest = cursor

    for doc in connector.fetch(cursor):
        stats.fetched += 1
        if newest is None or doc.updated_at > newest:
            newest = doc.updated_at
        existing = store.content_hash(doc.id)
        if existing == doc.content_hash:
            stats.unchanged += 1
            continue
        texts = chunk_text(doc.text) or [doc.title]
        # Embed outside any DB transaction: provider calls are slow and can fail.
        vectors = embedder.embed([f"{doc.title}\n{t}" for t in texts])
        store.upsert_document(doc, texts, vectors)
        stats.chunks_embedded += len(texts)
        if existing is None:
            stats.added += 1
        else:
            stats.updated += 1

    live = {f"{connector.name}:{ext_id}" for ext_id in connector.list_ids()}
    for gone in store.doc_ids(connector.name) - live:
        store.delete_document(gone)
        stats.deleted += 1

    # Only advance the cursor once everything above succeeded; a failed run is retried from the old cursor.
    store.set_cursor(connector.name, newest, asdict(stats))
    return stats
