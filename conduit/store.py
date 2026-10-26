"""SQLite persistence: documents, chunks with embeddings, sync cursors, the audit log and the approval queue."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from conduit.models import Chunk, Document

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id             TEXT PRIMARY KEY,
    source         TEXT NOT NULL,
    external_id    TEXT NOT NULL,
    title          TEXT NOT NULL,
    url            TEXT,
    allowed_groups TEXT NOT NULL,
    metadata       TEXT NOT NULL,
    content_hash   TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    synced_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS documents_source ON documents(source);

CREATE TABLE IF NOT EXISTS chunks (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id    TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ord       INTEGER NOT NULL,
    text      TEXT NOT NULL,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id);

CREATE TABLE IF NOT EXISTS sync_state (
    source      TEXT PRIMARY KEY,
    cursor      TEXT,
    last_run_at TEXT NOT NULL,
    last_stats  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_log (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      TEXT NOT NULL,
    user_id TEXT NOT NULL,
    event   TEXT NOT NULL,
    detail  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS actions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at   TEXT NOT NULL,
    requested_by TEXT NOT NULL,
    tool         TEXT NOT NULL,
    args         TEXT NOT NULL,
    status       TEXT NOT NULL CHECK (status IN ('pending', 'approved', 'rejected', 'executed', 'failed')),
    decided_by   TEXT,
    decided_at   TEXT,
    result       TEXT
);
"""


def now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class Store:
    def __init__(self, path: Path | str):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        # One connection shared across API worker threads, serialised by a lock.
        self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._lock = threading.RLock()
        self._conn.executescript(SCHEMA)

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")

    def _query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    # --- meta -------------------------------------------------------------

    def get_meta(self, key: str) -> str | None:
        rows = self._query("SELECT value FROM meta WHERE key = ?", (key,))
        return rows[0]["value"] if rows else None

    def set_meta(self, key: str, value: str) -> None:
        with self.transaction() as c:
            c.execute("INSERT INTO meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))

    def reset_index(self) -> None:
        """Drop all indexed content and cursors so the next sync rebuilds everything."""
        with self.transaction() as c:
            c.execute("DELETE FROM chunks")
            c.execute("DELETE FROM documents")
            c.execute("DELETE FROM sync_state")

    # --- documents and chunks ---------------------------------------------

    def content_hash(self, doc_id: str) -> str | None:
        rows = self._query("SELECT content_hash FROM documents WHERE id = ?", (doc_id,))
        return rows[0]["content_hash"] if rows else None

    def doc_ids(self, source: str) -> set[str]:
        return {r["id"] for r in self._query("SELECT id FROM documents WHERE source = ?", (source,))}

    def upsert_document(self, doc: Document, chunk_texts: list[str], embeddings: np.ndarray) -> None:
        """Replace a document and all its chunks in one transaction."""
        if len(chunk_texts) != len(embeddings):
            raise ValueError("one embedding per chunk is required")
        with self.transaction() as c:
            c.execute("DELETE FROM chunks WHERE doc_id = ?", (doc.id,))
            c.execute(
                """INSERT INTO documents(id, source, external_id, title, url, allowed_groups, metadata,
                                         content_hash, updated_at, synced_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                     title = excluded.title, url = excluded.url, allowed_groups = excluded.allowed_groups,
                     metadata = excluded.metadata, content_hash = excluded.content_hash,
                     updated_at = excluded.updated_at, synced_at = excluded.synced_at""",
                (
                    doc.id, doc.source, doc.external_id, doc.title, doc.url,
                    json.dumps(sorted(doc.allowed_groups)), json.dumps(doc.metadata),
                    doc.content_hash, doc.updated_at, now(),
                ),
            )
            c.executemany(
                "INSERT INTO chunks(doc_id, ord, text, embedding) VALUES (?, ?, ?, ?)",
                [
                    (doc.id, i, text, np.asarray(vec, dtype=np.float32).tobytes())
                    for i, (text, vec) in enumerate(zip(chunk_texts, embeddings))
                ],
            )

    def delete_document(self, doc_id: str) -> None:
        with self.transaction() as c:
            c.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

    def load_chunks(self) -> tuple[list[Chunk], np.ndarray]:
        rows = self._query(
            """SELECT ch.id, ch.doc_id, ch.ord, ch.text, ch.embedding, d.title, d.source, d.url, d.allowed_groups
               FROM chunks ch JOIN documents d ON d.id = ch.doc_id ORDER BY ch.id"""
        )
        chunks = [
            Chunk(
                id=r["id"], doc_id=r["doc_id"], ord=r["ord"], text=r["text"], title=r["title"],
                source=r["source"], url=r["url"], allowed_groups=frozenset(json.loads(r["allowed_groups"])),
            )
            for r in rows
        ]
        vectors = (
            np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
            if rows
            else np.zeros((0, 0), dtype=np.float32)
        )
        return chunks, vectors

    def counts(self) -> dict[str, int]:
        rows = self._query("SELECT source, COUNT(*) AS n FROM documents GROUP BY source ORDER BY source")
        return {r["source"]: r["n"] for r in rows}

    # --- sync cursors -----------------------------------------------------

    def get_cursor(self, source: str) -> str | None:
        rows = self._query("SELECT cursor FROM sync_state WHERE source = ?", (source,))
        return rows[0]["cursor"] if rows else None

    def set_cursor(self, source: str, cursor: str | None, stats: dict) -> None:
        with self.transaction() as c:
            c.execute(
                """INSERT INTO sync_state(source, cursor, last_run_at, last_stats) VALUES (?, ?, ?, ?)
                   ON CONFLICT(source) DO UPDATE SET cursor = excluded.cursor,
                     last_run_at = excluded.last_run_at, last_stats = excluded.last_stats""",
                (source, cursor, now(), json.dumps(stats)),
            )

    def sync_status(self) -> list[dict]:
        rows = self._query("SELECT * FROM sync_state ORDER BY source")
        return [{**dict(r), "last_stats": json.loads(r["last_stats"])} for r in rows]

    # --- audit log --------------------------------------------------------

    def audit(self, user_id: str, event: str, detail: dict) -> None:
        with self.transaction() as c:
            c.execute(
                "INSERT INTO audit_log(ts, user_id, event, detail) VALUES (?, ?, ?, ?)",
                (now(), user_id, event, json.dumps(detail, default=str)),
            )

    def audit_entries(self, limit: int = 100) -> list[dict]:
        rows = self._query("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,))
        return [{**dict(r), "detail": json.loads(r["detail"])} for r in rows]

    # --- approval queue ---------------------------------------------------

    def create_action(self, requested_by: str, tool: str, args: dict) -> int:
        with self.transaction() as c:
            cur = c.execute(
                "INSERT INTO actions(created_at, requested_by, tool, args, status) VALUES (?, ?, ?, ?, 'pending')",
                (now(), requested_by, tool, json.dumps(args)),
            )
            return int(cur.lastrowid)

    def get_action(self, action_id: int) -> dict | None:
        rows = self._query("SELECT * FROM actions WHERE id = ?", (action_id,))
        return self._action_row(rows[0]) if rows else None

    def list_actions(self, status: str | None = None) -> list[dict]:
        if status:
            rows = self._query("SELECT * FROM actions WHERE status = ? ORDER BY id", (status,))
        else:
            rows = self._query("SELECT * FROM actions ORDER BY id")
        return [self._action_row(r) for r in rows]

    def decide_action(self, action_id: int, decided_by: str, status: str) -> bool:
        """Atomically move a pending action to approved/rejected. False if it was not pending."""
        with self.transaction() as c:
            cur = c.execute(
                "UPDATE actions SET status = ?, decided_by = ?, decided_at = ? WHERE id = ? AND status = 'pending'",
                (status, decided_by, now(), action_id),
            )
            return cur.rowcount == 1

    def finish_action(self, action_id: int, status: str, result: dict) -> None:
        with self.transaction() as c:
            c.execute("UPDATE actions SET status = ?, result = ? WHERE id = ?", (status, json.dumps(result), action_id))

    @staticmethod
    def _action_row(r: sqlite3.Row) -> dict:
        d = dict(r)
        d["args"] = json.loads(d["args"])
        d["result"] = json.loads(d["result"]) if d["result"] else None
        return d
