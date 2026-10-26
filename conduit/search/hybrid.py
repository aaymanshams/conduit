"""Permission-aware hybrid search: BM25 keywords + vector similarity, fused with Reciprocal Rank Fusion.

Keyword search is good at exact identifiers ("TCK-501", "P1"); vector search
is good at paraphrases. RRF combines the two rankings without having to
calibrate their very different score scales against each other.

Access control is applied *before* ranking in both retrievers. A chunk the
user may not read is never scored, so it can never take a top-k slot, appear
in a result, or reach an LLM prompt.
"""

from __future__ import annotations

import threading

import numpy as np

from conduit.embeddings import Embedder
from conduit.models import Chunk, Hit, User
from conduit.search.bm25 import BM25Index
from conduit.search.text import tokenize

RRF_K = 60  # standard constant from Cormack et al. (2009)


class HybridIndex:
    def __init__(self, embedder: Embedder, candidates: int = 20):
        self.embedder = embedder
        self.candidates = candidates
        self.min_cosine = embedder.min_cosine
        self._lock = threading.Lock()
        self._chunks: list[Chunk] = []
        self._vectors = np.zeros((0, 0), dtype=np.float32)
        self._by_id: dict[int, int] = {}
        self._bm25 = BM25Index()

    def rebuild(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        bm25 = BM25Index()
        for c in chunks:
            bm25.add(c.id, tokenize(f"{c.title}\n{c.text}"))
        by_id = {c.id: i for i, c in enumerate(chunks)}
        with self._lock:  # swap atomically so concurrent searches see old or new, never half
            self._chunks, self._vectors, self._by_id, self._bm25 = chunks, vectors, by_id, bm25

    def __len__(self) -> int:
        return len(self._chunks)

    def search(self, query: str, user: User, k: int = 5, sources: set[str] | None = None) -> list[Hit]:
        with self._lock:
            chunks, vectors, by_id, bm25 = self._chunks, self._vectors, self._by_id, self._bm25
        if not chunks or k <= 0:
            return []

        visible = np.fromiter(
            (user.can_see(c.allowed_groups) and (sources is None or c.source in sources) for c in chunks),
            dtype=bool,
            count=len(chunks),
        )
        if not visible.any():
            return []

        # Keyword leg (C++ when available). Only chunks with a real term match come back.
        allowed_ids = [chunks[i].id for i in np.flatnonzero(visible)]
        keyword = bm25.search(tokenize(query), self.candidates, allowed_ids)
        keyword_rank = {cid: r for r, (cid, _) in enumerate(keyword)}

        # Vector leg: mask invisible rows to -inf before taking the top candidates.
        q = self.embedder.embed([query])[0]
        sims = vectors @ q
        masked = np.where(visible, sims, -np.inf)
        top = np.argsort(-masked)[: self.candidates]
        vector_rank = {
            chunks[i].id: r for r, i in enumerate(i for i in top if visible[i] and sims[i] >= self.min_cosine)
        }

        fused: dict[int, float] = {}
        for ranks in (keyword_rank, vector_rank):
            for cid, r in ranks.items():
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + r + 1)

        ordered = sorted(fused.items(), key=lambda x: (-x[1], x[0]))[:k]
        return [
            Hit(
                chunk=chunks[by_id[cid]],
                score=score,
                keyword_rank=keyword_rank.get(cid),
                vector_rank=vector_rank.get(cid),
                cosine=float(sims[by_id[cid]]),
            )
            for cid, score in ordered
        ]
