"""BM25 keyword index. Uses the C++ extension when it was built, else this pure-Python twin."""

from __future__ import annotations

import math
from collections import Counter


class PyBM25Index:
    """Same maths and tie-breaking as cpp/bm25.cpp, so results match exactly."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._slot_of: dict[int, int] = {}
        self._doc_ids: list[int] = []
        self._doc_len: list[float] = []
        self._total_len = 0.0
        self._postings: dict[str, list[tuple[int, int]]] = {}

    def add(self, doc_id: int, tokens: list[str]) -> None:
        if doc_id in self._slot_of:
            raise ValueError(f"duplicate doc id {doc_id}")
        slot = len(self._doc_len)
        self._slot_of[doc_id] = slot
        self._doc_ids.append(doc_id)
        self._doc_len.append(float(len(tokens)))
        self._total_len += len(tokens)
        for term, freq in Counter(tokens).items():
            self._postings.setdefault(term, []).append((slot, freq))

    def search(self, query: list[str], k: int, allowed: list[int]) -> list[tuple[int, float]]:
        if not self._doc_len or k == 0:
            return []
        allowed_slots = {self._slot_of[i] for i in allowed if i in self._slot_of}
        n = float(len(self._doc_len))
        avgdl = self._total_len / n if self._total_len > 0 else 1.0
        scores: dict[int, float] = {}
        for term in dict.fromkeys(query):  # dedupe, keep order
            postings = self._postings.get(term)
            if not postings:
                continue
            df = float(len(postings))
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            for slot, freq in postings:
                if slot not in allowed_slots:
                    continue
                norm = self.k1 * (1.0 - self.b + self.b * self._doc_len[slot] / avgdl)
                scores[slot] = scores.get(slot, 0.0) + idf * freq * (self.k1 + 1.0) / (freq + norm)
        ranked = sorted(((self._doc_ids[s], sc) for s, sc in scores.items()), key=lambda x: (-x[1], x[0]))
        return ranked[:k]

    def __len__(self) -> int:
        return len(self._doc_len)


try:
    from conduit._bm25 import BM25Index  # type: ignore[attr-defined]

    BACKEND = "cpp"
except ImportError:  # extension not compiled on this machine
    BM25Index = PyBM25Index  # type: ignore[misc,assignment]
    BACKEND = "python"
