"""Embedding providers. All return L2-normalised float32 rows, so a dot product is cosine similarity."""

from __future__ import annotations

import hashlib
import time
from collections import Counter
from typing import Protocol

import httpx
import numpy as np

from conduit.search.text import tokenize


class EmbeddingError(RuntimeError):
    pass


class Embedder(Protocol):
    model_id: str
    # Cosine below which a vector match is treated as noise. Each model's score
    # distribution differs, so this is a per-model heuristic, not a calibrated value.
    min_cosine: float

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


class HashingEmbedder:
    """Offline embedder: hashed unigrams and bigrams (the "hashing trick").

    It needs no network or model download, which keeps tests deterministic,
    but it is lexical: it will not match synonyms. Configure a real embedding
    model for semantic search.
    """

    min_cosine = 0.1

    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.model_id = f"local-hashing-{dim}"

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = tokenize(text)
            features = Counter(tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])])
            for feat, tf in features.items():
                h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "little")
                sign = 1.0 if h & 1 else -1.0
                out[row, (h >> 1) % self.dim] += sign * (1.0 + np.log(tf))
        return _normalise(out)


class OpenAICompatibleEmbedder:
    """Calls POST {base_url}/embeddings on any OpenAI-compatible API."""

    def __init__(self, base_url: str, api_key: str, model: str, client: httpx.Client | None = None, min_cosine: float = 0.25):
        self.model_id = model
        self.min_cosine = min_cosine
        self.client = client or httpx.Client(base_url=base_url, timeout=30.0)
        self.client.headers["Authorization"] = f"Bearer {api_key}"

    def embed(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        rows: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            data = self._post({"model": self.model_id, "input": batch})["data"]
            rows.extend(item["embedding"] for item in sorted(data, key=lambda d: d["index"]))
        if not rows:
            return np.zeros((0, 0), dtype=np.float32)
        return _normalise(np.asarray(rows, dtype=np.float32))

    def _post(self, payload: dict) -> dict:
        for attempt in range(4):
            try:
                resp = self.client.post("/embeddings", json=payload)
            except httpx.TransportError as exc:
                if attempt == 3:
                    raise EmbeddingError(f"embedding provider unreachable: {exc}") from exc
            else:
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code != 429 and resp.status_code < 500:
                    raise EmbeddingError(f"embedding provider returned {resp.status_code}")
            time.sleep(2**attempt)
        raise EmbeddingError("embedding provider kept failing; giving up after 4 attempts")
