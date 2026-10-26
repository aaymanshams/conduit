"""Tokenization and chunking shared by the keyword index and the local embedder."""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    "a an and are as at be but by can do does for from has have how i if in into is it its "
    "me my no not of on or our so than that the their them then there these they this to "
    "us was we were what when where which who why will with you your".split()
)

_TOKEN = re.compile(r"[a-z0-9]+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _stem(token: str) -> str:
    # Deliberately tiny: plural folding only, so "tickets" matches "ticket".
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


def chunk_text(text: str, max_chars: int = 700) -> list[str]:
    """Split on paragraphs, then pack paragraphs into chunks of at most max_chars.

    A paragraph longer than max_chars is split on sentence boundaries. A single
    sentence longer than max_chars becomes its own (oversized) chunk rather
    than being cut mid-sentence.
    """
    pieces: list[str] = []
    for para in (p.strip() for p in re.split(r"\n\s*\n", text)):
        if not para:
            continue
        if len(para) <= max_chars:
            pieces.append(para)
        else:
            pieces.extend(_pack(_SENTENCE_END.split(para), max_chars, " "))
    return _pack(pieces, max_chars, "\n\n")


def _pack(parts: list[str], max_chars: int, sep: str) -> list[str]:
    chunks: list[str] = []
    current = ""
    for part in parts:
        candidate = f"{current}{sep}{part}" if current else part
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = part
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks
