"""Embedding backends.

The default :class:`HashingEmbedder` is fully deterministic and dependency-free
so AgentMem runs (and its tests pass) with **zero API keys**. Swap in a real
provider by passing any object with an ``embed(texts) -> list[list[float]]``
method to :class:`~agentmem.memory.MemoryStore`.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, Sequence

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class Embedder(Protocol):
    """Anything that turns text into fixed-length vectors."""

    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        ...


class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder (the hashing trick).

    No model download, no network, stable across runs. Good enough for
    retrieval-by-similarity in tests and small deployments; replace with a
    real embedding model for production semantic quality.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in _tokenize(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h >> 8) & 1 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)
