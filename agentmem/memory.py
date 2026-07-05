"""Bounded, self-consolidating long-term memory for LLM agents.

Design follows two ideas from the mid-2026 agent-memory literature:

* **Memory as a cognitive skill** (AutoMem, 2026) — the agent *decides* what is
  worth storing and actively consolidates related items instead of dumping every
  turn into a vector store.
* **Bounded memory** (AgenticSTS, 2026) — real agents operate under a fixed
  budget, so retention is a policy problem: what to keep, merge, or forget.

``MemoryStore`` ties these together: writes are salience-gated, retrieval blends
semantic similarity with recency and importance, and when the store exceeds its
budget it consolidates the most redundant items before evicting the weakest.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

from .embeddings import Embedder, HashingEmbedder, cosine
from .llm import LLM, ExtractiveLLM


@dataclass
class MemoryItem:
    id: int
    content: str
    embedding: list[float]
    salience: float
    created_at: float
    last_used: float
    uses: int = 0
    sources: list[int] = field(default_factory=list)  # ids merged into this one

    def strength(self, now: float, half_life: float) -> float:
        """Decayed retention strength — importance that fades unless reused."""
        age = max(0.0, now - self.last_used)
        decay = 0.5 ** (age / half_life) if half_life > 0 else 1.0
        return self.salience * decay * (1.0 + 0.1 * self.uses)


@dataclass
class Retrieved:
    item: MemoryItem
    score: float


class MemoryStore:
    """A bounded long-term memory with salience-gated writes and consolidation.

    Parameters
    ----------
    capacity:
        Maximum number of items retained. Exceeding it triggers consolidation
        then eviction.
    write_threshold:
        Minimum salience required to store a new observation.
    half_life_seconds:
        Recency half-life used for decay-aware retrieval and eviction.
    dedup_threshold:
        Cosine similarity above which a *write* is treated as a near-exact repeat
        and reinforces the existing memory instead of creating a new item.
    merge_threshold:
        Looser similarity above which *consolidation* fuses related items into a
        single summary. Should be <= ``dedup_threshold``.
    """

    def __init__(
        self,
        capacity: int = 128,
        write_threshold: float = 0.35,
        half_life_seconds: float = 3600.0,
        dedup_threshold: float = 0.95,
        merge_threshold: float = 0.82,
        embedder: Optional[Embedder] = None,
        llm: Optional[LLM] = None,
        clock=time.time,
    ) -> None:
        self.capacity = capacity
        self.write_threshold = write_threshold
        self.half_life = half_life_seconds
        self.dedup_threshold = dedup_threshold
        self.merge_threshold = merge_threshold
        self.embedder = embedder or HashingEmbedder()
        self.llm = llm or ExtractiveLLM()
        self._clock = clock
        self._items: dict[int, MemoryItem] = {}
        self._next_id = 0

    # ------------------------------------------------------------------ write
    def write(self, content: str, salience: Optional[float] = None) -> Optional[MemoryItem]:
        """Store an observation if it clears the salience gate.

        Returns the stored (or updated) item, or ``None`` if it was judged not
        worth remembering. Near-duplicate writes reinforce the existing memory
        instead of creating a new one.
        """
        content = content.strip()
        if not content:
            return None
        if salience is None:
            salience = self.llm.score_salience(content)
        if salience < self.write_threshold:
            return None

        emb = self.embedder.embed([content])[0]
        dup = self._nearest(emb)
        if dup and cosine(emb, dup.embedding) >= self.dedup_threshold:
            # reinforce rather than duplicate
            dup.salience = max(dup.salience, salience)
            dup.uses += 1
            dup.last_used = self._clock()
            return dup

        now = self._clock()
        item = MemoryItem(
            id=self._next_id,
            content=content,
            embedding=emb,
            salience=salience,
            created_at=now,
            last_used=now,
        )
        self._items[item.id] = item
        self._next_id += 1
        if len(self._items) > self.capacity:
            self._compact()
        return item

    # -------------------------------------------------------------- retrieve
    def retrieve(
        self,
        query: str,
        k: int = 5,
        recency_weight: float = 0.2,
        salience_weight: float = 0.2,
    ) -> list[Retrieved]:
        """Return the top-``k`` memories for a query.

        Score = similarity + recency_weight*recency + salience_weight*salience.
        Retrieving an item counts as a use, strengthening it against decay.
        """
        if not self._items:
            return []
        q = self.embedder.embed([query])[0]
        now = self._clock()
        scored: list[Retrieved] = []
        for it in self._items.values():
            sim = cosine(q, it.embedding)
            age = max(0.0, now - it.last_used)
            recency = 0.5 ** (age / self.half_life) if self.half_life > 0 else 1.0
            score = sim + recency_weight * recency + salience_weight * it.salience
            scored.append(Retrieved(it, score))
        scored.sort(key=lambda r: r.score, reverse=True)
        top = scored[:k]
        for r in top:
            r.item.uses += 1
            r.item.last_used = now
        return top

    # -------------------------------------------------------------- maintain
    def _compact(self) -> None:
        """Consolidate redundant memories, then evict the weakest to budget."""
        self.consolidate()
        while len(self._items) > self.capacity:
            now = self._clock()
            weakest = min(self._items.values(), key=lambda i: i.strength(now, self.half_life))
            del self._items[weakest.id]

    def consolidate(self) -> int:
        """Merge highly-similar memories into summaries. Returns #merges done."""
        items = list(self._items.values())
        merges = 0
        used: set[int] = set()
        for i, a in enumerate(items):
            if a.id in used or a.id not in self._items:
                continue
            group = [a]
            for b in items[i + 1 :]:
                if b.id in used or b.id not in self._items:
                    continue
                if cosine(a.embedding, b.embedding) >= self.merge_threshold:
                    group.append(b)
                    used.add(b.id)
            if len(group) > 1:
                self._merge(group)
                merges += 1
        return merges

    def _merge(self, group: Sequence[MemoryItem]) -> None:
        summary = self.llm.summarize([g.content for g in group])
        emb = self.embedder.embed([summary])[0]
        now = self._clock()
        merged = MemoryItem(
            id=self._next_id,
            content=summary,
            embedding=emb,
            salience=max(g.salience for g in group),
            created_at=min(g.created_at for g in group),
            last_used=now,
            uses=sum(g.uses for g in group),
            sources=[s for g in group for s in (g.sources or [g.id])],
        )
        self._next_id += 1
        for g in group:
            self._items.pop(g.id, None)
        self._items[merged.id] = merged

    def _nearest(self, emb: list[float]) -> Optional[MemoryItem]:
        best, best_sim = None, -1.0
        for it in self._items.values():
            s = cosine(emb, it.embedding)
            if s > best_sim:
                best, best_sim = it, s
        return best

    # --------------------------------------------------------------- dunders
    def __len__(self) -> int:
        return len(self._items)

    def all(self) -> list[MemoryItem]:
        return list(self._items.values())
