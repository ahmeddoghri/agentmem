"""LLM backends used for salience scoring and memory consolidation.

The default :class:`ExtractiveLLM` is offline and deterministic: it scores
salience with a lightweight information-density heuristic and consolidates by
extractive summarization. Pass any object exposing ``score_salience(text)`` and
``summarize(texts)`` to use a real model instead.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Protocol, Sequence

_WORD = re.compile(r"[A-Za-z0-9']+")
_STOP = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "to",
    "of", "in", "on", "at", "for", "with", "as", "by", "it", "this", "that",
    "i", "you", "he", "she", "they", "we", "my", "your", "me", "so", "do",
}


class LLM(Protocol):
    def score_salience(self, text: str) -> float:
        """Return importance in [0, 1], how worth remembering this is."""
        ...

    def summarize(self, texts: Sequence[str]) -> str:
        """Compress several memories into one consolidated memory."""
        ...


class ExtractiveLLM:
    """Offline, deterministic stand-in for a real LLM.

    ``score_salience`` rewards content words, named-entity-ish tokens, and
    numbers (facts worth keeping) and penalises pure filler. ``summarize``
    keeps the most information-dense sentences.
    """

    def score_salience(self, text: str) -> float:
        words = _WORD.findall(text)
        if not words:
            return 0.0
        content = [w for w in words if w.lower() not in _STOP]
        if not content:
            return 0.0
        density = len(content) / len(words)
        # absolute information matters: "ok sounds good" is dense but worthless
        length_factor = min(1.0, len(content) / 8)
        has_number = any(any(c.isdigit() for c in w) for w in words)
        has_proper = any(w[0].isupper() for w in words[1:])  # mid-sentence caps
        score = 0.5 * length_factor
        score += 0.2 if has_number else 0.0
        score += 0.2 if has_proper else 0.0
        score += 0.1 * density
        return max(0.0, min(1.0, score))

    def summarize(self, texts: Sequence[str]) -> str:
        joined = " ".join(texts)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", joined) if s.strip()]
        if len(sentences) <= 1:
            return joined.strip()
        freq = Counter(
            w.lower() for w in _WORD.findall(joined) if w.lower() not in _STOP
        )
        def sent_score(s: str) -> float:
            toks = [w.lower() for w in _WORD.findall(s)]
            return sum(freq[t] for t in toks) / (len(toks) or 1)
        ranked = sorted(sentences, key=sent_score, reverse=True)
        keep = ranked[: max(1, len(sentences) // 2)]
        # preserve original order for readability
        keep_set = set(keep)
        return " ".join(s for s in sentences if s in keep_set)
