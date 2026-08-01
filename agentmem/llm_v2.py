"""A salience scorer that doesn't mistake "I" for a proper noun.

``ExtractiveLLM.score_salience`` rewards a sentence for containing a
mid-sentence capitalized word, on the theory that named entities ("Toronto",
"UA482") mark a specific, memorable claim. The check is ``w[0].isupper()``
over every word after the first, and "I" is always capitalized whether or not
it introduces anything worth remembering. "Hmm, I'm not sure about that."
contains "I'm" and clears the 0.35 write threshold at 0.596. Run the real
scorer, no hand-picked overrides, over a synthetic support-chat stream and
297 of 800 filler turns (37%) get written to memory.

The fix is not "ignore capitalization"; capitalization is still the cheapest
signal this scorer has for named entities. It's excluding the specific
closed class of words that are capitalized for grammatical reasons having
nothing to do with being a name: "I" and its contractions, plus the
sentence-initial position, which the original correctly already skips via
``words[1:]`` but which still lets "I'm" at position 1 count as a proper noun
everywhere else in the sentence.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Sequence

from .llm import _STOP, _WORD

# Capitalized purely by grammar, never because they name something. Checked
# case-insensitively against the raw token, before the has_proper test.
_PRONOUN_FORMS = {
    "i", "i'm", "i've", "i'll", "i'd", "i's",
}


def _is_pronoun_form(word: str) -> bool:
    return word.lower() in _PRONOUN_FORMS


# Fixing has_proper alone was not enough: "Hmm, I'm not sure about that."
# still clears 0.35 on length and density once the false proper-noun credit
# is removed, because "not", "sure", and "about" are not stopwords in the
# original list and inflate content-word count for a sentence that conveys
# no retrievable information. These are hedges and discourse markers, the
# vocabulary people actually use to fill space while thinking, and they
# function as stopwords for salience even though they are not for grammar.
_HEDGE_WORDS = {
    "i'm", "i've", "i'll", "i'd", "i's",
    "not", "sure", "about", "really", "honestly", "guess", "suppose",
    "mean", "well", "think", "yeah", "okay", "ok", "hmm", "fine",
    "basically", "probably", "pretty", "kind", "sort", "actually",
    "just", "maybe", "might", "could", "should", "would", "fair",
    "enough", "right", "wrong", "opinion", "strong",
    "either", "know", "something", "anything",
}
_STOP_V2 = _STOP | _HEDGE_WORDS


class ExtractiveLLMV2:
    """Salience scoring and consolidation, with the pronoun-capitalization fix."""

    def score_salience(self, text: str) -> float:
        words = _WORD.findall(text)
        if not words:
            return 0.0
        content = [w for w in words if w.lower() not in _STOP_V2]
        if not content:
            return 0.0
        density = len(content) / len(words)
        length_factor = min(1.0, len(content) / 8)
        has_number = any(any(c.isdigit() for c in w) for w in words)
        # The fix: a mid-sentence capital only counts as a named-entity signal
        # if the word is not one of the grammatically-capitalized pronoun
        # forms. "I'm not sure" no longer reads as "contains a proper noun".
        has_proper = any(
            w[0].isupper() and not _is_pronoun_form(w) for w in words[1:]
        )
        score = 0.5 * length_factor
        score += 0.2 if has_number else 0.0
        score += 0.2 if has_proper else 0.0
        score += 0.1 * density
        return max(0.0, min(1.0, score))

    def summarize(self, texts: Sequence[str]) -> str:
        # Unchanged from ExtractiveLLM: the bug was in salience, not
        # summarization, and there is no reason to duplicate working code.
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
        keep_set = set(keep)
        return " ".join(s for s in sentences if s in keep_set)
