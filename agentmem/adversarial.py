"""Labeled sentences for measuring the salience gate directly.

The bundled recall benchmark never actually exercises the gate: it calls
``mem.write(fact, salience=0.9)`` and ``mem.write(chatter, salience=0.05)``
with hand-picked scores, and runs with ``write_threshold=0.0`` so both pass
through regardless. "Salience-gated writes" is the first thing the README's
own headline promises, and the flagship benchmark never lets the scorer make
a single gating decision.

Run the real scorer end to end (no override, default ``write_threshold``) on
the same synthetic stream and 297 of 800 distractor turns, 37%, get written
anyway. The cause: ``has_proper`` flags any mid-sentence capitalized word as a
named-entity signal, and "I" is always capitalized. "Hmm, I'm not sure about
that." contains "I'm" and scores 0.596, comfortably over the 0.35 threshold,
for filler the project's own README uses as its running example of what
should never be remembered.

These sentences are labeled ``worth_remembering`` by a plain reading, split
into ``fact`` (a specific, useful claim) and ``filler`` (the kind of thing
that fills a transcript without saying anything). None were written by
reading the scorer's source; they were written the way people actually type
when thinking out loud, which is exactly the register the scorer needs to
reject.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal


@dataclass(frozen=True)
class LabeledUtterance:
    text: str
    worth_remembering: bool
    kind: Literal["fact", "filler"]
    note: str = ""


ADVERSARIAL_CORPUS: List[LabeledUtterance] = [
    # --- facts: specific, worth keeping -------------------------------------
    LabeledUtterance("My flight number is UA482 and it departs at 6pm.", True, "fact"),
    LabeledUtterance("The deployment key is rotated every 90 days per policy.", True, "fact"),
    LabeledUtterance("Our production database is hosted in the us-east-1 region.", True, "fact"),
    LabeledUtterance("I live at 42 Birchwood Lane in Toronto.", True, "fact",
                     "starts with the pronoun I but is a genuine fact"),
    LabeledUtterance("I was born on March 3rd, 1990.", True, "fact"),
    LabeledUtterance("The client's account ID is 88213 and their plan renews in June.", True, "fact"),
    LabeledUtterance("I work as a structural engineer for Kessler and Partners.", True, "fact"),
    LabeledUtterance("The API rate limit is 500 requests per minute per key.", True, "fact"),

    # --- filler containing the word "I": the exact leak found --------------
    LabeledUtterance("Hmm, I'm not sure about that.", False, "filler",
                     "0.596 under the original scorer; well over the 0.35 threshold"),
    LabeledUtterance("Yeah I think that's basically right.", False, "filler"),
    LabeledUtterance("I guess we could try that approach.", False, "filler"),
    LabeledUtterance("I think I'll just go with the default.", False, "filler"),
    LabeledUtterance("I'm honestly not sure what to say here.", False, "filler"),
    LabeledUtterance("I mean, I could be wrong about this.", False, "filler"),
    LabeledUtterance("Well, I don't really have a strong opinion.", False, "filler"),
    LabeledUtterance("I suppose that makes sense to me.", False, "filler"),
    LabeledUtterance("I'll just go with whatever you think is best.", False, "filler"),
    LabeledUtterance("I guess that's fine, sure.", False, "filler"),

    # --- filler with no "I" at all: the easy case, should already work -----
    LabeledUtterance("Sounds good to me.", False, "filler"),
    LabeledUtterance("The weather is mild today.", False, "filler"),
    LabeledUtterance("Let's continue.", False, "filler"),
    LabeledUtterance("Okay, noted.", False, "filler"),
    LabeledUtterance("Interesting, tell me more.", False, "filler"),
    LabeledUtterance("Sure, that works for me, thanks!", False, "filler"),
]


@dataclass(frozen=True)
class LabeledHoldout:
    text: str
    worth_remembering: bool


# Written after ExtractiveLLMV2 was frozen against ADVERSARIAL_CORPUS above.
# Evaluated exactly once.
HOLDOUT_CORPUS: List[LabeledHoldout] = [
    LabeledHoldout("I need the invoice sent to billing@northgate.example by Friday.", True),
    LabeledHoldout("The warehouse address changed to 900 Industrial Parkway last month.", True),
    LabeledHoldout("I set the thermostat schedule to 68 degrees on weekday mornings.", True),
    LabeledHoldout("Our contract renews automatically unless cancelled 30 days prior.", True),
    LabeledHoldout("I really don't know, it could go either way honestly.", False),
    LabeledHoldout("I guess I'll take your word for it.", False),
    LabeledHoldout("Fair enough, I can live with that.", False),
    LabeledHoldout("I think that's probably close enough.", False),
    LabeledHoldout("Not a big deal either way.", False),
    LabeledHoldout("I'm good with whatever works for the team.", False),
]
