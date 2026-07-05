"""A bounded-memory long-horizon recall benchmark (AgenticSTS-style).

We stream many "facts" past an agent, interleaved with distractor chatter, while
the memory store is held to a small budget. Later we quiz the agent on a random
subset of the injected facts and measure recall@k. This isolates the exact thing
bounded-memory agents get wrong: silently forgetting salient facts under budget
pressure.

Run it::

    python -m agentmem.eval --capacity 32 --facts 200 --distractors 4
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

from .memory import MemoryStore

_SUBJECTS = ["Ava", "Ben", "Chen", "Diego", "Eli", "Fatima", "Grace", "Hiro"]
_CITIES = ["Toronto", "Berlin", "Lagos", "Osaka", "Lima", "Oslo", "Cairo", "Perth"]
_DISTRACTORS = [
    "The weather is mild today.",
    "Let's continue.",
    "Interesting, tell me more.",
    "Okay, noted.",
    "Hmm, I'm not sure about that.",
    "Sounds good to me.",
]


@dataclass
class EvalResult:
    facts: int
    capacity: int
    recall_at_k: float
    consolidations: int
    final_size: int


def run(capacity: int = 32, n_facts: int = 200, distractors: int = 4,
        k: int = 5, seed: int = 0) -> EvalResult:
    rng = random.Random(seed)
    mem = MemoryStore(capacity=capacity, write_threshold=0.0, half_life_seconds=1e9)

    facts: list[tuple[str, str]] = []  # (question, expected_answer)
    consolidations = 0
    for i in range(n_facts):
        subj = rng.choice(_SUBJECTS) + str(i)
        city = rng.choice(_CITIES)
        fact = f"{subj} lives in {city} and works as engineer number {i}."
        mem.write(fact, salience=0.9)
        facts.append((f"Which city does {subj} live in?", city))
        for _ in range(distractors):
            mem.write(rng.choice(_DISTRACTORS), salience=0.05)

    # quiz on a random sample
    sample = rng.sample(facts, min(50, len(facts)))
    hits = 0
    for q, expected in sample:
        retrieved = mem.retrieve(q, k=k)
        if any(expected in r.item.content for r in retrieved):
            hits += 1
    recall = hits / len(sample)
    return EvalResult(
        facts=n_facts,
        capacity=capacity,
        recall_at_k=recall,
        consolidations=consolidations,
        final_size=len(mem),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capacity", type=int, default=32)
    p.add_argument("--facts", type=int, default=200)
    p.add_argument("--distractors", type=int, default=4)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    res = run(args.capacity, args.facts, args.distractors, args.k, args.seed)
    print(f"facts={res.facts}  capacity={res.capacity}  final_size={res.final_size}")
    print(f"recall@{args.k} = {res.recall_at_k:.1%}")


if __name__ == "__main__":
    main()
