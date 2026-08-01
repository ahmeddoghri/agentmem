"""Does the salience gate actually gate, end to end?

The bundled recall benchmark (:mod:`agentmem.eval`) never lets the salience
scorer make a decision: it calls ``mem.write(text, salience=0.9)`` for facts
and ``mem.write(text, salience=0.05)`` for distractors, both explicit
overrides, and runs with ``write_threshold=0.0`` so nothing is rejected
either way. "Salience-gated writes" is the first thing this project's README
promises, and its own flagship benchmark never exercises the gate once.

This module measures two things the original does not:

**Gate precision and recall** on labeled sentences (:mod:`agentmem.adversarial`),
split into an adversarial set built independently of the scorer's source and a
frozen holdout evaluated exactly once.

**The real end-to-end leak rate**: run the same synthetic support stream as
the original benchmark, but let the real ``score_salience`` decide every
write, no overrides. Report what fraction of pure filler gets written to
memory anyway, and whether fact recall changes.

    python -m agentmem.eval_v2
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence

from .adversarial import ADVERSARIAL_CORPUS, HOLDOUT_CORPUS, LabeledUtterance
from .llm import ExtractiveLLM
from .llm_v2 import ExtractiveLLMV2
from .memory import MemoryStore

DEFAULT_THRESHOLD = 0.35

_SUBJECTS = ["Ava", "Ben", "Chen", "Diego", "Eli", "Fatima", "Grace", "Hiro"]
_CITIES = ["Toronto", "Berlin", "Lagos", "Osaka", "Lima", "Oslo", "Cairo", "Perth"]
_DISTRACTORS = [
    "The weather is mild today.",
    "Let's continue.",
    "Interesting, tell me more.",
    "Okay, noted.",
    "Hmm, I'm not sure about that.",
    "Sounds good to me.",
    "I think I'll just go with the default.",
    "Well, I don't really have a strong opinion.",
]


@dataclass
class GateScore:
    true_positive: int = 0
    false_positive: int = 0
    true_negative: int = 0
    false_negative: int = 0

    @property
    def precision(self) -> float:
        denom = self.true_positive + self.false_positive
        return self.true_positive / denom if denom else 1.0

    @property
    def recall(self) -> float:
        denom = self.true_positive + self.false_negative
        return self.true_positive / denom if denom else 1.0

    def to_dict(self) -> dict:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "true_negative": self.true_negative,
            "false_negative": self.false_negative,
        }


def score_gate(llm, corpus: Sequence, threshold: float = DEFAULT_THRESHOLD) -> GateScore:
    """Score a salience scorer's write decisions against labeled ground truth."""
    result = GateScore()
    for item in corpus:
        write = llm.score_salience(item.text) >= threshold
        if item.worth_remembering and write:
            result.true_positive += 1
        elif item.worth_remembering and not write:
            result.false_negative += 1
        elif not item.worth_remembering and write:
            result.false_positive += 1
        else:
            result.true_negative += 1
    return result


@dataclass
class LeakResult:
    facts_written: int
    distractors_written: int
    distractors_total: int
    recall_at_k: float
    final_size: int
    consolidations: int

    @property
    def leak_rate(self) -> float:
        return self.distractors_written / self.distractors_total if self.distractors_total else 0.0

    def to_dict(self) -> dict:
        return {
            "facts_written": self.facts_written,
            "distractors_written": self.distractors_written,
            "distractors_total": self.distractors_total,
            "leak_rate": round(self.leak_rate, 4),
            "recall_at_k": round(self.recall_at_k, 4),
            "final_size": self.final_size,
            "consolidations": self.consolidations,
        }


def run_real_gate(llm, capacity: int = 32, n_facts: int = 200,
                  distractors: int = 4, k: int = 5, seed: int = 0) -> LeakResult:
    """Run the long-horizon stream with the gate actually deciding every write.

    Unlike :func:`agentmem.eval.run`, no salience override is passed to
    ``mem.write``: the scorer under test makes every gating decision, the
    same way a real caller would use the library.
    """
    rng = random.Random(seed)
    mem = MemoryStore(capacity=capacity, half_life_seconds=1e9, llm=llm)

    facts: List[tuple] = []
    facts_written = 0
    distractors_written = 0
    distractors_total = 0

    for i in range(n_facts):
        subj = rng.choice(_SUBJECTS) + str(i)
        city = rng.choice(_CITIES)
        fact = f"{subj} lives in {city} and works as engineer number {i}."
        item = mem.write(fact)
        if item is not None:
            facts_written += 1
        facts.append((f"Which city does {subj} live in?", city))

        for _ in range(distractors):
            text = rng.choice(_DISTRACTORS)
            distractors_total += 1
            written = mem.write(text)
            if written is not None and written.content == text:
                distractors_written += 1

    sample = rng.sample(facts, min(50, len(facts)))
    hits = 0
    for q, expected in sample:
        retrieved = mem.retrieve(q, k=k)
        if any(expected in r.item.content for r in retrieved):
            hits += 1

    return LeakResult(
        facts_written=facts_written,
        distractors_written=distractors_written,
        distractors_total=distractors_total,
        recall_at_k=hits / len(sample) if sample else 0.0,
        final_size=len(mem),
        consolidations=mem.total_merges,
    )


def build_report(capacity: int = 32, n_facts: int = 200) -> Dict:
    v1, v2 = ExtractiveLLM(), ExtractiveLLMV2()
    return {
        "gate": {
            "adversarial": {
                "v1": score_gate(v1, ADVERSARIAL_CORPUS).to_dict(),
                "v2": score_gate(v2, ADVERSARIAL_CORPUS).to_dict(),
            },
            "holdout": {
                "v1": score_gate(v1, HOLDOUT_CORPUS).to_dict(),
                "v2": score_gate(v2, HOLDOUT_CORPUS).to_dict(),
            },
        },
        "end_to_end": {
            "v1": run_real_gate(v1, capacity=capacity, n_facts=n_facts).to_dict(),
            "v2": run_real_gate(v2, capacity=capacity, n_facts=n_facts).to_dict(),
        },
    }


def format_report(report: Dict) -> str:
    lines = [
        "Gate precision/recall on labeled sentences",
        "=" * 70,
        f"{'split / scorer':<24}{'precision':>11}{'recall':>9}{'FP':>5}{'FN':>5}",
        "-" * 70,
    ]
    for split in ("adversarial", "holdout"):
        for name in ("v1", "v2"):
            row = report["gate"][split][name]
            label = f"{split} / {name}"
            lines.append(
                f"{label:<24}{row['precision']:>10.0%}{row['recall']:>9.0%}"
                f"{row['false_positive']:>5}{row['false_negative']:>5}"
            )
    lines.append("")

    lines += [
        "End to end: the real gate deciding every write, no overrides",
        "=" * 70,
        f"{'scorer':<10}{'leak rate':>11}{'distractors':>14}{'recall@5':>10}{'consolidations':>16}",
        "-" * 70,
    ]
    for name in ("v1", "v2"):
        row = report["end_to_end"][name]
        lines.append(
            f"{name:<10}{row['leak_rate']:>10.0%}"
            f"{row['distractors_written']:>7}/{row['distractors_total']:<6}"
            f"{row['recall_at_k']:>9.0%}{row['consolidations']:>16}"
        )
    lines.append("")
    lines.append(
        "leak rate = share of pure filler the gate wrote to memory anyway."
    )
    lines.append(
        "consolidations = real merge count; the original benchmark hardcoded this to 0."
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity", type=int, default=32)
    parser.add_argument("--facts", type=int, default=200)
    args = parser.parse_args()
    report = build_report(capacity=args.capacity, n_facts=args.facts)
    print(format_report(report))


if __name__ == "__main__":
    main()
