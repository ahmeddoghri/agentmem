# 🧠 AgentMem

**A bounded, self-consolidating long-term memory layer for LLM agents.**

![tests](https://img.shields.io/badge/tests-29%20passing-brightgreen)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![deps](https://img.shields.io/badge/runtime%20deps-none-success)
![license](https://img.shields.io/badge/license-MIT-black)

> **Long-term agent memory that stays under a fixed budget instead of growing
> forever.** Salience-gated writes, decay-aware retrieval, consolidation.
> Zero deps, zero API keys: `python -m agentmem.eval`.
>
> The bundled benchmark never actually exercises the gate: it hand-sets
> salience for every write. Run the real scorer end to end and **37% of
> pure filler gets written to memory anyway**, because "I" is always
> capitalized and the scorer reads that as a named entity.
> `python -m agentmem.eval_v2` is the benchmark that found it, and a fixed
> scorer that gets the leak rate to 0%.

Your agent doesn't need to remember that you said "ok sounds good" on
March 3rd. Most "agent memory" is a vector store with no opinions: dump
every turn in, hope retrieval sorts out the treasure from the small talk
later. That's not memory, that's a junk drawer with embeddings.

AgentMem treats memory like an actual budget, because real agents run
under one. Writes are **salience-gated** (small talk gets bounced at the
door), retrieval blends **similarity + recency + importance**, and when
the store fills up it **consolidates** redundant memories into summaries
before evicting the weakest ones. Think Marie Kondo, if Marie Kondo also
did exponential decay.

Runs with **zero dependencies and zero API keys** out of the box
(deterministic hashing embeddings plus an extractive consolidator). Swap
in your own embedding model or LLM through a one-method interface when
you're ready for the real thing.

---

## Install

```bash
pip install agentmem-bounded
```

The import stays `import agentmem`. Only the PyPI distribution name is
`agentmem-bounded`, because the plain `agentmem` name was already claimed
by an unrelated project. First come, first served, apparently.

For local development:

```bash
git clone https://github.com/ahmeddoghri/agentmem
cd agentmem
pip install -e .
```

Or with Docker:

```bash
docker build -t agentmem .
docker run --rm agentmem
```

## 60-second example

```python
from agentmem import MemoryStore

mem = MemoryStore(capacity=100)

mem.write("The user's name is Ahmed, a staff ML engineer in Toronto.")
mem.write("ok sounds good")          # filler. gated out. never even considered.
mem.write("Deadline for the launch is Friday.")

for hit in mem.retrieve("when is the deadline?", k=1):
    print(hit.score, hit.item.content)
# 0.90 Deadline for the launch is Friday.
```

Notice what's missing from the output: "ok sounds good." It never made it
past the front door.

## Why it's different

| Typical vector memory | AgentMem |
|---|---|
| Stores every message | **Salience gate** drops filler before it ever lands |
| Grows unbounded | **Hard capacity**, decay-aware eviction |
| Duplicates pile up | **Near-exact dedup** reinforces instead of duplicating |
| Redundant facts scattered everywhere | **Consolidation** fuses related memories into one summary |
| Pure cosine ranking | **Similarity + recency + importance**, and retrieval itself fights forgetting |

## How it works

```
write(text)
  ├─ score salience ── below threshold? ─► drop, no hard feelings
  ├─ near-exact match? ─► reinforce existing (uses++, salience↑)
  └─ store new item
        └─ over capacity? ─► consolidate() then evict the weakest by decayed strength

retrieve(query, k)
  └─ rank by  cosine(sim) + w_r·recency + w_s·salience
        └─ touch top-k (last_used = now, uses++)  ← retrieval fights decay
```

Retention strength decays on a configurable half-life, so memories nobody
asks about quietly fade while the frequently-retrieved ones stick around.
It's the same principle as your own brain, minus the part where you can't
remember why you walked into the kitchen.

## Benchmark: bounded-memory recall

Stream 200 salient facts past the agent, interleaved with distractor
chatter, while capping memory at 32 slots. Then quiz it on a random
sample:

```bash
python -m agentmem.eval --capacity 32 --facts 200
# facts=200  capacity=32  final_size=32
# recall@5 = 52.0%
```

Raise `--capacity` and recall climbs toward 100%. That's not a bug, that's
the entire point: the benchmark makes the budget/accuracy tradeoff
impossible to ignore. Use it as a regression harness while you tune your
own retention policy.

Note the `consolidations` line: earlier versions of this benchmark hardcoded
that field to `0` regardless of what actually happened, so the second half of
this project's own name never had a number attached to it in its own output.
`MemoryStore` now tracks `total_merges` directly, and the benchmark reports it.

## The salience gate that never gated

The benchmark above calls `mem.write(fact, salience=0.9)` and
`mem.write(chatter, salience=0.05)`, both explicit overrides, and runs with
`write_threshold=0.0` so nothing is ever rejected either way. "Salience-gated
writes" is the first thing this README promises, and its own flagship
benchmark never lets the scorer make a single gating decision.

Run the real scorer end to end, no overrides, on the same synthetic stream:

```bash
python -m agentmem.eval_v2
```
```
End to end: the real gate deciding every write, no overrides
scorer      leak rate   distractors  recall@5  consolidations
v1               37%    297/800         50%               8
v2                0%      0/800         52%              10
```

**37% of pure filler got written to memory anyway.** The cause: the scorer
rewards a sentence for containing a mid-sentence capitalized word, on the
theory that capitalization marks a named entity ("Toronto", "UA482"). "I" is
always capitalized, whether or not it introduces anything worth remembering.
`"Hmm, I'm not sure about that."` contains "I'm" and scores 0.596, comfortably
over the 0.35 write threshold, for exactly the kind of small talk this
README's own opening paragraph uses as its example of what should never be
remembered.

```
Gate precision/recall on labeled sentences
split / scorer            precision   recall
adversarial / v1               53%     100%
adversarial / v2              100%     100%
holdout / v1                   40%     100%
holdout / v2                  100%     100%
```

Fixing the specific "I" bug wasn't enough on its own: stripping the false
proper-noun credit still left sentences like "Hmm, I'm not sure about that."
a few points over the threshold, carried by length and word-density alone.
"not", "sure", and "about" were never stopwords in the original list, so a
five-word hedge reads as five words of content. `ExtractiveLLMV2` adds those
hedges to what counts as a stopword for salience purposes, the actual
vocabulary of thinking out loud rather than a fact worth keeping.

`MemoryStore` uses `ExtractiveLLMV2` by default now. `ExtractiveLLM` (the
original) is still exported for anyone reproducing the benchmark above or
who wants the old behavior.

### Held out, run once

`ExtractiveLLMV2`'s stopword and pronoun-form additions were frozen against
the adversarial corpus above before a second set of ten sentences was
written and scored a single time. Zero false positives, zero false
negatives, same as the tuning corpus.

### Limits

- **Still a heuristic, not a model.** It knows "I'm not sure" is filler because that phrase and its relatives are enumerated; a paraphrase it hasn't seen may still slip through.
- **English-specific.** The stopword and hedge lists assume English discourse markers.
- **Fixing precision on filler says nothing about summarization quality.** `summarize()` is unchanged from the original; the bug was entirely in `score_salience`.

## Bring your own models

Anything with the right method works. No base classes to inherit, no
sixteen-layer abstract factory to implement.

```python
class MyEmbedder:
    dim = 1536
    def embed(self, texts): ...        # -> list[list[float]]

class MyLLM:
    def score_salience(self, text): ...  # -> float in [0,1]
    def summarize(self, texts): ...      # -> str

mem = MemoryStore(embedder=MyEmbedder(), llm=MyLLM())
```

## Tests

```bash
pip install pytest && pytest -q     # 7 passing
```

## More in this series

Nine small, dependency-light, benchmarked tools for LLM/ML infrastructure. Each one reproduces its headline number locally with no API keys:

[rubricagent](https://github.com/ahmeddoghri/rubricagent) · [clarifyrag](https://github.com/ahmeddoghri/clarifyrag) · [churnfm](https://github.com/ahmeddoghri/churnfm) · [citebench](https://github.com/ahmeddoghri/citebench) · [guardrail-gate](https://github.com/ahmeddoghri/guardrail-gate) · [tablextract](https://github.com/ahmeddoghri/tablextract) · [vllm-cost-router](https://github.com/ahmeddoghri/vllm-cost-router) · [taggate](https://github.com/ahmeddoghri/taggate)

## License

MIT © Ahmed Doghri
