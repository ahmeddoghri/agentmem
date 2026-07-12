# 🧠 AgentMem

**A bounded, self-consolidating long-term memory layer for LLM agents.**

![CI](https://github.com/ahmeddoghri/agentmem/actions/workflows/ci.yml/badge.svg)
![tests](https://img.shields.io/badge/tests-7%20passing-brightgreen)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![deps](https://img.shields.io/badge/runtime%20deps-none-success)
![license](https://img.shields.io/badge/license-MIT-black)

> **Long-term agent memory that stays under a fixed budget instead of growing
> forever.** Salience-gated writes, decay-aware retrieval, consolidation.
> Zero deps, zero API keys: `python -m agentmem.eval`.

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
