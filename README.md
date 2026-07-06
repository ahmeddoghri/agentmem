# 🧠 AgentMem

**A bounded, self-consolidating long-term memory layer for LLM agents.**

![CI](https://github.com/ahmeddoghri/agentmem/actions/workflows/ci.yml/badge.svg)
![tests](https://img.shields.io/badge/tests-7%20passing-brightgreen)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![deps](https://img.shields.io/badge/runtime%20deps-none-success)
![license](https://img.shields.io/badge/license-MIT-black)

Most "agent memory" is just an unbounded vector store — dump every turn in, hope
retrieval sorts it out. Real agents run under a **fixed budget** and need to
*decide* what's worth keeping. AgentMem treats memory as a policy: writes are
**salience-gated**, retrieval blends **similarity + recency + importance**, and
when the store fills up it **consolidates redundant memories into summaries**
before evicting the weakest ones.

Runs with **zero dependencies and zero API keys** out of the box (deterministic
hashing embeddings + an extractive consolidator). Swap in your own embedding
model or LLM with a one-method interface when you want production quality.

> **Inspired by the mid-2026 agent-memory literature:**
> - *AutoMem: Automated Learning of Memory as a Cognitive Skill* (2026) — memory as an active, learned skill rather than passive storage.
> - *AgenticSTS: A Bounded-Memory Testbed for Long-Horizon LLM Agents* (2026) — evaluating agents under realistic memory budgets.

---

## Install

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
mem.write("ok sounds good")          # ← filler, gated out (never stored)
mem.write("Deadline for the launch is Friday.")

for hit in mem.retrieve("when is the deadline?", k=1):
    print(hit.score, hit.item.content)
# 0.87 Deadline for the launch is Friday.
```

## Why it's different

| Typical vector memory | AgentMem |
|---|---|
| Stores every message | **Salience gate** drops filler before it ever lands |
| Grows unbounded | **Hard capacity** with decay-aware eviction |
| Duplicates pile up | **Near-exact dedup** on write reinforces instead of duplicating |
| Redundant facts scattered | **Consolidation** fuses related memories into summaries |
| Pure cosine ranking | **Similarity + recency + importance**, and retrieval strengthens memories against forgetting |

## How it works

```
write(text)
  ├─ score salience ── below threshold? ─► drop
  ├─ near-exact match? ─► reinforce existing (uses++, salience↑)
  └─ store new item
        └─ over capacity? ─► consolidate() then evict weakest by decayed strength

retrieve(query, k)
  └─ rank by  cosine(sim) + w_r·recency + w_s·salience
        └─ touch top-k (last_used = now, uses++)  ← retrieval fights decay
```

Retention strength decays with a configurable half-life, so unused memories fade
while frequently-retrieved ones persist — the eviction policy that keeps a small
budget useful over long horizons.

## Benchmark: bounded-memory recall (AgenticSTS-style)

Stream 200 salient facts past the agent, interleaved with distractor chatter,
while capping memory at 32 slots. Then quiz on a random sample:

```bash
python -m agentmem.eval --capacity 32 --facts 200
# facts=200  capacity=32  final_size=32
# recall@5 = 46.0%
```

Raise `--capacity` and recall climbs toward 100% — the benchmark makes the
budget/accuracy tradeoff explicit, which is the whole point of bounded-memory
evaluation. Use it as a regression harness when tuning your own retention policy.

## Bring your own models

Anything with the right method works — no base classes to inherit:

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

## License

MIT © Ahmed Doghri
