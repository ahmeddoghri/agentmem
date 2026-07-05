"""Minimal end-to-end demo. Run: python examples/quickstart.py"""
from agentmem import MemoryStore

mem = MemoryStore(capacity=50)

# The agent observes a long conversation; only salient bits get stored.
for line in [
    "The user's name is Ahmed, a staff ML engineer in Toronto.",
    "ok sounds good",                       # filler -> gated out
    "Ahmed is building three open-source agent projects this week.",
    "cool",                                 # filler -> gated out
    "Deadline for the AgentMem repo is Friday.",
]:
    stored = mem.write(line)
    print(("STORED " if stored else "skipped") + f": {line}")

print("\nMemories retained:", len(mem))
print("\nQ: what is the user building?")
for r in mem.retrieve("what is the user building?", k=2):
    print(f"  [{r.score:.2f}] {r.item.content}")

print("\nQ: when is the deadline?")
for r in mem.retrieve("when is the deadline?", k=1):
    print(f"  [{r.score:.2f}] {r.item.content}")
