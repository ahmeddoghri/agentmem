"""AgentMem — a bounded, self-consolidating memory layer for LLM agents.

Quickstart
----------
>>> from agentmem import MemoryStore
>>> mem = MemoryStore(capacity=100)
>>> mem.write("The user's name is Ahmed and he lives in Toronto.")
>>> mem.write("Ahmed prefers concise answers.")
>>> hits = mem.retrieve("where does the user live?")
>>> hits[0].item.content
"The user's name is Ahmed and he lives in Toronto."
"""
from .embeddings import Embedder, HashingEmbedder, cosine
from .llm import LLM, ExtractiveLLM
from .memory import MemoryItem, MemoryStore, Retrieved

__all__ = [
    "MemoryStore",
    "MemoryItem",
    "Retrieved",
    "Embedder",
    "HashingEmbedder",
    "LLM",
    "ExtractiveLLM",
    "cosine",
]
__version__ = "0.1.0"
