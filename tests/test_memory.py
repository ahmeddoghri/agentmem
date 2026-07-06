from agentmem import MemoryStore, HashingEmbedder, cosine


def test_write_and_retrieve():
    mem = MemoryStore(write_threshold=0.0)
    mem.write("The user's name is Ahmed and he lives in Toronto.")
    mem.write("Ahmed prefers concise, fast answers.")
    mem.write("The capital of France is Paris.")
    hits = mem.retrieve("where does the user live?", k=1)
    assert hits
    assert "Toronto" in hits[0].item.content


def test_salience_gate_drops_filler():
    mem = MemoryStore(write_threshold=0.5)
    stored = mem.write("ok")  # low information -> below threshold
    assert stored is None
    assert len(mem) == 0


def test_near_duplicate_reinforces_not_duplicates():
    mem = MemoryStore(write_threshold=0.0)
    a = mem.write("Project deadline is March 3 for the Instagram engagement app.")
    b = mem.write("Project deadline is March 3 for the Instagram engagement app.")
    assert a is b
    assert len(mem) == 1
    assert a.uses >= 1


def test_bounded_capacity_never_exceeded():
    mem = MemoryStore(capacity=10, write_threshold=0.0, merge_threshold=0.99)
    for i in range(100):
        mem.write(f"Distinct salient fact number {i} about topic {i}.", salience=0.9)
    assert len(mem) <= 10


def test_consolidation_merges_similar():
    # dedup is near-exact (0.99) so both are stored; consolidation (0.5) fuses them
    mem = MemoryStore(capacity=100, write_threshold=0.0,
                      dedup_threshold=0.99, merge_threshold=0.5)
    mem.write("Ava lives in Toronto and works as a software engineer.", salience=0.9)
    mem.write("Ava works in Toronto as an engineer building web apps.", salience=0.9)
    assert len(mem) == 2
    merges = mem.consolidate()
    assert merges >= 1
    assert len(mem) == 1


def test_retrieval_strengthens_against_eviction():
    mem = MemoryStore(capacity=5, write_threshold=0.0, merge_threshold=0.99,
                      half_life_seconds=1.0)
    mem.write("Critical fact: the launch code is alpha-seven-niner.", salience=0.9)
    for _ in range(5):
        mem.retrieve("what is the launch code?")
    for i in range(20):
        mem.write(f"Unrelated filler observation {i}.", salience=0.9)
    assert any("launch code" in it.content for it in mem.all())


def test_hashing_embedder_is_deterministic():
    e = HashingEmbedder(dim=64)
    v1 = e.embed(["hello world"])[0]
    v2 = e.embed(["hello world"])[0]
    assert v1 == v2
    assert cosine(v1, v2) > 0.999
