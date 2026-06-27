"""Memory tests with a fake embedder (no model). Includes cross-'session' check."""

from agent.memory import MemoryStore
from rag.store import VectorStore


class FakeEmbedder:
    """Maps keyword presence to a tiny vector; last dim is a nonzero bias so
    cosine distance is always defined."""

    def _vec(self, text: str) -> list[float]:
        t = text.lower()
        return [
            1.0 if "apple" in t else 0.0,
            1.0 if "vector" in t else 0.0,
            1.0 if "license" in t else 0.0,
            0.1,
        ]

    def embed_passages(self, texts):
        return [self._vec(x) for x in texts]

    def embed_query(self, q):
        return self._vec(q)


def _mem(tmp_path):
    store = VectorStore(path=str(tmp_path / "mem"), collection="mem_test")
    return MemoryStore(embedder=FakeEmbedder(), store=store)


def test_remember_and_recall(tmp_path) -> None:
    mem = _mem(tmp_path)
    mem.remember("I like apple pie", tags=["food"])
    mem.remember("vector databases are useful", tags=["tech"])

    assert mem.count() == 2
    top = mem.recall("apple", k=1)
    assert len(top) == 1
    assert "apple" in top[0].text
    assert top[0].tags == ["food"]


def test_all_and_forget(tmp_path) -> None:
    mem = _mem(tmp_path)
    mid = mem.remember("uses the Apache license", tags=["legal"])
    assert mem.count() == 1
    assert any(m.id == mid for m in mem.all())

    mem.forget(mid)
    assert mem.count() == 0


def test_persists_across_sessions(tmp_path) -> None:
    # "Session 1": write a fact, then drop the store object.
    s1 = VectorStore(path=str(tmp_path / "mem"), collection="mem_test")
    MemoryStore(embedder=FakeEmbedder(), store=s1).remember("apple fact", tags=[])

    # "Session 2": a brand-new store at the same path must see it.
    s2 = VectorStore(path=str(tmp_path / "mem"), collection="mem_test")
    mem2 = MemoryStore(embedder=FakeEmbedder(), store=s2)
    assert mem2.count() == 1
    assert "apple" in mem2.recall("apple", k=1)[0].text
