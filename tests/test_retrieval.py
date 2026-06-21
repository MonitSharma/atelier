"""Hybrid-retrieval unit tests: BM25 ranking + RRF fusion (no model)."""

from rag.lexical import BM25Index
from rag.retrieve import _rrf_fuse

DOCS = [
    "the quick brown fox jumps",
    "vector databases store embeddings for search",
    "the lazy dog sleeps all day",
]
METAS = [{"source": f"d{i}.md"} for i in range(len(DOCS))]


def test_bm25_ranks_keyword_match_first() -> None:
    idx = BM25Index(DOCS, METAS)
    hits = idx.search("vector embeddings search", n=3)
    assert hits, "expected matches"
    assert "vector databases" in hits[0]["text"]


def test_bm25_returns_only_matches() -> None:
    idx = BM25Index(DOCS, METAS)
    hits = idx.search("quick fox", n=5)
    assert all(h["score"] > 0 for h in hits)
    assert "quick brown fox" in hits[0]["text"]


def test_rrf_rewards_agreement() -> None:
    dense = [{"text": "A"}, {"text": "B"}, {"text": "C"}]
    lexical = [{"text": "C"}, {"text": "A"}, {"text": "D"}]
    fused = _rrf_fuse(dense, lexical, k=4, rrf_k=60)
    # A is high in both lists -> should win; C (also in both) second.
    assert fused[0]["text"] == "A"
    assert fused[1]["text"] == "C"
    assert "fused_score" in fused[0]
