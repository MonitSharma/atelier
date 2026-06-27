"""Retrieval: embed a query, fetch nearest chunks, format them for the prompt."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from atelier.config import settings
from rag.embed import get_embedder
from rag.store import VectorStore


def _rrf_fuse(
    dense: list[dict[str, Any]],
    lexical: list[dict[str, Any]],
    k: int,
    rrf_k: int,
) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion: combine two ranked lists by rank, not by score.

    RRF is robust precisely because it ignores the (incomparable) raw scores of
    dense vs. BM25 and uses only positions: score = sum 1/(rrf_k + rank).
    """
    table: dict[str, dict[str, Any]] = {}
    for rank, hit in enumerate(dense):
        entry = table.setdefault(hit["text"], {"hit": hit, "score": 0.0})
        entry["score"] += 1.0 / (rrf_k + rank)
    for rank, hit in enumerate(lexical):
        entry = table.setdefault(hit["text"], {"hit": hit, "score": 0.0})
        entry["score"] += 1.0 / (rrf_k + rank)
    fused = sorted(table.values(), key=lambda e: -e["score"])
    out: list[dict[str, Any]] = []
    for entry in fused[:k]:
        hit = dict(entry["hit"])
        hit["fused_score"] = round(entry["score"], 5)
        out.append(hit)
    return out


def retrieve(
    query: str,
    k: int | None = None,
    store: VectorStore | None = None,
    *,
    hybrid: bool | None = None,
    rerank: bool | None = None,
) -> list[dict[str, Any]]:
    """Retrieve the top-k chunks for a query.

    Pipeline: dense (always) [+ BM25 fused via RRF if hybrid] [→ cross-encoder
    rerank if enabled]. Defaults come from config; pass explicit flags to override.
    """
    store = store or VectorStore()
    k = k or settings.retrieval_k
    use_hybrid = settings.use_hybrid if hybrid is None else hybrid
    do_rerank = settings.rerank if rerank is None else rerank

    n = max(settings.hybrid_candidates, k)
    pool = n if do_rerank else k

    dense = store.query(get_embedder().embed_query(query), k=n)
    if use_hybrid:
        from rag.lexical import get_bm25

        lexical = get_bm25(store).search(query, n)
        candidates = _rrf_fuse(dense, lexical, pool, settings.rrf_k)
    else:
        candidates = dense[:pool]

    if do_rerank and candidates:
        from rag.rerank import rerank as _do_rerank

        candidates = _do_rerank(query, candidates, k)

    return candidates[:k]


def format_context(hits: list[dict[str, Any]], max_chars: int | None = None) -> str:
    """Render retrieved chunks into a numbered, citable context block."""
    max_chars = max_chars or settings.max_context_chars
    blocks: list[str] = []
    used = 0
    for i, hit in enumerate(hits, start=1):
        src = hit["metadata"].get("source", "?")
        name = Path(src).name if src != "?" else "?"
        section = hit["metadata"].get("section", "")
        header = f"[{i}] {name}" + (f"  ({section})" if section else "")
        body = hit["text"]
        block = f"{header}\n{body}"
        if used + len(block) > max_chars and blocks:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n---\n\n".join(blocks)


def citations(hits: list[dict[str, Any]]) -> list[str]:
    """Short, de-duplicated source list for display under an answer."""
    seen: list[str] = []
    for hit in hits:
        name = Path(hit["metadata"].get("source", "?")).name
        if name not in seen:
            seen.append(name)
    return seen
