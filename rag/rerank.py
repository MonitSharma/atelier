"""Optional cross-encoder reranking — the precision pass.

Bi-encoder retrieval (dense/BM25) is fast but coarse: it scores query and
passage independently. A cross-encoder reads the (query, passage) pair together
and scores relevance directly — much sharper, but too slow to run over the whole
corpus. So we use it the standard way: retrieve a wide candidate set cheaply,
then rerank just those. Opt-in via ``ATELIER_RERANK=1`` (downloads a small model
on first use). Fully local.
"""

from __future__ import annotations

from typing import Any

from atelier.config import settings

_model = None


def get_reranker():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder

        from rag.embed import _pick_device

        _model = CrossEncoder(settings.rerank_model, device=_pick_device(settings.embed_device))
    return _model


def rerank(query: str, hits: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    if not hits:
        return hits
    model = get_reranker()
    scores = model.predict([(query, h["text"]) for h in hits])
    for h, s in zip(hits, scores, strict=False):
        h["rerank_score"] = float(s)
    hits.sort(key=lambda h: -h["rerank_score"])
    return hits[:top_k]
