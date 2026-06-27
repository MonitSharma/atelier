"""Long-term, semantic memory (Phase 3).

Naive memory — "stuff every past message into the context" — breaks: it blows
the window and buries signal. Instead Atelier stores discrete *facts* as embedded
records in their own ChromaDB collection, and recalls them by semantic
similarity when relevant. This is persistent across sessions (it lives on disk
next to the knowledge store) and shares the same local embedding model.

A fact is one short, self-contained statement: "Monit prefers Apache-2.0",
"the brain model is qwen3:14b". Recall returns the closest facts to a query.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from atelier.config import settings
from rag.store import VectorStore

MEMORY_COLLECTION = "atelier_memory"


@dataclass
class Memory:
    id: str
    text: str
    tags: list[str]
    created_at: str
    score: float | None = None


def _new_id(text: str) -> str:
    return hashlib.sha1(f"{text}:{time.time_ns()}".encode()).hexdigest()[:16]


class MemoryStore:
    """Persistent semantic memory. ``embedder`` is injectable for testing."""

    def __init__(self, embedder: Any = None, store: VectorStore | None = None) -> None:
        self._embedder = embedder
        self._store = store or VectorStore(collection=MEMORY_COLLECTION)

    @property
    def embedder(self):
        if self._embedder is None:
            from rag.embed import get_embedder

            self._embedder = get_embedder()
        return self._embedder

    def remember(self, text: str, tags: list[str] | None = None) -> str:
        text = text.strip()
        if not text:
            raise ValueError("cannot remember empty text")
        mid = _new_id(text)
        embedding = self.embedder.embed_passages([text])[0]
        self._store.upsert_raw(
            ids=[mid],
            documents=[text],
            embeddings=[embedding],
            metadatas=[{
                "tags": ",".join(tags or []),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }],
        )
        return mid

    def recall(self, query: str, k: int = 5) -> list[Memory]:
        if self._store.count() == 0:
            return []
        embedding = self.embedder.embed_query(query)
        hits = self._store.query(embedding, k=k)
        return [
            Memory(
                id="",  # query() doesn't return ids; recall is about content
                text=h["text"],
                tags=[t for t in h["metadata"].get("tags", "").split(",") if t],
                created_at=h["metadata"].get("created_at", ""),
                score=round(float(h["score"]), 3),
            )
            for h in hits
        ]

    def all(self) -> list[Memory]:
        got = self._store.get_all()
        out: list[Memory] = []
        for mid, doc, meta in zip(
            got.get("ids", []), got.get("documents", []), got.get("metadatas", []), strict=False
        ):
            out.append(Memory(
                id=mid,
                text=doc,
                tags=[t for t in (meta or {}).get("tags", "").split(",") if t],
                created_at=(meta or {}).get("created_at", ""),
            ))
        return out

    def forget(self, memory_id: str) -> None:
        self._store.delete([memory_id])

    def count(self) -> int:
        return self._store.count()


_default: MemoryStore | None = None


def get_memory() -> MemoryStore:
    global _default
    if _default is None:
        _default = MemoryStore()
    return _default
