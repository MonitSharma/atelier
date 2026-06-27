"""Agent-facing memory tools: remember a fact, recall relevant facts.

These let the agent build up durable knowledge about the user and the work
across sessions — the difference between a stateless chatbot and an assistant
that remembers your preferences and past decisions.
"""

from __future__ import annotations

from typing import Any

from tools.base import Tool


def run_remember(arguments: dict[str, Any]) -> dict[str, Any]:
    text = arguments.get("text")
    tags = arguments.get("tags", [])
    if not isinstance(text, str) or not text.strip():
        return {"status": "error", "error_type": "invalid_arguments",
                "message": "remember requires a non-empty string 'text'."}
    if isinstance(tags, str):
        tags = [tags]
    if not isinstance(tags, list):
        tags = []

    from agent.memory import get_memory

    try:
        mid = get_memory().remember(text, [str(t) for t in tags])
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error_type": "memory_error", "message": str(exc)}
    return {"status": "success", "tool": "remember", "id": mid}


def run_recall(arguments: dict[str, Any]) -> dict[str, Any]:
    query = arguments.get("query")
    k = arguments.get("k", 5)
    if not isinstance(query, str) or not query.strip():
        return {"status": "error", "error_type": "invalid_arguments",
                "message": "recall requires a non-empty string 'query'."}
    if not isinstance(k, int) or k <= 0 or k > 20:
        k = 5

    from agent.memory import get_memory

    try:
        memories = get_memory().recall(query, k=k)
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error_type": "memory_error", "message": str(exc)}
    return {
        "status": "success",
        "tool": "recall",
        "results": [{"text": m.text, "tags": m.tags, "score": m.score} for m in memories],
    }


REMEMBER_TOOL = Tool(
    name="remember",
    description=(
        "Store a durable fact in long-term memory (persists across sessions). "
        "Use for user preferences, decisions, or facts worth recalling later. "
        "Keep each fact short and self-contained."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The fact to remember."},
            "tags": {"type": "array", "items": {"type": "string"},
                     "description": "Optional labels for the fact."},
        },
        "required": ["text"],
        "additionalProperties": False,
    },
    function=run_remember,
)


RECALL_TOOL = Tool(
    name="recall",
    description=(
        "Search long-term memory for facts relevant to a query and return the "
        "closest matches. Use at the start of a task to check what you already "
        "know about the user or the work."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to recall, in natural language."},
            "k": {"type": "integer", "description": "How many memories to return (default 5)."},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    function=run_recall,
)
