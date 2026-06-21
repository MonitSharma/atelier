"""Router tests — heuristic backend + role mapping (no MLX model needed)."""

from agent.router import HeuristicRouter, Router


def test_heuristic_flags_hard_tasks() -> None:
    h = HeuristicRouter()
    assert h.classify("Refactor the auth module across the codebase") == "hard"
    assert h.classify("Migrate the data store and update all callers") == "hard"


def test_heuristic_flags_easy_tasks() -> None:
    h = HeuristicRouter()
    assert h.classify("What is 47 * 89?") == "easy"
    assert h.classify("Convert 'hello' to uppercase") == "easy"


def test_router_role_mapping() -> None:
    r = Router(backend="heuristic")
    assert r.role("What is the capital of France?") == "worker"
    assert r.role("Redesign the plugin system across multiple files") == "brain"


def test_router_never_crashes_on_weird_input() -> None:
    r = Router(backend="heuristic")
    # Empty / odd input should still yield a valid role, not raise.
    assert r.role("") in {"worker", "brain"}
