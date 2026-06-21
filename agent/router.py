"""The router: send easy subtasks to a cheap model, hard ones to the brain.

Phase 6 — "model as a component." A fine-tuned 0.5B classifier (trained in
``models/router/``) labels a task easy/hard in a few milliseconds; the router
maps that to a model *role* (worker vs brain). If the adapter isn't present, it
degrades to a transparent keyword heuristic, so nothing depends on the fine-tune
at runtime.

The win we measure (PROJECT.md §8 Phase 6): routing cuts brain calls on easy
tasks without hurting success rate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from atelier.config import settings
from models.router.make_dataset import PROMPT

Difficulty = Literal["easy", "hard"]
ADAPTER_PATH = settings.root / "models" / "router" / "adapter"
BASE_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

# Words that strongly signal a multi-step / cross-cutting task.
_HARD_HINTS = ("refactor", "across", "multiple files", "design", "architecture",
               "migrate", "race condition", "end-to-end", "redesign", "diagnose",
               "profil", "strategy", "spanning", "root cause", "concurrent")
_EASY_HINTS = ("what is", "convert", "one-line", "one line", "rename", "round",
               "capital of", "is ", "reverse", "format", "compute", "add a docstring")


class HeuristicRouter:
    """No model. Fast, transparent, always available."""

    def classify(self, task: str) -> Difficulty:
        t = task.lower()
        if any(h in t for h in _HARD_HINTS):
            return "hard"
        if any(h in t for h in _EASY_HINTS):
            return "easy"
        # Fall back on length: short asks are usually easy.
        return "easy" if len(task) < 90 else "hard"


class FineTunedRouter:
    """Loads the LoRA-fine-tuned 0.5B classifier via MLX (lazy, cached)."""

    def __init__(self, base: str = BASE_MODEL, adapter: Path = ADAPTER_PATH) -> None:
        self.base = base
        self.adapter = adapter
        self._model = None
        self._tok = None

    def _ensure(self) -> None:
        if self._model is None:
            from mlx_lm import load

            self._model, self._tok = load(self.base, adapter_path=str(self.adapter))

    def classify(self, task: str) -> Difficulty:
        from mlx_lm import generate

        self._ensure()
        out = generate(self._model, self._tok, prompt=PROMPT.format(task=task),
                       max_tokens=2, verbose=False)
        return "hard" if "hard" in out.lower() else "easy"


class Router:
    """Picks the classifier backend and maps difficulty → model role."""

    def __init__(self, backend: str = "auto") -> None:
        if backend == "heuristic":
            self.backend: HeuristicRouter | FineTunedRouter = HeuristicRouter()
        elif backend == "finetuned":
            self.backend = FineTunedRouter()
        else:  # auto: use the fine-tune if it's been trained, else heuristic
            self.backend = FineTunedRouter() if ADAPTER_PATH.exists() and any(
                ADAPTER_PATH.glob("*.safetensors")
            ) else HeuristicRouter()

    @property
    def name(self) -> str:
        return type(self.backend).__name__

    def classify(self, task: str) -> Difficulty:
        try:
            return self.backend.classify(task)
        except Exception:  # noqa: BLE001 - never let routing crash a task
            return HeuristicRouter().classify(task)

    def route(self, task: str) -> str:
        """Return the model *role* to use for this task."""
        return settings.worker_model if self.classify(task) == "easy" else settings.brain_model

    def role(self, task: str) -> str:
        return "worker" if self.classify(task) == "easy" else "brain"
