"""Measure the fine-tune: base vs LoRA-adapted accuracy on the held-out test set.

Runs the 0.5B model twice over ``test.jsonl`` — once as the stock base model,
once with the trained adapter — and reports difficulty-classification accuracy
for each. The lift (adapted − base) is the evidence that the fine-tune actually
taught the router something.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
ADAPTER = HERE / "adapter"


def _load_test() -> list[dict]:
    lines = (HERE / "data" / "test.jsonl").read_text().splitlines()
    return [json.loads(ln) for ln in lines if ln.strip()]


def _predict(model, tok, prompt: str) -> str:
    from mlx_lm import generate

    out = generate(model, tok, prompt=prompt, max_tokens=2, verbose=False)
    return "hard" if "hard" in out.lower() else "easy"


def _accuracy(adapter: Path | None) -> float:
    from mlx_lm import load

    model, tok = load(BASE_MODEL, adapter_path=str(adapter) if adapter else None)
    rows = _load_test()
    correct = 0
    for r in rows:
        gold = r["completion"].strip()
        pred = _predict(model, tok, r["prompt"])
        correct += int(pred == gold)
    return correct / len(rows)


def main() -> dict:
    base = _accuracy(None)
    tuned = _accuracy(ADAPTER) if (ADAPTER / "adapters.safetensors").exists() or any(
        ADAPTER.glob("*.safetensors")
    ) else None
    result = {
        "base_accuracy": round(base, 3),
        "finetuned_accuracy": round(tuned, 3) if tuned is not None else None,
        "lift": round(tuned - base, 3) if tuned is not None else None,
        "n_test": len(_load_test()),
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
