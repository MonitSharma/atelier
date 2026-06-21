# Atelier

A fully **local**, **zero-cost**, **dual-mode** AI agent that runs on a single
MacBook (Apple Silicon, 36 GB). It does two things over one shared harness:

- **Knowledge mode** — ask questions grounded in *your own* notes, PDFs, and
  code (local RAG). Your documents never leave the machine.
- **Build mode** — read a repo, edit code, run tests, and fix failures, proving
  each change with a green test run.

No cloud APIs, no subscriptions, no rented GPUs. Everything runs against a local
[Ollama](https://ollama.com) server and a local embedding model.

> Design rationale, scope, and roadmap live in [`PROJECT.md`](PROJECT.md).

## Requirements

- macOS on Apple Silicon (built/tested on M3 Pro, 36 GB)
- [Ollama](https://ollama.com) running locally (`ollama serve`)
- Python 3.11+ and [`uv`](https://github.com/astral-sh/uv)
- Models pulled locally:
  ```bash
  ollama pull qwen3:14b   # brain (reasoning / build)
  ollama pull qwen3:4b    # fast worker / router
  ollama pull gemma4:26b  # optional heavy reasoner
  ```

## Setup

```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .            # provides the `atelier` command
atelier doctor                 # verify models + store + embeddings
```

## Knowledge mode — quickstart

```bash
# 1. Index your notes (point it anywhere — an Obsidian vault, a papers folder…)
atelier ingest ~/Notes ./Project.md

# 2. Ask, grounded in what you indexed
atelier ask "What did I decide about the embedding model and why?"

# 3. Or chat interactively
atelier chat
```

Useful flags: `atelier ask -k 8 --show-context "<q>"` (more context, show
passages), `--heavy` (use the larger reasoning model), `atelier sources` (list
what's indexed), `atelier ingest --reset <path>` (rebuild the index).

## Build mode — the autonomous agent

The full dual-mode agent reasons and uses tools (file read/write/edit, repo map,
sandboxed code exec, pytest runner, semantic note search) to complete tasks:

```bash
atelier tools                              # see the toolbox
atelier agent "Fix the failing test in sample_task/ and prove it passes"
atelier agent "Using my notes, summarize the project's non-goals" 
```

Each run streams its steps and writes a full trace to `data/traces/`. Add
`--shell` to enable the (powerful, lightly-guarded) shell tool, `--heavy` for the
bigger model, `--max-steps N` to bound the loop, `--memory` to recall relevant
long-term memories first.

## Memory — it remembers across sessions

Atelier has persistent, semantic long-term memory (its own local vector
collection):

```bash
atelier remember "I prefer Apache-2.0 and pytest over unittest" --tags prefs
atelier recall "what license does the user want?"
atelier memory                              # list everything stored
atelier agent --memory "scaffold a new module the way I like it"
```

The agent also has `remember`/`recall` tools, so it can choose to persist facts
mid-task.

## Hybrid retrieval

Knowledge mode fuses **dense** (embedding) and **lexical** (BM25) retrieval via
Reciprocal Rank Fusion — meaning *and* exact terms. An optional local
cross-encoder reranker sharpens the top results:

```bash
ATELIER_RERANK=1 atelier ask "..."         # enable reranking (downloads ~80MB once)
ATELIER_USE_HYBRID=0 atelier ask "..."     # dense-only, for comparison
```

## Use Atelier's tools from any MCP client

```bash
atelier mcp        # serves the full toolbox over MCP (stdio)
```

Point Claude Desktop/Code (or any MCP host) at the command `atelier mcp` to give
it Atelier's local tools: semantic note search, file edit, sandboxed code exec,
the pytest runner, repo map, and memory.

## Routing — a fine-tuned cheap component

A LoRA-fine-tuned **0.5B** model classifies a task easy/hard so easy subtasks can
go to the cheap worker and the 14B brain is reserved for hard ones:

```bash
atelier route "what is 47 * 89?"                       # -> easy  -> qwen3:4b
atelier route "refactor auth across the codebase"      # -> hard  -> qwen3:14b
make train-router                                      # reproduce the fine-tune (~1 min)
```

The fine-tune lifts the 0.5B router from **43.8% → 100%** held-out accuracy. See
[`docs/WRITEUP.md`](docs/WRITEUP.md).

## Reproduce everything

```bash
make reproduce        # env -> models -> tests -> eval -> train + evaluate router
make help             # all tasks
```

## Testing

A complete, ability-by-ability test playbook lives in
[`docs/TESTING.md`](docs/TESTING.md). The fast automated suite:

```bash
pytest -q        # 40+ tests, no model required
```

## Reliability evaluation

Atelier ships frozen task suites and a scored eval harness — the part that makes
it a *measured* system, not just a demo:

```bash
atelier eval                 # score both modes, save a JSON report
atelier eval --judge         # add a local LLM-as-judge for groundedness
```

Current baseline (M3 Pro, `qwen3:14b`): **doc-QA 8/8 correct** (citation 6/8),
**code 2/3 solved** — the unsolved task maps the local model's reliability
boundary (multi-line structural edits). Full results and honest analysis in
[`docs/EVAL.md`](docs/EVAL.md).

## Configuration

Everything is overridable via environment variables (prefix `ATELIER_`) or a
`.env` file. See [`atelier/config.py`](atelier/config.py). Common knobs:

| Variable | Default | Meaning |
|---|---|---|
| `ATELIER_BRAIN_MODEL` | `qwen3:14b` | main reasoning model |
| `ATELIER_EMBED_MODEL` | `BAAI/bge-base-en-v1.5` | local embedding model |
| `ATELIER_RETRIEVAL_K` | `6` | chunks retrieved per query |
| `ATELIER_CHUNK_SIZE` | `1000` | chunk size (characters) |
| `ATELIER_USE_HYBRID` | `true` | fuse dense + BM25 retrieval (RRF) |
| `ATELIER_RERANK` | `false` | cross-encoder reranking (downloads a small model) |

## Layout

```
atelier/   config, CLI, MCP server, banner
agent/     ReAct engine (react.py), brain (Ollama client), long-term memory
tools/     tool registry + tools (files, code_exec, test_runner, repo_map,
           search, search_notes, remember/recall, calculator, shell)
rag/       knowledge mode: ingest → chunk → embed → store → retrieve (hybrid) → answer
eval/      reliability harness: frozen suites, metrics, runner, regression gate, routing
models/    router/ — LoRA fine-tune (data, train, evaluate, adapter)
scripts/   reproduce.sh   ·   Makefile (make help)
docs/      ARCHITECTURE.md, TESTING.md, EVAL.md, WRITEUP.md
```

## License

Apache-2.0.
