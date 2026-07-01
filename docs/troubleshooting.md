# Troubleshooting: Common Warnings and Errors

None of the warnings below require action unless stated otherwise — they're listed here so they don't alarm new users.

---

## W1 — sqlglot unsupported syntax (SHOW / SET commands)

```
'SHOW TRANSACTION ISOLATION LEVEL' contains unsupported syntax. Falling back to parsing as a 'Command'.
```

**When:** During `process-history` with a PostgreSQL connector.

**Why:** `pg_stat_statements` records every query the database sees, including internal driver handshake queries (e.g. `psycopg2` checking session settings). sqlglot doesn't recognise PostgreSQL `SHOW` commands and logs this before falling back to a generic representation.

**Risk:** None — these rows are discarded by the filter step before they can influence the knowledge base.

**Action:** None required.

---

## W2 — Hugging Face Hub unauthenticated requests

```
Warning: You are sending unauthenticated requests to the HF Hub.
```

**When:** First run of any embedding-dependent command (`--embed`, `doc-ingest`, semantic cache warm-up). Does not appear on later runs once the model is cached.

**Why:** The `all-MiniLM-L6-v2` sentence-transformer model downloads from Hugging Face Hub on first use and is cached at `~/.cache/huggingface/hub/`. Without `HF_TOKEN`, the download is unauthenticated and subject to shared rate limits.

**Risk:** Low — the model downloads once and is cached permanently. The only real risk is a `429` rate-limit error during the initial download.

**Action recommended:** Set a free Hugging Face token (Settings → Access Tokens → New token, Read access) via `HF_TOKEN=hf_...`. Not needed once the model is cached.

---

## W3 — Qdrant connection refused

```
Connection refused (os error 111)
# Windows: [WinError 10061] No connection could be made because the target machine actively refused it
```

**When:** `--embed` is passed to `process-history`, or the semantic cache tries to connect, and Qdrant isn't running.

**Risk:** None to your data — pipeline stages before the embed step already completed; only the Qdrant upload was skipped.

**Action:** Start Qdrant (see [qdrant-setup.md](qdrant-setup.md)), or rerun with `--no-embed` to continue without it.

---

## W4 — Docker daemon not running (Windows)

```
failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine
```

**When:** Any `docker` command on Windows when Docker Desktop is installed but not started.

**Action:** Open Docker Desktop and wait for the whale icon in the system tray to stop animating. Alternatively, use the native Qdrant binary ([qdrant-setup.md](qdrant-setup.md#option-d--native-binary-no-docker)), which needs no Docker at all.

---

## W5 — LLM authentication error

```
Could not resolve authentication method. Expected one of api_key, auth_token, or credentials to be set.
```

**When:** `--annotate` runs (or defaults on) with no LLM API key set.

**Why:** NLQueries defaults to Anthropic. Setting `OPENAI_API_KEY` alone does **not** switch the provider — you also need `LLM_PROVIDER=litellm` and `LLM_MODEL=openai/gpt-4o`.

**Action:**

```bash
# Anthropic
export ANTHROPIC_API_KEY=sk-ant-...

# OpenAI
export LLM_PROVIDER=litellm
export LLM_MODEL=openai/gpt-4o
export OPENAI_API_KEY=sk-proj-...
```

To skip annotation entirely, use `--no-annotate`.

---

## W6 — pydantic v1 incompatibility (Python 3.14+)

```
UserWarning: Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.
```

**When:** Any command that imports `langchain_text_splitters` on Python 3.14+ — that is, document ingestion only.

**Why:** Some LangChain-ecosystem packages use a `pydantic.v1` compatibility shim that relies on CPython internals removed in 3.14.

**Update (2026-07-01):** multi-agent/hybrid query routing (`query`) previously depended on `langgraph`/`langchain_core` for a simple classify → dispatch → merge flow. That dependency has been removed — routing is now plain async dispatch with no LangGraph/LangChain-core import, so `query`, intent classification, and hybrid queries are **no longer affected** by this issue on any Python version. `pyproject.toml` also now caps `requires-python` at `<3.14` as a backstop.

The remaining exposure is document ingestion, which still uses `langchain_text_splitters` for chunking:

| Operation | Command | Impact |
|---|---|---|
| PDF / Word ingestion | `doc-ingest` | Chunking step may break at runtime |
| Notion / Confluence sync | `doc-sync-notion` / `doc-sync-confluence` | Chunking step may break at runtime |

**Not affected:** `query`, `ask`, `connect`, `process-history`, `export-kb`, `doc-ask` (querying already-ingested documents), embedding and Qdrant operations, intent classification, hybrid queries.

**Action required:** `pip install` refuses to install on Python 3.14+ (see `pyproject.toml`'s `requires-python`), so this should only come up in an editable/source install that bypasses that check. Use Python 3.11 or 3.12 if you hit it:

```bash
python3.12 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
pip install -e ./core
```

Fully removing `langchain_text_splitters` from the four document connectors (replacing it with a small custom chunker) would close this out entirely; that's a reasonable fast-follow rather than a launch blocker, since document ingestion is a secondary feature relative to the core query path.

---

## W7 — "Loading weights" progress bar on every command

```
Loading weights: 100%|████████████████| 103/103 [00:00<00:00, 12833.48it/s]
```

**When:** Every command that uses embeddings, even after the model is already downloaded.

**Why:** Each CLI invocation is a fresh process with no shared memory, so the ~80 MB model loads from local disk cache every time. This completes in well under a second.

**Risk:** None — loading from local cache, not downloading.

**Action:** None required. If startup time matters, run the embedding daemon (`nlqueries embed-server start`, see [cli-reference.md](cli-reference.md#embed-server)) or run NLQueries as a persistent server (MCP mode or Docker), so the model loads once and stays resident.
