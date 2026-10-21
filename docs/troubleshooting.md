# Troubleshooting: Common Warnings and Errors

None of the warnings below require action unless stated otherwise — they're listed here so they don't alarm new users.

---

## W1 — sqlglot unsupported syntax (SHOW / SET commands)

```
'SHOW TRANSACTION ISOLATION LEVEL' contains unsupported syntax. Falling back to parsing as a 'Command'.
```

**When:** During `process-history` with `--verbose`. Without `--verbose`, `process-history` sets the `sqlglot` logger to `ERROR`, so these warnings are not printed.

**Why:** `pg_stat_statements` records every query the database sees, including internal driver handshake queries (e.g. `psycopg2` checking session settings). The filter step discards statements that start with `SHOW`, `SET`, `RESET`, `BEGIN`, `COMMIT`, `ROLLBACK`, `DEALLOCATE`, `DECLARE`, `CLOSE`, `FETCH` or `DISCARD` before they reach sqlglot, so those no longer produce the warning; other statements sqlglot cannot parse fully can still produce it under `--verbose`.

**Risk:** None — statements that are not a `SELECT` are discarded by the filter step before they can influence the knowledge base.

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
✗ --embed requires Qdrant to be running at http://localhost:6333
  Start it with: docker run -d --name qdrant -p 127.0.0.1:6333:6333 qdrant/qdrant
  Or skip embedding with --no-embed and run it later.
  See docs/qdrant-setup.md for full setup instructions.
```

**When:** `process-history --embed` checks Qdrant's `/healthz` at `QDRANT_URL` before it reads any query history, and exits with status 1 and this message if Qdrant does not answer. `doc-ingest`, `doc-sync-notion` and `doc-sync-confluence` store their chunks in Qdrant and, if that fails, print `✗ Could not store the chunks in Qdrant: …` followed by `The earlier version was not deleted. Is Qdrant running?` and exit with status 1.

**Risk:** None to your data — `process-history` stops before the pipeline runs, and a failed document store leaves the earlier version in place.

**Action:** Start Qdrant (see [qdrant-setup.md](qdrant-setup.md)). For `process-history`, you can instead run without `--embed` (`--no-embed` is the default).

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
✗ --annotate requires an LLM credential.
  Set ANTHROPIC_API_KEY or OPENAI_API_KEY, or configure Bedrock, before running this command.
  To skip annotation, run with --no-annotate.
```

**When:** `process-history` runs with `--annotate` (the default) and no LLM credential is configured: no `ANTHROPIC_API_KEY`, no `OPENAI_API_KEY`, and neither `LLM_PROVIDER=bedrock` nor a `bedrock/` model in `LLM_MODEL`. The command exits with status 1 before it reads any query history.

**Why:** NLQueries defaults to Anthropic. When `LLM_PROVIDER` is not set, `OPENAI_API_KEY` is set and `ANTHROPIC_API_KEY` is not, the provider is detected as `litellm` and the model defaults to `openai/gpt-4o`, so the OpenAI key alone is enough. If both keys are set, Anthropic is used unless you set `LLM_PROVIDER` and `LLM_MODEL` explicitly.

**Action:**

```bash
# Anthropic
export ANTHROPIC_API_KEY=sk-ant-...

# OpenAI (LLM_PROVIDER and LLM_MODEL are only needed if ANTHROPIC_API_KEY is also set,
# or to pick a model other than openai/gpt-4o)
export OPENAI_API_KEY=sk-proj-...
export LLM_PROVIDER=litellm
export LLM_MODEL=openai/gpt-4o
```

To skip annotation entirely, use `--no-annotate`.

---

## W6 — pydantic v1 incompatibility (Python 3.14+)

**Status: Resolved** (see below).

```
UserWarning: Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.
```

**When (historical):** Any command that imported `langchain_text_splitters` on Python 3.14+ — that is, document ingestion only.

**Why:** Some LangChain-ecosystem packages use a `pydantic.v1` compatibility shim that relies on CPython internals removed in 3.14.

**Update (2026-07-01):** multi-agent/hybrid query routing (`query`) previously depended on `langgraph`/`langchain_core` for a simple classify → dispatch → merge flow. That dependency has been removed — routing is now plain async dispatch with no LangGraph/LangChain-core import, so `query`, intent classification, and hybrid queries are **no longer affected** by this issue on any Python version. `pyproject.toml` capped `requires-python` at `<3.14` as a backstop at the time; that cap was lifted on 2026-08-27 once the last LangChain-ecosystem dependency was gone and CI began testing 3.13 and 3.14.

**Resolved:** `langchain_text_splitters` has now been fully removed from the four document
connectors (PDF, Word, Notion, Confluence), replaced with a small built-in chunker
(`nlqueries.document_connectors.chunker`). Document ingestion no longer depends on any
LangChain-ecosystem package, so Python 3.14 is now fully supported across the entire project.

This section is kept for historical context. Prior impact, before the fix:

| Operation | Command | Impact |
|---|---|---|
| PDF / Word ingestion | `doc-ingest` | Chunking step could break at runtime |
| Notion / Confluence sync | `doc-sync-notion` / `doc-sync-confluence` | Chunking step could break at runtime |

---

## W7 — "Loading weights" progress bar on every command

```
Loading weights: 100%|████████████████| 103/103 [00:00<00:00, 12833.48it/s]
```

**When:** Every command that uses embeddings, even after the model is already downloaded.

**Why:** Each CLI invocation is a fresh process with no shared memory, so the ~80 MB model loads from local disk cache every time. This completes in well under a second.

**Risk:** None — loading from local cache, not downloading.

**Action:** None required. If startup time matters, run the embedding daemon (`nlqueries embed-server start`, see [cli-reference.md](cli-reference.md#embed-server)) or run NLQueries as a persistent server (MCP mode or Docker), so the model loads once and stays resident.

---

## W8 — Failed to read connectors file

```
Error: Failed to read connectors file: line 4: mapping values are not allowed here (in /home/you/.nlqueries/connectors.yaml)
```

**When:** Any command that reads `connectors.yaml` (`connectors` and every command given a connector ID or alias) finds that the file is not valid YAML. The command exits with status 1. `health` reports the same message as a failed `Database` check, runs its other checks, and exits with status 1; `health --json` still prints the full report. The MCP `list_connectors` tool returns the same message without the path.

**Why:** The file was edited by hand and a line no longer parses. A common cause is a value that contains `: `, such as `password: a: b`, which needs quotes.

**Risk:** None. Nothing is written, and the message gives the line number and the problem but not the line itself, which may hold a password.

**Action:** Open the file at the line named and fix it. Quote any value that contains `: ` or starts with a special character (`'a: b'`).
