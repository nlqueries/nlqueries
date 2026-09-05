# Configuration Reference

All settings are read from environment variables, or a `.env` file in the working directory (copy `.env.example` to `.env` to start).

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | One of these two | — | Anthropic API key |
| `OPENAI_API_KEY` | One of these two | — | OpenAI API key |
| `LLM_MODEL` | No | `claude-sonnet-4-5` | LLM model identifier |
| `LLM_PROVIDER` | No | Auto-detected | `anthropic`, `openai`, or any LiteLLM provider. **Setting `OPENAI_API_KEY` alone does not switch the provider** — also set `LLM_PROVIDER=litellm` and `LLM_MODEL=openai/<model>` to use OpenAI. |
| `DATABASE_URL` | No | — | Connection string for the database being queried, e.g. `postgresql+psycopg2://user:password@localhost:5432/mydb` |
| `SSL_MODE` | No | `require` | TLS mode for the source database connection. `require` encrypts but verifies no certificate; use `verify-full` (with a CA) in production. `disable` restores plaintext, explicitly. |
| `SSL_CA_CERT` | No | — | Path to an SSL CA certificate bundle (e.g. for AWS RDS/Aurora with `verify-full`) |
| `QDRANT_URL` | No | `http://localhost:6333` | Qdrant URL. Required for `--embed`, the semantic cache, and document connectors. |
| `QDRANT_API_KEY` | No | — | Required if using Qdrant Cloud |
| `QDRANT_COLLECTION` | No | `nlqueries` | Qdrant collection name |
| `NLQ_STATE_DIR` | No | `~/.nlqueries` | Root for everything NLQueries keeps between runs. The paths below default under it; set them individually to override. Set this when the home directory is not writable — a read-only root filesystem otherwise stops the embedding server starting, and with it every natural-language query |
| `KB_PATH` | No | `~/.nlqueries/knowledge_base` | Local path for exported knowledge base files |
| `KB_REFRESH_INTERVAL` | No | `3600` | Seconds between auto-refresh of the KB (`0` disables) |
| `CONNECTORS_FILE` | No | `~/.nlqueries/connectors.yaml` | Path to the connector registry |
| `CAPSULES_DIR` | No | `~/.nlqueries/capsules` | Path to saved query capsules |
| `FEEDBACK_DIR` | No | `~/.nlqueries/feedback` | Path to feedback JSONL files |
| `NOTION_API_TOKEN` | Only for Notion sync | — | Notion integration token |
| `CONFLUENCE_URL` / `CONFLUENCE_USER` / `CONFLUENCE_API_TOKEN` | Only for Confluence sync | — | Confluence connection details |
| `HF_TOKEN` | No | — | Hugging Face token — avoids rate limits on the one-time embedding model download. See [troubleshooting.md](troubleshooting.md#w2--hugging-face-hub-unauthenticated-requests). |
| `EMBED_SERVER_PORT` | No | `8765` | Port the embedding daemon listens on |
| `LOG_LEVEL` | No | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | No | — | OTLP endpoint for traces (disabled if unset) |
| `LLM_MODEL_FAST` | No | `claude-haiku-4-5-20251001` (Anthropic) / `openai/gpt-4o-mini` (OpenAI) | Smaller/cheaper model used for short-output auxiliary calls (intent classifier, follow-up resolver). Set to any LiteLLM-supported model string. |
| `QUERY_HISTORY_LIMIT` | No | `500` | Maximum number of useful queries returned by `process-history` after filtering. Override per-run with `--max-queries`. |
| `EMBED_BACKEND` | No | `torch` | Embedding backend for the embed-server daemon. `torch` uses sentence-transformers/PyTorch (no extra deps); `onnx` uses ONNX Runtime via `optimum[onnxruntime]` (faster cold start, no PyTorch). |
| `NLQ_SCHEMA_FORMAT` | No | `compact` | Schema format injected into the system prompt. `compact` uses M-Schema (`【Table】 …`) — fewer tokens; `verbose` uses full Markdown (`### Table: …`) — backward compatible. |
| `NLQ_SELF_CONSISTENCY` | No | `off` | Self-consistency mode. `off` — disabled; `hard` — run N parallel SQL candidates only for queries classified as hard; `all` — always run N candidates and pick the majority answer. |
| `NLQ_CACHE_ANSWER_THRESHOLD` | No | `0.97` | Cosine similarity threshold for the Tier 1 answer cache. Questions within this distance of a cached question get the cached answer directly. |
| `NLQ_CACHE_TEMPLATE_THRESHOLD` | No | `0.90` | Cosine similarity threshold for the Tier 2 template cache. |
| `NLQ_CACHE_MAX_QUESTION_CHARS` | No | `500` | Longest question that may be **written** to the semantic cache; `0` disables the limit. A question over it is still answered normally, just not stored. Long questions are near-unique, so they cost a write and are rarely hit again — and a padded prompt injection has exactly that shape. |
| `NLQ_CACHE_ANSWER_TIERS` | No | `0,1,2` | Which cache tiers may serve an answer. `0` is an exact match on the normalised question; `1` is cosine similarity over stored answers; `2` binds the question's entities into a stored SQL template. Set to `0` for a sensitive agent to keep exact-match caching without the tiers that can serve one user's answer to another user's differently-worded question. |
| `NLQ_CACHE_COSINE_CANDIDATES` | No | `5` | How many neighbours the Tier 1 and Tier 2 searches fetch. The cache-context equality is applied after the search — Qdrant's filter can require the caller's keys but not the absence of others — so fetching one point lets a neighbour from another context shadow a valid entry ranked below it. Raise it on a busy conversational agent, where scoped entries accumulate one per conversation context for the same question. |
| `NLQ_CACHE_PRUNE_INTERVAL_SECONDS` | No | `3600` | How often a cache collection is swept for points past the TTL; `0` disables it. Nothing else deletes from the cache — the TTL is applied on read — so without a sweep a collection grows for the life of the agent, and expired points go on consuming the `NLQ_CACHE_COSINE_CANDIDATES` slots a lookup scans. The sweep runs on write, at most once per collection per interval. |
| `NLQ_EXPLAIN_VALIDATION` | No | `false` | When `true`, runs `EXPLAIN` on the final generated SQL via the connector to validate query plans before returning an answer. |
| `NLQ_GLOSSARY_QUESTION_SCOPED` | No | `false` | When `true`, glossary terms are injected **per question** — only terms the question mentions, plus their [hierarchy](cli-reference.md#glossary-hierarchy) ancestors and descendants (depth 3) — instead of the whole glossary in the cached static prompt. Business rules are always injected in full. Off by default (the full glossary ships in the static block, exactly as before). |

**Windows note:** `~` in default paths resolves to `C:\Users\<YourUsername>` in PowerShell. To set a variable for the current session use `$env:VAR = "value"`; to persist it, use **System Properties → Environment Variables** or add it to your PowerShell profile.

---

## Docker Compose services and volumes

Running `docker compose up` from `core/` starts:

| Service | Port | Purpose |
|---|---|---|
| `qdrant` | 6333 (REST), 6334 (gRPC) | Vector store for embeddings, semantic cache, document search |
| `nlqueries-core` | 8080 | MCP server + CLI engine |

| Volume | Persists |
|---|---|
| `qdrant-data` | Qdrant collections across restarts |
| `nlqueries-data` | Knowledge bases, connector config, capsules, and feedback (mounted at `/data/nlqueries` in the container) |

Run CLI commands inside the container with `docker exec -it nlqueries-core nlqueries <command>` — when connecting to a database on your host machine, use `--host host.docker.internal`.
