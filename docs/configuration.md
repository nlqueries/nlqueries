# Configuration Reference

All settings are read from environment variables, or a `.env` file in the working directory (copy `.env.example` to `.env` to start).

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | One of these two | — | Anthropic API key |
| `OPENAI_API_KEY` | One of these two | — | OpenAI API key |
| `LLM_MODEL` | No | `claude-sonnet-4-5` | LLM model identifier |
| `LLM_PROVIDER` | No | Auto-detected | `anthropic`, `openai`, `bedrock`, or any LiteLLM provider. **Setting `OPENAI_API_KEY` alone does not switch the provider** — also set `LLM_PROVIDER=litellm` and `LLM_MODEL=openai/<model>` to use OpenAI. An `LLM_MODEL` starting `bedrock/` selects Bedrock on its own; see [Amazon Bedrock](#amazon-bedrock). |
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
| `LLM_MODEL_FAST` | No | `claude-haiku-4-5-20251001` (Anthropic) / `openai/gpt-4o-mini` (OpenAI) / same as `LLM_MODEL` (Bedrock) | Smaller/cheaper model used for short-output auxiliary calls (intent classifier, follow-up resolver). Set to any LiteLLM-supported model string. On Bedrock it defaults to your `LLM_MODEL` rather than to a Haiku id that does not exist there — correct but not cheap, so set a Bedrock Haiku id to get the cheap tier back. |
| `QUERY_HISTORY_LIMIT` | No | `500` | Maximum number of useful queries returned by `process-history` after filtering. Override per-run with `--max-queries`. |
| `EMBED_BACKEND` | No | `torch` | Embedding backend for the embed-server daemon. `torch` uses sentence-transformers/PyTorch (no extra deps); `onnx` uses ONNX Runtime via `optimum[onnxruntime]` (faster cold start, no PyTorch). |
| `NLQ_SCHEMA_FORMAT` | No | `compact` | Schema format injected into the system prompt. `compact` uses M-Schema (`【Table】 …`) — fewer tokens; `verbose` uses full Markdown (`### Table: …`) — backward compatible. |
| `NLQ_SELF_CONSISTENCY` | No | `off` | Self-consistency mode. `off` — disabled; `hard` — run N parallel SQL candidates only for queries classified as hard; `all` — always run N candidates and pick the majority answer. |
| `NLQ_CACHE_ANSWER_THRESHOLD` | No | `0.97` | Cosine similarity threshold for the Tier 1 answer cache. Questions within this distance of a cached question get the cached answer directly. |
| `NLQ_CACHE_TEMPLATE_THRESHOLD` | No | `0.90` | Cosine similarity threshold for the Tier 2 template cache. |
| `NLQ_CACHE_MAX_QUESTION_CHARS` | No | `500` | Longest question that may be **written** to the semantic cache; `0` disables the limit. A question over it is still answered normally, just not stored. Long questions are near-unique, so they cost a write and are rarely hit again — and a padded prompt injection has exactly that shape. |
| `NLQ_CACHE_ANSWER_TIERS` | No | `0,1,2` | Which cache tiers may serve an answer. `0` is an exact match on the normalised question; `1` is cosine similarity over stored answers; `2` binds the question's entities into a stored SQL template. Set to `0` for a sensitive agent to keep exact-match caching without the tiers that can serve one user's answer to another user's differently-worded question. |
| `NLQ_CACHE_COSINE_CANDIDATES` | No | `5` | How many neighbours the Tier 1 and Tier 2 searches fetch. The caller's cache context is matched inside the query, so these candidates already belong to the caller; the window exists because a candidate can still fail signature verification, the TTL, or entity binding. Raising it helps only where those failures are common. |
| `NLQ_CACHE_PRUNE_INTERVAL_SECONDS` | No | `3600` | How often a cache collection is swept for points past the TTL; `0` disables it. Nothing else deletes from the cache — the TTL is applied on read — so without a sweep a collection grows for the life of the agent, and expired points go on consuming the `NLQ_CACHE_COSINE_CANDIDATES` slots a lookup scans. The sweep runs on write, at most once per collection per interval **per process** — so a long-lived server sweeps hourly, while a CLI invocation sweeps once. The delete is issued without waiting, and `created_at` is indexed as a datetime on collections created from now on, so the work is a range query rather than a scan. |
| `NLQ_EXPLAIN_VALIDATION` | No | `false` | When `true`, runs `EXPLAIN` on the final generated SQL via the connector to validate query plans before returning an answer. |
| `NLQ_GLOSSARY_QUESTION_SCOPED` | No | `false` | When `true`, glossary terms are injected **per question** — only terms the question mentions, plus their [hierarchy](cli-reference.md#glossary-hierarchy) ancestors and descendants (depth 3) — instead of the whole glossary in the cached static prompt. Business rules are always injected in full. Off by default (the full glossary ships in the static block, exactly as before). |

### Amazon Bedrock

Bedrock is reached through LiteLLM, so there is no separate provider to install.
Set `LLM_MODEL` to a `bedrock/` model id and NLQueries routes there:

```bash
LLM_MODEL=bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0
AWS_REGION=us-east-1
```

**The region never fails loudly, so set it deliberately.** LiteLLM takes the
first of these that it finds: a region passed in code, one embedded in a model
ARN, `AWS_REGION_NAME`, `AWS_REGION`, then whatever a `boto3.Session()` resolves
(`AWS_DEFAULT_REGION`, or a profile in `~/.aws/config`). If none of them
produce a region it does **not** raise — it falls back to a built-in
`us-west-2`. A deployment that forgot the region therefore calls a real region
it never chose, where the model is very likely not enabled, and the error says
the model was not found. `AWS_REGION` above is honoured; `AWS_REGION_NAME` beats
it if both are set.

Authentication is boto3's, not an API key. NLQueries passes no AWS credentials of
its own, so the ordinary chain applies: `AWS_ACCESS_KEY_ID` /
`AWS_SECRET_ACCESS_KEY` (and `AWS_SESSION_TOKEN` for temporary credentials) in
the environment, a shared `~/.aws/config` profile, or — on EC2, ECS or EKS — the
instance profile, task role, or IRSA role of the host. On a machine that already
has an AWS role, no credential settings are needed at all.

Three things to check when it does not work, because each fails differently:

- **The model must be enabled** for your account in that region, under Bedrock →
  Model access in the AWS console. Until it is, every call returns AccessDenied
  however correct the credentials are.
- **The region must match** where the model is enabled. A model id from one
  region simply does not exist in another.
- **Newer Anthropic models require a cross-region inference profile**, which is
  the `us.` or `eu.` prefix inside the id — `bedrock/us.anthropic.claude-…`, not
  `bedrock/anthropic.claude-…`. The bare id is rejected as not found.

The IAM role needs `bedrock:InvokeModel` and
`bedrock:InvokeModelWithResponseStream` on the model or inference-profile ARN.

`LLM_PROVIDER=bedrock` also resolves to LiteLLM, but it does **not** replace the
model id: naming the provider does not choose a model, and the default
`LLM_MODEL` is an Anthropic id that LiteLLM would route to Anthropic. So
`LLM_PROVIDER=bedrock` requires an `LLM_MODEL` beginning `bedrock/`, and without
one the first LLM call fails with a message saying so. It fails there rather
than at startup deliberately — `connect`, `extract-schema` and the diagnostics
you would run to find the problem never touch an LLM, and should still work
while you are fixing it. Setting only the model works on its own and is the
shorter path.

**The model decides, wherever it is set.** A `bedrock/` id selects Bedrock even
when something else names a provider — before the request is built, not at the
API. That matters because the Anthropic client does not reject a `bedrock/`
model id: it would send the system prompt, the schema and the question to
`api.anthropic.com` and only then report that the model does not exist. For a
deployment that chose Bedrock to keep traffic inside its AWS account, the data
would already have left. A provider explicitly naming something other than
Bedrock alongside a `bedrock/` model is refused as the contradiction it is.

The same ordering applies to key-based detection: the `bedrock/` prefix is
checked before `ANTHROPIC_API_KEY`, so a Bedrock deployment that still has an
Anthropic key in its environment for something else keeps going to Bedrock.

---

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
