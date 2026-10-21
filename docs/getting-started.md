# Getting Started with NLQueries Core

This guide walks you from zero to your first natural-language query in six steps.

---

## Prerequisites

- Python 3.11–3.14 **or** Docker + Docker Compose.
- A running PostgreSQL, MySQL, Snowflake, BigQuery, Redshift, SQL Server/Azure SQL, DuckDB, or SQLite database (SQLite and DuckDB are file-based and need no server)
- An API key for an LLM provider (Anthropic, OpenAI, or any [LiteLLM-supported model](https://docs.litellm.ai/docs/providers))
- *(Optional)* A running [Qdrant](https://qdrant.tech/) instance — required for `--embed`, the semantic cache, and document connectors. The Docker Compose stack starts one automatically. See [qdrant-setup.md](qdrant-setup.md) if you're not using Docker.

---

## Step 1 — Install nlqueries-core

### Option A: Docker (includes Qdrant, no clone needed)

Pulls the published [`nlqueries/core`](https://hub.docker.com/r/nlqueries/core) image from Docker Hub:

```bash
curl -O https://raw.githubusercontent.com/nlqueries/nlqueries/main/docker-compose.yml
```

Create a `.env` file next to it with an LLM key, a Qdrant key and an MCP
token. Compose refuses to start without `QDRANT_API_KEY` or
`NLQ_MCP_STATIC_TOKEN`:

```bash
ANTHROPIC_API_KEY=sk-ant-...    # or OPENAI_API_KEY
QDRANT_API_KEY=...              # any random secret: openssl rand -hex 32
NLQ_MCP_STATIC_TOKEN=...        # another one: openssl rand -hex 32
```

An MCP client connects to `http://localhost:8080/sse` and sends
`Authorization: Bearer <your NLQ_MCP_STATIC_TOKEN>`. See
[MCP authentication](mcp-authentication.md) to give different people
different access.

Start the stack:

```bash
docker compose up -d
```

This reads `docker-compose.yml` in the current directory automatically — no `-f` flag needed — and pulls `nlqueries/core:latest` alongside Qdrant. Open a shell into the container for the steps below:

```bash
docker exec -it nlqueries-core bash
```

Or run individual commands directly without a shell:

```bash
docker exec -it nlqueries-core nlqueries health
```

The image includes the PostgreSQL, SQLite, Snowflake, BigQuery and DuckDB drivers. It does **not** include the MySQL, Redshift or SQL Server drivers, or the `[docs]` (PDF, Word, Excel) and `[wiki]` (Notion, Confluence) extras. For those, use Option B or C and install the extra you need.

### Option B: pip install

```bash
pip install nlqueries-core
```

Set environment variables:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export QDRANT_URL=http://localhost:6333   # if you have Qdrant running
```

Or put the same lines (without `export`) in a `.env` file in the directory you run `nlqueries` from. A `.env` in a parent directory is not read. A variable already set in your environment takes precedence over the file.

Verify the install:

```bash
nlqueries health
# or the shorter alias:
nlq health
```

`health` probes every service NLQueries depends on (LLM key, Qdrant, embedding daemon, config) and prints a pass/fail summary. See [cli-reference.md](cli-reference.md#health) for details.

### Option C: Clone and install from source (no Docker)

For contributing, or to run against unreleased changes:

```bash
git clone https://github.com/nlqueries/nlqueries.git
cd nlqueries
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=sk-ant-...
nlqueries health
```

See [CONTRIBUTING.md](../CONTRIBUTING.md#development-setup) for linting and test commands.

---

## Step 2 — Connect a Database

Register your database connection. NLQueries stores the connection metadata in `~/.nlqueries/connectors.yaml` so you only need to do this once per database. The password itself is stored separately, in your OS keychain (via the `keyring` package), not in that file — see [cli-reference.md](cli-reference.md#connect) for what happens if keyring isn't available on your machine. Nothing is sent anywhere except the database itself.

```bash
nlqueries connect postgres \
    --host localhost \
    --database mydb \
    --user alice \
    --password secret \
    --alias dev
```

If you run the CLI inside the Docker container (Option A), `localhost` is the container itself, not your machine: pass the database's host name or address as the container sees it. On Docker Desktop, `host.docker.internal` reaches a database running on your machine.

The `--alias dev` lets you type `dev` instead of the full connector ID (`postgres:localhost:mydb`) on every later command. See [cli-reference.md](cli-reference.md#connect) for MySQL, Snowflake, BigQuery, Redshift, SQL Server, and DuckDB connection examples, and [connectors.md](connectors.md) for per-database setup notes (e.g. enabling query history capture).

On success you'll see:

```
✓ Connection successful.
  Connector registered as 'postgres:localhost:mydb'
  Alias               : dev
  Config saved to ~/.nlqueries/connectors.yaml
  ✓ Password stored in OS keychain (not written to the config file).
```

Where no keychain service is available, as inside the Docker container, the last line is a note instead: the password is kept in the config file.

---

## Step 3 — Extract the Schema

```bash
nlqueries extract-schema dev
```

Expected output:

```
✓ Schema extraction complete
  Database: mydb
  Tables  : 12
  Columns : 87 total across all tables

  Schema   Table            Columns  Rows
  public   orders           8        142,871
  public   order_items      6        398,204
  public   customers        9        24,503
  ...
```

---

## Step 4 — Process Query History

Build the knowledge base by processing your database's recent query history. This reads raw query logs, deduplicates and parameterizes them, clusters queries by intent, and emits `QueryCapsule` objects — annotated query templates that give the LLM contextual examples of how your schema is actually used.

```bash
nlqueries process-history dev --days 30 --annotate
```

> **Getting zero or very few capsules?** The default `--min-executions 3` drops any query executed fewer than 3 times. On a fresh or lightly-used database, lower it to `--min-executions 1`. Also make sure you've actually run some representative business queries against the database first — see [cli-reference.md](cli-reference.md#process-history) for why a brand-new database often produces nothing useful on the first pass.

> Add `--embed` to immediately upload the capsules to Qdrant for semantic search (requires Qdrant running — see [qdrant-setup.md](qdrant-setup.md)):
> ```bash
> nlqueries process-history dev --days 30 --annotate --embed
> ```

Expected output:

```
✓ Pipeline complete.
  Queries scanned    : 142
  Capsules produced   : 12
  Annotated           : 12 / 12
  Saved to            : ~/.nlqueries/capsules/postgres_localhost_mydb.json
```

Files are named after the connector ID, not the alias, with `:` and other unsafe characters replaced by `_`.

If your database has little or no query history (e.g. a fresh dev DB), the pipeline produces few or zero capsules — that's fine. SQLite, DuckDB and the generic SQLAlchemy connector (which MySQL uses) have no query history to read, so they always produce zero. Where the history cannot be read at all — PostgreSQL without the `pg_stat_statements` extension, for example — `process-history` stops with `✗ Query history unavailable:` and the reason. Either enable history capture (see [connectors.md](connectors.md#postgresql--enabling-query-history-capture)) or go straight to Step 5. Either way the knowledge base still includes full schema context.

---

## Step 5 — Export the Knowledge Base

This step is required before `query` or `ask` will work, and must be re-run any time you run `process-history` again.

```bash
nlqueries export-kb dev
```

Expected output:

```
✓ Knowledge base written to ~/.nlqueries/knowledge_base/postgres_localhost_mydb.yaml
  Tables   : 12
  Columns  : 87
  Capsules : 12
```

The knowledge base is human-readable YAML — you can manually annotate tables and columns (`description:` fields) to improve SQL generation accuracy. Re-run `export-kb` after edits; manual annotations are preserved. Check its coverage any time with:

```bash
nlqueries kb-stats dev
```

---

## Step 6 — Ask a Question

Two commands answer questions — pick based on what you need:

| Command | What it does |
|---|---|
| `nlqueries query` | Generates SQL, **executes it against the database**, and prints the result rows. Use this to get data. |
| `nlqueries ask` | Generates and validates SQL **without executing it**. Use this to preview what SQL will be produced. |

```bash
nlqueries query dev "How many orders did we ship last month?"
```

Output:

```
Agent type : sql
SQL        : SELECT COUNT(*) FROM orders WHERE shipped_at >= '2026-05-01' ...
┏━━━━━━━┓
┃ count ┃
┡━━━━━━━┩
│ 4382  │
└───────┘
1 row(s) in 12.4 ms
Latency    : 1243 ms
```

Add `--json` for the full structured result including rows:

```bash
nlqueries query dev "How many orders did we ship last month?" --json
```

Try a few more:

```bash
nlqueries query dev "Top 10 customers by total revenue this year"
nlqueries ask dev "Average order value by product category"   # preview SQL only, no execution
```

> **Every query takes ~9 s longer than expected?** Without the embedding daemon, each `nlqueries` command that embeds text loads the embedding model itself first. Start the daemon once (`nlqueries embed-server start`) and later commands use it instead, at ~10 ms per embedding call. See [cli-reference.md](cli-reference.md#embed-server).

---

## What's Next

### Give feedback on an answer

```bash
nlqueries feedback dev --question "Orders last month" --thumbs-up
nlqueries feedback dev --question "Orders last month" --thumbs-down --corrected-sql "SELECT ..."
nlqueries feedback-stats dev
```

`feedback-stats` accepts the alias, like `feedback`. For a connector with no feedback recorded it prints "No feedback recorded yet" — that's expected, not an error.

### Ingest documents and ask about them

```bash
nlq doc-ingest q1-report ./report.pdf
nlq doc-ask doc_q1-report_chunks "What did the Q1 report say about churn?"
```

`doc-ingest` stores the report's chunks in Qdrant under `doc_q1-report_chunks`, and `doc-ask` answers from them with citations. PDF, Word and Excel need the `[docs]` extra; Markdown and plain text work on a base install. Questions to `nlqueries query` are answered from the database. See [connectors.md](connectors.md#document-connectors).

### Connect an AI assistant via MCP

The Docker Compose stack runs an MCP server on port 8080. See [cli-reference.md](cli-reference.md#mcp-server) for the Claude Desktop config.

### Run as a library

`agent_id` is the full connector ID: the library does not resolve aliases, the CLI does. By default the library generates SQL without running it (`result.sql_result` is `None`); pass `execution=ExecutionPolicy.execute_read_only()` to run it and get the rows.

```python
# Async
import asyncio
from nlqueries.execution import ExecutionPolicy
from nlqueries.orchestrator.sync_runner import run_query


async def main() -> None:
    result = await run_query(
        question="How many orders last month?",
        agent_id="postgres:localhost:mydb",  # connector ID, not an alias
        execution=ExecutionPolicy.execute_read_only(),
    )
    print(result.sql, result.sql_result, result.agent_type)


asyncio.run(main())
```

```python
# Synchronous (blocking)
from nlqueries.execution import ExecutionPolicy
from nlqueries.orchestrator.sync_runner import run_query_sync

result = run_query_sync(
    question="How many orders last month?",
    agent_id="postgres:localhost:mydb",
    execution=ExecutionPolicy.execute_read_only(),
)
print(result.sql, result.sql_result, result.agent_type)
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Connector 'X' not found` | Run `nlqueries connect <db-type> ...` first |
| `Connection failed: ...` | Check host/port/credentials; verify the DB is reachable from your machine (or from inside Docker) |
| `No capsules found` | Run `nlqueries process-history <connector>` first |
| `✗ Query failed: ...` with an authentication error, or `✗ --annotate requires an LLM credential.` | Check your API key is set and valid; see [troubleshooting.md#w5](troubleshooting.md#w5--llm-authentication-error) |
| Qdrant connection refused | Start Qdrant (see [qdrant-setup.md](qdrant-setup.md)). For `process-history`, you can instead leave out `--embed` (it is off by default) |
| `feedback-stats` errors on a new connector | Should print "No feedback recorded yet" — if it errors instead, please [open an issue](https://github.com/nlqueries/nlqueries/issues) |

For the full list of warnings you may see and what they mean, see [troubleshooting.md](troubleshooting.md).

For more detail see the [README](../README.md), the [CLI reference](cli-reference.md), or open an [issue](https://github.com/nlqueries/nlqueries/issues).
