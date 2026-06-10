# Getting Started with NLQueries Core

This guide walks you from zero to your first natural-language SQL query in five steps.

**Time to complete:** ~10 minutes (excluding data-loading time in your database).

---

## Prerequisites

- Python 3.11+ **or** Docker + Docker Compose
- A running PostgreSQL, MySQL, BigQuery, or Snowflake database
- An API key for an LLM provider (OpenAI, Anthropic, or any [LiteLLM-supported model](https://docs.litellm.ai/docs/providers))
- *(Optional)* A running [Qdrant](https://qdrant.tech/) instance for semantic search (the Docker Compose stack starts one automatically)

---

## Step 1 — Install nlqueries-core

### Option A: Docker Compose (includes Qdrant)

```bash
git clone https://github.com/nlqueries/nlqueries.git
cd nlqueries
cp .env.core.example .env.core
```

Edit `.env.core` and set at minimum:

```bash
OPENAI_API_KEY=sk-...          # or ANTHROPIC_API_KEY / any LiteLLM key
LLM_MODEL=gpt-4o-mini          # or claude-3-5-haiku-latest, etc.
```

Start the stack:

```bash
docker compose -f infra/docker-compose.core.yml up -d
```

Open a shell into the container for the steps below:

```bash
docker exec -it nlqueries-core bash
```

### Option B: pip install

```bash
pip install nlqueries-core
```

Set environment variables:

```bash
export OPENAI_API_KEY=sk-...
export LLM_MODEL=gpt-4o-mini
export QDRANT_URL=http://localhost:6333   # if you have Qdrant running
```

Verify the install:

```bash
nlqueries --version
```

---

## Step 2 — Connect a Database

Register your database connection. NLQueries stores the connection config in `~/.nlqueries/connectors.yaml` so you only need to do this once per database.

### PostgreSQL

```bash
nlqueries connect postgres \
    --host localhost \
    --database mydb \
    --user alice \
    --password secret
```

### MySQL

```bash
nlqueries connect mysql \
    --host localhost \
    --database mydb \
    --user alice \
    --password secret
```

### Snowflake

```bash
nlqueries connect snowflake \
    --account acme-prod \
    --database PROD \
    --user bob \
    --password s3cr3t \
    --warehouse COMPUTE_WH \
    --schema PUBLIC
```

### BigQuery (using Application Default Credentials)

```bash
nlqueries connect bigquery \
    --project-id acme-prod \
    --dataset-id analytics
```

On success you'll see:

```
✓ Connection successful.
  Connector registered as 'postgres:localhost:mydb'
```

The connector ID (e.g. `postgres:localhost:mydb`) is used in all subsequent commands.

---

## Step 3 — Extract the Schema

Inspect your database schema. NLQueries reads table definitions, column types, primary keys, and foreign key relationships.

```bash
nlqueries extract-schema postgres:localhost:mydb
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

Build the knowledge base by processing your database's recent query history. This step reads raw query logs, deduplicates and parameterizes them, clusters queries by intent, and emits `QueryCapsule` objects — annotated query templates that give the LLM contextual examples of how your schema is actually used.

```bash
nlqueries process-history postgres:localhost:mydb \
    --days 90 \
    --min-executions 3
```

> **Tip:** Add `--embed` to immediately upsert the capsules into Qdrant for semantic search:
> ```bash
> nlqueries process-history postgres:localhost:mydb --days 90 --embed
> ```

Expected output:

```
✓ Pipeline complete.
  Capsules produced : 47
  Annotated         : 47 / 47
  Saved to          : ~/.nlqueries/capsules/postgres:localhost:mydb.jsonl
```

If your database has no query history (e.g. a fresh dev DB), the pipeline produces zero capsules — that's fine. The knowledge base still includes full schema context.

---

## Step 5 — Export the Knowledge Base

Generate the YAML knowledge base. This bundles your schema, sample rows, and query capsules into a single file optimised for LLM context injection.

```bash
nlqueries export-kb postgres:localhost:mydb --output kb.yaml
```

Expected output:

```
✓ Knowledge base written to kb.yaml
  Tables   : 12
  Columns  : 87
  Capsules : 47
```

Inspect the file:

```bash
head -60 kb.yaml
```

The knowledge base is human-readable YAML — you can manually annotate tables and columns (`description:` fields) to improve SQL generation accuracy. Re-run `export-kb` after edits to refresh the file; manual annotations are preserved.

---

## Step 6 — Ask a Question

You're ready. Ask a natural-language question and watch NLQueries stream a reasoning response followed by validated SQL:

```bash
nlqueries ask postgres:localhost:mydb \
    "How many orders did we ship last month?"
```

Output:

```
To answer this question I'll look at the orders table and filter by the
shipped_at timestamp for the previous calendar month...

{"sql": "SELECT COUNT(*) AS orders_shipped\nFROM public.orders\nWHERE DATE_TRUNC('month', shipped_at) = DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')\n  AND status = 'shipped'", "is_valid": true, "dialect": "postgres", "attempt_count": 1}
```

Try a few more:

```bash
nlqueries ask postgres:localhost:mydb "Top 10 customers by total revenue this year"
nlqueries ask postgres:localhost:mydb "Average order value by product category" --dialect postgres
```

---

## What's Next

### Annotate capsules manually

If you ran `process-history --no-annotate`, annotate later:

```bash
nlqueries annotate postgres:localhost:mydb
```

### View feedback stats

After using the enterprise chat UI or API, review captured feedback:

```bash
nlqueries feedback-stats postgres:localhost:mydb
```

### Connect an AI assistant via MCP

The Docker Compose stack runs an MCP server on port 8080. Configure your MCP-compatible AI assistant (Claude, Cursor, etc.) to connect to `http://localhost:8080` to expose NLQueries tools directly.

### Run as a library

```python
import asyncio
from nlqueries.orchestrator import Orchestrator

async def main() -> None:
    orchestrator = Orchestrator()
    async for token in orchestrator.handle_question(
        "How many orders last month?",
        agent_id="postgres:localhost:mydb",
        dialect="postgres",
    ):
        print(token, end="", flush=True)

asyncio.run(main())
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Connector 'X' not found` | Run `nlqueries connect <db-type> ...` first |
| `Connection failed: ...` | Check host/port/credentials; verify the DB is reachable from your machine (or from inside Docker) |
| `No capsules found` | Run `nlqueries process-history <connector-id>` first |
| `LLM call failed` | Check your API key is set and valid; verify `LLM_MODEL` matches your provider |
| Qdrant connection refused | Start Qdrant (`docker compose ... up qdrant`) or unset `QDRANT_URL` to skip embedding |

---

For more detail see the [README](../README.md) or open an [issue](https://github.com/nlqueries/nlqueries/issues).
