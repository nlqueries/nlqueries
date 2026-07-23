# CLI Command Reference

Every command is available as `nlqueries <command>` or the shorter `nlq <command>` — both entry points are identical.

```bash
nlqueries --help
nlqueries <command> --help
```

Anywhere a connector ID (`<type>:<host>:<database>`) is accepted, a configured `--alias` works too — see [Connector aliases](#connector-aliases).

---

## connect

Register a database connection. Connection metadata (host, port, database, user) is written to `~/.nlqueries/connectors.yaml` (mode `0600`).

**Password storage:** the password itself does not go in that file. `connect` stores it in your OS keychain (macOS Keychain, Windows Credential Manager, or Linux Secret Service, via the `keyring` package) and prints `✓ Password stored in OS keychain` on success. If keyring is unavailable on the machine — headless CI without a secret service, for example — it falls back to embedding the password in the stored connection URL instead, and prints a warning telling you so. To rotate a password later without re-running `connect`, use `nlqueries update-password <connector-id>`.

**Password handling:** don't put `--password` on the command line if you can avoid it — it lands in shell history. Two safer options: omit `--password` entirely and you'll get an interactive, hidden prompt, or use `--password-env VAR` to read it from an already-set environment variable. (`--password` is shown in the examples below only for brevity.)

**PostgreSQL / MySQL**

```bash
nlqueries connect postgres --host db.example.com --port 5432 --database prod --user alice --alias prod   # prompts for password
nlqueries connect postgres --host db.example.com --database prod --user alice --password-env PGPASSWORD --alias prod
nlqueries connect mysql --host localhost --database mydb --user alice --password secret
```

SSL for PostgreSQL/MySQL connections is configured via the `SSL_MODE` / `SSL_CA_CERT` environment variables, not a `connect` flag — see [configuration.md](configuration.md).

**Snowflake** — install `pip install "nlqueries-core[snowflake]"` first

```bash
nlqueries connect snowflake --account myorg.us-east-1 --database ANALYTICS --schema PUBLIC --warehouse COMPUTE_WH --user alice --password s3cr3t
```

**BigQuery** — install `pip install "nlqueries-core[bigquery]"` first (uses Application Default Credentials if `--service-account-json` is omitted)

```bash
nlqueries connect bigquery --project-id my-gcp-project --dataset-id my_dataset --service-account-json /path/to/service-account.json
```

BigQuery has no password — `--project-id` (or `--database`, used as a fallback) is the only required flag.

**Amazon Redshift** — install `pip install "nlqueries-core[redshift]"` first

```bash
nlqueries connect redshift --host my-cluster.abc123.us-east-1.redshift.amazonaws.com --port 5439 --database dev --user awsuser --password s3cr3t --alias redshift-prod
```

**SQL Server / Azure SQL** — install `pip install "nlqueries-core[mssql]"` first

```bash
nlqueries connect mssql --host my-server.database.windows.net --port 1433 --database mydb --user alice --password s3cr3t --alias sql-prod
```

**DuckDB** — install `pip install "nlqueries-core[duckdb]"` first; file-based, no host/port

```bash
nlqueries connect duckdb --database /data/warehouse.db --alias duck-local   # persistent
nlqueries connect duckdb --alias duck-mem                                   # in-memory, transient
```

See [connectors.md](connectors.md) for per-database query-history and schema-introspection caveats (Redshift, MSSQL, and DuckDB in particular).

---

## update-password

Rotate the stored password for a connector without re-running `connect`. The new password replaces whatever is in the OS keychain (or the config-file fallback); the connector's other settings are untouched.

```bash
nlqueries update-password postgres:localhost:mydb   # prompts for the new password, hidden input
nlqueries update-password dev --password-env DB_PASSWORD
```

`CONNECTOR_ID` accepts an alias. This does not re-test the connection — run `nlqueries extract-schema <connector-or-alias>` afterwards to confirm the new password actually works. If keyring is unavailable, the command prints a warning instead of silently writing the password to the config file.

---

## Connector aliases

```bash
nlqueries alias postgres:db.example.com:prod prod        # add/update an alias
nlqueries alias postgres:db.example.com:prod ""           # remove an alias
nlqueries connectors                                       # list connectors and aliases
```

Once set, `prod` works anywhere `postgres:db.example.com:prod` would.

---

## extract-schema

Inspect a registered connector and print schema statistics — table/column counts, types, foreign keys, row estimates. No data is read.

```bash
nlqueries extract-schema <connector-or-alias>
```

---

## health

Probes every dependent service and prints a pass/fail summary. Run after a fresh install or in a CI health gate. Exit code `0` if all checks pass, `1` otherwise.

```bash
nlqueries health
nlqueries health --connector <alias>   # check a single connector only
nlqueries health --verbose             # full error traces
nlqueries health --json                # for CI / scripting
```

Status meanings: `[OK]` reachable, `[FAIL]` unreachable/misconfigured (exit 1), `[WARN]` not required but affects performance (e.g. embedding daemon not running), `[SKIP]` not applicable yet (e.g. no connectors registered).

---

## process-history

Runs the query-capsule pipeline over recent query history: reads raw logs, deduplicates and parameterizes queries, clusters by intent, optionally annotates with LLM-generated descriptions and embeds into Qdrant.

```bash
nlqueries process-history <connector-or-alias> --days 30 --annotate --embed
```

| Flag | Default | Description |
|---|---|---|
| `--days` | `90` | How far back to read query history |
| `--annotate` | on | Generate LLM intent descriptions per cluster |
| `--embed` | off | Upload embeddings to Qdrant (requires Qdrant running) |
| `--min-executions` | `3` | Minimum executions for a query cluster to be kept |
| `--max-queries` | `500` (env: `QUERY_HISTORY_LIMIT`) | Cap on useful queries to process after filtering |
| `--verbose` | off | Print sqlglot parse warnings and LLM provider log lines |

Supported history sources: PostgreSQL (`pg_stat_statements`, must be enabled — see [connectors.md](connectors.md#postgresql)), Snowflake (`QUERY_HISTORY`), BigQuery (`INFORMATION_SCHEMA.JOBS`), MySQL (performance schema). Redshift, SQL Server, and DuckDB have their own caveats — see [connectors.md](connectors.md).

A fresh or lightly-used database will produce few or zero capsules until it has real query history to learn from — run some representative business queries first, and consider `--min-executions 1` on a dev database.

---

## export-kb

Reads the capsules saved by `process-history` together with the live schema and writes the KB YAML that `query`/`ask` read. **Required before `query`/`ask` will work**, and must be re-run after every `process-history` run.

```bash
nlqueries export-kb <connector-or-alias> [--output kb.yaml] [--sample-rows 3] [--no-include-samples] [--describe-columns]
```

Default output path: `~/.nlqueries/knowledge_base/<connector-id>.yaml` (`:` replaced with `_`).

| Flag | Default | Description |
|---|---|---|
| `--output` / `-o` | `~/.nlqueries/knowledge_base/<id>.yaml` | Path to write the YAML KB |
| `--include-samples` / `--no-include-samples` | on | Include sample rows per table |
| `--sample-rows` | `3` | Number of sample rows per table |
| `--describe-columns` | off | Use the LLM to auto-populate column descriptions from sample data (skips surrogate-key columns; requires LLM key) |

---

## kb-stats

Structured coverage and quality report for the exported knowledge base: schema coverage, description coverage, query capsule / intent coverage, FK join coverage, cache entries, and feedback history.

```bash
nlqueries kb-stats <connector-or-alias>
nlqueries kb-stats <connector-or-alias> --verbose   # per-table/per-column breakdown
nlqueries kb-stats <connector-or-alias> --json
```

Exit code `0` if the KB exists, `1` if missing — useful as a CI gate before deployment. Works from the local KB file alone if the database is unreachable (DB-derived fields show `N/A`).

---

## annotate

Backfills LLM intent descriptions on capsules that don't have one yet (e.g. after `process-history --no-annotate`).

```bash
nlqueries annotate <connector-or-alias>
```

---

## query

Generates SQL, **executes it against the database**, and returns the natural-language answer plus result rows.

```bash
nlqueries query <connector-or-alias> "Top 10 customers by revenue this year"
nlqueries query <connector-or-alias> "..." --json        # structured output including rows
nlqueries query <connector-or-alias> "..." --no-execute  # generate SQL without running it
nlqueries query <connector-or-alias> "..." --new-session # discard prior conversation context
```

Output includes agent type, answer, generated SQL, and latency. The orchestrator routes each question to a SQL agent, a document agent, or both in parallel (hybrid) — see [Multi-agent routing](#multi-agent-routing).

| Flag | Default | Description |
|---|---|---|
| `--json` | off | Emit the full result as raw JSON (includes rows, citations, latency) |
| `--execute` / `--no-execute` | on | Execute the generated SQL and display rows |
| `--session` / `--no-session` | on | Carry conversation context across queries for follow-up questions |
| `--new-session` | off | Start a fresh conversation, discarding prior context |

Conversational context is persisted per agent in `~/.nlqueries/sessions/` and carried forward automatically between calls.

---

## eval

Regression-checks an agent's knowledge: re-asks each mined **query capsule** (and, with `--golden`, a golden question set), then checks that the generated SQL **parses** and **references only tables the agent knows**. Exits non-zero if any case fails, so it works as a CI gate.

```bash
nlqueries eval <connector-or-alias>
nlqueries eval <connector-or-alias> --golden tests/golden/questions.yaml
nlqueries eval <connector-or-alias> --json        # machine-readable per-case results
```

| Flag | Default | Description |
|---|---|---|
| `--golden PATH` | none | A golden questions YAML (`- q: … / agent: … / expect_tables: […]`); cases scoped to this agent are added on top of the KB's capsules |
| `--dialect` | `postgres` | SQL dialect used for generation + parsing (`postgres`, `snowflake`, `bigquery`) |
| `--max-cases N` | `50` | Cap on cases evaluated per run |
| `--json` | off | Emit JSON: `{total, passed, failed, cases: [{question, source, ok, reason, sql}]}` |

Each case passes when the generated SQL parses as a `SELECT` referencing only known tables (CTE names excepted); golden cases additionally require their `expect_tables` to appear. This is the community check — the enterprise tier adds QueryGraph-shape drift classification (`same_shape` / `benign_diff` / `drift`).

---

## ask

Generates and validates SQL **without executing it** — no database connection is made. Use this to preview what SQL a question would produce.

```bash
nlqueries ask <connector-or-alias> "Top 10 customers by revenue"
```

Streams reasoning tokens, then a JSON chunk with the validated SQL.

---

## Multi-agent routing

The orchestrator classifies each question and dispatches accordingly:

| Mode | When used |
|---|---|
| `sql` | Answerable from structured database data |
| `document` | Answerable from ingested documents |
| `hybrid` | Needs both — SQL agent and document agent run in parallel, results merged and ranked by confidence |
| `auto` | Orchestrator decides (default) |

---

## Document connectors

```bash
# Ingest a local file — SOURCE_ID is an opaque slug you choose (e.g. a UUID)
nlq doc-ingest <source_id> <file_path>
# e.g.
nlq doc-ingest my-q1-report /path/to/report.pdf

# Ask a question against an ingested collection
# COLLECTION is the Qdrant collection name: doc_{source_id}_chunks
nlq doc-ask <collection> "What did the report say about churn?"
# e.g.
nlq doc-ask doc_my-q1-report_chunks "What did the report say about churn?"

# Sync a Notion page — requires NOTION_API_TOKEN env var
nlq doc-sync-notion <source_id> <page_id>
# e.g.
NOTION_API_TOKEN=secret_... nlq doc-sync-notion my-wiki-src abc123def456

# Sync a Confluence space — requires CONFLUENCE_API_TOKEN env var
nlq doc-sync-confluence <source_id> <space_key> --base-url <url> --username <user>
# e.g.
CONFLUENCE_API_TOKEN=... nlq doc-sync-confluence my-src ENG \
    --base-url https://acme.atlassian.net --username alice@acme.com
```

Formats: PDF, Word (`.docx`), Excel (`.xlsx`), Notion pages, Confluence spaces. Install with `pip install "nlqueries-core[docs]"` (PDF/Word/Excel) or `pip install "nlqueries-core[wiki]"` (Notion/Confluence). See [connectors.md](connectors.md#document-connectors).

---

## feedback

Submit thumbs-up/down feedback, optionally with a corrected SQL statement. Stored locally as JSONL under `~/.nlqueries/feedback/`.

```bash
nlqueries feedback <connector-or-alias> --question "Orders last month" --thumbs-up
nlqueries feedback <connector-or-alias> --question "Orders last month" --thumbs-down --corrected-sql "SELECT ..."
```

`--question` is required; `--generated-sql` is an optional third flag to attach the SQL the agent originally produced (useful context alongside a correction). A thumbs-down with `--corrected-sql` is saved but not applied automatically — re-run `export-kb` to fold it into the knowledge base.

## feedback-stats

```bash
nlqueries feedback-stats <connector-or-alias>
```

Prints total rated, thumbs-up/down rate, and recent corrections. Prints "No feedback recorded yet" for a connector with none — this is expected on first use, not an error.

---

## promote-feedback

Promotes all thumbs-up feedback for an agent into the verified Qdrant collection (`agent_{id}_verified`), so future `ask` and `query` calls can blend those (question, SQL) pairs into the prompt as high-confidence examples.

```bash
nlqueries promote-feedback <connector-or-alias>
```

This is also called automatically at the end of every `export-kb` run. Run it manually after submitting a batch of thumbs-up ratings between KB exports.

---

## verify-oidc-token

Verifies an OIDC ID token and prints the decoded claims as JSON. Useful for debugging SSO setups before wiring up the enterprise auth flow.

```bash
nlqueries verify-oidc-token <discovery_url> <client_id> <id_token>
```

```bash
# Example — Google OIDC
nlqueries verify-oidc-token \
  https://accounts.google.com/.well-known/openid-configuration \
  my-client-id \
  eyJhbGci...
```

Validates the token signature, expiry, audience, and issuer. Exits 1 on any verification failure.

---

## cache

Semantic cache management — two questions are considered the same if their embeddings have cosine similarity ≥ 0.97. The cache is enabled automatically once Qdrant is reachable.

```bash
nlqueries cache list <connector-or-alias> [--limit 50]   # list cached questions and their generated SQL
nlqueries cache stats <connector-or-alias>                # cache statistics
nlqueries cache clear <connector-or-alias>                 # invalidate all cached entries
```

Cached answers include `from_cache: true` and typically respond in under 50 ms. Entries are written automatically on every successful query — no setup beyond having Qdrant running.

---

## embed-server

Every embedding-dependent command (`query`, `ask`, `process-history --embed`) loads the sentence-transformer model on startup (~9 s, ~130 MB RAM). The daemon loads it once and serves requests over `localhost:8765`, cutting per-call latency to ~10 ms.

```bash
nlqueries embed-server start [--port 8765] [--foreground]   # --foreground blocks and is useful for debugging
nlqueries embed-server status   # not running | starting | running
nlqueries embed-server stop
```

Not restarted automatically on reboot — add the `start` command to your shell profile or a startup task if you want it always on. Override the default port with `--port` or the `EMBED_SERVER_PORT` environment variable.

---

## mcp-server

Manage and start the NLQueries MCP server. Two transports are supported:

```bash
nlqueries mcp-server start                        # stdio — for Claude Desktop (default)
nlqueries mcp-server start --sse --port 8000      # SSE — for network/browser clients
nlqueries mcp-server start --sse --host 0.0.0.0 --port 8000
```

**Claude Desktop config** (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "nlqueries": {
      "command": "nlqueries",
      "args": ["mcp-server", "start"]
    }
  }
}
```

Tools exposed: `list_agents`, `get_agent_schema`, `query`, `submit_feedback`, `health`, `invalidate_cache`, `list_connectors`, `get_query_history`, `get_cache_stats`.
