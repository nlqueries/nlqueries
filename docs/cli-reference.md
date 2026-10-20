# CLI Command Reference

Every command is available as `nlqueries <command>` or the shorter `nlq <command>` — both entry points are identical.

```bash
nlqueries --help
nlqueries <command> --help
```

Anywhere a connector ID (`<type>:<host>:<database>` by default — see [connect](#connect)) is accepted, a configured `--alias` works too — see [Connector aliases](#connector-aliases).

---

## connect

Register a database connection. Connection metadata (host, port, database, user) is written to `~/.nlqueries/connectors.yaml` (mode `0600`).

`connect` tests the connection, then prints the connector ID it registered. By default the ID is `<type>:<host>:<database>`; DuckDB and SQLite use `<type>:<path>` (`<type>::memory:` for an in-memory database), and the generic connector `sqlalchemy:<driver>:<host>:<database>`, leaving out any part the URL does not have (`sqlalchemy:<driver>:db` if it has neither). `--connector-id NAME` registers it under a name of your choosing instead.

**Password storage:** the password itself does not go in that file. `connect` stores it in your OS keychain (macOS Keychain, Windows Credential Manager, or Linux Secret Service, via the `keyring` package) and prints `✓ Password stored in OS keychain` on success. If keyring is unavailable on the machine — headless CI without a secret service, for example — it falls back to embedding the password in the stored connection URL instead, and prints a warning telling you so. To rotate a password later without re-running `connect`, use `nlqueries update-password <connector-id>`.

**Password handling:** don't put `--password` on the command line if you can avoid it — it lands in shell history. Two safer options: omit `--password` entirely and you'll get an interactive, hidden prompt, or use `--password-env VAR` to read it from an already-set environment variable. (`--password` is shown in the examples below only for brevity.)

**PostgreSQL / MySQL**

```bash
nlqueries connect postgres --host db.example.com --port 5432 --database prod --user alice --alias prod   # prompts for password
nlqueries connect postgres --host db.example.com --database prod --user alice --password-env PGPASSWORD --alias prod
nlqueries connect mysql --host localhost --database mydb --user alice --password secret
```

TLS is not a `connect` flag. For PostgreSQL the connector requires TLS by default; set `ssl_mode` and `ssl_ca_cert` in the connector's entry — see [database-hardening.md](database-hardening.md#transport-security). `connect mysql` sets no TLS: to require it, register through the generic connector with the options in the URL — see [connectors.md](connectors.md#mysql--tls).

**Snowflake** — install `pip install "nlqueries-core[snowflake]"` first

```bash
nlqueries connect snowflake --account myorg.us-east-1 --database ANALYTICS --schema PUBLIC --warehouse COMPUTE_WH --user alice --password YOUR_PASSWORD
```

**BigQuery** — install `pip install "nlqueries-core[bigquery]"` first (uses Application Default Credentials if `--service-account-json` is omitted). `--service-account-json` takes a service account key or a Workload Identity Federation credential configuration; to run as an impersonated service account, see [connectors.md](connectors.md#bigquery-authentication)

```bash
nlqueries connect bigquery --project-id my-gcp-project --dataset-id my_dataset --service-account-json /path/to/service-account.json
```

BigQuery has no password — `--project-id` (or `--database`, used as a fallback) is the only required flag.

**Amazon Redshift** — install `pip install "nlqueries-core[redshift]"` first

```bash
nlqueries connect redshift --host my-cluster.abc123.us-east-1.redshift.amazonaws.com --port 5439 --database dev --user awsuser --password YOUR_PASSWORD --alias redshift-prod
```

**SQL Server / Azure SQL** — install `pip install "nlqueries-core[mssql]"` first

```bash
nlqueries connect mssql --host my-server.database.windows.net --port 1433 --database mydb --user alice --password YOUR_PASSWORD --alias sql-prod
```

**DuckDB** — install `pip install "nlqueries-core[duckdb]"` first; file-based, no host/port

```bash
nlqueries connect duckdb --database /data/warehouse.db --alias duck-local   # persistent
nlqueries connect duckdb --alias duck-mem                                   # in-memory, transient
```

**SQLite** — no install needed (stdlib); file-based, no host/port

```bash
nlqueries connect sqlite --database /data/app.db --alias app-local   # persistent
nlqueries connect sqlite --alias app-mem                             # in-memory, transient
```

**Generic SQLAlchemy** — for anything without a dedicated type above (MariaDB, Oracle, ...); install that driver yourself

```bash
nlqueries connect sqlalchemy --url "mysql+pymysql://alice:secret@db.internal:3306/shop" --alias shop
nlqueries connect sqlalchemy --url "oracle+cx_oracle://alice:secret@host:1521/?service_name=ORCL"
```

`--url` takes a whole SQLAlchemy URL and is the only way to configure this type; it is rejected on any other db-type, which composes its URL from the flags above. Host, port, database and user are read back out of the URL for the config file, and a password inside the URL is moved into the OS keychain exactly as `--password` would be.

`--password` and `--password-env` work here too, and are the better habit: a URL is a command-line argument, so a password inside it lands in shell history. Leave it out of the URL and supply it separately — passing it both ways is refused rather than silently preferring one.

```bash
export SHOP_PW=...
nlqueries connect sqlalchemy --url "mysql+pymysql://alice@db.internal:3306/shop" --password-env SHOP_PW
```

Schema comes from SQLAlchemy's dialect-agnostic inspector and there is no query history — see [connectors.md](connectors.md).

See [connectors.md](connectors.md) for per-database query-history and schema-introspection caveats (Redshift, MSSQL, DuckDB, and SQLite in particular).

---

## update-password

Rotate the stored password for a connector without re-running `connect`. The new password is written to the OS keychain, replacing what is there. A connector whose password was embedded in its stored URL (the fallback `connect` uses without keyring) is switched to the keychain and the old password removed from the URL; the connector's other settings are untouched.

```bash
nlqueries update-password postgres:localhost:mydb   # prompts for the new password, hidden input
nlqueries update-password dev --password-env DB_PASSWORD
```

`CONNECTOR_ID` accepts an alias. This does not re-test the connection — run `nlqueries extract-schema <connector-or-alias>` afterwards to confirm the new password actually works. If keyring is unavailable, nothing is stored: the command prints a warning saying so and that the password has to be set by hand in the connector's URL in `connectors.yaml`.

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

Connect to a registered connector, read its schema, and print a summary: the database name, the table count, the total column count, the extraction time, and a table of the first 20 tables with each one's schema, column count and row count. Column types and foreign keys are not printed. Row counts are what the connector reports: a catalogue figure on most engines (an estimate on PostgreSQL and DuckDB), an exact `COUNT(*)` of each table on SQLite, and none (`—`) from the generic SQLAlchemy connector.

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

Supported history sources: PostgreSQL (`pg_stat_statements`, must be enabled — see [connectors.md](connectors.md#postgresql--enabling-query-history-capture)), Snowflake (`QUERY_HISTORY`), BigQuery (`INFORMATION_SCHEMA.JOBS_BY_PROJECT`). MySQL has no history source yet. Redshift, SQL Server, and DuckDB have their own caveats — see [connectors.md](connectors.md).

A fresh or lightly-used database will produce few or zero capsules until it has real query history to learn from — run some representative business queries first, and consider `--min-executions 1` on a dev database.

---

## export-kb

Reads the capsules saved by `process-history` together with the live schema and writes the KB YAML that `query`/`ask` read. **Required before `query`/`ask` will work**, and must be re-run after every `process-history` run.

```bash
nlqueries export-kb <connector-or-alias> [--output kb.yaml] [--sample-rows 3] [--no-include-samples] [--values-max-rows 10000000] [--describe-columns]
```

Default output path: `~/.nlqueries/knowledge_base/<connector-id>.yaml` (`:` replaced with `_`).

| Flag | Default | Description |
|---|---|---|
| `--output` / `-o` | `~/.nlqueries/knowledge_base/<id>.yaml` | Path to write the YAML KB |
| `--include-samples` / `--no-include-samples` | on | Store column values in the KB, so the prompt shows how they are spelled. A text column with at most 20 distinct values (and 200 characters in all) is stored whole, marked `values_complete: true` and shown as `values: [...]`; any other column gets up to `--sample-rows` samples, shown as `samples: [...]`, unless they average over 80 characters. Keys and personal-data columns (email, phone, address, ...) never get values. Read-only; a query that fails or takes over 10 s leaves that column without values. **What it costs:** one sample query per table and one `SELECT DISTINCT` per text column, and the `DISTINCT` reads every row of a column with few values. On a warehouse billed by bytes read, BigQuery for one, each is charged for the whole column it names, or the whole table for the sample query, whatever its `LIMIT`. `--values-max-rows` skips the largest tables, and `--no-include-samples` runs none of these queries |
| `--sample-rows` | `3` | Rows sampled per table, and so the most sample values stored per column |
| `--values-max-rows` | `10000000` | A table the schema records as having more rows than this gets no column values, and none of the queries that collect them. `0` for no cap; a table whose size the connector does not report is collected. The size is the one the connector reports, and the generic SQLAlchemy connector (MySQL, MariaDB, Oracle and the rest) reports none, so on those engines this cap skips nothing; nor do the others report one in every case (a Postgres table never analysed, a Redshift role without access to `SVV_TABLE_INFO`). `export-kb` says how many tables report no row count. |
| `--describe-columns` | off | Use the LLM to auto-populate column descriptions from sample data (skips surrogate-key columns; requires LLM key) |

---

## import-dbt

Merge a dbt project's model and column documentation into a knowledge base. If your data is modelled in dbt, its docs are usually better than anything generated from the schema alone — this grounds the KB in them without a `dbt-core` dependency (it reads the artifacts with stdlib JSON only).

```bash
nlqueries import-dbt <artifacts> --kb kb.yaml [--output merged.yaml]
```

`<artifacts>` is a dbt target directory (containing `manifest.json`, and `semantic_manifest.json` if the project uses the Semantic Layer) or a `manifest.json` file directly.

| Flag | Default | Description |
|---|---|---|
| `--kb` | *(required)* | KB YAML to merge dbt docs into (created if missing) |
| `--output` / `-o` | the `--kb` path (in place) | Where to write the merged KB |

Descriptions merge with source precedence **manual edit > dbt doc > database schema > LLM-generated**, recorded per field in `description_source`. A human edit is never overwritten, and a later re-import updates only dbt-sourced fields. Semantic-layer metrics are listed in the output (they map to Beacons in the enterprise Nexus UI), not written to the KB.

---

## kb-stats

Structured coverage and quality report for the exported knowledge base: schema coverage, description coverage, query capsule / intent coverage, FK join coverage, glossary hierarchy, cache entries, and feedback history.

```bash
nlqueries kb-stats <connector-or-alias>
nlqueries kb-stats <connector-or-alias> --verbose   # adds a per-table breakdown (rows, columns, column-description %, table description)
nlqueries kb-stats <connector-or-alias> --json
```

Exit code `0` if the KB exists, `1` if missing — useful as a CI gate before deployment. Works from the local KB file alone if the database is unreachable: the database-derived lines (tables and columns in the database, join coverage) are left out, and are `null` in `--json`.

### Glossary hierarchy

A glossary term in the KB may carry an optional single `parent:` naming another term, giving a lightweight `is-a` hierarchy (`ActiveCustomer is-a Customer`). It is fully backward compatible — a glossary with no `parent:` keys behaves exactly as a flat list.

```yaml
business_context:
  glossary:
    - term: Customer
      definition: A person or organization that has placed at least one order.
    - term: ActiveCustomer
      definition: A customer with an order in the last 90 days.
      parent: Customer          # optional · single string · absent = a root term
```

When a term is injected into the SQL prompt, its parent is included as one line of context (`ActiveCustomer (a kind of Customer): …`). `kb-stats` reports the term count, how many declare a parent, the maximum hierarchy depth, and any **orphans** (a `parent:` that names a term which doesn't exist — allowed, treated as a root, and surfaced so you can fix the typo). A **cycle** (a term that is its own ancestor) is rejected when the KB is generated and flagged by `kb-stats`. Multiple inheritance is not supported — `parent:` is a single string.

By default the whole glossary is injected into every prompt. Setting `NLQ_GLOSSARY_QUESTION_SCOPED=true` (see [configuration.md](configuration.md)) switches to **question-scoped** injection: only terms the question mentions are included, expanded with their hierarchy ancestors (for context) and descendants (as related candidates, up to depth 3). Business rules are always injected in full.

---

## annotate

Backfills LLM intent descriptions on capsules that don't have one yet (e.g. after `process-history --no-annotate`).

```bash
nlqueries annotate <connector-or-alias>
```

---

## query

Generates SQL, **executes it against the database**, and prints the result rows.

```bash
nlqueries query <connector-or-alias> "Top 10 customers by revenue this year"
nlqueries query <connector-or-alias> "..." --json        # structured output including rows
nlqueries query <connector-or-alias> "..." --no-execute  # generate SQL without running it
nlqueries query <connector-or-alias> "..." --new-session # discard prior conversation context
nlqueries query <connector-or-alias> "..." --paraphrase  # also print the SQL as one English sentence
```

Plain output shows the agent type, the generated SQL, the result rows as a table with their count and execution time, and the latency (plus the resolved question when a follow-up was rewritten). The natural-language answer text is printed only when the question was not answered with SQL and there are no rows to show; `--json` always includes it. `query` answers from the database; to ask about ingested documents, use [`doc-ask`](#document-connectors).

| Flag | Default | Description |
|---|---|---|
| `--json` | off | Emit the full result as raw JSON (includes rows, citations, latency) |
| `--execute` / `--no-execute` | on | Execute the generated SQL and display rows |
| `--session` / `--no-session` | on | Carry conversation context across queries for follow-up questions |
| `--new-session` | off | Start a fresh conversation, discarding prior context |
| `--explain` | off | Collect answer provenance (route, capsules, cache, checks, timings) and include it in the `--json` output |
| `--paraphrase` | off | Render the generated SQL back into one plain-English sentence (deterministic, no LLM); printed as `In English`, and included in the `--json` output |
| `--dialect` | the connector's type, else `postgres` | SQL dialect used for generation: `postgres`, `mysql`, `snowflake`, `bigquery`, `redshift`, `mssql`, `duckdb` or `sqlite`. Without it, the agent's connector type decides (for a `sqlalchemy` connector, the engine its URL names), and `postgres` when that names none |

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
| `--dialect` | the connector's type, else `postgres` | SQL dialect used for generation + parsing: `postgres`, `mysql`, `snowflake`, `bigquery`, `redshift`, `mssql`, `duckdb` or `sqlite`. Without it, the agent's connector type decides (for a `sqlalchemy` connector, the engine its URL names), and `postgres` when that names none |
| `--max-cases N` | `50` | Cap on cases evaluated per run |
| `--json` | off | Emit JSON: `{total, passed, failed, cases: [{question, source, ok, reason, sql}]}` |

Each case passes when the generated SQL parses as a `SELECT` referencing only known tables (CTE names excepted); golden cases additionally require their `expect_tables` to appear. This is the community check — the enterprise tier adds QueryGraph-shape drift classification (`same_shape` / `benign_diff` / `drift`).

---

## ask

Generates and validates SQL **without executing it**. Use this to preview what SQL a question would produce. The generated statement is never run, but with literal grounding on (the default, `NLQ_LITERAL_GROUNDING`) `ask` can still open the agent's connector, read-only, to look up the values a column stores and check the statement's string literals against them — see [configuration.md](configuration.md).

```bash
nlqueries ask <connector-or-alias> "Top 10 customers by revenue"
```

On a terminal, it waits for the whole response, prints its text, then prints the validated SQL as a `SQL: …` line; the JSON chunk is not shown. When output is piped or redirected, it streams every token as it arrives, ending with a JSON chunk carrying the validated SQL, for scripts to parse. `--dialect` works as it does for [`query`](#query): any of `postgres`, `mysql`, `snowflake`, `bigquery`, `redshift`, `mssql`, `duckdb` or `sqlite`, defaulting to the connector's type, else `postgres`.

---

## Document connectors

```bash
# Ingest a local file — SOURCE_ID is an opaque slug you choose (e.g. a UUID).
# The chunks are embedded and stored in the Qdrant collection doc_{source_id}_chunks,
# replacing what an earlier run stored for the same SOURCE_ID. A read that produces
# no text stores nothing and exits 1; --allow-empty stores the empty result instead.
nlq doc-ingest <source_id> <file_path>
# e.g.
nlq doc-ingest my-q1-report /path/to/report.pdf

# Ask a question against an ingested collection
# COLLECTION is the Qdrant collection name: doc_{source_id}_chunks
# --source-id restricts retrieval to one source; --top-k sets how many chunks are retrieved (default 5)
nlq doc-ask <collection> "What did the report say about churn?" [--source-id <source_id>] [--top-k 5]
# e.g.
nlq doc-ask doc_my-q1-report_chunks "What did the report say about churn?"

# Sync a Notion page — requires NOTION_API_TOKEN env var. Stored in doc_{source_id}_chunks,
# replacing the earlier copy. --since <ISO 8601 timestamp> leaves the stored copy as it is
# when the page has not changed since then; --allow-empty works as for doc-ingest.
nlq doc-sync-notion <source_id> <page_id> [--since 2024-01-15T00:00:00+00:00] [--allow-empty]
# e.g.
NOTION_API_TOKEN=secret_... nlq doc-sync-notion my-wiki-src abc123def456

# Sync a Confluence space — requires CONFLUENCE_API_TOKEN env var. Stored in doc_{source_id}_chunks,
# replacing the earlier copy. With --since, if any page changed after that timestamp the
# space is re-read in full, and if none did the stored copy is left; --allow-empty as above.
nlq doc-sync-confluence <source_id> <space_key> --base-url <url> --username <user> [--since <timestamp>] [--allow-empty]
# e.g.
CONFLUENCE_API_TOKEN=... nlq doc-sync-confluence my-src ENG \
    --base-url https://acme.atlassian.net --username alice@acme.com
```

Formats: Markdown (`.md`, `.markdown`, sectioned at `#` and `##` headings) and plain text (`.txt`, UTF-8) on a base install; PDF, Word (`.docx`), Excel (`.xlsx`), Notion pages, Confluence spaces with an extra. Install with `pip install "nlqueries-core[docs]"` (PDF/Word/Excel) or `pip install "nlqueries-core[wiki]"` (Notion/Confluence). See [connectors.md](connectors.md#document-connectors).

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

Prints the total number of entries, the thumbs-up and thumbs-down counts, and, when there are corrections, how many and the five most recent. Prints "No feedback recorded yet" for a connector with none — this is expected on first use, not an error.

---

## promote-feedback

Promotes thumbs-up feedback for an agent into the verified Qdrant collection (`agent_{id}_verified`), so future `ask` and `query` calls can blend those (question, SQL) pairs into the prompt as high-confidence examples. Each pair's SQL is validated against the current knowledge base schema first.

```bash
nlqueries promote-feedback <connector-or-alias>
nlqueries promote-feedback <connector-or-alias> --include-anonymous
```

Feedback submitted over the MCP transport, and feedback recorded before origins were tracked, is skipped, because who gave the rating cannot be established and a promoted pair steers generation for everyone using the agent. Review those records and pass `--include-anonymous` to promote them as well.

This is also called automatically at the end of every `export-kb` run, without `--include-anonymous`. Run it manually after submitting a batch of thumbs-up ratings between KB exports.

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

Semantic cache management. A question is served from the cache by an exact match on its normalised text (Tier 0), by a stored answer whose question's embedding has cosine similarity ≥ `NLQ_CACHE_ANSWER_THRESHOLD` (default 0.97, Tier 1), or by a stored SQL template at ≥ `NLQ_CACHE_TEMPLATE_THRESHOLD` (default 0.90, Tier 2) with the question's entities bound into it. `NLQ_CACHE_ANSWER_TIERS` limits which tiers serve — see [configuration.md](configuration.md). The cache is enabled automatically once Qdrant is reachable.

```bash
nlqueries cache list <connector-or-alias> [--limit 50]   # list cached questions and their generated SQL
nlqueries cache stats <connector-or-alias>                # cache statistics
nlqueries cache clear <connector-or-alias>                 # invalidate all cached entries
```

Cached answers include `from_cache: true`. The cache serves `query` and the MCP `query` tool; `ask` does not use it. A hit skips the LLM: the stored answer is replayed and the stored SQL is run again (not with `--no-execute`), so the rows are current and the database is still queried. Entries are written automatically after a query whose SQL passed validation — no setup beyond having Qdrant running.

---

## embed-server

Every embedding-dependent command (`query`, `ask`, `process-history --embed`) loads the sentence-transformer model itself (~9 s) when no daemon is running. The daemon loads it once and serves requests on `localhost` at `EMBED_SERVER_PORT` (default 8765), cutting per-call latency to ~10 ms.

```bash
nlqueries embed-server start [--port 8765] [--foreground]   # --foreground blocks and is useful for debugging
nlqueries embed-server status [--port 8765]   # not running | starting | running
nlqueries embed-server stop
```

Not restarted automatically on reboot — add the `start` command to your shell profile or a startup task if you want it always on. To change the port, set `EMBED_SERVER_PORT`: it is the port the daemon listens on and the port clients, `status` and `health` look for it on. `start --port` and `status --port` override it for that command only; `start` warns when its port differs from `EMBED_SERVER_PORT`, because clients would not find the daemon there.

---

## mcp-server

Manage and start the NLQueries MCP server. Two transports are supported:

```bash
nlqueries mcp-server start                        # stdio — for Claude Desktop (default)
nlqueries mcp-server start --sse --port 8000      # SSE on 127.0.0.1
```

With `--sse`, `--host` sets the bind address (default `127.0.0.1`) and `--port` the port (default `8000`).

The SSE listener **will not start without authentication**: configure an
identity provider or a pre-shared token as
[mcp-authentication.md](mcp-authentication.md) describes. The refusal names what
to set. `NLQ_ALLOW_UNAUTHENTICATED_MCP=1` starts it without authentication, and
then every tool is reachable by anything that can connect, including `query`,
which runs SQL against a configured database.

It also binds loopback and **refuses a wildcard address**, with authentication
on or off. For remote access, put a proxy in front of the loopback port. If you
want the wildcard bind anyway, set `NLQ_ALLOW_INSECURE_BIND=1`; the server
starts and logs it each time.

The Docker image and `python -m nlqueries.mcp_server` choose the transport from
`MCP_TRANSPORT` (`stdio`, `sse` or `streamable-http`) instead of a flag, under
the same rules.

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
