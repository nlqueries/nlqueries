# Connectors

NLQueries reads two kinds of sources: databases (for structured SQL answers) and documents (answered with `doc-ask`).

---

## Before you connect: the role to use

Give NLQueries a login that cannot write and cannot read more than you would put
in an answer. What the connectors do on their side differs by engine. TLS: the
`postgres` connector requires it unless its entry sets a weaker `ssl_mode`, the
generic connector applies `ssl_mode`/`ssl_ca_cert` only to libpq PostgreSQL URLs,
`redshift` always connects with `ssl=True`, and `mssql` and `connect mysql` set no
TLS options at all ([Transport security](database-hardening.md#transport-security)).
Read-only: PostgreSQL and Redshift are asked for a read-only transaction (directly
or through the generic connector), and SQLite and DuckDB files are opened
read-only; every other engine gets at most a transaction that is never committed
([What each connector enforces](database-hardening.md#what-each-connector-enforces)).
None of this limits *what* the role can read, and none of it is a substitute for a
least-privilege grant.

How much the connector can enforce differs by engine, and the gap worth knowing
is **DDL**. A rolled-back transaction undoes an `INSERT` on a transactional
table; it does not undo a `CREATE` or `DROP` on an engine that commits
implicitly around DDL. (Nor does it undo an `INSERT` into a **MyISAM or MEMORY**
table on MySQL or MariaDB — those engines have no transaction to roll back, and
the server reports only warning 1196.) That
is **Snowflake**, and also **MySQL, MariaDB and Oracle** behind the generic
SQLAlchemy connector; **SQLite** is a third variant, running DDL outside the
transaction altogether. **BigQuery** has no transaction to roll back at all, so a
non-`SELECT` statement type is logged after the job has run rather than
prevented. On all of these the grant is doing work the connector cannot, which is
why the role matters more than the transaction does.

[docs/database-hardening.md](database-hardening.md) has the SQL, per engine.

## Database connectors

| Connector | Install | Query history source |
|---|---|---|
| PostgreSQL | included | `pg_stat_statements` extension: `SELECT`/`WITH` statements recorded for this database, most-called first (see below for the privilege needed to see other roles' statements) |
| MySQL | `pip install "nlqueries-core[mysql]"` | None yet: `connect mysql` runs on the generic SQLAlchemy connector, so the knowledge base comes from the schema alone. It sets no TLS; see below |
| Snowflake | `pip install "nlqueries-core[snowflake]"` | `SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY` (requires `IMPORTED PRIVILEGES` on the `SNOWFLAKE` database); if that cannot be read, the `INFORMATION_SCHEMA.QUERY_HISTORY()` table function, which covers the last 7 days only. If neither can be read, `process-history` stops with an error |
| BigQuery | `pip install "nlqueries-core[bigquery]"` | `INFORMATION_SCHEMA.JOBS_BY_PROJECT` (requires `bigquery.jobs.listAll`, for example through `roles/bigquery.resourceViewer`), completed `SELECT` jobs only, read from one region: see below |
| Amazon Redshift | `pip install "nlqueries-core[redshift]"` | `STL_QUERY` (requires superuser or `pg_read_all_stats`; provisioned clusters only, see below) |
| SQL Server / Azure SQL | `pip install "nlqueries-core[mssql]"` | `sys.dm_exec_query_stats` + `sys.dm_exec_sql_text` (requires `VIEW SERVER STATE` / `VIEW DATABASE STATE`) |
| DuckDB | `pip install "nlqueries-core[duckdb]"` | None — file-based, no persisted history |
| SQLite | included (stdlib `sqlite3`) | None — file-based, no persisted history |
| Generic SQLAlchemy | included (SQLAlchemy is a base dependency); install the URL's own driver | None — no portable source across dialects |

The generic `sqlalchemy` type reaches anything with a SQLAlchemy dialect installed — MariaDB, Oracle, and the engines above if you would rather supply a URL than the discrete flags. It is configured by `--url` rather than the per-field flags, and introspects schema through SQLAlchemy's dialect-agnostic inspector rather than engine-specific catalog SQL. That reflection is narrower than the dedicated connectors': it reads the tables (not views) of the connection's default schema only, keeps column comments as descriptions, and leaves row counts and table descriptions empty. `--password` and `--password-env` still apply, and are the better habit — a password inside the URL is a command-line argument and lands in shell history. See [cli-reference.md](cli-reference.md#connect).

See [cli-reference.md](cli-reference.md#connect) for `connect` examples per type.

### MySQL — TLS

`nlqueries connect mysql` composes a `mysql+pymysql://` URL with no TLS options, and the generic connector it runs on cannot add them (its `ssl_mode` and `ssl_ca_cert` settings are mapped only for PostgreSQL). The session is therefore unencrypted unless the server itself insists. To require TLS, register the database through the generic connector with the TLS options in the URL's query string, which the PyMySQL driver reads:

```bash
nlqueries connect sqlalchemy --url "mysql+pymysql://alice@db.internal:3306/shop?ssl_ca=/etc/ssl/certs/db-ca.pem"
```

Configuring the server to require TLS (`require_secure_transport=ON`) refuses an unencrypted session whichever way the client connects.

### PostgreSQL — enabling query history capture

`process-history` requires the `pg_stat_statements` extension. Check whether it's already enabled before changing anything:

```sql
SHOW shared_preload_libraries;                                  -- library loaded at server level?
SELECT extname FROM pg_extension WHERE extname = 'pg_stat_statements';  -- extension created in this DB?
```

**Managed databases (AWS RDS, Google Cloud SQL, Supabase, Neon, Azure Database for PostgreSQL)** pre-load the library — just run:

```sql
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
```

**Self-hosted PostgreSQL** requires a restart, since `shared_preload_libraries` is a startup-only parameter:

1. Add `shared_preload_libraries = 'pg_stat_statements'` to `postgresql.conf` (append with a comma if other libraries are already listed)
2. Restart PostgreSQL
3. Run `CREATE EXTENSION IF NOT EXISTS pg_stat_statements;` once per database
4. Verify: `SELECT count(*) FROM pg_stat_statements;`

The view is cluster-wide; `process-history` reads only this database's rows, and only statements that begin with `SELECT` or `WITH`, ordered by call count. For a login that is neither a superuser nor a member of `pg_read_all_stats`, PostgreSQL shows other roles' statements as `<insufficient privilege>`; those rows are skipped, so the history holds only the login's own statements. To include everyone's queries, `GRANT pg_read_all_stats TO <login>;`.

Note: `--days` has no effect on PostgreSQL — `pg_stat_statements` doesn't record per-query timestamps, so all history since the last `pg_stat_statements_reset()` is returned. `--days` is honoured on Snowflake, BigQuery, Redshift and SQL Server, which do track execution time. (On SQL Server it filters on each cached plan's last execution time.)

### BigQuery: query history region

`JOBS_BY_PROJECT` is a regional view, and the connector reads one region of it: the location of the `--dataset-id` dataset when one is given and its location can be read, otherwise `region-us`. Jobs that ran in another location are not in that view, so for a project whose data is outside the US multi-region, register the connector with `--dataset-id`.

### BigQuery: authentication

The connector finds its credentials in one of three ways, chosen by what its
entry in `connectors.yaml` holds:

| `service_account_json` | What it authenticates as |
|---|---|
| omitted | Application Default Credentials (ADC): `gcloud auth application-default login`, `GOOGLE_APPLICATION_CREDENTIALS`, or the workload identity of a GKE pod or Cloud Run service. The client library finds them; nothing is read by the connector. |
| a service account key (`"type": "service_account"`) | That service account. |
| a Workload Identity Federation credential configuration (`"type": "external_account"`) | The federated identity, with no service account key: a workload on AWS, Azure, GitHub Actions or another OIDC or SAML provider exchanges its own token for a Google one. Generate the file with `gcloud iam workload-identity-pools create-cred-config`. |

`service_account_json` may be the path to the JSON file (as `--service-account-json`
stores it) or the parsed JSON itself. The connector reads the file's `type` to
choose; any other type is handed to the service-account loader, which refuses it
as before.

A federation configuration names where its token comes from (`credential_source`),
and `google-auth` reads it when the connector first authenticates: a local file,
a URL, or an executable. Treat the configuration as you would a key, and use only
one you generated. An executable source runs only when
`GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES=1` is set.

**Impersonation.** Add `impersonate_service_account` with a service account
email and the credential above (key, federation configuration or ADC) is used only
to mint short-lived tokens for that account, which every query then runs as, with
the BigQuery scope. Register the connector with `nlqueries connect bigquery`, then
add the key to the entry it wrote:

```yaml
bigquery:localhost:acme-prod:
  db_type: bigquery
  # ... the other keys `connect` wrote ...
  project_id: acme-prod
  impersonate_service_account: nlqueries-reader@acme-prod.iam.gserviceaccount.com
```

The identity you start from needs `roles/iam.serviceAccountTokenCreator` on that
service account, and the service account needs the BigQuery roles described in
[database-hardening.md](database-hardening.md). This keeps one narrowly granted
account as the one queries run as, whichever machine or person starts them. A
value that is empty or not an email is refused when the connector opens, rather
than ignored, since ignoring it would run queries as the starting identity. There
is no `connect` flag for it yet, and running `nlqueries connect bigquery` again for
the same connector rewrites its whole entry, so add the key back afterwards.

### Amazon Redshift

The connector does not read table or column comments (`COMMENT ON`), so descriptions come from `import-dbt` or from editing the knowledge base. Row counts come from `SVV_TABLE_INFO` (requires table-owner or superuser; falls back to a permission-free list if inaccessible).

Query history comes from `STL_QUERY`, which only provisioned clusters have. On a **Serverless** workgroup `process-history` stops with an error; build the knowledge base from the schema with `export-kb`.

`REDSHIFT_SOCKET_TIMEOUT_SECONDS` bounds the whole connection, not just the
handshake: the driver sets it on the socket before connecting and never clears
it, so it also limits how long a query may go without sending data. It therefore
defaults to your statement timeout plus 30 seconds, and never less than 60 (150
with the default 120-second `CONNECTOR_STATEMENT_TIMEOUT_SECONDS`), and must stay
above it — below, a long query is killed by the client and reported as a network
fault instead of being cancelled by the server.

It also caps any per-query budget: a caller passing `timeout_seconds=300` against
the default 150 still dies on the socket at 150. Raise this above the largest
per-query budget you intend to allow. Setting `CONNECTOR_STATEMENT_TIMEOUT_SECONDS=0`
disables the socket ceiling too, so a deliberately unbounded query stays
unbounded; `REDSHIFT_SOCKET_TIMEOUT_SECONDS=0` does the same on its own.

Lowering it to fail faster on an unreachable host shortens every query budget by
the same amount.

The same value covers a **Serverless** workgroup resuming from zero, which it has
to do before it answers the first connection after an idle period.

### SQL Server / Azure SQL

Use `alice@my-server` as `--user` for Azure SQL with SQL authentication — the same connector covers on-premises SQL Server and Azure SQL since the T-SQL dialect is identical. If the account lacks `VIEW SERVER STATE`/`VIEW DATABASE STATE`, `process-history` stops with an error that names the missing permission; grant it, or build the knowledge base from the schema alone with `export-kb`.

### DuckDB

No query history across connections — `process-history` always returns an empty list; the KB comes from schema introspection only. Primary keys are detected via `duckdb_constraints()`; foreign keys are skipped (rarely declared in DuckDB analytics workloads).

### SQLite

Built on the standard-library `sqlite3` driver, so no extra install is needed. The `database` credential is a file path (`/data/app.db`) or `:memory:` for a transient in-process database — there is no host, port, or credentials. No query history (`process-history` returns an empty list); the KB is built from schema introspection via `PRAGMA table_info` (columns + primary keys) and `PRAGMA foreign_key_list` (foreign keys). SQLite has no server-side statement timeout, so a runaway query is bounded best-effort by a watchdog that interrupts it.

> Already have a SQLite URL? The generic `sqlalchemy` connector also reaches SQLite (`--url "sqlite:///path.db"`); the dedicated `sqlite` type just gives you the simpler file-path form.

---

## Document connectors

| Connector | Format | Requires |
|---|---|---|
| PDF | `.pdf` | `pip install "nlqueries-core[docs]"` |
| Word | `.docx` | `pip install "nlqueries-core[docs]"` |
| Excel | `.xlsx` | `pip install "nlqueries-core[docs]"` |
| Markdown | `.md`, `.markdown` (UTF-8; sectioned at `#` and `##` headings) | included |
| Plain text | `.txt` (UTF-8) | included |
| Notion | Notion pages | `pip install "nlqueries-core[wiki]"`, `NOTION_API_TOKEN` |
| Confluence | Confluence spaces | `pip install "nlqueries-core[wiki]"`, `CONFLUENCE_API_TOKEN` (the URL and user are the `--base-url` and `--username` options) |

```bash
# SOURCE_ID is an opaque slug you choose (e.g. a UUID or short name)
nlq doc-ingest <source_id> <file_path>
# e.g.
nlq doc-ingest q1-report ./report.pdf

# Notion — requires NOTION_API_TOKEN env var; PAGE_ID is the Notion page or database ID
nlq doc-sync-notion <source_id> <page_id>
# e.g.
NOTION_API_TOKEN=secret_... nlq doc-sync-notion my-wiki-src abc123def456

# Confluence — requires CONFLUENCE_API_TOKEN env var
nlq doc-sync-confluence <source_id> <space_key> --base-url <url> --username <user>
# e.g.
CONFLUENCE_API_TOKEN=... nlq doc-sync-confluence my-src ENG \
    --base-url https://acme.atlassian.net --username alice@acme.com
```

Each command chunks the source, embeds the chunks and stores them in the Qdrant collection `doc_{source_id}_chunks` (Qdrant is required — see [qdrant-setup.md](qdrant-setup.md)). A later run for the same `SOURCE_ID` replaces what the earlier one stored. The new chunks are written, in batches of 256, before the old ones are removed: if the first batch fails, the command exits with an error and the earlier copy is untouched; if a later batch fails, or only the removal of the old version's leftover chunks fails, the command says so and exits with an error, and `doc-ask` may return text from both versions until a rerun succeeds.

A read that produces no text (a mistyped space key, a lost permission, a scanned PDF with no text layer) stores nothing and exits with an error, leaving the stored copy as it was. If the source really is empty now, pass `--allow-empty` to remove its stored chunks. (`doc-sync-notion --since` is the exception: there an empty read is taken to mean the page is unchanged, and the command leaves the stored copy as it was without an error.) A Confluence space that cannot be read to the end stops the command with an error before anything is stored.

A global space key must be letters, digits and underscores (`ENG`). A personal space key is `~` and the owner's username or account id, which may also contain `.`, `-`, `@` and `:` (`~john.smith`, `~557058:f2d5e7a1-...`). Anything else is refused before a query is sent. An `http://` base URL to a host other than `localhost` or a loopback address logs a warning, since the API token is then sent unencrypted. A sync reads at most `NLQ_MAX_CONFLUENCE_PAGES` pages (5,000) and `NLQ_MAX_CONFLUENCE_BYTES` of page body (200 MiB) from a space, and at most `NLQ_MAX_NOTION_BLOCKS` blocks (10,000) and `NLQ_MAX_NOTION_BYTES` of block text (50 MiB) from a Notion page; a source over a cap stops the command with an error before anything is stored, rather than storing part of it. See [configuration.md](configuration.md).

With `--since`, an unchanged Notion page or Confluence space is left as it is. For Confluence, one search checks whether any page changed since then; if one did, the whole space is read, so the stored copy stays whole. `--since` only sees pages modified since then: a page that was deleted, moved to another space or had its permissions withdrawn is not noticed, and its chunks stay searchable until a full sync (no `--since`). Run a full sync periodically.

Ask with `nlq doc-ask doc_{source_id}_chunks "..."`. It retrieves the most relevant chunks and answers with citations (source document, page/section). `nlqueries query` and the MCP `query` tool answer from the database only; they do not search documents.

Document ingestion no longer depends on `langchain_text_splitters` — chunking uses a small built-in chunker (`nlqueries.document_connectors.chunker`), so the Python 3.14 limitation that previously affected document connectors no longer applies (see [troubleshooting.md](troubleshooting.md#w6--pydantic-v1-incompatibility-python-314) for history).
