# Connectors

NLQueries reads two kinds of sources: databases (for structured SQL answers) and documents (for the document and hybrid agents).

---

## Database connectors

| Connector | Install | Query history source |
|---|---|---|
| PostgreSQL | included | `pg_stat_statements` extension |
| MySQL | `pip install "nlqueries-core[mysql]"` | Performance schema `events_statements_summary_by_digest` |
| Snowflake | `pip install "nlqueries-core[snowflake]"` | `QUERY_HISTORY` view in `INFORMATION_SCHEMA` |
| BigQuery | `pip install "nlqueries-core[bigquery]"` | `INFORMATION_SCHEMA.JOBS` |
| Amazon Redshift | `pip install "nlqueries-core[redshift]"` | `STL_QUERY` (requires superuser or `pg_read_all_stats`) |
| SQL Server / Azure SQL | `pip install "nlqueries-core[mssql]"` | `sys.dm_exec_query_stats` + `sys.dm_exec_sql_text` (requires `VIEW SERVER STATE` / `VIEW DATABASE STATE`) |
| DuckDB | `pip install "nlqueries-core[duckdb]"` | None — file-based, no persisted history |

See [cli-reference.md](cli-reference.md#connect) for `connect` examples per type.

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

Note: `--days` has no effect on PostgreSQL — `pg_stat_statements` doesn't record per-query timestamps, so all history since the last `pg_stat_statements_reset()` is returned. `--days` is honoured on Snowflake and BigQuery, which do track execution time.

### Amazon Redshift

Schema descriptions are not available (no equivalent of PostgreSQL's `pg_description`). Row counts come from `SVV_TABLE_INFO` (requires table-owner or superuser; falls back to a permission-free list if inaccessible).

### SQL Server / Azure SQL

Use `alice@my-server` as `--user` for Azure SQL with SQL authentication — the same connector covers on-premises SQL Server and Azure SQL since the T-SQL dialect is identical. If the account lacks `VIEW SERVER STATE`/`VIEW DATABASE STATE`, `process-history` returns empty history and the KB is built from schema introspection only.

### DuckDB

No query history across connections — `process-history` always returns an empty list; the KB comes from schema introspection only. Primary keys are detected via `duckdb_constraints()`; foreign keys are skipped (rarely declared in DuckDB analytics workloads).

---

## Document connectors

| Connector | Format | Requires |
|---|---|---|
| PDF | `.pdf` | `pip install "nlqueries-core[docs]"` |
| Word | `.docx` | `pip install "nlqueries-core[docs]"` |
| Excel | `.xlsx` | `pip install "nlqueries-core[docs]"` |
| Notion | Notion pages | `pip install "nlqueries-core[wiki]"`, `NOTION_API_TOKEN` |
| Confluence | Confluence spaces | `pip install "nlqueries-core[wiki]"`, `CONFLUENCE_URL`, `CONFLUENCE_USER`, `CONFLUENCE_API_TOKEN` |

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

After ingestion, documents are chunked and embedded into Qdrant (required — see [qdrant-setup.md](qdrant-setup.md)). The document agent retrieves relevant chunks automatically when answering; citations (source document, page/section) are included in the answer.

Query with `nlq doc-ask doc_{source_id}_chunks "..."` for a document-only answer (the collection name follows the pattern `doc_{source_id}_chunks`), or use `nlqueries query` for the orchestrator to route automatically (including hybrid SQL + document answers).

**Python 3.14 note:** document ingestion depends on `langchain_text_splitters`, which is affected by the Python 3.14 / pydantic v1 compatibility issue — see [troubleshooting.md](troubleshooting.md#w6--pydantic-v1-incompatibility-python-314).
