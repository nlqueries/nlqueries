# Changelog

All notable changes to `nlqueries-core` are documented here. Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.2.0] — 2026-07-07

### Added

- MCP server: 7 additional tools beyond the initial `list_agents`/`query` pair,
  including SQL execution that returns result rows, a `mcp_entry.py` entry
  point for Claude Desktop on Windows, and a Glama MCP server manifest.
- ONNX Runtime embedding backend (`optimum[onnxruntime]`) as a lighter
  alternative to the PyTorch backend for the `embed-server` daemon, plus an
  LRU cache and Qdrant scalar quantization for reduced memory/disk footprint.
- Query pipeline performance improvements (Phases 1–6C): concurrent
  dynamic-context Qdrant searches and other latency reductions across the
  embed/search/cache path.

### Changed

- `sentence-transformers` is a mandatory dependency again (was briefly moved to
  an optional `[torch]` extra). Embedding is on the critical path for `ask`,
  `query`, `process-history --embed`, and the semantic cache, and the
  in-process fallback used whenever the `embed-server` daemon isn't running
  imports `sentence_transformers` unconditionally — without the extra
  installed, that raised a raw `ModuleNotFoundError` instead of a clear error.
  Plain `pip install nlqueries-core` now works out of the box, matching the
  README. The `[torch]` extra has been removed as redundant; `[onnx]` remains
  for anyone who wants the lighter ONNX Runtime backend for the `embed-server`
  daemon instead.
- Replaced `langchain-text-splitters` with a small built-in chunker
  (`nlqueries.document_connectors.chunker`). Document connectors (PDF, Word, Notion,
  Confluence) no longer depend on langchain. Python 3.14 is now fully supported for
  all connectors including document ingestion.
- Snowflake and BigQuery connectors now lazy-register into `CONNECTOR_REGISTRY`
  instead of importing their drivers unconditionally, so a plain install no
  longer fails to import `nlqueries.connectors` when those optional driver
  packages aren't present.

### Fixed

- Tier 2 semantic cache entity binding for multi-number questions (e.g. two
  numeric filters in the same query) no longer mis-binds values.
- MCP query tool timeout handling replaced `asyncio.wait_for` with
  `anyio.fail_after`, and a `LIMIT` string-literal bug was corrected.
- SQL results containing `Decimal` or date/datetime column values now
  serialize correctly to JSON instead of raising.
- Resolved all mypy strict-mode errors across the codebase.

## [0.1.0] — Initial release

First public release of NLQueries Core.

### Added

- Natural-language-to-SQL query engine with two access modes: `query` (executes and returns results) and `ask` (previews validated SQL without executing)
- Database connectors: PostgreSQL, MySQL, Snowflake, BigQuery, Amazon Redshift, SQL Server / Azure SQL, DuckDB
- Document connectors: PDF, Word, Excel, Notion, Confluence — with a document agent and hybrid SQL+document routing
- Query pipeline (`process-history`, `annotate`) that builds a YAML knowledge base from schema and query history
- `kb-stats` command for knowledge base coverage and quality reporting
- Semantic cache backed by Qdrant, and an embedding daemon (`embed-server`) to avoid per-invocation model load latency
- Connector aliases, `health` service-check command, local JSONL feedback store (`feedback`, `feedback-stats`)
- MCP server exposing query execution and schema/knowledge lookup as tools for MCP-compatible AI assistants
- CLI available as both `nlqueries` and `nlq`
- Docker Compose stack (Qdrant + core service) using the published [`nlqueries/core`](https://hub.docker.com/r/nlqueries/core) image

### Known limitations

- Python 3.14+ is not supported for document ingestion (`doc-ingest`, `doc-sync-notion`, `doc-sync-confluence`) — see [docs/troubleshooting.md](docs/troubleshooting.md#w6--pydantic-v1-incompatibility-python-314)
- `--days` has no effect on PostgreSQL query history (`pg_stat_statements` doesn't record per-query timestamps) — see [docs/connectors.md](docs/connectors.md#postgresql--enabling-query-history-capture)

[0.2.0]: https://github.com/nlqueries/nlqueries/releases/tag/v0.2.0
[0.1.0]: https://github.com/nlqueries/nlqueries/releases/tag/v0.1.0
