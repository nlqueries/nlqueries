# Changelog

All notable changes to `nlqueries-core` are documented here. Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `NLQ_CACHE_MAX_QUESTION_CHARS` (default 500) and `NLQ_CACHE_ANSWER_TIERS`
  (default `0,1,2`). The first caps the length of a question that may be written
  to the semantic cache; the second selects which tiers may serve an answer, so
  an operator can run exact-match-only caching for a sensitive agent without
  turning the cache off. Existing deployments are unaffected by the defaults.

### Security

- The semantic cache no longer stores an entry whose answer is empty or is this
  system reporting its own failure, nor one whose question is over the length
  limit above. None of these is an authorisation boundary -- a user who can
  query an agent can still write a short, plausible question into its cache, and
  the blast radius is other users of that same agent, who are already entitled
  to its answers. They refuse the shapes that are never worth storing, one of
  which is where a padded prompt injection sits.

### Fixed

- Semantic cache Tier 2 template hits returned SQL that did not parse. A stored
  template already quotes its placeholder (`d >= '[d:DATE]'`), and the binder
  quoted the value again, so a date bound as `>= ''2024-06-01''` and failed on
  every dialect — the hit then served the cached answer text beside "Cached SQL
  failed revalidation and was not executed". String values were doubly wrong:
  the entity patterns captured the question's quote characters as part of the
  value, so `"East"` compared against `"East"` rather than `East`.

### Security

- Cache template binding no longer builds SQL by string substitution. Values are
  bound as literal nodes in a parsed statement and rendered by sqlglot for the
  target dialect, so quoting and escaping follow the engine's own rules rather
  than a hand-written one. Three defences, each independently tested: values are
  type-checked against their placeholder before binding, they can only ever
  become literals, and a bound statement whose shape differs from its template is
  discarded rather than executed.

  This closes FINDING-001 of the September 2026 external review. The reported
  injection was not exploitable — for three separate reasons, all accidental —
  but the protection was two bugs cancelling out, and the obvious fix for the
  parsing failure above would have made it real. No advisory is warranted.

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
