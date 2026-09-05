# Changelog

All notable changes to `nlqueries-core` are documented here. Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed

- **The shipped `docker-compose.yml` pinned a Qdrant that could not serve any
  vector search.** NLQueries searches through the Universal Query API
  (`query_points`), which Qdrant added in v1.10; the compose file pinned v1.9.3
  and the benchmarks compose v1.9.2, so every search returned `404`. The
  semantic cache degraded to exact-match hits only, dynamic context injection
  found nothing and document retrieval returned nothing — and because several of
  those paths treat a failed search as an empty result, the symptom was a system
  answering slowly and without context rather than reporting an error. Both
  files now pin v1.18.2, matching what enterprise ships.

- The semantic cache no longer reports a failed search as a cache miss. A
  rejected request and an empty cache were the same `None` to the caller while
  meaning opposite things. A failure is now logged once per collection and tier,
  naming the version requirement, since a 404 from a pre-v1.10 server is its
  most likely cause.

### Added

- `NLQ_CACHE_PRUNE_INTERVAL_SECONDS` (default 3600; `0` disables). The semantic
  cache now sweeps points past the TTL on write, at most once per collection per
  interval. Nothing previously deleted anything — the TTL is applied on read, and
  `invalidate()` drops the whole collection — which stopped being survivable when
  point IDs gained the cache context, since an id that never recurs is never
  overwritten either. Expired points were also still ranked by the vector search
  and consumed the `NLQ_CACHE_COSINE_CANDIDATES` slots a lookup scans.

- `NLQ_CACHE_MAX_QUESTION_CHARS` (default 500) and `NLQ_CACHE_ANSWER_TIERS`
  (default `0,1,2`). The first caps the length of a question that may be written
  to the semantic cache; the second selects which tiers may serve an answer, so
  an operator can run exact-match-only caching for a sensitive agent without
  turning the cache off. Existing deployments are unaffected by the defaults.

### Fixed

- Tier 2 template lookups now validate each candidate in turn, as Tier 1 does. An
  expired or unverifiable template ranked above a usable one ended the lookup
  instead of continuing past it — which mattered increasingly, since expired
  points were never deleted.

### Security

- The semantic cache no longer stores an entry whose answer is empty or is this
  system reporting its own failure, nor one whose question is over the length
  limit above. None of these is an authorisation boundary -- a user who can
  query an agent can still write a short, plausible question into its cache, and
  the blast radius is other users of that same agent, who are already entitled
  to its answers. They refuse the shapes that are never worth storing, one of
  which is where a padded prompt injection sits.

### Changed

- Cache entry signatures now cover the caller's `cache_context`. Previously the
  HMAC covered the answer and its SQL but not the context keys, so anyone with
  write access to Qdrant and no access to the signing key could move a valid
  entry between contexts by editing them -- no forgery required. The context is
  appended to the signed message only when non-empty, so entries written without
  one keep verifying and the cache does not go cold on upgrade; only
  context-carrying entries miss once.

- Semantic cache point IDs now include the caller's `cache_context`. **This
  removes what bounded a collection's size**: nothing deletes points (the TTL is
  applied on read), so a repeated question used to upsert over its own id.
  Entries written under a context that changes per turn now accumulate
  indefinitely, are still searched after expiry, and pre-existing scoped entries
  are orphaned rather than paying a one-off miss. The sweep added above reclaims
  them; see "Cache partitioning and authorisation" in `docs/architecture.md`. Two callers
  in different contexts asking the same question previously derived the same id
  and upserted over one another; with the equality match below, neither then read
  the survivor back. Not a leak -- the partition holds -- but both missed
  indefinitely. Entries written without a context keep the id they had.

- A `cache_context` naming a key the cache writes itself (`kind`, `sql`, …) is
  now refused on both read and write, with a warning. Such a key was overwritten
  during the write, so the entry was stored unscoped -- readable by every
  context-free caller and not by the caller that asked to be scoped.

- The semantic cache's `cache_context` (seam S2) is now matched by equality
  rather than as a subset. A caller that passes no context previously matched
  entries written under *any* context, while the reverse correctly missed --
  so the case the mechanism exists to catch, a caller that forgets to pass its
  context on read, was the one that silently succeeded. A context-free read now
  sees only entries written without a context. In practice this affects
  follow-up-scoped entries, which are no longer served to standalone questions;
  standalone turns still share with each other.

  Because the equality is applied client-side -- Qdrant's filter can require the
  caller's keys but not the absence of others -- the cosine tiers now fetch a
  few candidates and take the first that clears both the similarity threshold
  and the context. Fetching one would let a nearer entry from another context
  shadow a valid one ranked just below it, which for a context-free read is any
  follow-up-scoped entry at all. See "Cache partitioning and authorisation" in
  `docs/architecture.md`.

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

- Every connector now runs the query in the most restrictive execution its
  engine offers, rather than only the four that already did. SQL Server and the
  generic SQLAlchemy connector no longer use `engine.begin()`, which commits on
  exit; they run on a connection that is never committed and is rolled back
  whether the statement succeeded or failed. The SQLAlchemy connector also
  issues `SET TRANSACTION READ ONLY` on the dialects that have it. Snowflake
  wraps the query in `BEGIN`/`ROLLBACK`. This matters because every validator in
  front of a connector asks whether the root node is a `SELECT`, and
  `SELECT some_volatile_function(...)` satisfies that while writing.

  What the rollback does not cover is documented rather than implied. The gap is
  mostly DDL, and on MySQL also the storage engine: an `INSERT` into a MyISAM or
  MEMORY table survives the rollback outright, with only warning 1196. An engine that commits implicitly around DDL keeps a `CREATE` or
  `DROP` whatever the transaction does -- Snowflake, and MySQL, MariaDB and
  Oracle behind the generic connector -- and SQLite runs DDL outside the
  transaction altogether. BigQuery has no transaction at all: its jobs are
  pinned to standard SQL with no session, and a non-`SELECT` statement type is
  logged after the fact as an audit signal, not prevented. On all of these the
  database grant is doing work the connector cannot. MySQL additionally keeps
  the rollback only, since `SET SESSION TRANSACTION READ ONLY` is refused inside
  an open transaction and SQLAlchemy has already begun one.

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
