# Changelog

All notable changes to `nlqueries-core` are documented here. Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed

- **`--dialect` accepts every engine `nlqueries connect` registers, and
  defaults to the connector's.** On `ask`, `query` and `eval` it used to offer
  only `postgres`, `snowflake` and `bigquery`, so a SQLite connector, a BIRD-SQL
  database for instance, could not be asked for SQLite SQL at all. It now takes
  `postgres`, `mysql`, `snowflake`, `bigquery`, `redshift`, `mssql`, `duckdb` and
  `sqlite`. Without the flag, the dialect is the connector's type (for a
  `sqlalchemy` connector, the engine its URL names), and `postgres` only when
  that names none. An explicit `--dialect` still wins. A non-Postgres connector
  used without the flag therefore now gets its own dialect where it used to
  get Postgres; pass `--dialect postgres` to keep the old behaviour.
- **The default models are Claude Sonnet 5.5** (`claude-sonnet-5-5`) **and
  Haiku 5.5** (`claude-haiku-5-5`, the fast tier), replacing Sonnet 4.5 and
  Haiku 4.5. Only installs that leave `LLM_MODEL` / `LLM_MODEL_FAST` unset
  change; a configured model, older ones included, is used as before. On ten
  real questions the pair answered all ten correctly, against eight for Sonnet
  4.6 with Haiku 4.5, in a third of the time and for about two thirds of the
  cost. The Bedrock examples in the docs, and the example in the error for a
  non-Bedrock model on a Bedrock install, name
  `bedrock/us.anthropic.claude-sonnet-4-6` for the main model and
  `bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0` for the fast one.
- **Claude 5 and later are sent an effort, and no temperature.** They think
  adaptively, and `output_config.effort` caps how much: the new `LLM_EFFORT`
  setting, `low` by default, applied through the Anthropic API only. They also
  reject a `temperature` with a 400, so the self-consistency candidates no
  longer send them one.
- **`LLM_MAX_OUTPUT_TOKENS` defaults to 4096** (was 1024). Thinking is billed
  from the same allowance, and at 1024 Sonnet 5.5 can run out part-way through
  its SQL. Set it to 1024 to keep the old ceiling.

### Fixed

- **A hand-edited connector entry that is not a mapping no longer crashes the
  CLI.** `agent-a: postgresql://host/db` (the URL where its settings belong)
  raised `AttributeError` from alias resolution, so from every command given an
  alias, and from `doctor`, `nlqueries connectors`, `kb-stats` and the MCP
  `list_connectors` tool. The listings now name the entry. `doctor` reports it
  as a failed check, and a command that opens the connector refuses with a
  message that names it and says to register it again.
- **A temperature no longer fails the Anthropic client on anthropic 1.x.** The
  SDK dropped `temperature` from `messages.create()`, so the self-consistency
  candidates that pass one raised `TypeError` before sending anything, and
  each such candidate was silently dropped. It is now sent in the request body.

## [0.3.0] — 2026-10-02

### Upgrading from 0.2.0

- **The Docker quickstart needs an MCP token.** Add `NLQ_MCP_STATIC_TOKEN`
  (generate it like the Qdrant key: `openssl rand -hex 32`) to `.env` next to
  `QDRANT_API_KEY`. MCP clients send it as `Authorization: Bearer <token>`.
- **A networked MCP server requires authentication.** `--transport sse` and
  `--transport streamable-http` refuse to start without an identity provider or
  a pre-shared token, `NLQ_MCP_RESOURCE_URL`, and a grants file. stdio (Claude
  Desktop) is unaffected. `NLQ_ALLOW_UNAUTHENTICATED_MCP=1` restores the old
  behaviour, with a warning on every start. See
  [docs/mcp-authentication.md](docs/mcp-authentication.md).
- **Semantic-cache entries are signed.** Entries written by 0.2.0 do not verify
  and are treated as misses, so the cache refills. Set `NLQ_CACHE_SIGNING_KEY`
  or `NLQ_CACHE_SIGNING_KEY_FILE` to keep one key across containers and
  restarts.
- **The compose file's Qdrant is now v1.18.2**, and a `qdrant-data` volume
  written by v1.9.x will not open — see *Fixed* below and
  [docs/qdrant-setup.md](docs/qdrant-setup.md).
- **Python 3.14 is supported.** 0.1.0 and 0.2.0 declared `<3.14`, so pip on 3.14
  installed 0.0.1 instead.
- **`nlqueries process-history` exits 1** when the database's query history
  cannot be read, and names the grant or install that fixes it, instead of
  succeeding with nothing.
- **Feedback whose origin cannot be established is no longer promoted.** Pass
  `--include-anonymous` to `promote-feedback` to promote it anyway.

### Added

- Amazon Bedrock works as an LLM provider. An `LLM_MODEL` beginning `bedrock/`
  (or `LLM_PROVIDER=bedrock`) routes through LiteLLM, which authenticates with
  boto3 — so environment credentials, a shared profile, or the instance/task
  role of the host are all usable, and no API key is involved. The model-prefix
  check runs ahead of `ANTHROPIC_API_KEY`, which a Bedrock deployment often
  still has set for something else. See
  [docs/configuration.md](docs/configuration.md#amazon-bedrock).

- `LLM_MAX_OUTPUT_TOKENS` (default `1024`), the answer budget and the two
  budgets derived from it, with `LLMOverride.max_tokens` as the per-request
  equivalent for a host application. The correction tier takes half of it
  (floor 512) and the classification tier an eighth (floor 200), so the
  defaults are exactly the
  numbers the call sites hard-coded before, and raising one setting raises the
  whole pipeline. This matters on a **reasoning** model, which bills its private
  reasoning from the same allowance and spends it first: measured on one, a
  200-token classification used 52 tokens reasoning with four to spare, and a
  5-token health check returned an empty string with `finish_reason=length`.
  See [docs/configuration.md](docs/configuration.md).

- `LLMOverride.extra`, a dict of provider-specific keyword arguments forwarded
  verbatim to the completion call, and the matching `extra=` on `LiteLLMClient`.
  This is how a host application supplies settings core has no model for — an
  AWS region and credentials, say — without core learning any one cloud's
  vocabulary. An explicit `api_key`/`api_base` still wins over the same name in
  `extra`.

  `extra` may not carry a name the client already passes to LiteLLM (`model`,
  `messages`, `max_tokens`, `stream`, `temperature`); the constructor rejects
  those. Allowing them was inconsistent and half silent — a duplicate keyword
  raised `TypeError` on the sync and streaming paths, but on `acomplete` the
  same entry was merged after the caller's and quietly replaced it, so a host
  that put `max_tokens` in `extra` would have capped every async completion
  without an error.

  Supplying `extra` for a provider that cannot forward it now raises
  `ValueError` rather than dropping it. Dropping was the dangerous outcome: with
  no `provider` on the override it resolves to `LLM_PROVIDER`, so an override
  built for Bedrock would go out to the public Anthropic API under the
  process-level key — the exact egress a deployment chose Bedrock to avoid, with
  a correct-looking answer and nothing to notice.

- `LLM_PROVIDER=bedrock` now requires an `LLM_MODEL` beginning `bedrock/`, and
  the first LLM call raises without one. Naming the provider does not choose a
  model, and the default is an Anthropic id that LiteLLM routes to Anthropic;
  there is no safe default to substitute, since Bedrock ids differ per region
  and only work once enabled for the account. The check sits at the client
  rather than at import so that `connect`, `extract-schema` and the diagnostics
  you would use to find the misconfiguration still run.

- The CLI no longer refuses a Bedrock host for having no API key. `doctor`,
  `process-history --annotate` (the default) and `export-kb --describe-columns`
  each gated on `ANTHROPIC_API_KEY`/`OPENAI_API_KEY` being present, so the
  deployment these notes recommend most — an IAM role and no key at all — was
  rejected by commands that would have worked, `doctor` included: it reported the
  LLM as misconfigured on a working host, in the command you run to find out why
  something is wrong. They now ask `config.llm_credentials_available()`, which
  counts a Bedrock configuration as a credential route because boto3 supplies
  one.

  `--describe-columns` was additionally checking `LLM_API_KEY`, a variable this
  product does not define anywhere; it now accepts the same credentials as
  everything else.

  `doctor`'s Config line and the MCP server's health check asked the same
  question a third and fourth way, so `doctor` on a Bedrock host would have
  printed a passing LLM line beside a Config line warning of a missing key. Both
  now use the same helper. The MCP check was reporting "no ANTHROPIC_API_KEY set"
  and was therefore already wrong for an OpenAI-only deployment.

- On a Bedrock deployment, a `LLM_MODEL_FAST` that is not a Bedrock id is now
  refused rather than routed elsewhere. The two tiers could previously disagree
  about which cloud they were talking to: with Bedrock selected by the model
  prefix, an `LLM_MODEL_FAST` left over from a previous Anthropic setup sent
  every auxiliary call — intent classification and follow-up resolution, which
  carry the question and the conversation history — to `api.anthropic.com` under
  the leftover key, with no error, while the default tier stayed on Bedrock. The
  check is made against the default model, so both ways of selecting Bedrock
  agree, and the message names `LLM_MODEL_FAST` rather than `LLM_MODEL`.

- A `bedrock/` model id selects Bedrock even when the provider says otherwise —
  when nothing names a provider it resolves to LiteLLM, and an explicit
  contradiction is refused. Previously such an override built an
  `AnthropicClient`, which does not reject a `bedrock/` id: it would transmit
  the system prompt, the schema and the user's question to `api.anthropic.com`
  before failing with a model-not-found. This was reachable from the deployment
  the docs recommend most — an IAM role and no stored credentials, so no `extra`
  to trigger the other guard. `LLMOverride(provider="bedrock", …)` is also
  accepted now, matching what `LLM_PROVIDER=bedrock` already did.

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

- **Python 3.14 support.** `requires-python` is now `>=3.11,<3.15`, and CI tests
  3.11 through 3.14.

- **MCP authentication and authorisation.** A networked transport authenticates
  callers with an identity provider (`NLQ_MCP_OIDC_DISCOVERY_URL`,
  `NLQ_MCP_OIDC_CLIENT_ID`) or a pre-shared token (`NLQ_MCP_STATIC_TOKEN` or
  `NLQ_MCP_STATIC_TOKEN_FILE`), with `NLQ_MCP_RESOURCE_URL`. Every tool call is
  authorised against a grants file (`NLQ_MCP_GRANTS_FILE`) by subject, agent and
  action, and audited. Each caller is also limited to
  `NLQ_MCP_RATE_LIMIT_PER_MINUTE` calls (default 60) and
  `NLQ_MCP_MAX_CONCURRENT` at once (default 8); `0` disables either. See
  [docs/mcp-authentication.md](docs/mcp-authentication.md).

- **One correction when the database rejects a generated statement.** The model
  gets the database's error, the failed SQL and the question once; a correction
  that passes the same validation as a first attempt runs on the same
  connector. It starts only when the caller's deadline leaves time for it, and
  cached-SQL replay does not get it.

- **Markdown and plain-text documents.** `doc-ingest` reads `.md`, `.markdown`
  and `.txt`.

- **`nlqueries connect sqlalchemy --url <url>`** registers the generic
  SQLAlchemy connector from the CLI.

- **`NLQ_STATE_DIR`** (default `~/.nlqueries`) is the root for everything kept
  between runs. The knowledge base, connector file, capsules and feedback
  default under it and keep their own overrides; the embedding server's pid
  file, the cache signing key and the CLI's session transcripts now live under
  it too.

- **`NLQ_EMBED_MODEL`** names the embedding model by hub name or local path. A
  model whose vectors are the wrong width is refused at load. The default is
  unchanged.

- **`REDSHIFT_SOCKET_TIMEOUT_SECONDS`** replaces the hardcoded fifteen-second
  Redshift socket timeout, which capped every query at fifteen seconds and could
  expire while a Serverless workgroup resumed. It bounds the whole connection,
  not only the handshake, so its default is derived from
  `CONNECTOR_STATEMENT_TIMEOUT_SECONDS` with headroom (that plus 30, at least
  60 — 150 by default) and is `0`, unbounded, when the statement timeout is
  disabled.

- **Limits on document extraction.** A zip-based document (Excel, Word) is
  refused before parsing if it would expand beyond
  `NLQ_MAX_DOCUMENT_EXPANDED_BYTES` (400 MiB) or `NLQ_MAX_DOCUMENT_EXPANSION_RATIO`
  (100×). Extraction is bounded by `NLQ_MAX_DOCUMENT_ROWS` (50,000) and
  `NLQ_MAX_EXTRACTION_SECONDS` (120). Running out of time raises
  `DocumentExtractionTimeout`, a subclass of `DocumentTooComplexError`, so it
  can be retried apart from the final refusals.

- **Table schemas in the knowledge base.** Each table records its `schema`, and
  the prompt names it `schema.table`.

- **Partial column lists are said so.** When a knowledge-base table lists only
  some of its columns, the prompt and the MCP `get_agent_schema` tool say to
  select those columns by name, never with `*`.

- **`connector_class_for` and `set_connector_resolver`**, one seam through which
  every connector class is resolved, so an embedding application can substitute
  its own.

### Changed

- On Bedrock, `LLM_MODEL_FAST` now defaults to whatever `LLM_MODEL` is instead of
  `claude-haiku-4-5-20251001`. That id does not exist on Bedrock, so the old
  default failed every auxiliary call — the intent classifier and the follow-up
  resolver — while the main model kept working, which reads as a broken product
  rather than one unset variable. The fallback is correct but not cheap; set
  `LLM_MODEL_FAST` to a Bedrock Haiku id to get the cheap tier back.

- The caller's `cache_context` is now matched inside the Qdrant query rather than
  after it. `put()` writes a digest of the context under a reserved payload key
  and `get()` filters on that single value, so a lookup's candidates already
  belong to the caller. Previously the context was compared key by key, which
  Qdrant can only evaluate as a subset test — entries from other callers came
  back and were discarded after the search, consuming the
  `NLQ_CACHE_COSINE_CANDIDATES` slots a lookup scans. On a busy conversational
  agent that starved context-free reads outright.

  Entries written before the key existed carry no digest, so a **context-free**
  read matches either its absence or the empty-context digest, and those entries
  stay readable. The disjunction is on that branch only: a scoped read takes the
  exact-match path, so a pre-digest entry that carries a context is no longer
  reachable through Tier 1 or Tier 2. That costs nothing in practice, because the
  signature and point-ID changes above already invalidate such entries — an entry
  written with a context by any released version fails verification today. The digest is derived rather than signed, so it can
  misdirect a lookup but cannot get a foreign entry served — `_payload_matches`
  still compares the real, signed context before anything is returned.


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

- A model whose prefix names a provider other than Anthropic is refused when
  the provider resolves to the native Anthropic client, rather than being sent
  to api.anthropic.com and failing there. `AnthropicClient` does not inspect
  the model id, so `LLM_MODEL=deepseek/deepseek-chat` alongside an Anthropic
  key used to transmit the system prompt, the schema and the question before
  the provider reported the model missing. The equivalent check already existed
  for `bedrock/` only.

  An `anthropic/`-prefixed id is normalised to the bare form rather than
  refused, so `anthropic/claude-sonnet-4-5` and `claude-sonnet-4-5` behave
  identically; passing it through had the same leak-then-fail shape, since no
  Anthropic model id contains a slash.

  Two cases are deliberately left alone. A **bare** model name is not judged:
  LiteLLM's registry spells DeepSeek's keys bare and Mistral's prefixed, so
  placing one needs that registry, which this path does not consult. And the
  check is skipped when `api_base` or `ANTHROPIC_BASE_URL` points the client at
  a gateway, because the request does not reach api.anthropic.com and prefixed
  ids may be exactly what that gateway expects. See
  [docs/configuration.md](docs/configuration.md#provider-and-model-must-agree).

- **The Docker quickstart configures MCP authentication.** `docker-compose.yml`
  requires `NLQ_MCP_STATIC_TOKEN` and gives that token full access through a
  grants file it writes at start; before this, the image's SSE server would
  have refused to start under the shipped compose file.

- **Postgres query history covers this database's reads only.** Statements from
  other databases on the server, and writes, no longer use up the history
  budget.

- **Personal data is not sampled for column descriptions.** Columns whose names
  mark them as personal data are described from their name and type, without
  sample values. The sample query names each table with its schema.

- **An unreadable query history is an error.** Postgres, SQL Server, BigQuery,
  Redshift and Snowflake raise `QueryHistoryUnavailable`, saying what could not
  be read and the grant or install that fixes it, instead of returning nothing.

- **Schema extraction degrades instead of returning nothing.** Snowflake,
  Redshift and SQL Server return tables, columns and row counts without key
  information when key metadata cannot be read, and the generic SQLAlchemy
  connector keeps a table whose keys cannot be read.

- **Hybrid answers carry their data.** The answer includes the rows the SQL step
  returned, the statement it ran (under `sql`) and whether the rows were
  truncated, as the SQL path's answers already did.

- **The generic SQLAlchemy connector applies TLS settings, or refuses them.**
  For libpq-based PostgreSQL drivers they become connection arguments; for a
  driver whose TLS parameters are not mapped, connecting raises instead of
  silently ignoring them.

- **A connector's stored configuration reaches the database** in every CLI path
  — health check, schema extraction, history and queries — rather than a subset
  of its fields.

- **The licensor is Theorence Labs Private Limited**, in `LICENSE` and the
  contributor licence agreement. The licence terms are unchanged.

### Fixed

- **The shipped `docker-compose.yml` pinned a Qdrant that could not serve any
  vector search.** NLQueries searches through the Universal Query API
  (`query_points`), which Qdrant added in v1.10; the compose file pinned v1.9.3
  and the benchmarks compose v1.9.2, so every search returned `404`. The
  semantic cache degraded to exact-match hits only, dynamic context injection
  found nothing and document retrieval returned nothing — and because several of
  those paths treat a failed search as an empty result, the symptom was a system
  answering slowly and without context rather than reporting an error. Measured
  on the version this file pinned and the one it pins now: `v1.9.3 -> MISS`,
  `v1.18.2 -> CACHED`, driving a Tier 1 paraphrase so only the cosine tier can
  serve it (v1.12.4 was measured too, and also serves it). Both
  files now pin v1.18.2, matching enterprise and the locked client, and the
  `qdrant-client` floor moves from `>=1.9` to
  `>=1.10` — the client gained `query_points` at the same release, so a resolved
  1.9.x raised `AttributeError` at every call site and reached the same silent
  empty result.

  **An existing `qdrant-data` volume written by v1.9.x must be removed.**
  v1.18.2 panics on startup against it, and there is no data-preserving path
  forward: stepping v1.9.3 → v1.12.4 → v1.18.2 panics identically at the last
  hop, and staying on v1.12.4 makes `qdrant-client` 1.19.0 report the server as
  incompatible on every construction. The failure is loud — the container exits
  — and `docs/qdrant-setup.md` records the measurements, the exact error, and
  what rebuilding costs. The cache regenerates itself; document chunks need
  re-ingesting.


- The semantic cache no longer reports a failed search as a cache miss. A
  rejected request and an empty cache were the same `None` to the caller while
  meaning opposite things. A failure is now logged once per collection and tier,
  naming the version requirement, since a 404 from a pre-v1.10 server is its
  most likely cause.


- Tier 2 template lookups now validate each candidate in turn, as Tier 1 does. An
  expired or unverifiable template ranked above a usable one ended the lookup
  instead of continuing past it — which mattered increasingly, since expired
  points were never deleted.


- Semantic cache Tier 2 template hits returned SQL that did not parse. A stored
  template already quotes its placeholder (`d >= '[d:DATE]'`), and the binder
  quoted the value again, so a date bound as `>= ''2024-06-01''` and failed on
  every dialect — the hit then served the cached answer text beside "Cached SQL
  failed revalidation and was not executed". String values were doubly wrong:
  the entity patterns captured the question's quote characters as part of the
  value, so `"East"` compared against `"East"` rather than `East`.

- **`docker compose up` could not start the stack.** The Qdrant healthcheck used
  `wget`, which the Qdrant image does not ship, so Qdrant never reported healthy
  and the core service never started.

- **Redshift transactions.** `extract_schema`, `test_connection` and
  `extract_query_history` end their transaction whether they succeed or fail,
  and the row-count fallback actually runs.

- **Connectors release the driver handle on close**, not only a SQLAlchemy
  engine, without cutting off a call still in progress.

- **Snowflake accepts a pasted host** and reduces it to the account identifier.

- **Opening a connector says why it failed**, rather than returning nothing.

- **An authenticated MCP server no longer logs that it has no
  authentication** when bound to every interface.

### Security

- The semantic cache no longer stores an entry whose answer is empty or is this
  system reporting its own failure, nor one whose question is over the length
  limit above. None of these is an authorisation boundary -- a user who can
  query an agent can still write a short, plausible question into its cache, and
  the blast radius is other users of that same agent, who are already entitled
  to its answers. They refuse the shapes that are never worth storing, one of
  which is where a padded prompt injection sits.


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

- **Signed semantic-cache entries.** Entries are signed with HMAC-SHA256 over
  their contents and the context they were produced in (agent, connector,
  dialect, schema, policy version) and verified on read; an entry this
  deployment did not write is a miss. The key comes from
  `NLQ_CACHE_SIGNING_KEY`, `NLQ_CACHE_SIGNING_KEY_FILE`, or one generated under
  the state directory.

- **A SQL policy decision is bound to the statement it was made about**, by
  digest and dialect, so a decision for one statement cannot authorise another.

- **OIDC tokens.** A provider whose discovery document has no `issuer` is
  refused rather than verified with the issuer check off, and a token with no
  `sub` is refused rather than becoming an empty identity.

- **Feedback records where it came from**, and promotion skips records whose
  origin cannot be established.

- **The embedding server validates its requests** and answers malformed ones
  with `400` instead of dropping the connection; state files are written with
  restricted permissions.

- **Hardened containers.** The composed core service runs with a read-only root
  filesystem, a `noexec,nosuid` `/tmp`, all capabilities dropped,
  `no-new-privileges`, and pid and memory limits. The image installs its
  dependencies from a hashed lock (`requirements/core.lock`) on a base image
  pinned by digest.

- **Documented and benchmark services bind to loopback**, and a test keeps
  every published port naming an interface.

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

[0.3.0]: https://github.com/nlqueries/nlqueries/releases/tag/v0.3.0
[0.2.0]: https://github.com/nlqueries/nlqueries/releases/tag/v0.2.0
[0.1.0]: https://github.com/nlqueries/nlqueries/releases/tag/v0.1.0
