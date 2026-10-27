# Changelog

All notable changes to `nlqueries-core` are documented here. Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Upgrading from 0.4.0

- **A Docker Compose stack started from the 0.4.0 release runs 0.3.0.** The
  `docker-compose.yml` in the v0.4.0 tag pins `nlqueries/core:0.3.0`, and the
  README and getting-started at that tag say so. The `pip` package and the
  `nlqueries/core:0.4.0` image are 0.4.0. Run `docker exec nlqueries-core pip
  show nlqueries-core` to see which you have. To move to 0.4.0 without
  upgrading further, set the `image:` line to `nlqueries/core:0.4.0` and run
  `docker compose up -d`, then follow "Upgrading from 0.3.0" under 0.4.0.
  This release's compose file pins its own version.
- **Run `nlqueries process-history <agent> --embed` again for each agent
  whose capsules you embedded.** Capsules embedded by earlier releases sit in
  the collection named by `QDRANT_COLLECTION` (default `nlqueries`), which no
  question ever read, and are unsigned. The new run writes them where the
  agent's questions search. The old collection can then be deleted in Qdrant.
- **The schema collection is renamed for an agent ID with a character the
  storage name replaces.** Nothing to do. For an ID Qdrant accepted under the
  old name (one with `@` or a space, say), the raw-ID collection is still
  read until the new name holds its tables, and calling
  `generate_knowledge_base(..., embed=True)` again moves them and deletes it.
  Qdrant refuses `:` in a collection name, so for a connector ID such as
  `postgres:localhost:mydb` there was no such collection to move.

### Security

- **Query capsules from Qdrant are signed and checked.** Capsule SQL templates
  are shown to the model as example queries, like verified examples, and
  anything written into an agent's schema collection was shown. `process-history
  --embed` now signs each capsule with the cache's signing key
  (`NLQ_CACHE_SIGNING_KEY`), and questions use only capsules whose signature
  verifies, of their own agent; others are skipped, with one warning per
  collection. The signature does not cover the vector a capsule is found by,
  so each one's text is embedded again and must score against the question
  as its stored vector did; a capsule moved onto another question's vector
  is left out.

- **pip 26.2 in the image.** The `Dockerfile` pinned pip 24.0, which the
  image keeps after installing the lock (CVE-2025-8869, fixed in 25.3;
  CVE-2026-1703, fixed in 26.0; CVE-2026-3219 and CVE-2026-6357, fixed in
  26.1; CVE-2026-8643, fixed in 26.1.2; CVE-2026-13346, fixed in 26.2). The
  application never runs pip, so only installing packages inside a running
  container was exposed. Images up to 0.4.0 carry pip 24.0.

### Fixed

- **A DeepSeek key counts as an LLM credential.** The credential check knew
  `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` and Bedrock. On a host with only
  `DEEPSEEK_API_KEY`:
  - `nlqueries health` and the MCP `health` tool reported no credentials;
  - `process-history` refused to annotate;
  - `export-kb --describe-columns` skipped the descriptions.
- **The direct Qdrant probes send `QDRANT_API_KEY`.** `nlqueries health`, the
  MCP `health` tool, `process-history --embed`'s preflight and the `kb-stats`
  cache count call Qdrant's REST API without building a client, and they sent
  no key. On a Qdrant that requires one:
  - `health` reported "connection refused";
  - the cache count showed n/a and said nothing.
- **A release's compose file runs that release.** A test now holds the
  `docker-compose.yml` image pin, and the tag that README and getting-started
  quote, to the version in `pyproject.toml`. The release workflow refuses a
  final tag that does not match that version. Together they stop a tag from
  publishing a compose file pinned to another release, as 0.4.0's was.
- **Setting `MCP_PORT` in `.env` no longer makes the Docker Compose MCP
  server unreachable.** It was passed into the container as well, so the
  server listened on the new port behind a mapping to 8080 and nothing
  answered, while the container still reported healthy. The service now
  keeps the server on 8080 inside the container, and `MCP_PORT` picks only
  the host port. `MCP_HOST` is fixed to `0.0.0.0` the same way:
  `MCP_HOST=127.0.0.1` in `.env` bound the container's own loopback, with
  the same result.
- **`LOG_LEVEL` sets the log level.** It was documented up to 0.3.0 and read
  by nothing. It now applies to the CLI, the MCP server and the embedding
  daemon. Unset, nothing changes: the MCP server and the daemon log at INFO,
  and the CLI sets up no logging, so only warnings and errors reach stderr.
  An unrecognised value is ignored, with a warning.
- **`mcp-server start --sse` no longer announces a server that then refuses
  to start.** The green "listening on" line was printed before the startup
  checks (the cache signing key, authentication, the wildcard-bind guard).
  It is now printed once they pass, and says "starting on", since the port
  is bound after it.
- **Redshift query history: the permission advice was wrong.** The error and
  `docs/connectors.md` named `pg_read_all_stats`, a PostgreSQL role Redshift
  does not have. Every login can read `STL_QUERY` but sees only its own
  statements unless it is a superuser or has `SYSLOG ACCESS UNRESTRICTED`;
  the error now says what it usually means, a Serverless workgroup, which
  has no `STL_QUERY`.
- **`process-history --embed` capsules are used.** They were written to
  `QDRANT_COLLECTION`, which nothing read, while questions searched the
  agent's `agent_<id>_schema` collection for example queries. They now go
  there, replacing the agent's earlier capsules on each run, with their
  intent and description, which were not stored, so a capsule found by a
  search was shown as a bare SQL template. The search keeps to capsule points,
  where the collection's table and column points used to take its places.
  `process_query_history(..., embed=True)` needs `agent_id=` for the same
  reason, and raises `ValueError` without it. The capsules file is now saved
  before the embed, so a Qdrant failure there no longer discards them.
- **Writing a state file on Windows no longer fails now and then with "Access is
  denied".** Every connectors, knowledge-base and feedback file is written to a
  temporary file and renamed into place, and Windows refuses the rename while
  another process, typically an antivirus scanner or the indexer, is reading a
  file just written. A command that writes the same file twice in quick
  succession (`agent docs add`, `import-dbt`) hit it. The rename is now retried
  a few times over about half a second, on Windows only.
- **Redshift query history reads only the connector's database.**
  `STL_QUERY` is cluster-wide, and the history query had no `database`
  filter, so a superuser or `SYSLOG ACCESS UNRESTRICTED` login built the
  knowledge base from every database's statements on the cluster. It now
  keeps to `current_database()`, as the PostgreSQL connector does, compared
  on its first 32 characters, which is all `STL_QUERY` stores of the name.
- **`doc-sync-confluence --since` notices pages that are gone.** It re-read
  the space only when a page had been modified since the timestamp, so a page
  deleted, moved to another space or hidden by a permission change kept its
  chunks, and `doc-ask` kept citing it, until a full sync. The space's page IDs
  are now compared with the stored ones, and a stored page that is no longer
  there re-reads the space. Listing the IDs fetches no page bodies.
- **`doc-sync-notion --since` no longer keeps a page whose text was removed.**
  An empty read was taken to mean "unchanged", so a page edited down to no
  text kept its old chunks. The sync now asks whether the page was edited
  (`NotionConnector.has_changes_since`), and an edited page is read and stored
  as without `--since`: an empty one is refused unless `--allow-empty`.
- **The knowledge base's own capsules are shown whenever no example was
  found.** They were used only when nothing at all had been found, so a
  relevant-tables hint alone left the prompt with no example queries. An
  "Example Queries" heading with nothing under it is no longer added.

### Added

- **`docs/mcp-clients.md`: connecting MCP clients.** Claude Desktop and Cursor
  over stdio, including the `env` block a client-started server needs for its
  LLM key, and a client over the network with a bearer token. It also lists
  the tools, what `query` returns, and the usual first-run errors.
- **`docs/llm-providers.md`: the LLM providers NLQueries has been run
  against.** Anthropic, OpenAI, Amazon Bedrock and DeepSeek: how each is
  selected, its models, effort and thinking settings, prompt caching, and
  the output budget and timeout every provider shares.
- **`docs/observability.md`: what NLQueries reports about itself.** Logs and
  `LOG_LEVEL`, audit events, the OpenTelemetry spans and metrics and how to
  export them (core installs no exporter of its own), health checks, and
  per-answer provenance.
- **`nlqueries agent docs add|remove|list`: the document sources an agent may
  use.** The list is kept as `document_sources` on the agent's entry in the
  connectors file, under either form of its ID or its alias, and survives
  registering the connector again. `add` warns about a source not ingested
  yet; `remove` keeps the source's stored chunks.
- **`query` and the MCP `query` tool answer from an agent's documents.** For
  an agent that lists document sources, each question is answered from the
  database, the listed sources, or both (a hybrid answer), as the router
  judges it, with citations naming the source; routing to documents was
  reachable only from the Python API. The document agent searches every
  listed source and keeps the nearest chunks overall, skipping one not
  ingested. An agent that lists none answers from its database only, as
  before. On the MCP server, `query:execute` covers document answers. `query`
  now prints a hybrid answer's text, which it showed only when no rows came
  back, and names each source.

### Changed

- **One agent per storage name, for verified examples and capsules too.** The
  schema collection (`agent_<id>_schema`) used the raw agent ID while every
  other store used the storage name, and verified examples were matched on the
  raw ID. For a connector ID with `:`, such as `postgres:localhost:mydb`,
  Qdrant refuses the raw-ID name, so its table and column embeddings and its
  capsules could not be stored, and `process-history --embed` failed for it.
  For any ID the storage name changes, the CLI and the MCP server
  (`postgres_localhost_mydb`, from the knowledge-base file name) used
  different names, and the MCP server never saw a pair promoted from the CLI. The schema collection now uses the storage name, and pairs and
  capsules are matched by it, so both see the same ones. The agent is passed
  to `assemble_prompt` and `assemble_prompt_async` as `agent_id=` rather than
  read back out of the collection name; a caller passing `collection=` alone
  works as before.

### Removed

- **`QDRANT_COLLECTION`.** It named only the collection `--embed` wrote and
  nothing read; capsules now go to the agent's schema collection.

- **`NLQ_EXPLAIN_VALIDATION` and `KB_REFRESH_INTERVAL`.** Both were
  documented up to 0.3.0 and read by nothing, so setting them never changed
  anything. The EXPLAIN check is still available to library callers as
  `validate_and_repair(..., explain_check=True)`.

## [0.4.0] — 2026-10-25

Mostly security fixes: see *Security* below, and the steps to take first.

### Upgrading from 0.3.0

- **Run `nlqueries export-kb` again for each agent.** The knowledge base's
  example queries are rebuilt on every run, and corrections accepted under
  0.3.0's rules are dropped; corrections recorded over MCP are used only with
  `--include-anonymous`.
- **Run `nlqueries promote-feedback` (or `export-kb`) once per agent.**
  Verified question/SQL pairs are now signed, and pairs promoted by 0.3.0 are
  unsigned and ignored, with a warning, until then. An agent whose ID contains
  a `.` starts with an empty verified-examples collection and needs this too.
  Stored pairs that would not be promoted today are removed, unless every
  pair fails at once. A deployment whose embedding model has changed since its
  pairs were promoted drops them all, with a log line per agent saying so;
  promoting again restores them.
- **`nlqueries health` works again.** In 0.3.0 its `SELECT 1` probe could not
  run, so every registered database was reported as "connection failed".
- **The Snowflake driver is now 5.x** (`snowflake-connector-python>=5.0` in the
  `snowflake` extra), and a Snowflake query's timeout is split between waiting
  in the warehouse queue, set by the new `SNOWFLAKE_QUEUE_TIMEOUT_SECONDS`
  (default 30, at most half the budget), and running.
- **`mcp` 1.30 or later and `filelock` 4.2.1 or later are required.** The
  filelock floor holds a fresh `pip install` on litellm 1.103.8 until litellm
  allows filelock 4; the image is not affected.
- **`.env` is read from the working directory only**, not from parent
  directories.
- **OIDC providers must be reached over `https`** (loopback excepted), and a
  discovery URL must belong to the issuer it names. Microsoft Entra ID needs its
  tenant-specific v2.0 URL; see `docs/mcp-authentication.md`.
- **`nlqueries update-password` exits 1** when no keyring can store the
  password, instead of 0.
- **Cache entries miss once** after upgrading, as the SQL policy version and
  the cache's schema binding have changed.

### Security

- **filelock 4.2.4 and setuptools 84.2.1 in the image.** The lock pinned
  filelock 3.32.4 (CVE-2026-112907, CVE-2026-112913, CVE-2026-112940, fixed
  in 4.2.1) and setuptools 84.0.0 (GHSA-xmj3-9gf7-h52w, fixed in 84.1.0).
  The Snowflake driver 4.x capped filelock below 4, so the `snowflake` and
  `dev` extras now require **`snowflake-connector-python` 5.0 or later**;
  5.x no longer depends on filelock, and it needs protobuf below 7, so the
  lock moves protobuf from 7.36.0 to 6.33.6. Packages only the 4.x driver
  used (asn1crypto, platformdirs, pyOpenSSL, sortedcontainers) leave the
  image. **The Snowflake query timeout is now sent as Snowflake's
  `STATEMENT_QUEUED_TIMEOUT_IN_SECONDS` and `STATEMENT_TIMEOUT_IN_SECONDS`
  statement parameters**, so the server cancels the statement. It was the
  driver's `execute(timeout=...)` keyword, a client-side limit on queued and
  running time together, which 5.x removed; its `execute()` takes `**kwargs`
  and ignores it, so with the old call every Snowflake query would have run
  unbounded. Snowflake bounds queueing and running separately, so the budget
  is split and `timeout_seconds` stays a total: **`SNOWFLAKE_QUEUE_TIMEOUT_SECONDS`**
  (new, default 30, at most half the budget) for the warehouse queue and the
  rest for running, 30 and 90 of the default 120 seconds. Raise it for a
  warehouse that queues or resumes from suspension. A query cancelled for
  either reason is reported with Snowflake's timeout error (000630, SQLSTATE
  57014), which is now recognised as a timeout and not handed back to the
  model to be rewritten and re-run, however soon it came back. The rest of what
  nlqueries uses from the driver (`connect`, `DictCursor`, `execute(sql,
  params)` with `%s` placeholders, `description`, iterating, `fetchone`,
  `fetchall`, `close` and `errors.ProgrammingError`) is unchanged in 5.x.
  **`filelock>=4.2.1` is now a dependency of the package**, though nothing
  in nlqueries imports it: litellm 1.104 and later cap filelock below 4, so
  without the floor `pip install nlqueries-core` resolves the vulnerable
  3.x. With it, pip installs litellm 1.103.8, the last release without that
  cap, until litellm allows filelock 4. The image is not affected; its
  pinned litellm does not depend on filelock.
- **The MCP `query` tool checks SQL in the grammar of the agent's own
  database.** The SQL policy parsed the generated statement in the dialect the
  MCP client passed (default `postgres`), not the dialect of the engine that ran
  it, and a statement is only checked when it is parsed the way the engine will
  read it. The server now takes the dialect from the agent's connector, through
  the same lookup the CLI's `--dialect` default uses
  (`connectors.loader.dialect_for_agent`). The tool's `dialect` argument is kept
  so clients that send it keep working, but it is ignored; a value that
  disagrees with the connector is logged as a warning.
- **SQL Server: a reserved word used as an unquoted name is refused.** SQL
  Server accepts its reserved keywords as identifiers only when quoted, while
  sqlglot's T-SQL grammar reads some unquoted ones as an alias, so
  `SELECT a FROM t COMMIT` parsed as one query. Such statements are now refused
  with a reason naming the word; `[COMMIT]` is still an ordinary name.
  `POLICY_VERSION` is now `6`, so semantic-cache entries made under version 5
  miss once.
- **OIDC verification no longer lets a request stall the MCP server.** A
  token whose key id was not in the cached key set made the verifier refetch
  the provider's JWKS, on every such request, with a blocking call on the
  event loop. Those refetches now happen at most once a minute, under a lock,
  and the MCP verifier runs off the event loop. When the hourly refresh fails,
  the cached keys stay in use for up to 24 hours and the refresh is retried
  a minute later instead of on every request. A reply that is not a key set,
  or a key set with no keys in it, counts as a failed refresh rather than
  replacing the working keys. With no usable keys, a request waits at most 2
  seconds for another request's fetch, then is refused rather than queued.
- **Malformed bearer tokens are rejected, not answered with a 500.** A header
  PyJWT refuses for any reason (a `kid` that is not a string, say) and a
  static token containing non-ASCII characters both raised out of the verifier.
- **Stricter checks on the identity provider and its tokens** (MCP
  authentication and `verify-oidc-token`):
  - the discovery URL and `jwks_uri` must be `https`; plain `http` is
    accepted only for localhost or a loopback address;
  - the discovery document's `issuer` must be the one its URL belongs to
    (OpenID Connect Discovery 1.0, section 4.3);
  - either document over 1 MiB is refused, and a key published for
    encryption (`use: enc`) is not used to check signatures;
  - `exp`, `iss` and `aud` must be present; a token whose `azp` names
    another client is refused, and a token with several audiences must carry
    `azp`.

  A deployment whose provider is reached over plain `http` on another host, or
  whose discovery URL is not derived from its issuer, stops verifying tokens
  until that is corrected. Microsoft Entra ID's v1.0 endpoint, and its v2.0
  `common` and `organizations` endpoints, are such cases: use the
  tenant-specific v2.0 discovery URL with the tenant's ID. See
  `docs/mcp-authentication.md`.
- **`mcp` 1.30 or later is required.** The server tells it not to apply its own
  RFC 8707 resource check (`validate_token_resource=False`), which 1.30 added:
  the verifiers check each token's audience themselves, and OIDC providers put
  the client id there, not the server's URL.
- **Verified question/SQL pairs are signed, and unsigned ones are ignored.**
  Pairs reach the prompt as user-confirmed examples, and anyone able to write
  to Qdrant could add one. `promote-feedback` now signs each pair with the
  semantic-cache signing key, and the reader uses only pairs whose signature
  verifies for the agent asking. It reads four times as many candidates as it
  needs, so a few unsigned pairs (planted, or stale ones never re-signed) do
  not take the real ones' places; more than that nearer the question leave
  the prompt with fewer verified examples, never with planted ones. **After
  upgrading, run
  `nlqueries promote-feedback` (or `export-kb`) once per agent**: pairs
  promoted before this release are unsigned and are ignored, with a warning,
  until then.
- **A cache entry is served only for the question it was made for.** Tier 1
  and tier 2 hits are found by vector, and the vector is not signed, so a
  writer to Qdrant could move a genuine signed entry onto another question's
  vector and have it served for that question. The entry's stored question,
  which is signed, is now embedded again and must itself be close to the
  question asked. This costs one embedding per cache hit.
- **Verified examples are read from Qdrant under the same API-key rule as
  everything else.** The lookup that adds user-confirmed question-and-SQL
  pairs to the prompt built its own Qdrant client, so it read anonymously from
  a remote Qdrant that every other path refused to use without a key. It now
  goes through `build_qdrant_client`; without a key it adds no examples and
  logs why once. A test fails if any other module constructs a `QdrantClient`.
- **"On this machine" means on this machine.** `is_loopback` treated a URL
  without a scheme (`qdrant:6333`, `10.0.0.5:6333`) and any host name starting
  with `127.` as local, so those reached a remote Qdrant without a key. Only
  `localhost` and loopback addresses count now; a host that cannot be read
  counts as remote. A key sent over plain `http` to another host is logged as
  a warning.
- **The semantic-cache signing key must be at least 32 bytes.** An empty or
  short key, from the environment, a mounted file or the state directory, was
  accepted, and an empty key file is what a crash during first use left behind.
  A short key now stops the MCP server at startup; elsewhere the cache is
  disabled and the reason logged. A generated key is written to a private
  temporary file and linked into place, so no other process ever reads it
  half-written, and two processes starting together end up sharing one key.
- **Cache entries are bound to their agent's schema again for almost every
  agent.** The binding named the knowledge-base file by dropping every
  character other than letters, digits, `-` and `_`, where `export-kb` replaces
  them with `_` (and keeps dots). For any id containing such a character,
  which includes every id `nlqueries connect` produces (`postgres:localhost:shop`
  became `postgreslocalhostshop.yaml`), it fingerprinted a file that does not
  exist, and the entry was bound to no schema. Those agents' cache entries miss
  once after upgrading.
- **Thumbs-down corrections reach the knowledge base only from known sources,
  and only as safe examples.** `export-kb` put every correction in the feedback
  file at the front of the agent's example queries, which the model sees for
  every question, whoever submitted it over MCP and whatever its SQL. A
  correction is now used only when it was recorded with `nlqueries feedback`
  or the Python API (`--include-anonymous` opts in to the rest, as for
  `promote-feedback`), and only when its SQL is a single read-only query that
  passes the SQL policy and reads tables in the knowledge base (a table named
  with a schema must match it where the knowledge base records one, so
  `other.orders` does not pass for `public.orders`, and one named with a
  database or project too must name the one the knowledge base was built
  from). Its question is
  kept on one line of at most 200 characters, since it is shown as a comment
  and a line break would end the comment, and its literals become
  placeholders. `export-kb` reports how many it skipped and why.
  **After upgrading, run `nlqueries export-kb` again for each agent**: the
  knowledge base's example queries are rebuilt on every run, so this drops
  corrections accepted under the old rules.
- **Promotion refuses SQL it cannot check.** `promote-feedback` treated a
  pair as valid when the knowledge base had no schema or the SQL did not
  parse. A pair is now promoted only when its SQL passes the SQL policy and is
  one read-only query that names at least one table, all of them in the
  knowledge base (compared with their schema, as for corrections), with no
  line break left inside a name or value; its question is stored on one line
  and its SQL without comments. After promoting, `promote-feedback` (and
  `export-kb`, which promotes too) checks the agent's stored pairs the same
  way and **deletes** those that fail, including any pair promoted under the
  old rules that does not pass the new ones and any stored before pairs were
  normalised, which would otherwise sit beside the normalised copy. Each
  removed pair is logged, and the count reported. **Nothing is removed when
  every stored pair fails at once**: that points at the check, not the pairs
  (a connector whose engine sqlglot cannot parse is read as Postgres, and then
  no pair parses), so they are kept and a warning says so. A knowledge base that cannot
  be read is logged as such, rather than reported only as having no schema.
- **Verified examples are stored under the agent's own name.** Their
  collection turned `.` into `_` in the agent ID, so `sales.eu` and `sales_eu`
  shared one set of verified examples. It now uses the rule every other store
  does. **An agent whose ID contains a `.` starts with an empty collection**:
  run `nlqueries promote-feedback` for it again.
- `submit_feedback` no longer tells MCP callers that `promote-feedback`
  applies a correction; that command handles thumbs-up only.

### Added

- **Cached answers and document text can be encrypted in Qdrant.** With
  `NLQ_CACHE_ENCRYPTION=true`, the semantic cache's answer and each document
  chunk's text are encrypted with AES-256-GCM before they are stored. The key
  comes from `NLQ_CACHE_ENCRYPTION_KEY` or `NLQ_CACHE_ENCRYPTION_KEY_FILE`, or is
  generated in the state directory at mode `0600`. Off by default. The setting
  governs writing only: text stored unencrypted is still read, and encrypted text
  is read whenever the key is available. Text that cannot be decrypted is a
  cache miss or a skipped chunk. Vectors, questions and SQL are stored as before.
  See `docs/qdrant-setup.md`.
- **PDF, Word and Excel documents are parsed in a child process with time and
  memory limits.** The extraction budget is checked between pages, paragraphs
  and rows, so a parser stuck inside one of them, or one allocating without
  bound, held the process that asked. Now the parse runs in a child that is
  killed at the extraction deadline (plus 15 seconds to start) and runs under
  a memory limit (`NLQ_DOCUMENT_PARSE_MEMORY_BYTES`, 2 GiB: an address-space
  limit on Linux and macOS, a Job Object on Windows) and a CPU-time limit
  (`NLQ_DOCUMENT_PARSE_CPU_SECONDS`, 180: `RLIMIT_CPU` on Linux and macOS, the
  Job Object on Windows). Running out of memory
  raises `DocumentTooComplexError`, including when the XML parser reports it as
  a parse error; the deadline or the CPU limit raises
  `DocumentExtractionTimeout`, as before. A child whose caller dies without
  stopping it stops itself at the same deadline, and what the parser starts
  ends with the parse. On Linux and macOS the child leads a session of its
  own, so a signal sent to the caller's process group (a supervisor's
  `kill -TERM -<pgid>`, a hang-up) no longer reaches it: it runs on until that
  deadline. `NLQ_DOCUMENT_PARSE_ISOLATION=false` parses in process, as before.
- **`nlqueries health` reports over-privileged logins for MySQL, SQL Server
  and Snowflake.** The identity line the PostgreSQL connector already gave is
  now printed for MySQL and MariaDB (through the `sqlalchemy` connector, from
  `SHOW GRANTS`), SQL Server (server and database role membership and the
  effective permissions from `fn_my_permissions`) and Snowflake (the grants of
  the current role and every role it inherits). It names the login and any
  write, DDL or administrative right it holds, says `least privilege` when
  there is none, and is a failure when the grants could not be read. It
  reports and does not refuse a connection. See
  [docs/database-hardening.md](docs/database-hardening.md).
- **Audit events for query execution, and a file to keep them in.** Every
  statement sent to a database through the connector an agent's query opens
  (`PermittedConnector.execute_query`) is recorded as a `query_execution` event
  on the `nlqueries.audit` logger, from the CLI, the MCP server, the library
  and cache replays alike. It carries the request id, surface (`cli`, `mcp` or
  `library`), principal when known, agent, connector type and dialect, a
  SHA-256 digest and the byte length of the statement, the SQL policy's
  version and decision, and the outcome, row count, truncation, duration and
  exception class, plus the digest and length of the statement that actually
  ran when a connector wrapper rewrote it. It never carries the question, the SQL text, the rows or a
  credential. Authorization decisions now carry the request id and surface as
  well, so a tool call's decision and the statements it ran share an id, and
  both kinds of event put their JSON (sorted keys, one line, control
  characters escaped) in the log message as well as in `extra`. Set
  `NLQ_AUDIT_LOG` to a path to have each event appended to it as one JSON
  line; the file is created at mode `0600`. A symbolic link at that path is
  not followed, and anything but a regular file is refused without blocking.
  A file with more than one hard link, a file owned by another user, or one
  group or others can still reach after narrowing to `0600` gets no events,
  with a warning on `nlqueries.audit.sink` at the first, 10th, 100th...
  event it missed and the total once it can be written again. See "Audit events" in
  `docs/configuration.md`. `SQLAlchemyConnector.dialect_name` is new: the
  backend SQLAlchemy reports for the connection (`postgresql`, `mysql`, ...),
  or `None` before `connect`; the event's dialect for that connector comes
  from it.
- **`BIGQUERY_MAX_BYTES_BILLED`: a cost cap for BigQuery answers.** Set to a
  number of bytes, it is carried on every query job that answers a question as
  `maximum_bytes_billed`, so BigQuery fails a job that would bill more before
  running it and the answer reports the error. Unset, the default, means no
  cap, as before. A value that is not a whole number greater than zero makes
  every BigQuery query fail with an error naming the variable, rather than
  run without the cap; other commands, `health` and `doctor` included, still
  run. Schema and query-history reads are not capped. See
  [docs/connectors.md](docs/connectors.md#bigquery-cost-cap).
- **SQL Server encryption setting.** The `mssql` connector passes pymssql's
  `encryption` argument on every connection, from a new `encryption` key in
  the connector's entry: `require`, `request` or `off`, defaulting to
  `request`. Any other value is refused when the connector opens. Measured
  against SQL Server 2022, which offers TLS without forcing it, only `require`
  encrypts the session; `request` and `off` leave it unencrypted, as before.
  pymssql applies the setting only from 2.4.3, so the `mssql` extra now
  requires `pymssql>=2.4.3`; with an older pymssql installed the connector
  refuses `require`, which that release would ignore. `nlqueries health`
  prints the level as a transport line for `mssql` connectors, always as a
  warning, since certificate verification is FreeTDS configuration the
  connector does not set (under `require` it accepted a self-signed
  certificate), and says when the installed pymssql ignores the setting. See
  [docs/database-hardening.md](docs/database-hardening.md#sql-server).
- **SBOMs for each release.** The Docker image is built with a BuildKit SBOM
  attestation, so each platform in a pushed `nlqueries/core` manifest carries
  an SPDX SBOM (`docker buildx imagetools inspect nlqueries/core:<version>
  --format '{{ json .SBOM }}'`). The scanner that produces it,
  `docker/buildkit-syft-scanner`, is pinned by digest. The wheel gets a
  CycloneDX SBOM: a separate `wheel-sbom` job installs it into a virtualenv created
  without pip, describes that environment with `cyclonedx-py`, and the GitHub
  release attaches `nlqueries_core-<version>-py3-none-any.whl.cdx.json`. It is
  not uploaded to PyPI. It records the resolution on linux/amd64 with CPython
  3.11 at release time; other platforms and Pythons resolve differently. PyPI
  publishing does not wait on the SBOM job, so a failure there holds back only
  the GitHub release, which can be re-run. The manual dry run exports its
  image as an OCI tarball to `/dev/null`, so it builds the image's SBOM
  attestations too without writing the image to the runner's disk. See
  "SBOMs" in `docs/dependency-locking.md`.
- **`QueryResult.executed_sql`: the statement the database actually ran.**
  `None`, the default, means the statement passed to `execute_query` ran
  unchanged, and no connector in core sets it, because none rewrites. A
  connector wrapper that rewrites the statement on its way to the database
  (adding a row filter's predicate, say) sets it to the text it sent; the
  `PermittedConnector` docstring states this beside `cache_scope`. When such a
  statement fails, the one-correction retry now shows the model both its own
  SQL and the statement the database ran, the latter framed as information
  like the database error, and still asks it to correct its own SQL, which
  goes back through the wrapper. Before, the model saw its own SQL beside an
  error about different text. When `executed_sql` is unset or the same
  statement however laid out, the correction prompt is byte-identical to
  before. A correction that repeats the statement the database ran is treated
  as returned unchanged and not run, since it would be rewritten a second time
  and its filter shown to the caller. Core never puts `executed_sql` in a
  frame, so the frame's `sql` names the statement the caller asked for. The
  model does see it, though, and a correction that keeps the wrapper's
  predicate while changing something else runs, so its text, predicate
  included, reaches the frame. A wrapper should not put there anything a
  caller must never see.
- **BigQuery: Workload Identity Federation and service account
  impersonation.** `service_account_json` now also accepts a Workload Identity
  Federation credential configuration (`"type": "external_account"`), as a path
  or as parsed JSON, so a workload on AWS, Azure, GitHub Actions or another
  identity provider can reach BigQuery without a service account key. It is
  loaded by `google.auth`; a service account key goes to the same loaders as
  before. A new optional `impersonate_service_account` (a service account
  email) wraps whichever credential the connector resolved, whether key,
  federation configuration or Application Default Credentials, in an
  impersonated credential with the BigQuery scope, so queries run as one
  narrowly granted account. An empty or non-email value is refused when the
  connector opens rather than ignored. A connector with no key and no
  impersonation is unchanged: the client still finds ADC itself. The
  `bigquery` extra now requires `google-auth>=2.20`, the first release with
  `load_credentials_from_dict`. See
  [docs/connectors.md](docs/connectors.md#bigquery-authentication).
- **JSON log lines for the embedding daemon, when asked.** Set
  `NLQ_LOG_FORMAT=json`, or pass `--log-format json` to
  `python -m nlqueries.embeddings.embed_server`, and each log line is one JSON
  object with `asctime`, `name`, `levelname`, `message`, and `exc_info` for a
  traceback. An application that runs the daemon beside its own JSON-logging
  services can then read the daemon's lines, failures included, with the same
  tools. Plain text is still the default and unchanged; an unknown value is
  plain text with a warning. `nlqueries embed-server start --foreground`
  honours the variable too, and an exception that escapes a request handler (a
  client gone mid-response) is logged in the same format rather than printed
  as a raw traceback.
- **`connection_problem()` on every connector: why a connection is not
  usable.** `test_connection()` answers yes or no, and every built-in
  connector caught the driver's error inside it and only logged it, so an
  application showing a person why a connection failed had nothing to show.
  `connection_problem()` returns `None` for a usable connection, otherwise the
  driver's error as its type and first line (`OperationalError: ... port 5432
  failed: Connection refused`); later lines, where drivers quote what they were
  sent, are left out. Each built-in connector implements it with its own test,
  and `test_connection()` is now `connection_problem() is None`. A third-party
  connector that implements only `test_connection()` keeps working: the
  default says only that the test failed.
- **`application_name` for the PostgreSQL connector.** Optional; it labels
  every session in `pg_stat_activity`. A caller that cancels a running
  statement by finding it there needs it: while a streamed read runs, the
  session's text is the cursor's `FETCH`, not the statement.
- **String literals are checked against the stored values before a statement
  runs (`NLQ_LITERAL_GROUNDING`, on by default).** A model writes `status = 'legal'`
  where the column holds `'Legal'`, or copies `' = '` from a hint where it
  holds `'='`. The statement is valid, runs, and returns no rows, a wrong
  answer with nothing to say so.
  - **The check.** Each literal compared to a text column with `=`, `IN` or a
    wildcard-free `LIKE` is looked up first.
  - **Exactly one stored value matches**, ignoring case and surrounding spaces
    or, failing that, also punctuation at either end (a title stored with its
    question mark and quoted without it, or the reverse): the literal is
    replaced with it, and the substitution is recorded in the answer's
    provenance (`literals_grounded`).
    A literal shorter than three characters without its end punctuation gets
    no punctuation pass.
  - **None or several match:** the statement is left alone, and an LLM repair,
    if one runs, is told the nearby stored values.
  - **Scope.** Keys and columns the knowledge base does not know are skipped,
    and so are columns whose names mark them as personal data, the ones
    `export-kb` will not sample (email, phone, address, tokens, secrets and
    the like): a lookup would copy other people's values into the repair
    prompt and provenance.
  - **Cost.** Where the knowledge base holds a column's complete value list,
    the list answers: a literal in it needs no query, and one that is not
    gets one exact-match query, in case it was stored since the export.
    The list was read with no row restriction, so under a per-request row
    filter the lookups go through the filter instead. Otherwise a literal
    that matches nothing costs up to four queries, most of them scans, so a
    table the knowledge base records as larger than
    `NLQ_LITERAL_GROUNDING_MAX_ROWS` (1,000,000 by default; `0` for no cap)
    is not queried, and only a stored list grounds its columns. A table of
    unknown size is queried.
    The check is read-only and bounded per pass: 8 literals, 1 s a lookup,
    3 s in all. No lookup is waited for past the 3 s: one still running then
    is abandoned. A statement gets at most two passes, one before a repair
    and one on the repaired statement, so up to 16 literals and 6 s.
    Outcomes are cached per database for 15 minutes, and any error skips the
    check, as does a statement whose literals would not come back unchanged
    from its re-render. On SQLite the case-insensitive comparison is written
    `TRIM(col) = TRIM('x') COLLATE NOCASE` and the search for values
    containing a literal `col LIKE '%x%'`, which stay fast when several
    lookups run at once, where `LOWER()` on the column did not.
  - **When it runs.** Whatever the execution policy, for any agent with a
    registered connector. Generate-only forbids running the statement, not
    reading a column's values, and the statement is never run by the check.
    An agent with no registered connector opens nothing.
  - **The agent's connector, as every query reads it.** The lookups go through
    `open_connector_for_agent` with read-only permission, so whatever wraps that
    function for a request, such as a row filter, applies to them too; their
    threads run in the caller's context so request-bound state reaches them.
    Outcomes are shared only where no such wrapper restricts rows, or within
    the `cache_scope` a restricting wrapper declares.
  - **A valid statement whose literal matches nothing gets one repair call
    (`NLQ_LITERAL_GROUNDING_REPAIR`, on by default).** Such a statement used to
    run and return no rows, the notes discarded. When a note offers nearby
    values, one LLM call with the question, the statement and the notes asks
    for the same statement with only the literal corrected. The answer is used
    only if it validates and differs in nothing but the values of the literals
    grounding flagged, and is grounded again without a second call. Recorded
    in provenance as `literal_repair`, with the reason
    `rejected: non-literal change` when the answer changed more than literals,
    and `rejected: changed a literal grounding did not flag` when it changed
    any other literal.

- **Bounds on document and wiki ingestion.** A PDF larger than
  `NLQ_MAX_PDF_BYTES` (100 MiB) is refused before it is parsed, and one with
  more than `NLQ_MAX_PDF_PAGES` (2,000) pages before any page is extracted;
  both raise `DocumentTooComplexError`. The expansion-ratio limit for Word and
  Excel files (`NLQ_MAX_DOCUMENT_EXPANSION_RATIO`) now also applies to each
  member that expands to 4 MiB or more, so one highly compressed member is
  not averaged away by a large incompressible one. A smaller member is cheap
  to inflate whatever its ratio, so it is held only to the whole-archive
  limits. Word extraction checks its
  time budget every 1,000 paragraphs, not only between sections.
  `doc-sync-confluence` refuses a space key that is not letters, digits and
  underscores, or for a personal space `~` and a username or account id that
  may also contain `.`, `-`, `@` and `:` (`~john.smith`). It logs a warning
  for an `http://` base URL to a non-loopback host, and refuses a space over
  `NLQ_MAX_CONFLUENCE_PAGES` (5,000) pages or `NLQ_MAX_CONFLUENCE_BYTES`
  (200 MiB) of page body. `doc-sync-notion` refuses a page of more than
  `NLQ_MAX_NOTION_BLOCKS` (10,000) blocks or `NLQ_MAX_NOTION_BYTES` (50 MiB)
  of block text. A refused sync stores nothing and leaves the earlier copy in
  place. Refusal messages give sizes in bytes, KiB or MiB by magnitude, so a
  small limit never reads as zero; where a size and its limit would print
  alike ("100 MiB, over the 100 MiB limit"), both are given in exact bytes.

### Changed

- **A URL the package opens must be `http` or `https`.** `urllib` opens
  `file:` and `ftp:` URLs as readily as web ones, so a `QDRANT_URL` of
  `file:///etc/passwd` would have been read. `nlqueries health`, the MCP
  `health` tool, KB stats and the embedding client now refuse any other scheme
  before opening it. Failures that used to pass without a word are reported: a
  stored keychain password that could not be put into a connector's URL (the
  connection then failed as a bad password), a password left in the connectors
  file when `update-password` moved it to the keychain, KB stats that could not
  be counted, and an optional connector that fails to load for any reason but a
  missing driver. The SQL Server query-history limit is taken as an integer, as
  the other engines' are, and a BigQuery dataset location is used in the
  history query only in the shape of one. Ruff's security and print rules are
  now part of the lint.
- **SQL policy version 4: a generated statement may contain only what is listed
  for its dialect.** The policy used to refuse a function sqlglot does not model
  unless it was allowlisted, so anything sqlglot did model was allowed. It now
  lists, per dialect, the sqlglot classes a read query may contain
  (`nlqueries/sql_constructs.py`) and the unmodelled functions it may call
  (`ALLOWED_ANONYMOUS`), and refuses the rest, including session variables and
  parameters (`@@version`, `@x`, `$1`, `?`) and `CURRENT_USER`. An unmodelled
  aggregate is now checked by name too. A statement whose rendered form the
  policy reads differently when parsed again is refused. MySQL's `now()` and
  SQLite's `julianday()`, refused as unrecognised before, are allowed. The
  queries the lists must keep allowing are in `tests/sql_corpus.py`;
  `nlqueries.sql_policy_report` lists what the policy would refuse among a
  deployment's own queries. See `docs/database-hardening.md`.
- **SQL policy version 5: a qualified function call is judged by its whole
  name.** `some_schema.age(a, b)` was judged as `age`, which PostgreSQL's list
  allows, though the function that runs is whichever `some_schema` holds. It is
  now judged as `some_schema.age`, and refused unless listed as written in
  `ALLOWED_ANONYMOUS`. The same holds for a function sqlglot models: a qualified
  call is no longer allowed by its class, so `FROM
  some_schema.generate_series(1, 3)` is refused unless
  `some_schema.generate_series` is listed. A qualifier on what parses as an
  operator rather than a call, as `FROM public.mod(5, 2)` does, is refused. An
  unmodelled function whose name cannot be read is refused. When a statement is refused because its rendered
  form is, the functions and constructs only the rendered form uses are
  recorded on the decision, so `nlqueries.sql_policy_report` counts them.
- **The default expanded-size cap for `.xlsx` and `.docx` is 128 MiB** (was
  400 MiB). Parsing a `.docx` costs several times its expanded size: one of
  250 MiB of ordinary paragraphs peaked at about 2 GiB. At 128 MiB an ordinary
  document fits within the parsing child's memory limit, and one over the cap
  is refused from the zip directory before anything is parsed. A sheet of
  200,000 rows by 10 columns (about 109 MiB) still passes. The same setting is
  the largest `.txt` or `.md` file ingested, so that is 128 MiB too.
  `NLQ_MAX_DOCUMENT_EXPANDED_BYTES` raises it, together with
  `NLQ_DOCUMENT_PARSE_MEMORY_BYTES`.
- **The container image is built on the current `python:3.11-slim-bookworm`**
  (Python 3.11.17), pinned by digest as before. It picks up Debian's security
  updates to `perl-base` and `libpcre2-8-0`, which the image scan reported as
  high and critical findings with fixes available.
- **The generic `sqlalchemy` connector refuses SQLite and DuckDB URLs.** A
  `sqlite://` or `duckdb://` URL is refused when the connector opens, with a
  message naming the dedicated `sqlite` or `duckdb` connector, which opens the
  file read-only and sandboxed. Through the generic connector the same file
  was opened read-write, with `ATTACH` and DuckDB's file functions available
  to a query. A connector registered that way must be registered again with
  its dedicated type.
- **MySQL and MariaDB run each query in a read-only transaction, and their
  URLs lose two settings.** Through the generic connector, each query now
  starts with `START TRANSACTION READ ONLY`, so the server refuses a write to
  any table, MyISAM and MEMORY tables included, where the rollback alone left
  such a write standing. DDL still commits implicitly. `LOAD DATA LOCAL
  INFILE` is disabled (`local_infile=0` for `mysqldb`, `pymysql` and
  `mariadbconnector`, or `allow_local_infile=False` for `mysqlconnector`; a
  URL naming another MySQL driver, such as `mysql+pyodbc`, is refused, since
  the setting cannot be applied to it) and any URL parameter enabling it is
  dropped; the
  multi-statements bit is cleared from a `client_flag` or `client_flags`
  parameter, and a non-integer flag is refused. `init_command` and the other
  parameters are unchanged. Measured against MySQL 8.0 and 8.4 and MariaDB
  10.11; see
  [docs/database-hardening.md](docs/database-hardening.md#what-each-connector-enforces).
- **The SQL policy refuses file reads, and its version is now 3.** The
  file-reading functions sqlglot models (`read_csv`, `read_parquet`, and
  Snowflake's `DIRECTORY` and `TO_FILE`) are refused wherever they appear, on
  every dialect; `read_csv` was previously allowed because sqlglot recognises
  it. So is a string literal used as a table name, such as DuckDB's
  `FROM 'orders.csv'`, told apart from a quoted identifier (`FROM "orders"`) by
  the token it was parsed from. If the installed sqlglot records no token
  positions on table names, a statement with both a string and a table name is
  refused, since the rule cannot tell them apart. Cached answers signed under
  policy version 2 no longer verify and are regenerated.
- **Query history is limited to the connector's own database or dataset.** SQL
  Server history keeps only plans compiled in the connected database (the
  plan's `dbid` attribute, which unlike `dm_exec_sql_text.dbid` is set for ad
  hoc statements), so a login with `VIEW SERVER STATE` no longer mines the
  server's other databases. SQL Server history also comes back at all now: its
  rows were read after the connection had been returned to the pool, and
  against a real server the result was always empty. BigQuery history, when the connector has a
  `dataset_id`, keeps only jobs whose `referenced_tables` include a table in
  that dataset; the dataset and project are bound as query parameters.
- **Redshift queries on one connector run one at a time.** The connector holds
  one connection, and two overlapping queries could send a second
  `SET TRANSACTION READ ONLY` inside the first one's transaction, where
  Redshift refuses it, or have the first one's rollback end the second's
  transaction. The read-only, statement and rollback sequence is now held
  under a lock, with the same bounded wait as Snowflake's: the statement
  timeout plus 30 seconds, or five minutes with no statement timeout, after
  which the query fails with an error saying the connector was busy.
- **DuckDB spills to a private directory.** The connector sets `temp_directory`
  to a directory it creates once per process under the system temporary
  directory, with mode `0700`, removed when the process exits. DuckDB's default
  is `<database>.tmp` beside the database file. External access is now switched
  off and the configuration locked by statement right after connecting; a
  second connector on a file already open in the process shares that locked
  instance and is accepted once the sandbox is confirmed to hold.
- **Snowflake catalog queries quote the database name and bind their values.**
  The configured database is written as a quoted identifier, upper-cased first
  when it is a plain unquoted name so it still names the database Snowflake
  opened (`analytics` becomes `"ANALYTICS"`, a name given in double quotes is
  kept exactly), and the schema names and the database name in the
  `POLICY_REFERENCES` lookup are bound rather than interpolated. A database
  configured in lower case now matches its `POLICY_REFERENCES` rows, which it
  did not before.
- **`analyze_query` checks the statement with the SQL policy before running
  `EXPLAIN ANALYZE` on it.** `EXPLAIN ANALYZE` executes the statement it wraps;
  a statement the policy refuses is no longer explained, and the analysis
  comes back with no plan.
- **`nlqueries query` no longer executes SQL itself.** The orchestrator already
  executes the generated SQL when execution is asked for. The command had a
  fallback that ran the statement again on a connector it opened directly
  whenever the orchestrator returned no result, which included the
  orchestrator declining to run it. When the orchestrator returns no rows, the
  command now shows none.
- **Reference data renders on one line in prompts.** Table and column
  descriptions, glossary terms and capsule intents have their whitespace
  collapsed and are capped at 300 characters, so a description holding
  `\n## Business Rules` stays on its table's line instead of opening a
  section. Business rules and glossary definitions are collapsed the same way
  but capped at 1,000 characters, since a rule cut mid-sentence can change its
  meaning; a rule or glossary definition cut at that cap is logged once per
  process as a warning naming its start, so the operator can shorten or split it. Sample values are single-quoted with embedded quotes doubled
  (`'O''Brien'`); a sample over 80 characters is left out of the list rather
  than cut, and a complete `values:` list that loses one is shown as
  `samples:`, since it is no longer complete. The SQL prompt's preamble
  now says that descriptions, values, glossary entries, examples and excerpts
  are reference data and never change its instructions, and the verified
  example label reads `-- verified example`, without "user-confirmed correct".
- **Document excerpts and conversation turns are delimited.** The document
  prompt and hybrid synthesis put each excerpt in an
  `<excerpt source="...">` block with its source label, and follow-up
  resolution puts each earlier turn in a `<turn source="User|Assistant">`
  block. Inside a block, only the markers themselves are neutralised: any
  `<excerpt`, `</excerpt`, `<turn` or `</turn`, in any letter case, has its
  `<` written as `&lt;`. Everything else, `&`, `<` and `>` included, reaches
  the model as written, so "R&D" and "Smith & Sons" are not turned into
  entities that the model then copies into answers or rewritten questions.
  The source label, which the model cites, is put on one line (capped at 300
  characters like a description) with its markers neutralised, and only `"`
  is escaped (as `&quot;`), so `Director's Report.pdf` and
  `Profit & Loss.xlsx` are cited as written. The SQL data in the hybrid
  prompt, cell values and column names included, has its markers neutralised
  the same way, so a database row cannot pass itself off as a cited document. A hybrid answer's
  document-only passthrough text is unchanged.
- **`NLQ_SEND_RESULT_ROWS_TO_LLM` decides whether hybrid synthesis sees rows.**
  On (the default), up to 20 rows are sent as before, except that columns
  whose names mark them as personal data are now always left out and named as
  withheld; when that is every column, the prompt still gets the row count and
  the fact of any failure. Off, the synthesis prompt gets the column names and
  the row count only. While values are withheld (off, or every column
  withheld), a failure is stated without the database's error text, which can
  quote a value. When no query ran, the generated statement is sent whatever
  the setting says, since it is not data. The synthesis prompt also tells the
  model to state no figure it has not been shown and, where values are
  withheld, to refer the reader to the result table. The result table returned
  to the reader is the same either way. See
  "What leaves this machine" in `docs/configuration.md`.
- **Document and hybrid answers are not written to the semantic cache.** A
  cached document answer, or a hybrid answer synthesised from the same
  excerpts, would be replayed after its source changed, for the whole TTL.
  Entries written before this change still expire with the TTL.
- **`import-dbt` matches models by schema and name.** A knowledge-base table
  that records its schema takes the docs of the dbt model built into that
  schema; otherwise the bare name still matches, but only when one model has
  it and no other table took that model by schema. Both comparisons ignore
  letter case, so Snowflake's `ORDERS` matches dbt's `orders`, and column
  descriptions are matched the same way (`ORDER_ID` takes `order_id`'s).
  dbt model and column descriptions are cut to 2,000 characters wherever dbt
  docs are read (`parse_dbt_docs` and `parse_dbt_models`), so the cap applies
  to `import-dbt` and to docs passed to `generate_knowledge_base` alike. A cut
  description ends with `...`, within the 2,000, and each model with a cut
  description is logged once at WARNING.
- **Knowledge bases, query capsules and `connectors.yaml` are written privately
  and atomically.** `export-kb` (including the fallback for a connector type
  with no registered class), `import-dbt`, `process-history` and `connect`
  now write through `nlqueries.state_files.write_private`: the content goes to
  a temporary file created at mode `0600` in the same directory, is flushed to
  disk, and is renamed over the target. The file is never readable by other
  accounts, not even between the write and a `chmod`, and an interrupted run
  leaves the previous file in place instead of a truncated one. A symlink at
  the target is replaced by the new file rather than written through. A
  directory the write needs is created at `0700`; an existing one, such as the
  directory of an `--output` path, keeps its permissions. On Windows the mode
  bits are not enforced, as before.
- **Knowledge-base YAML is written with `yaml.safe_dump`.** Every reader loads
  it with `safe_load`, and the writer now accepts only what that reader can
  load: a value that would have been written as a Python object tag is refused
  when the file is written.
- **`nlqueries/core:latest` moves only on a final release.** A tag of the form
  `vX.Y.Z` pushes the image as `X.Y.Z` and `latest`; a pre-release tag such as
  `v1.0.0-rc1` pushes `1.0.0-rc1` alone, so `docker pull nlqueries/core` no
  longer lands on a release candidate.
- **`docker-compose.yml` pins the image by version.** It now uses
  `nlqueries/core:0.3.0` instead of `:latest`, so `docker compose up` starts the
  release the file was written for and a later pull does not move it. Each
  final release bumps the tag; to run another release, change the `image:`
  line.
- **The image no longer contains `curl`.** Nothing in it used `curl`: the
  healthcheck is a Python TCP connect. The Dockerfile now installs no system
  packages over the base image. A script that ran `curl` inside the container
  needs another HTTP client, such as `python -c "import urllib.request ..."`.
- **`.env` is read from the working directory only.** `nlqueries` loads
  `./.env` when that file exists and no longer searches parent directories
  for one, so a `.env` belonging to an enclosing directory no longer sets this
  project's variables. Variables already set in the environment still win. If
  you relied on a `.env` higher up, copy it, or the lines you need, into the
  directory you run `nlqueries` from.
- **The embedding daemon checks a request's headers and size.** A POST to
  `/embed` or `/embed-batch` is refused with 415 unless its `Content-Type` is
  `application/json`, with 421 unless its `Host` is `127.0.0.1`, `localhost`
  or `[::1]` (any port), and with 413 when it carries more than 512 texts or
  more than 2,000,000 characters in all; a body over 10 MiB is refused with
  400, as before. An error raised by the encoder is now logged and answered
  with 500; a `KeyError` or `TypeError` from it used to be answered as a 400
  and not logged. The client now sends its JSON as UTF-8 rather than
  escaping non-ASCII characters, so non-English text costs its UTF-8 size
  instead of six bytes a character. `embed_batch` splits a batch into
  requests within all three limits, measuring each text's size as sent, so
  the built-in client and every caller that embeds a whole collection through
  it, a wide schema's long table texts included, are unaffected. A single
  text over the character or body limit is sent on its own and is the only
  request refused. If any part fails, or comes back with a different number
  of vectors than texts sent, the whole batch falls back as one request's
  failure did before, so no vector is paired with the wrong text. A text
  that cannot be encoded as UTF-8, such as one holding a lone surrogate, is
  never sent: it falls back the same way, or raises under
  `EMBED_SERVER_REQUIRED`. `GET /healthz` is unchanged.
- **`embed-server start` keeps the working directory off the daemon's import
  path.** The background daemon is now started as
  `python -P -m nlqueries.embeddings.embed_server`, so it imports the
  installed package rather than any module of the same name in the directory
  `start` was run from. `--foreground` runs in the CLI's own process, as
  before. `start` now fails at once if the port is already in use, and then
  watches the daemon for two seconds and fails, pointing at `--foreground`,
  if it exits in that time, where it used to report success for a daemon
  that had already died.
- **Query capsules carry no comments and no literal in any string form.**
  `process-history` already replaced plain `'...'` and number literals with
  typed placeholders. The parameterizer now also replaces `N'...'`, `X'...'`,
  `B'...'`, byte, raw and `U&'...'` strings, typed like a plain string
  (`DATE`, `TIMESTAMP`, the schema's type, else `VARCHAR`), and writes
  templates without comments. A representative statement that cannot be
  parsed is dropped instead of being kept as its raw text, so
  `parameterize_cluster` now returns `None` for it and `parameterize_clusters`
  leaves it out. In the query filter, normalization drops comments too, so
  statements that differ only in a comment are counted as one query, and the
  fingerprint treats every string form as a literal, so `N'...'` values no
  longer split a cluster.
- **More column names count as personal data.** `is_pii_column`, the rule
  behind `export-kb`'s refusal to sample a column and literal grounding's
  refusal to look one up, now also matches `pwd`, `credential`, `api_key`,
  `access_key`, `private_key`, `signing_key`, `encryption_key`, `ssh_key`,
  `master_key`, `license_key` / `licence_key`, `iban`, `account_number`,
  `national_id`, `passport`, `tax_id`, `first_name`, `last_name`, `full_name`,
  `given_name`, `family_name`, `session` and `cookie` anywhere in the name,
  plus `auth` anywhere except where it goes on to `or` (`oauth_code`,
  `region_auth`, not `author`, `authority` or `co_author`) and `authorization`
  / `authorisation` wherever they appear, `otp` at the start of a word
  (`user_otp`, not `footprint`), `totp` and `hotp` at the start of a word with
  no letter after (`totp_code`, `hotp_counter`, not `hotpot`), `account_no`
  when no letter follows (not `account_notes`), and `pin` as a whole word
  (`card_pin`, not `shipping`). Bare `key`, `name` and `account`, and a `_key`
  suffix, are not matched, so `keyword`, `product_name`, `account_type` and
  warehouse join keys such as `customer_key` and `date_key` keep their values;
  a key that holds a secret is named on its own, as above. The `export-kb`
  path for a connector with no registered connector class, which writes whole
  `sample_rows`, now leaves these columns out of each row as well; it used to
  copy every column.
- **A release waits on the full CI, and CI audits the shipped lock.**
  `release.yml` used to run its own copy of lint, type-check and tests before
  publishing. It now calls `ci.yml` as a reusable workflow, so building and
  publishing also wait on both pip-audit scans, the gitleaks secret scan and
  the import-boundary check. A new `lock-audit` job in `ci.yml` runs
  `pip-audit -r requirements/core.lock --require-hashes` against the pins the
  image installs. The `id-token: write` permission moves from the whole
  release workflow to the `publish-pypi` job, and `publish-docker` takes its
  Docker Hub credentials from the `dockerhub` environment. `release.yml` can
  also be run by hand (`gh workflow run release.yml --ref <branch>`) as a dry
  run: CI, the package build and a multi-platform image build with nothing
  pushed or published.
- **The image's dependency lock moves six packages to current fixed
  releases.** `requirements/core.lock`, which the Docker image installs from,
  now pins PyJWT 2.15.0 (was 2.13.0), urllib3 2.8.0 (was 2.7.0), aiohttp
  3.14.4 (was 3.14.3), mcp 1.30.0 (was 1.29.1), multidict 6.9.1 (was 6.7.1)
  and snowflake-connector-python 4.7.3 (was 4.7.2). yarl moves from 1.24.5 to
  1.25.1 with them, because aiohttp 3.14.4 requires it. Every other pin is
  unchanged, and so are the `pyproject.toml` floors, so a `pip install` of the
  package resolves as before.
- **certifi moves from 2026.7.22 to 2026.10.19 in the image's lock.**
- **OpenTelemetry moves from 1.44.0 to 1.45.1 in the image's lock**
  (`opentelemetry-api` 1.44.0 has CVE-2026-114603). 1.45.1 adds two packages
  of its own, `opentelemetry-exporter-otlp-common` and
  `opentelemetry-exporter-http-transport`, both 0.66b1.
- **A plain-http LLM gateway logs a warning.** When `ANTHROPIC_BASE_URL` or
  an `api_base` override (for the Anthropic or the LiteLLM client, including
  one passed in `extra`) uses `http://` to a host that is not `localhost` or a
  loopback IP address, the client logs a warning naming the setting and the
  endpoint, once per endpoint per process. The request is not refused, since
  in-cluster gateways commonly use http. Only the scheme, host and port are
  logged.
- **MCP server: binding, transport checks and input limits.**
  - A call that arrives as an HTTP request with no token is anonymous even on
    a server set up for stdio (the module's `mcp` object mounted in an app), so
    it is no longer treated as the process owner, and the local profile refuses
    it. stdio calls are unchanged.
  - The wildcard-bind check recognises every spelling of an unspecified address
    (`0`, `0.0`, `0x0`, `::0`, `0:0:0:0:0:0:0:0`, `[::]`, `::ffff:0.0.0.0`, ...),
    not only `0.0.0.0`, `::` and an empty host.
  - On a bind other than loopback, the server accepts in the `Host` header
    only `localhost`, `127.0.0.1` and `[::1]` (with any port or none) and the
    host of `NLQ_MCP_RESOURCE_URL` (with no port or its port; on any port if
    the URL names a non-default one), otherwise `421`. In `Origin` it accepts
    only the resource URL's origin, in the same spellings as its host, and
    `http://` on the same loopback names
    (with any port or none), otherwise `403`. Set `NLQ_MCP_RESOURCE_URL` to the
    URL clients use. The compose file keeps working at `localhost:8080` and
    `127.0.0.1:8080`.
  - SSE request bodies over 4 MiB get `413`, the limit the SDK already applies
    to streamable HTTP.
  - New `NLQ_MCP_MAX_QUESTION_CHARS` (default 4000) and `NLQ_MCP_MAX_SQL_CHARS`
    (default 8000): a tool given a longer `question`, `generated_sql` or
    `corrected_sql` returns a short refusal and does nothing.
  - `health` lists agent IDs only to a caller also granted `agents:list`, and
    gives the count to anyone else. "Agent not found" replies from `query` and
    `get_agent_schema` no longer list the available agents.
  - Requires `mcp>=1.29`.
- **The SQL prompt matches a quoted value as a value.** Read as exact, a
  value quoted with a stray space became a LIKE pattern with the space in it,
  and a "contains" became a case-sensitive INSTR, so the rows it named
  dropped out. The prompts now say the spaces around a quoted value are not
  part of it, that "contains" stays LIKE '%value%' rather than INSTR, with
  LOWER() on both sides where the database's LIKE compares case, and
  that definitions are still applied exactly.
- **The SQL prompt does not sum running totals.** A column that already holds
  a running total, such as a standings table's points or wins, an account
  balance or a cumulative_* figure, summed across rows gives a figure many
  times too large. Where a column's name says it holds one, the prompts name
  it and say an entity's total is the value on its latest row, not the SUM of
  its rows (MAX only where the total never falls), while a total across
  entities still adds up each one's latest value. A per-row amount whose
  name only looks like one (running_time, balance_due) is not listed, nor is
  a date, text or flag column (a contract's valid_to_date). Where
  the total restarts each period (year to date, a season), a span of periods
  adds up each period's last value.
- **The SQL prompt keeps coded values as stored.** A statement turned a code
  such as 'R' into a word in its output, or cast a code such as '65+' to the
  number 65, so neither matched the value stored. The prompts now say codes
  stay exactly as stored in SELECT and WHERE, that a coded column is not
  cast, and that a mapping explaining a code says which code to filter on,
  not that the output should be relabelled; relabelling only when the
  question asks for the relabelled output itself, as a "yes or no?"
  question does. A column the knowledge base shows holding numbers with a
  decoration is not treated as coded, and a number followed by a bare '+'
  or '-' ('65+') is no longer read as one: it is a code.
- **The SQL prompt adds a key column only to a per-entity figure that names
  nothing else.** An id added to an answer crowded out the columns the
  question asked for, and a url came back in place of a name. The prompts now
  say to add an entity's key only when the question asks for a figure for
  each entity and names no column for it (a name, title, label or
  description; "in each city" names the city column), and then
  that key once (every column of a two-column key) and no other id; never an
  id beside a name the question asked for, and never a url or a code in place
  of one. The rule against adding an id for context names this as its one
  exception.
- **The SQL prompt keeps a key column where the question puts it.** The
  prompts already said to return columns in the order the question names
  them, but a statement still moved an id the question named later to the
  front, out of the order asked for. They now say a key or id column stays
  where the question names it, never moved to the front.
- **The SQL prompt says to total before a join that repeats rows.** A sum or
  count over one table, joined to another with several rows per key (a
  customer's invoices joined to the same customer's shipments), counts each
  row once per match on the other side and still looks plausible. The prompts
  now say to total or count those rows per join key in a subquery first, then
  join the result.
- **The SQL prompt joins with an inner join unless the question's wording
  says otherwise.** An outer join where none was asked for adds rows with a
  count of 0, and an inner join where the question is about "at most two" or
  "with no orders" drops exactly the rows it asks about. The inner join is the
  stated rule, a bare "for each X, how many Y" included; LEFT JOIN is the
  exception, only for the listed wording ("including those with none", "zero",
  "at most", "fewer than", "no", "without"), a question about the X that have
  no Y ("never ordered", "missing a manager"), or every member of a dimension
  table, which a bare "for each X" does not ask for; then a row with no match
  counts as 0 where a count is asked for. If unsure, inner.
- **The SQL prompt asks for the name, not the id, of the thing asked about.**
  "Which customer" answered with customer ids is right and of no use to the
  person asking. The prompts now say to return the thing's name or title
  column instead of the id, or a description where it has neither, unless
  the question asks for the id.
- **The SQL prompt treats a given definition as exact.** A user who spells out
  how a segment, a rate or a period is worked out, in the question or in the
  knowledge base's business rules or glossary, wants that calculation, but the
  model sometimes applied the usual meaning of the term. The prompts now say
  such a definition is exact: every condition and boundary as written. The
  rule names the SQL prompt's own sections (Business Rules, Glossary), so the
  model can tell which text it means.
- **The SQL prompt cleans numbers stored as text only where the samples show
  a decoration.** On SQLite a text column holding '$1,200' or '1,200' casts,
  without an error, to 0 or to 1, so rows drop out of a filter or a total
  unnoticed; other databases reject the cast. Where a text column's sample
  values are numbers with one fixed decoration (a currency sign, a percent
  sign, a unit, a thousands separator), the prompts name the column and say to
  strip it inside the cast, keeping the decimal mark, but only when the
  question orders, compares or adds it. Any text column holding numbers is
  ordered, compared or added by its numeric cast but returned as stored, with
  anything besides digits and a decimal mark in the values shown removed inside
  the cast, and not cast at all when the question only asks for the value;
  dates written as free text are ordered as stored rather than parsed with
  CASE ladders.
- **The SQL prompt prefers joins and window functions to per-row subqueries.**
  Some databases run a subquery that refers to the outer row once for every
  row, and on a table of hundreds of thousands of rows a correct statement of
  that shape did not finish in two minutes. The prompts now say to prefer
  joins, GROUP BY and window functions.
- **A column the hint names is the one the SQL uses.** Given a hint such as
  "the cheapest refers to MIN(price)", the model sometimes ordered by another
  column holding the same measure in another form, a number of milliseconds
  for a time or a duration, and so answered a different question from the one
  the hint defines. The answer rules now say to use the column the hint names
  for what it names.

- **The compact schema shows each column's description.** The default
  schema format (M-Schema) rendered a column's name, type, keys and values
  but never the description the knowledge base holds for it, so an
  abbreviated or repeated name (`CRE`, `TG`, a `position` in two tables) was
  all the model had to choose by. A description now follows the column's
  keys, on one line and cut at 160 characters, and one that only repeats the
  column's name (`points` for `points`) is left out. The verbose format
  already showed them.

- **`export-kb` stores column values, so the prompt shows how they are
  spelled.** Until now it stored none. With `--include-samples`, which is on
  by default, it now stores:
  - **every value** of a text column with at most 20 distinct values (200
    characters in all), marked `values_complete: true` and shown in the prompt
    as `values: [...]`;
  - **up to `--sample-rows` samples** (3 by default) of any other column,
    shown as `samples: [...]`, unless they average over 80 characters.

  Keys and personal-data columns get none. Export now runs one sample query
  per table and one `SELECT DISTINCT` per text column, each bounded to 10 s.
  The `DISTINCT` reads every row of a column with few values, and on a
  warehouse billed by bytes read, BigQuery for one, a `LIMIT` does not reduce
  the charge, so a table the schema records as having more than
  `--values-max-rows` rows (10,000,000 by default; `0` for no cap) gets no
  values and none of these queries. A table of unknown size is collected. A
  knowledge base has to be exported again for its prompt to show the values.

  **Both row caps go by the size the connector reports.** The generic
  SQLAlchemy connector (MySQL, MariaDB, Oracle and the rest) reports none, so
  on those engines neither `--values-max-rows` nor
  `NLQ_LITERAL_GROUNDING_MAX_ROWS` skips a table; a Postgres table never
  analysed and a Redshift role without access to `SVV_TABLE_INFO` report
  none either. `export-kb` says how many tables report no row count.

  **An existing deployment starts storing values on its next `export-kb`.**
  `--include-samples` was on by default but stored nothing, so no export
  until now copied database values into a knowledge base. From this release
  the next export does, and from there they go to the model provider in every
  SQL prompt whose schema includes the column, and out through the MCP schema
  tool. The personal-data guard goes by column name only (passwords, secrets,
  tokens, hashes and salts, SSNs, card numbers and CVVs, dates of birth,
  email, phone, address), so a column whose values are sensitive under any
  other name, such as a diagnosis, an ethnicity or a disciplinary outcome, is
  stored like any other. Export with `--no-include-samples` to store no
  values.
- **SQL prompt: benchmark-measured idioms.** Covers COUNT over joined rows,
  ORDER BY LIMIT 1 for superlatives, projecting requested aggregates in the
  requested column order, hint literals as authoritative, and YES/NO literals
  where the question offers the choice. These come from re-running a
  text-to-SQL benchmark's misses, where the statement ran and was defensible
  but answered in a different shape than expected. The prompts now say:
  - a count, total, average or rank the question names is a column to select;
  - columns come in the order the question or its hint lists them;
  - a person's full name is first and last name;
  - count the rows the joins produce, with `COUNT(DISTINCT)` only when the
    question or hint says distinct, unique or different;
  - "the highest / lowest / latest ..." is `ORDER BY ... LIMIT 1`, unless every
    tie is asked for;
  - a quoted value in a hint is used exactly as written;
  - a question that offers the choice ("yes or no?", "true or false?") is
    answered with that literal text. One that only asks whether something
    holds returns the value the data stores.

  A rule against adding filters nobody asked for (such as `IS NOT NULL`) was
  measured and left out: it lost about twice as many correct answers as it
  gained.

  The counting rule is the one that trades: the benchmark's reference answers
  are themselves inconsistent on entities counted through a child table, some
  using `DISTINCT` and some not. Measured, keying `DISTINCT` to the wording
  fixed about twice as many misses as it lost, and no wording-based exception for
  that shape did better.
- **SQL prompt: project only the requested columns.** In a text-to-SQL
  benchmark run, 21% of the misses were correct answers returned with extra
  columns. A set-of-rows comparison scores
  those as wrong. Examples were the column it ranked by, a name added for
  context, every numbered variant of a column, or a count split into a
  labelled GROUP BY. Both SQL system prompts now tell the model to select only
  what the question asks for, and to count a field the question names
  ("include X", "along with Y") as asked for. That covers the one `run_query`
  caches and the one `generate_sql` and the repair step use. The cached block
  changes once, so the first question for each knowledge base after upgrading
  writes the cache again.
- **SQL policy version 2: the function allowlist answers to every spelling of
  a dialect.** `ALLOWED_ANONYMOUS` is keyed by sqlglot's names (`tsql`, not
  `mssql`; `mysql` gains an entry) and read through the same aliases the parser
  uses, so `mssql` and `tsql` share an entry, as do `postgresql` and `postgres`.
  The one decision that changes: `postgresql` now gets Postgres's allowlist
  (`age`, `jsonb_agg`, `regexp_matches`, `every`, `date_part`) where it used to
  get none. Semantic-cache entries are signed with the policy version, so
  those made under version 1 stop verifying: each such question misses once
  and is cached again.
- **`--dialect` accepts every engine `nlqueries connect` registers, and
  defaults to the connector's.** On `ask`, `query` and `eval` it used to offer
  only `postgres`, `snowflake` and `bigquery`, so a SQLite connector could not
  be asked for SQLite SQL at all. It now takes
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
- **DeepSeek is sent the effort too, and can be told not to think.** A
  `deepseek/` model through LiteLLM gets `LLM_EFFORT` as DeepSeek's
  `reasoning_effort` (`low` and `medium` send `low`, `high` sends `high`,
  `xhigh` and `max` send `max`), in `extra_body`: LiteLLM turns a top-level
  `reasoning_effort` for DeepSeek into thinking on or off and drops the
  level. The new `LLM_THINKING` setting, `on` by default, sends
  `thinking: {"type": "disabled"}` when `off`, and no effort with it. A host's
  own `extra_body` is merged with these, its keys winning.
- **Streamed answers on OpenAI-compatible routes record their real usage.**
  Such a stream reports no usage unless asked, so every streamed answer was
  recorded as an estimate. LiteLLM is now asked for it
  (`stream_options={"include_usage": True}`) on the routes it lists as
  OpenAI-compatible and accepting the option, DeepSeek and OpenAI among them,
  and the last chunk's usage is recorded. Other routes keep the estimate.
- **Reasoning tokens are visible on their own.** `UsageRecord` gains
  `reasoning_tokens`, read from `completion_tokens_details.reasoning_tokens`
  where the provider reports it, as DeepSeek does. They stay counted in
  `output_tokens`, which is what output is billed on.
- **`LLM_MAX_OUTPUT_TOKENS` defaults to 4096** (was 1024). Thinking is billed
  from the same allowance, and at 1024 Sonnet 5.5 can run out part-way through
  its SQL. Set it to 1024 to keep the old ceiling.

### Fixed

- **A missing Anthropic key is reported as missing, not as a connection
  error.** With `ANTHROPIC_API_KEY` unset and anthropic 2.0 or later (which
  `pip install nlqueries-core` resolves; the lock pins 1.x), every question
  failed with "Connection error.": the empty key was sent as an empty bearer
  token, which the HTTP library refuses before any request is made. The SDK
  now receives no key, and says that none is set.
- **`nlqueries update-password` fails when it cannot store the password.**
  Without a keyring it said the password could not be stored, and exited 0,
  so a script checking its status took the password for rotated. It now
  exits 1. `connect` given only `--password-env` for SQLite, DuckDB or
  BigQuery says the variable was never read, rather than that "the one
  given" was not used. The help of `extract-schema`, `kb-stats` and the
  document commands now describes what they do: row counts as the connector
  reports them, a per-table breakdown, and the collection name as stored.
- **`nlqueries health` reaches the databases it checks.** Its `SELECT 1` probe
  ran without permission to execute, which every connector requires, so each
  registered database was reported as "connection failed — is the database
  running?" whether it was or not. The probe now runs read-only, as the other
  commands that read from the database do.
- **The embedding fallback warning no longer says the daemon is required.**
  When the daemon could not be reached and `EMBED_SERVER_REQUIRED` was not set,
  the warning that the model was being loaded in-process quoted the message
  written for the required case: "EMBED_SERVER_REQUIRED is set, so this call
  will not load a model in-process".
- **`nlqueries connect` says a password was stored in the config file only
  when it was.** Connecting SQLite, or any database without a password, printed
  that note, and so did a password given to SQLite, DuckDB or BigQuery, whose
  URL has no place for one and drops it. Such a password, or a `--password-env`
  variable, which those types never read, is now said to be unused, and is not
  kept in the OS keychain either, where nothing read it.
- **A BigQuery dataset location that is not the shape of one is logged.** It
  was replaced by the default region without a word, and the history query then
  read another region's jobs, usually none.
- **`embed-server start`, `stop` and `status` on Windows no longer fail with a
  traceback when `tasklist` cannot be run.** A process whose state cannot be
  read is taken to be running, as when `tasklist` times out, and the failure
  is logged, once, with the path tried, as it does not pass the way a timeout
  does.
- **The CLI no longer fails when its output cannot carry a character.** With
  output redirected or piped on Windows, Python writes in the locale's code
  page, which has no check mark, and the first one raised `UnicodeEncodeError`.
  `connect` prints one after testing the connection and before saving it, so
  the connector was lost. Such characters are now written as a replacement.
  `--json` output, and the SQL chunk `ask` ends with when piped, is written in
  ASCII, with non-ASCII escaped (`\u00eb` for `ë`), so a payload carries
  the same data whatever the output's encoding. The chunk also starts a line
  of its own, so `| tail -1 | jq .sql` reads it whatever the answer before it
  ends with.
- **Every store names an agent the same way, so two agents no longer share
  a cache.** The semantic cache turned `.` into `_` in the agent ID, so
  `sales.eu` and `sales_eu` read and wrote one cache collection and could be
  served each other's answers. The knowledge base, the feedback and session
  files, and the cache and document collections now all use one storage
  name (`nlqueries.state_files.agent_storage_id`): letters, digits,
  `_`, `-` and `.` are kept and anything else becomes `_`. Document
  collections used the raw ID, so the CLI (`postgres:localhost:mydb`) and the
  MCP server (`postgres_localhost_mydb`, from the knowledge-base file name)
  wrote and read different collections for one agent; they now share one.
  **Upgrading:** knowledge-base, feedback and session files keep their names.
  A document collection written under the raw ID is still read until one
  under the new name holds the source's chunks; the next successful
  `doc-ingest` writes the new name and deletes the raw-ID collection, and
  `--allow-empty` removes both. The MCP server reaches such a source (an ID
  with `:`, `/` or a space) only after it is ingested again. A
  cache collection for an ID containing `.` starts empty under its new name;
  the old one is not read and can be deleted in Qdrant. The schema
  (`agent_<id>_schema`) and verified-examples (`agent_<id>_verified`)
  collections keep their current names for now. See
  [docs/qdrant-setup.md](docs/qdrant-setup.md).
- **Cache keys keep operators and numbers, and follow-ups are keyed on what
  they mean.** The Tier 0 exact-match key no longer strips `<`, `>`, `=`, `!`,
  `-`, `.` or `%` from the question, so `orders with total > 100` and
  `orders with total < 100`, `1.5` and `15`, or `!=` and `=` no longer share one
  cache entry and one answer. Other punctuation is still dropped. `run_query`
  now looks up and stores a follow-up under its resolved question rather than
  the text typed, so "and last month?" in one conversation is not answered
  with another conversation's entry. A stored `hit_count` that is not an
  integer is read as 0 instead of raising, and an exception from the cache
  lookup in `MultiAgentOrchestrator.handle_question` is treated as a miss, so
  the question is answered instead of failing. The first such failure for an
  agent and kind of error is logged at WARNING and later ones at DEBUG, since
  an unreachable cache fails the same way on every question. **Upgrading:**
  the Tier 0 point id now includes a version of the normalisation, so an entry
  stored under the old normalisation is never looked up for a question that
  now normalises to the same text it used to; each existing Tier 0 entry is a
  one-time Tier 0 miss and is stored under its new id the next time it is
  answered. Tiers 1 and 2 are keyed by embedding and are unaffected.
- **Follow-up detection matches whole words.** The signals that send a
  question to the follow-up resolver ("it", "that", "filter", "what about"
  and the rest) were matched inside other words, so "profit with tax" asked
  mid-conversation read as a follow-up (the "it" in "profit" and "with") and
  came back rephrased. They now match only as whole words.
- **The `export-kb` fallback writes sample values `yaml.safe_dump` cannot
  represent.** A sample value whose type is a subclass of `str`, `int` or
  `float` (a driver's own string or float type) was kept as it was and made
  the write fail with a traceback. Every value that is not exactly a `str`,
  `int`, `float`, `bool` or `None` is now written as its `str()`, as
  `Decimal` and `UUID` values already were.
- **A knowledge base built over SQLAlchemy reflection now loads.** The
  generic `sqlalchemy` connector and the `export-kb` fallback took table,
  column, schema and key names from SQLAlchemy's inspector as `quoted_name`, a
  `str` subclass that the PostgreSQL, Oracle and Snowflake dialects (among
  others) return. `yaml.dump` wrote each one as a
  `!!python/object/apply:sqlalchemy.sql.elements.quoted_name` tag, which
  `safe_load` refuses. They are now plain `str` where they leave the
  inspector, as the SQLite connector already did.
- **Errors no longer pass on what a driver or the connectors file quoted.**
  When the MCP `query` tool reports a failed query, it now shows the first
  line of the error and a short reference id. The server logs the same first
  line at WARNING with that reference (`[ref …]`), and the full error at DEBUG
  under the same reference only, so a log shipped elsewhere does not carry a
  DSN or password. An exception raised anywhere else in the query is logged
  the same way: its first line at ERROR, the traceback at DEBUG. The lines a
  driver adds after the
  first, where it quotes the statement, its parameters or a connection
  string, no longer reach the client. The same holds for a hybrid answer: when
  its query fails, the failure note in the SQL table (which is the whole
  answer when the documents found nothing, and the SQL part of the prompt for
  a combined answer) carries the first line only. Both use the new
  `connectors.base.short_error_text`. `QueryResult.error` itself is
  unchanged. A `connectors.yaml` that is not valid YAML is now reported by
  line number and problem, as `Failed to read connectors file: line 4:
  mapping values are not allowed here`, without the line the parser quoted,
  which may hold a password. That applies to the MCP `list_connectors` tool
  and to the CLI, where the parse error used to escape as a traceback and is
  now an ordinary error with exit status 1. `health` reports it as a failed
  `Database` check and still runs its other checks, so `health --json` still
  prints a report. The loader raises `ConnectorsFileError`, a `ValueError`,
  for it, and `list_connectors` now reads the file through the loader rather
  than parsing it itself. The loader reads the file as UTF-8 whatever the
  locale, so a non-ASCII password is read as written on a cp1252 Windows
  machine too, and a file that is not UTF-8 is reported as `Failed to read
  connectors file: it is not UTF-8 (the first invalid byte is at offset N)`,
  without the byte itself. `nlqueries connect` writes the file as UTF-8 too.
- **Promoted feedback reaches the prompt.** `promote-feedback`, and `export-kb`
  which runs it, write confirmed (question, SQL) pairs to
  `agent_<id>_verified` with the connector ID made safe for a collection name
  (`postgres:localhost:mydb` becomes `postgres_localhost_mydb`). The question
  path built the name from the raw ID instead, so for every ID `connect`
  produces it looked for a collection that never exists and the verified
  examples were never used. It now names the collection the same way, and
  `promote-feedback` prints the name it actually wrote to. Because two IDs
  that differ only in the replaced characters now share a collection, the
  lookup also matches the `agent_id` each pair was promoted for.
- **A `.env` in the working directory is read.** `load_dotenv()` with no
  path searches upward from the file that calls it, which for an installed
  package is `site-packages/nlqueries`, so the `nlqueries` command never found
  the `.env` a user created with `cp .env.example .env` (it did under
  `python -c`, which hid it). The search now starts from the working
  directory; variables already in the environment still win.
- **The CLI works inside the Docker image.** Compose runs the image on a
  read-only root filesystem with one volume at `/data/nlqueries`, but the
  state directory stayed at `~/.nlqueries`, so `nlqueries query` failed writing
  its session file (`[Errno 30] Read-only file system`) and `embed-server
  start` its PID file. The image now sets `NLQ_STATE_DIR=/data/nlqueries`. That
  also puts the generated semantic-cache signing key on the volume, so the CLI
  and the MCP server in one container verify each other's cache entries;
  before, each process held its own key in memory and nothing was shared.
- **`feedback-stats` accepts an alias.** `feedback dev` stores under the
  connector ID the alias names, while `feedback-stats dev` looked under `dev`
  and reported "No feedback recorded yet".
- **`embed-server start`, `status` and `health` agree on the port.** `status`
  always probed 8765, so a daemon started with `--port` read as "starting" for
  ever, and clients only ever connect to `EMBED_SERVER_PORT`. `start` now
  listens on `EMBED_SERVER_PORT` by default and warns when `--port` names a
  port clients will not use; `status` checks `EMBED_SERVER_PORT` and takes
  `--port`; `health` reports the port it reached. The daemon's own entry
  point (`python -m nlqueries.embeddings.embed_server`, and `serve()`) also
  defaults to `EMBED_SERVER_PORT`, and the client reads the port from
  `config` rather than the environment, so every path agrees.
- **`doc-ingest`, `doc-sync-notion` and `doc-sync-confluence` store what
  they ingest.** They chunked a source and printed a count, but nothing in
  core wrote the chunks anywhere, so `doc-ask doc_<source_id>_chunks`
  searched a collection no command had filled. Each now embeds its chunks and
  stores them in that collection through the new
  `qdrant_store.replace_chunks`, which stores the new chunks before removing
  the source's old ones, so a re-ingest replaces the earlier version and a
  failure before the write leaves it untouched; a failure of the clean-up
  after the write raises `StaleChunksRemain`, and the command says both
  versions may be searchable until a rerun. A command that cannot store its
  chunks exits 1 instead of reporting success. A read that produces no text
  stores nothing and exits 1, since a mistyped space key or a scanned PDF
  would otherwise delete a good copy; `--allow-empty` stores the empty
  result. With `--since`, an unchanged page or space is left as stored. For
  Confluence, `--since` only decides whether to read, through the new
  `ConfluenceConnector.has_changes_since` (one search, no page bodies), and
  any read is of the whole space, because a partial read numbers its chunks
  from zero and would overwrite other pages' chunks. The Confluence read now
  pages through the search instead of stopping at its first 200 results,
  following the next link's cursor where the installed client takes one and
  `start` otherwise. Pages are read in `created` order, which an edit does not
  change, so a page edited mid-read can no longer jump past the window
  already read and be skipped; a read that still ends with fewer distinct
  pages than Confluence reports, or stops advancing, is refused
  (`IncompleteSpaceRead`), since storing it would remove the missing pages.
  `replace_chunks` writes in batches of 256, so a large space stays under
  Qdrant's 32 MB request limit; a failure after the first batch raises
  `StaleChunksRemain`.
  The docs no longer say `nlqueries query`
  routes questions to documents: it and the MCP `query` tool answer from the
  database.
- **Every user-facing doc checked against the code.** The README, getting
  started, the CLI and configuration references, connectors, database
  hardening, Qdrant setup, MCP authentication, architecture, troubleshooting,
  the contributing guides, dependency locking and `SECURITY.md` were read
  claim by claim against the code, and what did not hold was corrected. The
  larger corrections:
  - `configuration.md` drops settings nothing reads (`NLQ_EXPLAIN_VALIDATION`,
    `KB_REFRESH_INTERVAL`, `LOG_LEVEL`, `DATABASE_URL`) and adds the ones it
    was missing: the embedding daemon, the cache signing key, the connector
    limits and pools, and `MCP_HOST`/`MCP_PORT`. `LLM_PROVIDER` takes
    `anthropic`, `litellm` or `bedrock`, and `OPENAI_API_KEY` alone selects
    OpenAI.
  - `cli-reference.md` documents every option `--help` lists, says what
    `query`, `ask`, `extract-schema` and `kb-stats` actually print, and
    explains `promote-feedback --include-anonymous`.
  - TLS is described per connector: only `postgres`, and libpq URLs on the
    generic connector, take `ssl_mode`/`ssl_ca_cert`, set by hand in
    `connectors.yaml`.
  - `qdrant-setup.md` lists every collection core writes and reads, and says
    plainly that capsules from `process-history --embed` are not yet read by
    the question path.
  - The Docker image's contents are stated: no MySQL, Redshift or SQL Server
    driver, and no `docs` or `wiki` extra.
  - `SECURITY.md` names everything core talks to, Qdrant and the one-time
    model download included.
  - The hint `process-history --embed` prints when Qdrant is down suggested
    `docker run -p 6333:6333`, publishing an unauthenticated Qdrant on every
    interface, and pointed at an "Appendix B" that does not exist. It now
    publishes on `127.0.0.1` and points at `docs/qdrant-setup.md`.
- **Documentation claims corrected against the code.**
  - The semantic cache does not answer "in under 50 ms with no LLM or
    database round-trip". A hit skips the LLM and replays the stored answer,
    and runs the stored SQL again unless execution is off, so the rows are
    current. Only `query` and the MCP `query` tool use the cache; `ask` does
    not.
  - MySQL is listed as running on the generic SQLAlchemy connector, which is
    what `connect mysql` uses.
  - BigQuery history reads `INFORMATION_SCHEMA.JOBS_BY_PROJECT`, which needs
    `bigquery.jobs.listAll` (for example through
    `roles/bigquery.resourceViewer`). The permission was not documented.
  - Without `VIEW SERVER STATE`/`VIEW DATABASE STATE`, SQL Server's
    `process-history` stops with an error naming the permission; the docs
    said it returned an empty history.
  - Redshift's `STL_QUERY` exists only on provisioned clusters, so
    `process-history` stops with an error on a Serverless workgroup. Redshift
    does keep comments in `pg_description`; the connector does not read them,
    rather than Redshift lacking them.
- **The MCP `query` tool runs the SQL it generates.** It passed no execution
  policy, so the run fell back to the generate-only default and every MCP
  client received SQL and no rows, though the tool's action is
  `query:execute`, its docstring promises an answer and `mcp-authentication.md`
  says it runs SQL. It now asks for read-only execution, as the CLI's `query`
  does, once the request has been authorised for `query:execute`.
- **`nlqueries connect mysql` and `connect postgresql` work after they
  connect.** `connect` accepts both and stores the type as given, but neither
  was in the connector registry, so `query`, `export-kb` and
  `process-history` could not open the connector again (`export-kb` took a
  reduced fallback path instead). `mysql` now runs on the generic SQLAlchemy
  connector and `postgresql` on the PostgreSQL one. That also tightens
  `connect postgresql`: it used to fall back to a raw connection under
  libpq's `prefer`, which accepts a server without TLS, and now goes through
  the PostgreSQL connector, which requires TLS, so such a server is refused
  under either spelling. `connect mysql` sets no TLS; connectors.md says how
  to require it. The docs no longer
  promise MySQL query history from the performance schema, or TLS settings
  from `SSL_MODE`/`SSL_CA_CERT`, neither of which anything read.
- **The LLM health check works on OpenAI's Responses-API and reasoning
  models.** `nlqueries health` and the MCP `health` tool asked the model for 5
  output tokens. OpenAI's Responses API, which LiteLLM uses for models such as
  `o3-pro`, `gpt-5-pro` and the codex models, refuses fewer than 16, so the
  check failed on every one of them. The probe now asks for 16
  (`nlqueries.llm.PROBE_MAX_TOKENS`). An OpenAI or Azure reasoning model that
  spends that budget before writing anything answers with an HTTP 400 ("Could
  not finish the message because max_tokens…") rather than an empty reply;
  the check now treats that as it treats an empty reply, as a successful round
  trip with a note, instead of reporting the provider as down.
- **The MCP docs describe the server as it is.** `docs/cli-reference.md` and
  `nlqueries mcp-server start --help` still said the SSE server has no
  authentication and its tools are reachable without credentials. Since
  0.3.0 it will not start without an identity provider or a pre-shared token
  unless `NLQ_ALLOW_UNAUTHENTICATED_MCP=1` is set. `docs/mcp-authentication.md`
  and the 0.3.0 entry below named a `--transport` flag that does not exist.
  The switches are `mcp-server start --sse`, and `MCP_TRANSPORT` for the
  Docker image and `python -m nlqueries.mcp_server`. The CLI reference's link
  to the PostgreSQL query-history section also pointed at a missing anchor.
- **SQL validation accepts the `mssql`, `postgresql` and `mariadb`
  dialect spellings.** The SQL policy already mapped these connector
  spellings to sqlglot's names, but the schema check's re-parse passed them
  through raw, and sqlglot raised `ValueError`. Every correction after a
  database error therefore failed for a caller that passed one of them (the
  CLI maps the dialect first, so it was not affected). `_same_statement`,
  which the correction prompt and the echo guard use, maps them too; before,
  it fell back to a text match, and a reindented echo of a rewritten
  statement would have run.
- **OIDC ID tokens: `email` and `email_verified` read strictly.**
  `OidcClaims.email` was `str()` of the claim, so `"email": null` became the
  address `"None"`, shared by every token that sent it; an email claim that
  is not a string is now no address, and a padded one is trimmed. `email_verified` was `bool()` of the claim, so
  a provider that sends booleans as strings had `"false"` read as verified;
  only `true` or `"true"` count now, and never without an address. Both
  decide whose account a sign-in links to.
- **OIDC ID tokens: a minute's clock difference, and any asymmetric
  algorithm.** `OidcTokenVerifier` allowed no difference between the
  provider's clock and this host's, so a provider one second ahead issued
  tokens that read as not yet valid, and every sign-in failed. It now allows
  60 seconds on `exp`, `nbf` and `iat` by default; `leeway_seconds` sets
  another. It also verified RS256 only, so a provider signing with ES256 or
  PS256 could not be used. It now accepts RS256/384/512, PS256/384/512,
  ES256/384/512 and EdDSA, building the key from the JWKS entry. The token's
  algorithm must agree with that key's type, with its `alg` when it states
  one, and with its curve, so a token cannot choose how its signature is
  checked. HMAC algorithms and `none` are refused.
- **The embedding daemon no longer grows without bound on Windows.** Each
  request is served on a new thread, and the encode ran on it. With the torch
  backend, MKL keeps buffers for each thread and does not free them when the
  thread exits, so texts of varying length grew the daemon by about 8 MB a
  call: one reached 38 GB in four days. Encodes now run on a fixed pool of
  threads, as many as `EMBED_SERVER_MAX_CONCURRENCY`, and the same 1,500 calls
  added about 90 MB. On Linux, encoding on a new thread per call levelled off
  as well.
- **PostgreSQL streams a read that opens with a comment.** A statement is
  read through a server-side cursor only when it is a read, and that was
  judged from its first characters. A statement opening with a comment was
  never streamed, so psycopg2 loaded its whole result before the row and
  byte budget saw a row. Every statement a SQL console tags for
  cancellation (`/* nlq-console:<id> */`) is one. The check now skips
  leading whitespace, parentheses and comments, nested as Postgres nests
  them, and compares the whole first word, so `withdraw_funds()` no longer
  counts as `with`.
- **A streamed PostgreSQL read is bounded by its timeout as a whole.**
  `statement_timeout` bounds each statement, and a streamed read is a
  `DECLARE` and then a `FETCH` per batch, so each batch could run for the
  whole budget again: a 2 s timeout let a 6 s read finish. Before each batch
  of 1,000 rows the timeout is now set to what is left, and a read that runs
  out of time fails as one that is not streamed does. This covers every
  streamed read, the agent's own included. The batches are now 1,000 rows
  from the first; they had started at a handful and grown.
- **A streamed read's `execution_time_ms` includes reading its rows.** It
  was taken once the cursor was declared, so a read that took a second
  reported a few milliseconds and was never classed as slow.
- **Query history keeps our own tagged runs once they stream.**
  `pg_stat_statements` records a streamed read only as its cursor's
  `DECLARE`, which history extraction drops along with every other cursor.
  A cursor declared for a run tagged `/* nlq-<kind>:<id> */` is now returned
  as its statement. Each such run is a row of its own with one call, and
  its time is the declaration's, not the read's.
- **The SQLite connector no longer deadlocks when threads share it.** The
  loader pools one connector per database and hands it to every thread that
  asks, and SQLite's is one `sqlite3` connection with a Python authorizer.
  Two threads on it at once could stop the whole process: one held the
  connection's mutex waiting for the GIL to run the authorizer, the other
  held the GIL waiting for the mutex. The connector now lets one thread use
  the connection at a time; SQLite never ran two statements on one
  connection at once anyway. A statement's timeout now starts when its
  thread has the connection, so a timeout running out while waiting can no
  longer interrupt another thread's statement.
- **The MCP schema tool shows primary and foreign keys.** `get_agent_schema`
  read `primary_key` and `foreign_key` from each column, names no knowledge
  base has carried; the generator writes `is_primary_key`, `is_foreign_key`
  and `references`. So no column ever showed a key flag. It now reads those,
  and, like the prompt, shows a foreign key only when its target is recorded.
- **The MCP schema tool no longer fails on a table that lists its foreign
  keys.** It read a per-table `foreign_keys` list of strings, which the
  generator never writes (it writes one list of `from`/`to` mappings under
  `schema`), so the line never showed anything, and a list in the
  generator's shape copied onto a table raised `TypeError`. The line is gone;
  each foreign key shows as its column's `FK→` flag.

- **A hand-edited connector entry that is not a mapping no longer crashes the
  CLI.** `agent-a: postgresql://host/db` (the URL where its settings belong)
  raised `AttributeError` from alias resolution, so from every command given an
  alias, and from `doctor`, `nlqueries connectors`, `kb-stats` and the MCP
  `list_connectors` tool. The listings now name the entry. `doctor` reports it
  as a failed check, and a command that opens the connector refuses with a
  message that names it and says to register it again.
- **A connector whose `db_type:` is blank, or not text, no longer crashes the
  CLI either.** A key present with no value read as None past the `""`
  default, and `.lower()` on it took down `doctor`'s health check and
  `extract-schema`, `process-history`, `export-kb`, `kb-stats` and `query`'s
  execution step. Each now reads the type as text, so a blank one reaches the
  command's own "no connector class" or "unsupported db-type" message.
- **A temperature no longer fails the Anthropic client on anthropic 1.x.** The
  SDK dropped `temperature` from `messages.create()`, so the self-consistency
  candidates that pass one raised `TypeError` before sending anything, and
  each such candidate was silently dropped. It is now sent in the request body.
- **`embed-server start` and `stop` handle a PID file left behind.** After an
  unclean shutdown (a reboot, a killed terminal), `start` took the leftover
  file to mean a daemon was running and told you to stop it, and `stop` then
  crashed on Windows with `OSError: [WinError 87]` on the dead PID. All three
  commands now check the PID: a stale file, or one that does not hold a PID, is
  removed with a one-line note, and `start` goes on to start the daemon. `stop`
  waits up to 5 s for the daemon to exit. If it cannot stop a live one, it says
  so in one line and exits 1.
- **SIGTERM stops the embedding daemon.** Its handler called
  `server.shutdown()` on the thread running the server loop, which waits for
  that loop to finish, so on Linux and macOS the daemon hung instead of
  exiting, and `stop` reported it stopped regardless.

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

[0.4.0]: https://github.com/nlqueries/nlqueries/releases/tag/v0.4.0
[0.3.0]: https://github.com/nlqueries/nlqueries/releases/tag/v0.3.0
[0.2.0]: https://github.com/nlqueries/nlqueries/releases/tag/v0.2.0
[0.1.0]: https://github.com/nlqueries/nlqueries/releases/tag/v0.1.0
