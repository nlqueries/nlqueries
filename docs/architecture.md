# Architecture

## Request flow

```mermaid
graph TD
    CLI["CLI (nlqueries / nlq)"]
    MCP["MCP Server"]
    ORCH["MultiAgentOrchestrator"]
    IC["Intent Classifier\n(sql / document / hybrid)"]
    SQLA["SQL Agent"]
    DOCA["Document Agent"]
    RM["Result Merger"]
    KB["Knowledge Base (YAML)"]
    CACHE["Semantic Cache"]
    EMBED["Embeddings\n(sentence-transformers + daemon)"]
    QDRANT["Qdrant (vector store)"]
    LLM["LLM Client\n(Anthropic / OpenAI / LiteLLM)"]
    CONN["DB Connectors\n(Postgres, MySQL, Snowflake, BigQuery,\nRedshift, MSSQL, DuckDB, SQLite)"]
    DOCCONN["Document Connectors\n(PDF, Word, Excel, Notion, Confluence)"]
    PROC["Query Pipeline\n(filter, cluster, parameterize, annotate)"]
    FB["Feedback Store (JSONL)"]

    CLI --> ORCH
    MCP --> ORCH
    ORCH --> IC
    IC --> SQLA
    IC --> DOCA
    SQLA --> RM
    DOCA --> RM
    ORCH --> CACHE
    CACHE --> QDRANT
    SQLA --> KB
    SQLA --> LLM
    DOCA --> QDRANT
    DOCA --> LLM
    CONN --> PROC
    PROC --> EMBED
    EMBED --> QDRANT
    CONN --> KB
    DOCCONN --> EMBED
    CLI --> CONN
    CLI --> DOCCONN
    CLI --> FB
```

A question enters through the CLI or the MCP server and reaches the `MultiAgentOrchestrator`. The orchestrator first checks the semantic cache; on a miss, an intent classifier routes the question to the SQL agent, the document agent, or both in parallel (hybrid), and a result merger combines and ranks the outputs by confidence before returning an answer.

## Module layout

```
nlqueries/
├── cli/                 CLI commands (click + rich) — connect, query, ask, health, kb-stats, etc.
├── connectors/           DB connector implementations + BaseConnector ABC
│   ├── postgres.py, mysql (via base), snowflake.py, bigquery.py
│   └── redshift.py, mssql.py, duckdb.py        (optional extras)
├── document_connectors/  PDF, Word, Excel, Notion, Confluence + BaseConnector ABC
├── processing/           Query filter, clusterer, parameterizer, intent annotator, pipeline
├── knowledge/            YAML knowledge base generator (kb_generator.py) + kb-stats report (kb_stats.py)
├── embeddings/           Sentence-transformer embedder, Qdrant store, embedding daemon (embed_server.py)
├── cache/                Semantic cache (semantic_cache.py)
├── llm/                  LLM client abstraction — Anthropic, generic client, LiteLLM
├── orchestrator/         Orchestrator, intent classifier, multi-agent + document orchestrators,
│                         prompt assembly, SQL generation + sqlglot validation, result merger,
│                         conversation / follow-up handling
├── analysis/             Query analyzer
├── auth/                 OIDC token verification utilities
├── feedback/             Local JSONL feedback store + models
├── mcp_server/           MCP server entry point
├── telemetry.py          OpenTelemetry integration
└── config.py             Environment-variable configuration
```

## Cache partitioning and authorisation

The semantic cache is partitioned by agent: one Qdrant collection per agent,
named `cache_{agent_id}`. **Every entry in a collection is readable by everyone
who may query that agent.** That is not a gap in the model, it is a restatement
of it -- authorisation is granted at agent level, row filters are a property of
the agent record rather than of the caller, and cached SQL is replayed through
the same filtered connector that produced it, so a replay is subject to the
filters a fresh run would be. Document answers are cached verbatim, and every
user of the agent is entitled to the same documents.

That reasoning holds only as long as nothing narrows what a caller may see
*below* the agent. The invariant to preserve:

> Anything that makes two callers of the same agent entitled to different
> results -- per-user row filters, per-user document ACLs, row-level security
> keyed on caller identity, per-user column masking -- must contribute its
> distinguishing value to `cache_context` on **both** `get()` and `put()`.
> If it cannot, it must not be built.

`cache_context` (seam S2) is stored in the entry payload on write and compared
on read. The comparison is an **equality**, not a subset test: a caller that
passes no context reads only entries written with none. That direction matters
more than it looks. The failure being guarded against is a caller that forgets
to pass its context on one `get()` path, and under a subset test that caller
would have matched entries scoped by every context -- failing open, silently,
in exactly the case the mechanism exists for.

The context is recovered from the payload as whatever keys the cache did not
write itself (`_RESERVED_PAYLOAD_KEYS` in `cache/semantic_cache.py`), so there
is no marker field to keep in sync and entries written before the check existed
are still read correctly. A consequence worth knowing: **adding a new key to a
stored payload without adding it to that set makes it count as caller context,
and every entry carrying it will miss.**

**The context is covered by the signature.** `envelope.py` protects a cache
entry against anyone with write access to Qdrant but no access to the signing
key. Until the context was part of the signed message, that protection stopped
at the partition: reaching another context did not require forging a tag, only
moving a valid one, by editing the context keys the HMAC did not cover. The
context is now appended to the signed message -- but only when it is non-empty,
so an entry written without one produces byte-identical output to before and
keeps verifying. Only context-carrying entries pay a one-off miss on upgrade,
rather than the whole cache going cold. Conditional inclusion is still sound
both ways: stripping a context makes the verifier build the short message
against a tag computed over the long one, and adding a context does the reverse.

**The point ID carries the context too.** `put()` used to derive point IDs from
the question alone (`_point_id_for_question(normalized)`, and `tmpl:{masked}` for
the template point), so two callers in different contexts asking the same
question upserted over one another. That is not a future problem: `cache_context`
is already in use for follow-up turns, so a scoped follow-up write and a
context-free write of the same normalised question collided. The partition still
held -- neither read the other's entry -- but each clobbered the other and both
then missed indefinitely, which is a collapsed hit rate rather than a leak and
correspondingly harder to attribute. The context is now part of the id, appended
only when non-empty so entries written without one keep the id they had.

**Putting the context in the point ID removes what bounded the collection's
size, and nothing here replaces it.** This is the most important operational
consequence on this page, so it is stated before the rest.

Nothing in the cache ever deletes a point. The TTL is applied on *read*, in
`_payload_to_entry`, and the only removal path is `invalidate()`, which drops the
whole collection. That was survivable while a repeated question upserted over its
own id: the point count tracked an agent's distinct question vocabulary and went
no further.

With the context in the id that no longer holds. Enterprise derives
`context_fingerprint` from the conversation summary, the last SQL and a window of
turns, so it changes on virtually every follow-up turn — and each such turn now
writes one or two points at an id that will essentially never recur, is never
overwritten, and is never removed. **Point count therefore grows linearly with
follow-up traffic for the life of the agent.** Three consequences follow:

- Storage grows without bound, and only `invalidate()` reclaims it — which
  discards the whole cache, including the entries worth keeping.
- Expired entries are still *searched*. The TTL discards them after the vector
  search has already ranked them, so they go on competing for the
  `NLQ_CACHE_COSINE_CANDIDATES` slots described below, and the starvation that
  section describes gets worse over time rather than staying level.
- Scoped entries written **before** this change are orphaned permanently. They
  keep their old context-free ids, which nothing will now look up, so they pay
  more than the one-off miss the signature change costs — they are never read
  again and never removed.

**A prune is required, and is deliberately not in this change.** The shape that
fits is a delete-by-filter on `created_at` older than the TTL, issued on write.
Qdrant can express it (`DatetimeRange` inside a `FilterSelector`), but whether it
behaves correctly against an unindexed string `created_at` — collections created
before this carry a payload index on `kind` only — is exactly the sort of thing
that must be measured against a real Qdrant rather than reasoned about. A delete
that matches the wrong points destroys live cache entries, so it should arrive
with a testcontainers-backed test and not on inference.

**What the candidate window bounds rather than eliminates.**
`NLQ_CACHE_COSINE_CANDIDATES` defaults to five. A context-free read on an agent
carrying more than five near-duplicate scoped entries, all above the threshold
and all ranked above the unscoped one, still falls through to a miss.

**Giving each context its own point ID makes that more likely, not less**, and
the two changes have to be read together. Scoped entries for one question used
to collapse onto a single point; they now accumulate one per context, so a busy
conversational agent can carry well past five follow-up-scoped entries for a
popular question, and from then on a context-free Tier 1 or Tier 2 lookup for it
is starved. Tier 0 hides this for a verbatim repeat but not for a paraphrase --
which is what tiers 1 and 2 exist for. Raising the setting is the immediate
answer, and the trade is payload transfer for candidates that are discarded.

The change that would remove it rather than bound it: store a digest of the
context as its own payload key, so the equality becomes an exact match Qdrant
*can* express, push it into the query filter and return the window to one. It is
not done here because every entry already stored lacks that key, so either the
whole cache goes cold for a TTL or the filter needs an `IsEmpty`-or-match
disjunction to keep matching them — worth doing deliberately rather than as the
tail of this change.

**Why the cosine tiers fetch more than one candidate.** Qdrant's filter is a
subset test: it can require the caller's keys but cannot require the absence of
others, so the equality is applied after the search rather than pushed into it.
Asking for a single point would therefore let the nearest neighbour decide the
outcome for everybody -- an entry from another context, ranked top, would consume
the only candidate slot and the lookup would fall through even with a
same-context entry immediately below it and above the threshold. That bites
hardest on context-free reads, where the pushed-down filter is `kind` alone and
every context-scoped entry competes freely for that slot. Both tiers take the
first candidate clearing the threshold *and* the context, and stop scanning at
the first below-threshold point so widening the window cannot lower the
similarity bar.

The invariant is guarded by `tests/test_cache_partitioning.py`, which asserts
across all three tiers that an entry written under one context is a miss for a
different context and for no context, and -- as the control that gives those
their meaning -- a hit for its own.

See [cli-reference.md](cli-reference.md) for what each CLI command does, [connectors.md](connectors.md) for connector-specific behavior, and [configuration.md](configuration.md) for the environment variables that wire these modules together.
