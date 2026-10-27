# Observability

What NLQueries tells you about itself, and where to look:

- **Logs**, with a level you set and, for the embedding daemon, a JSON format.
- **Audit events**: one structured record per authorization decision and per statement sent to a database.
- **OpenTelemetry spans and metrics** around the question path. NLQueries installs no exporter of its own, so they go nowhere until you run it under one.
- **Health checks** for every service it depends on, for people and for CI.
- **Per-answer detail**: how one answer was produced, and what the cache and knowledge base hold.

There is no Prometheus endpoint and no metrics server in core.

---

## Logs

`LOG_LEVEL` sets the level for all three processes: `DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL`. Any other value is ignored, with a warning that names it. Unset, each process keeps its own default:

| Process | Unset | Set |
|---|---|---|
| CLI (`nlqueries ...`) | No logging is configured, so only warnings and errors reach stderr | Records at that level and above go to stderr as `LEVEL logger.name: message` |
| MCP server (`nlqueries mcp-server start`) | `INFO` | That level |
| Embedding daemon (`nlqueries embed-server start`) | `INFO` | That level |

```bash
LOG_LEVEL=INFO nlqueries query dev "How many orders shipped last month?"
```

`process-history` quiets the `sqlglot`, `LiteLLM`, `litellm` and `httpx` loggers to `ERROR` unless you pass `--verbose`.

The embedding daemon writes plain text by default. Set `NLQ_LOG_FORMAT=json` for one JSON object per line, with the keys `asctime`, `name`, `levelname` and `message`, plus `exc_info` for a traceback and `stack_info` when there is one. This applies when the daemon runs in the foreground (`embed-server start --foreground`, or `python -m nlqueries.embeddings.embed_server`, whose `--log-format` overrides it). Any other value is plain text, with a warning.

---

## Audit events

Two events are recorded on the `nlqueries.audit` logger:

- `authorization`, for every decision the MCP server's guard makes, allowed or denied;
- `query_execution`, for every statement sent to a database for an agent, whether it came from a fresh question, a cache replay or literal grounding's value lookups.

The log message is the event's words followed by its JSON, on one line, with sorted keys:

```
INFO nlqueries.audit: query executed {"agent_id":"sqlite_demo","at":"2026-10-26T20:15:50.243609+00:00","connector_type":"sqlite","dialect":"sqlite","duration_ms":0.0,"error_class":null,"event":"query_execution","outcome":"ok","policy_decision":"allow","policy_version":"6","principal":"alice","request_id":"3aa3a3d53e60498285ac2a3d210060a9","row_count":1,"schema_version":1,"source":"local","sql_bytes":15,"sql_sha256":"70d501bdc85b04fc40fa92c599432fc63329dd6e35496a0970c77f6c8698ef30","surface":"cli","truncated":false}
```

An event carries a SHA-256 digest of the statement and its length, never the SQL text, the question, the rows or a credential. The same record is attached to the log record as `extra={"audit": ...}`, for a handler that wants the fields rather than the text.

To keep the events in a file of their own, set `NLQ_AUDIT_LOG` to a path: each event is appended as one JSON line, to a file created at mode `0600`. A file that cannot be written is reported on `nlqueries.audit.sink`, and the query is not failed.

**Execution events need `INFO` logging or the file.** A `query_execution` event is built only when the `nlqueries.audit` logger would emit an `INFO` record or `NLQ_AUDIT_LOG` is set. Otherwise it is skipped, and nothing says so. That catches three cases:

- the CLI with no `LOG_LEVEL`, since it then configures no logging at all;
- `LOG_LEVEL=WARNING` or higher, set to quieten the output, on the CLI or the MCP server;
- the MCP server at any level above its default of `INFO`.

For audit records with quiet logs, set `NLQ_AUDIT_LOG`: the file does not depend on the log level.

Every field, and the rules the file sink follows, are in [Audit events](configuration.md#audit-events). The `authorization` event's fields are in [What gets recorded](mcp-authentication.md#what-gets-recorded).

---

## OpenTelemetry traces and metrics

The question path is instrumented with OpenTelemetry under the instrumentation scope `nlqueries.core`. `nlqueries-core` installs the OpenTelemetry SDK and the OTLP exporter, but it never configures a tracer or meter provider. So by default every span and metric is a no-op, and setting `OTEL_EXPORTER_OTLP_ENDPOINT` on its own exports nothing.

To export, run NLQueries under OpenTelemetry's own launcher, which reads the standard `OTEL_*` variables and installs the providers before NLQueries starts:

```bash
pip install opentelemetry-distro      # provides opentelemetry-instrument

export OTEL_SERVICE_NAME=nlqueries
export OTEL_TRACES_EXPORTER=otlp
export OTEL_METRICS_EXPORTER=otlp
export OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317

opentelemetry-instrument nlqueries mcp-server start --sse
```

To see the output without a collector, use `OTEL_TRACES_EXPORTER=console` and `OTEL_METRICS_EXPORTER=console`. The console exporters write to stdout, so never use them with the stdio MCP server: stdout is its protocol channel, and the exporter's output would corrupt it. Use `otlp` there.

`opentelemetry-distro` is not part of `nlqueries-core` or its Docker image. Install it into the same environment as the package.

### Spans

| Span | Where | Attributes |
|---|---|---|
| `orchestrator.handle_question` | Generating, and when execution is on running, the SQL for one question | `agent_id`, `dialect`, `intent_type` (`sql`), `sql_valid`, `attempt_count`, `row_count`, and `execution_retry` when a correction was attempted after a database error |
| `document_orchestrator.handle_question` | Answering one question from a document source | `collection`, `intent_type` (`document`), `top_k`, and `source_id` when the search is limited to one source |
| `postgres_connector.execute_query` | One statement on the `postgres` connector. Other connectors have no span of their own | `db.system` (`postgresql`); `db.row_count` and `db.result_truncated` when it succeeds; `error` (`true`) when it fails |
| `qdrant_store.search_chunks` | One document-chunk search | `collection`, `top_k`, `result_count`, and `source_id_filter` when set |
| `qdrant_store.upsert_chunks` | Storing a document's chunks | `collection`, `chunk_count` |

### Metrics

| Metric | Type | Attributes |
|---|---|---|
| `nlqueries.queries.total` | Counter | `agent_type` (`sql` or `document`); `dialect` on SQL questions |
| `nlqueries.queries.latency_ms` | Histogram, ms | The same as `nlqueries.queries.total` |
| `nlqueries.chunks.search_latency_ms` | Histogram, ms | `collection` |

A question answered from the semantic cache is served before the SQL path runs. It produces no `orchestrator.handle_question` span and is not counted in `nlqueries.queries.total` or `nlqueries.queries.latency_ms`. Those two describe questions that went to the LLM. Use the audit log's `query_execution` events to count every statement, cache replays included.

---

## Health checks

`nlqueries health` checks each service NLQueries depends on and exits `1` if any check fails:

```bash
nlqueries health
nlqueries health --connector dev      # one connector only
nlqueries health --json               # for CI and scripts
```

It checks:

- **Qdrant**: reachability, version and round-trip time.
- **Each connector**: a `SELECT 1`. Where the connector supports them, it also reports the login's privileges and TLS, and what the connector enforces for that engine.
- **The LLM provider**: a real, small completion, which needs working credentials. It costs a few tokens.
- **The embedding daemon**: a warning, not a failure, when it is not running.
- **Configuration**: warnings for settings that will cause trouble later.

`--json` prints:

```json
{"checks": [{"service": "Qdrant", "status": "ok", "detail": "..."}, ...], "healthy": true}
```

`status` is `ok`, `warn`, `fail` or `skip`. With `--verbose`, a failed check also carries its `error`. See [health](cli-reference.md#health).

The MCP server's `health` tool checks the LLM configuration, Qdrant, the embedding daemon and the configuration, but no database. Its LLM line reports the configured provider and model without calling it, unless the caller passes `probe_llm=true`: the tool is reachable by anyone who can reach the server, and a check that spends tokens would let them spend yours.

---

## Per-answer detail

**How one answer was produced.** `nlqueries query --explain --json` adds a `provenance` object to the output:

- `route`;
- `capsules_used`;
- `prompt_sections`: which parts of the knowledge base were put in the prompt;
- `cache`: `hit`, `similarity` and `tier`;
- `validator`: warnings;
- `literals_grounded` and `literal_repair`;
- `timings`: `total_ms` for the whole question, and `sql_ms` for the SQL path.

The JSON also carries `latency_ms` either way. The MCP `query` tool takes an `explain` argument for the same record. See [query](cli-reference.md#query).

**What the cache holds.** `nlqueries cache stats <agent>`, and the MCP `get_cache_stats` tool, report the number of cached entries and the agent's Qdrant collection. They do not count hits and misses. For that, read `cache` in each answer's provenance. See [cache](cli-reference.md#cache).

**What the knowledge base covers.** `nlqueries kb-stats <agent>` reports schema, description, capsule and join coverage, the glossary, cache entries and feedback history. It exits `1` when the knowledge base is missing, so it works as a CI gate, and `--json` gives the same report for scripts. See [kb-stats](cli-reference.md#kb-stats).

**What people thought of the answers.** `nlqueries feedback-stats <agent>` counts thumbs-up, thumbs-down and corrections. The MCP `get_query_history` tool lists recent rated questions with their SQL. See [feedback-stats](cli-reference.md#feedback-stats).

---

## Token usage, from Python

The CLI and the MCP server do not report token usage. A program that uses NLQueries as a library can collect it: bind a list with `use_usage_sink`, and every LLM completion made inside the block appends a `UsageRecord`.

```python
from nlqueries.execution import ExecutionPolicy
from nlqueries.llm import use_usage_sink
from nlqueries.orchestrator.sync_runner import run_query_sync

usage = []
with use_usage_sink(usage):
    result = run_query_sync(
        question="How many orders last month?",
        agent_id="postgres:localhost:mydb",
        execution=ExecutionPolicy.execute_read_only(),
    )

for record in usage:
    print(record.model, record.input_tokens, record.output_tokens, record.estimated)
```

Each record has `model`, `input_tokens`, `output_tokens`, `cache_read_tokens` and `cache_write_tokens` (for prompt caching, `0` where the provider has none), and `reasoning_tokens`. `reasoning_tokens` is already counted in `output_tokens`, so do not add it again. `estimated` is `true` when the provider reported no usage and the counts are a rough estimate from the text length. The sink is bound per task, so concurrent requests each collect their own.
