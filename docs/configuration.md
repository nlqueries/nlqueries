# Configuration Reference

All settings are read from environment variables, or a `.env` file in the working directory (copy `.env.example` to `.env` to start). Only `./.env` is read: a `.env` in a parent directory is not. A variable already set in the environment wins over the same one in `.env`.

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | One of these two, except on [Bedrock](#amazon-bedrock) | — | Anthropic API key |
| `OPENAI_API_KEY` | One of these two, except on [Bedrock](#amazon-bedrock) | — | OpenAI API key |
| `ANTHROPIC_BASE_URL` | No | — | Endpoint for the Anthropic client, read by the Anthropic SDK — for a gateway in front of the API. While it is set, the check in [Provider and model must agree](#provider-and-model-must-agree) is skipped. An `http://` URL to a host other than `localhost` or a loopback address logs a warning once per endpoint, since prompts and the API key would cross the network unencrypted; it is not refused. |
| `LLM_MODEL` | No | `claude-sonnet-5-5`, or `openai/gpt-4o` when `OPENAI_API_KEY` selected the provider | LLM model identifier |
| `LLM_PROVIDER` | No | Auto-detected | `anthropic`, `litellm`, or `bedrock` (which resolves to `litellm`). Any other value, `openai` included, fails with `Unknown LLM provider` when an LLM client is created. Unset, the provider is detected in this order: an `LLM_MODEL` starting `bedrock/` selects Bedrock (see [Amazon Bedrock](#amazon-bedrock)); then `ANTHROPIC_API_KEY` selects `anthropic`; then `OPENAI_API_KEY` selects `litellm`, with `LLM_MODEL` defaulting to `openai/gpt-4o` and `LLM_MODEL_FAST` to `openai/gpt-4o-mini`. With both keys set Anthropic wins, so to use OpenAI then, set `LLM_PROVIDER=litellm` and `LLM_MODEL=openai/<model>`. |
| `QDRANT_URL` | No | `http://localhost:6333` | Qdrant URL. Required for `--embed`, the semantic cache, and document connectors. |
| `QDRANT_API_KEY` | For any Qdrant not on loopback | — | Required whenever `QDRANT_URL` is not on this machine — Qdrant Cloud and the Docker Compose stack included. On this machine means the host is `localhost` or a loopback address (`127.0.0.0/8`, `::1`); any other host, a DNS name that merely starts with `127.`, and a URL whose host cannot be read all count as remote. A URL without a scheme (`qdrant:6333`) is read as `http://qdrant:6333`, as the Qdrant client reads it. Without the key the Qdrant client refuses to connect rather than talk to a remote vector store anonymously. Set the same key on the Qdrant service (`QDRANT__SERVICE__API_KEY`). A key sent over plain `http` to another host is logged as a warning, since it crosses the network unencrypted. |
| `NLQ_STATE_DIR` | No | `~/.nlqueries` | Root for everything NLQueries keeps between runs. The paths below default under it; set them individually to override. Set this when the home directory is not writable — a read-only root filesystem otherwise stops the embedding server starting, and with it every natural-language query. The Docker image sets it to `/data/nlqueries` |
| `KB_PATH` | No | `~/.nlqueries/knowledge_base` | Local path for exported knowledge base files |
| `CONNECTORS_FILE` | No | `~/.nlqueries/connectors.yaml` | Path to the connector registry |
| `CAPSULES_DIR` | No | `~/.nlqueries/capsules` | Path to saved query capsules |
| `FEEDBACK_DIR` | No | `~/.nlqueries/feedback` | Path to feedback JSONL files |
| `NOTION_API_TOKEN` | Only for Notion sync | — | Notion integration token |
| `CONFLUENCE_API_TOKEN` | Only for Confluence sync | — | The API token for `doc-sync-confluence`; the site URL and user are its `--base-url` and `--username` options |
| `HF_TOKEN` | No | — | Hugging Face token — avoids rate limits on the one-time embedding model download. See [troubleshooting.md](troubleshooting.md#w2--hugging-face-hub-unauthenticated-requests). |
| `EMBED_SERVER_PORT` | No | `8765` | The one port setting for the embedding daemon: the port it listens on (`embed-server start`, `python -m nlqueries.embeddings.embed_server`) and the port clients, `embed-server status` and `health` look for it on. `embed-server start --port` overrides the listening port and warns when it differs from this, since clients would not find it |
| `NLQ_EMBED_MODEL` | No | `all-MiniLM-L6-v2` | The sentence-transformers model, by hub name or local path, used by both the daemon and in-process embedding. It must produce 384-dimension vectors; a model of another width is refused |
| `EMBED_SERVER_REQUIRED` | No | `false` | When `true`, a call the daemon cannot answer fails instead of loading the model in the calling process. Leave it off for the CLI; set it on a long-running server, where the fallback loads the model into that process and keeps it there |
| `EMBED_CLIENT_TIMEOUT_SECONDS` | No | `2.0` | How long a client waits for the embedding daemon before giving up on a request |
| `EMBED_SERVER_MAX_CONCURRENCY` | No | `min(4, CPU count)` | How many embeddings the daemon computes at once. `1` serialises them |
| `EMBED_SERVER_TORCH_THREADS` | No | `1` | Intra-op thread count for the `torch` backend inside the daemon. `0` leaves torch's own default |
| `LOG_LEVEL` | No | — | Log level of the CLI, the MCP server and the embedding daemon: `DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL`. Unset, the MCP server and the daemon log at `INFO`, and the CLI sets up no logging, so only warnings and errors reach stderr. Set, the CLI writes log records at that level and above to stderr. Any other value is ignored, with a warning |
| `NLQ_LOG_FORMAT` | No | `text` | Log line format of the embedding daemon, `text` or `json`, when it runs in the foreground (`embed-server start --foreground`, or `python -m nlqueries.embeddings.embed_server`, whose `--log-format` overrides it). Any other value is treated as `text`, with a warning |
| `NLQ_AUDIT_LOG` | No | — | A file that receives every audit event (authorization decisions and query executions) as one JSON line. Opened append-only for each event and created at mode `0600`. A symbolic link at the path is not followed (where the platform supports `O_NOFOLLOW`), anything but a regular file (a FIFO, a device) is refused without blocking, a file with more than one hard link is refused, a file owned by another user is refused, and on POSIX an existing file that group or others can reach is narrowed to `0600` and refused if that fails. A file that is refused or cannot be written gets no events and is reported on the `nlqueries.audit.sink` logger with the reason and the number of events it has not received: at the first, and again at the 10th, 100th, 1000th and so on, then once more with the total when writing to it works again. The events still go to `nlqueries.audit`, and the query is not failed. Unset, the events still go to the `nlqueries.audit` logger. See [Audit events](#audit-events). |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | No | — | Not read by NLQueries. Its OpenTelemetry instrumentation installs no tracer provider or exporter of its own, so spans and metrics go nowhere unless the process runs under an OpenTelemetry SDK setup that configures one (for example `opentelemetry-instrument`), which is what reads this variable. Set on its own, it exports nothing. See [Observability](observability.md#opentelemetry-traces-and-metrics) |
| `LLM_MAX_OUTPUT_TOKENS` | No | `4096` | Tokens an answer may generate, and the two derived budgets: a SQL correction or candidate set gets half (floor 512), intent classification and follow-up resolution an eighth (floor 200). At the default that is 4096 / 2048 / 512. The classification floor binds until this exceeds **1600**, so lowering it to, say, 1536 moves the answer and correction budgets but leaves classification at 200. The default was 1024 until the defaults moved to Claude Sonnet 5.5 and Haiku 5.5, which think before answering when a question calls for it; at 1024 Sonnet 5.5 can run out part-way through its SQL. The allowance is a ceiling: tokens are billed as generated. It does **not** govern knowledge-base column descriptions, which size their own allowance from the column count (`kb_generator`), nor the CLI and MCP health checks. **Raise it for a reasoning model** — such a model bills its private reasoning from the same allowance and spends it first, so a budget sized for the answer alone returns a short answer, or nothing at all on the smallest calls. Producing nothing now raises `OutputBudgetExhausted`, naming the model and the budget it spent, rather than returning an empty string that is indistinguishable from a model with nothing to say. Follow-up resolution and knowledge-base column descriptions are the exception: they log it at `WARNING` and carry on with a degraded result, because an unresolved follow-up is still worth asking. A non-positive value is ignored with a warning and the default applies. |
| `LLM_EFFORT` | No | `low` | How hard Claude 5 and later think before answering, sent as `output_config.effort`: `low`, `medium`, `high`, `xhigh` or `max`. Blank sends nothing, so the model uses its own default (`high` for Sonnet 5.5, `medium` for Haiku 5.5). An unrecognised value is ignored with a warning and `low` applies. Sent to Claude only through the Anthropic API (the Anthropic provider, or LiteLLM with an `anthropic/` or bare `claude-` model), and never to an earlier Claude model. Bedrock, Vertex and OpenRouter keep each model's default effort. Claude 5 and later also reject a `temperature`, so none is sent to them. DeepSeek through LiteLLM (a `deepseek/` model) is sent it as `reasoning_effort`, which takes `low`, `high` or `max`: `low` and `medium` send `low`, `high` sends `high`, `xhigh` and `max` send `max`. It goes in `extra_body`, since LiteLLM turns a top-level `reasoning_effort` for DeepSeek into thinking on or off and drops the level; none is sent while `LLM_THINKING` is `off`. Other models are sent nothing. |
| `LLM_THINKING` | No | `on` | Whether a DeepSeek model (`deepseek/` through LiteLLM) thinks before answering: `on` or `off`. DeepSeek thinks by default; `off` sends `thinking: {"type": "disabled"}` and no `LLM_EFFORT`. `on` sends nothing and leaves it to the model. Blank means `on`; an unrecognised value is ignored with a warning and `on` applies. Other models are sent nothing either way. |
| `LLM_TIMEOUT_SECONDS` | No | `180` | How long an LLM call may go **without producing data** before it fails. Not a rescue from an unbounded wait: both SDKs already impose 600s of their own (`anthropic`'s `DEFAULT_TIMEOUT`, litellm's `COMPLETION_HTTP_FALLBACK_SECONDS`), so what this buys is 600 → 180, configurable, plus a single `LLMTimeout` to catch in place of one exception type per SDK. **It is a read deadline, not a cap on the call**: httpx applies it per read, so it fires after this many seconds of *silence*. On the streaming paths the Anthropic Messages API sends periodic `ping` events that each reset the clock, so a stream that keeps trickling can outlast it — do not size an upstream budget as though this capped a question. 180 is deliberately generous rather than tight: a reasoning model on a long schema can take a minute or more before its first token, and a deadline that fires on those trades a hang for a broken deployment. Not to be confused with `CONNECTOR_STATEMENT_TIMEOUT_SECONDS`, which is the budget for the SQL, not for the model. A non-positive, non-finite or unparseable value is ignored with a warning — `inf` included, since it parses and would hand the SDKs a deadline that never fires. |
| `LLM_MODEL_FAST` | No | `claude-haiku-5-5` (Anthropic) / `openai/gpt-4o-mini` (OpenAI) / same as `LLM_MODEL` (Bedrock) | Smaller/cheaper model used for short-output auxiliary calls (intent classifier, follow-up resolver). Set to any LiteLLM-supported model string. On Bedrock it defaults to your `LLM_MODEL` rather than to a Haiku id that does not exist there — correct but not cheap, so set a Bedrock Haiku id to get the cheap tier back. |
| `QUERY_HISTORY_LIMIT` | No | `500` | Maximum number of useful queries returned by `process-history` after filtering. Override per-run with `--max-queries`. |
| `EMBED_BACKEND` | No | `torch` | Embedding backend for the embed-server daemon. `torch` uses sentence-transformers/PyTorch (no extra deps); `onnx` uses ONNX Runtime via `optimum[onnxruntime]` (faster cold start, no PyTorch). |
| `NLQ_SCHEMA_FORMAT` | No | `compact` | Schema format injected into the system prompt. `compact` uses M-Schema (`【Table】 …`) — fewer tokens; `verbose` uses full Markdown (`### Table: …`) — backward compatible. |
| `NLQ_SELF_CONSISTENCY` | No | `off` | Self-consistency for the LLM repair step, which runs only when the generated SQL fails validation and the mechanical (no-LLM) repair fails too. `off` — one repair call; `hard` — three repair candidates in parallel when the repair is judged hard (at least two complexity keywords such as `join` or `group by` in the correction request, which includes the failing SQL, or more than 20 tables in the knowledge base), otherwise one call; `all` — always three candidates. The candidates are normalised and the most frequent one wins, preferring those that validate. |
| `NLQ_CACHE_ANSWER_THRESHOLD` | No | `0.97` | Cosine similarity threshold for the Tier 1 answer cache. Questions within this distance of a cached question get the cached answer directly. |
| `NLQ_CACHE_TEMPLATE_THRESHOLD` | No | `0.90` | Cosine similarity threshold for the Tier 2 template cache. |
| `NLQ_CACHE_MAX_QUESTION_CHARS` | No | `500` | Longest question that may be **written** to the semantic cache; `0` disables the limit. A question over it is still answered normally, just not stored. Long questions are near-unique, so they cost a write and are rarely hit again — and a padded prompt injection has exactly that shape. |
| `NLQ_CACHE_ANSWER_TIERS` | No | `0,1,2` | Which cache tiers may serve an answer. `0` is an exact match on the normalised question; `1` is cosine similarity over stored answers; `2` binds the question's entities into a stored SQL template. Set to `0` for a sensitive agent to keep exact-match caching without the tiers that can serve one user's answer to another user's differently-worded question. |
| `NLQ_CACHE_COSINE_CANDIDATES` | No | `5` | How many neighbours the Tier 1 and Tier 2 searches fetch. The caller's cache context is matched inside the query, so these candidates already belong to the caller; the window exists because a candidate can still fail signature verification, the TTL, or entity binding. Raising it helps only where those failures are common. |
| `NLQ_CACHE_PRUNE_INTERVAL_SECONDS` | No | `3600` | How often a cache collection is swept for points past the TTL; `0` disables it. Nothing else deletes from the cache — the TTL is applied on read — so without a sweep a collection grows for the life of the agent, and expired points go on consuming the `NLQ_CACHE_COSINE_CANDIDATES` slots a lookup scans. The sweep runs on write, at most once per collection per interval **per process** — so a long-lived server sweeps hourly, while a CLI invocation sweeps once. The delete is issued without waiting, and `created_at` is indexed as a datetime on collections created from now on, so the work is a range query rather than a scan. |
| `NLQ_LITERAL_GROUNDING` | No | `true` | Before a generated statement runs, check each string literal it compares a column to (`=`, `IN`, `LIKE` without a wildcard) against the values the column stores. A literal that exists as written is left. One that matches exactly one stored value ignoring case and surrounding spaces, or failing that also ignoring punctuation at either end (`.,;:!?'"()[]{}-`, for literals of three characters or more once that is removed), is replaced with it, so `status = 'legal'` runs as `status = 'Legal'`; the substitution is recorded in the answer's provenance as `literals_grounded`. One that matches none, or several, is left, and if the statement goes on to an LLM repair, the repair is told which values are nearby. Only text columns the knowledge base knows, never keys, never a column whose name marks it as personal data (the names `export-kb` refuses to sample: names containing password, passwd, pwd, secret, token, hash, salt, ssn, credential, api key, access key, private key, signing key, encryption key, ssh key, master key, license or licence key, credit card, card number, cvv, iban, account number, national id, passport, tax id, dob, birth, email, phone, address, first, last, full, given or family name, session, cookie or auth (but not author or authority), authorization or authorisation, otp at the start of a word (`user_otp`, not `footprint`), totp or hotp at the start of a word with no letter after (`totp_code`, not `hotpot`), account no not followed by a letter (not `account_notes`), and the whole word pin (`card_pin`, not `shipping`); bare key, name and account, and a `_key` suffix, do not count, so `keyword`, `product_name` and a join key such as `customer_key` are not refused for their name), and literals up to 100 characters. Where the knowledge base holds a column's complete value list (`values_complete`, stored by `export-kb`), the list answers: a literal in it needs no query, and one that is not gets a single exact-match query, in case it was stored since the export, before the list supplies the match or the nearby values; the list was read with no row restriction, so under a per-request row filter the lookups go through the filter instead. A table the knowledge base records as larger than `NLQ_LITERAL_GROUNDING_MAX_ROWS` is not queried. Read-only and bounded per pass: at most 8 literals, 1 s a lookup, 3 s in all, and a lookup still running at 3 s is abandoned rather than waited for. A statement gets at most two passes, one before a repair and one on the repaired statement, so up to 16 literals and 6 s of lookups; outcomes are cached per database for 15 minutes, and only across requests no row-restricting wrapper separates; any error skips the check. Runs whatever the execution policy, for any agent with a registered connector, reading through the agent's connector (`open_connector_for_agent`, so a per-request wrapper such as a row filter applies) with read-only permission; the generated statement is never run by it. Set `false` to turn it off. |
| `NLQ_LITERAL_GROUNDING_MAX_ROWS` | No | `1000000` | Literal grounding sends no query to a table the knowledge base records as having more rows than this. Without a stored value list, a literal that matches nothing costs up to four queries, most of them scans: slow on a large table, and billed on a warehouse. A column of such a table whose complete value list the knowledge base holds is still grounded, from the list alone, where no row filter applies. A table whose size the knowledge base does not record is queried. `0` removes the cap. The size is the one the connector reports, and the generic SQLAlchemy connector (MySQL, MariaDB, Oracle and the rest) reports none, so on those engines this cap skips nothing; nor do the others report one in every case (a Postgres table never analysed, a Redshift role without access to `SVV_TABLE_INFO`). `export-kb` says how many tables report no row count. |
| `NLQ_LITERAL_GROUNDING_REPAIR` | No | `true` | When literal grounding leaves a valid statement with a literal no stored value matches, and offers nearby values for it, make one LLM call with the same cached system prompt, the question, the statement and grounding's notes, asking for the same statement with only the literal corrected if a stored value clearly matches the question. The answer replaces the statement only if it validates and differs from it in nothing but the values of the literals grounding flagged, those no stored value matches or several do: a changed column, join, aggregate or number, an added or removed predicate, or any other literal changed, such as a date format or a pattern elsewhere in the statement, keeps the original. The replacement is then grounded again, with no second call. Never loops. Recorded in the answer's provenance as `literal_repair` (`attempted`, `changed`, `notes`, and `reason`, which is `rejected: non-literal change` when the answer changed more than literals and `rejected: changed a literal grounding did not flag` when it changed another literal); the call's tokens are counted like any other. Notes with no nearby values make no call. Only meaningful when `NLQ_LITERAL_GROUNDING` is on. Set `false` to turn it off. |
| `NLQ_GLOSSARY_QUESTION_SCOPED` | No | `false` | When `true`, glossary terms are injected **per question** — only terms the question mentions, plus their [hierarchy](cli-reference.md#glossary-hierarchy) ancestors and descendants (depth 3) — instead of the whole glossary in the cached static prompt. Business rules are always injected in full. Off by default (the full glossary ships in the static block, exactly as before). |
| `NLQ_SEND_RESULT_ROWS_TO_LLM` | No | `true` | Whether executed result rows go to the LLM when a hybrid (SQL plus documents) answer is synthesised. `true`: up to 20 rows are sent, with every column whose name marks it as personal data (the names `export-kb` refuses to sample, listed under `NLQ_LITERAL_GROUNDING`) left out and named as withheld. That check reads the column name the query returns, not the column it came from, so `SELECT email AS contact` is not caught and its values are sent. `false`: the synthesis prompt gets the column names and the row count only, and no value from any row; this is the setting that holds whatever the columns are called. While values are withheld (the setting off, or every column withheld), a failed query is reported only as having failed, without the database's error text, since a driver's error can quote a value (`Duplicate entry 'bob@example.com'`). When no query ran (generate-only mode, a refused or failed execution, a statement that failed validation), the generated statement is sent whatever this says: it is not data. The result table the reader sees is the same either way. See [What leaves this machine](#what-leaves-this-machine). |
| `NLQ_MAX_DOCUMENT_EXPANDED_BYTES` | No | `134217728` (128 MiB) | Largest total **uncompressed** size permitted in a zip-based document (`.xlsx`, `.docx`). An upload is size-limited and a request body is capped, but neither bounds what comes out of a parser, and the size on disk says nothing about it: measured on ordinary content, 200k rows by 10 columns is 5.2 MiB on disk and 108.8 MiB expanded. Parsing costs a multiple of the expanded size (a `.docx` of 250 MiB of ordinary paragraphs peaked at about 2 GiB), so at 128 MiB an ordinary document fits within `NLQ_DOCUMENT_PARSE_MEMORY_BYTES`; raise the two together. Also the largest `.txt` or `.md` file ingested. Read from the zip directory before any parsing, so nothing is decompressed to find out. A malformed value falls back to the default rather than failing the ingest. |
| `NLQ_MAX_DOCUMENT_EXPANSION_RATIO` | No | `100` | Largest expanded-over-compressed ratio permitted in a zip-based document, checked for the archive as a whole and, for any member that expands to 4 MiB or more, for that member on its own. Complementary to the byte ceiling rather than redundant with it: a small file that expands 1000× is a bomb even when the result fits under the ceiling, and a large file that expands 2× is ordinary. Checked per member as well because one member that expands 1000× beside a large incompressible one (an image, say) averages out under the limit for the whole archive. Smaller members are exempt from the per-member check because the risk is the expanded size: a few MiB costs little memory however well it compressed, and ordinary documents have such members (20,000 identical Word paragraphs give a `document.xml` of about 4 MB at about 250×). 100× is roughly three times the highest ratio measured on real content. |
| `NLQ_MAX_DOCUMENT_ROWS` | No | `50000` | Most spreadsheet rows extracted from one document, across all its sheets. The expansion limits above read what a zip archive declares about itself; this is counted while reading, so it bounds a document whatever its directory claims. Measured: 200,000 rows by 10 columns took 87 seconds to ingest from a 5.2 MiB file. At the connector's batch size of 50, 50,000 rows is already a thousand chunks from one document. |
| `NLQ_MAX_EXTRACTION_SECONDS` | No | `120` | Wall-clock budget for extracting one document, in every connector. The general bound on cost: per-format counts are guesses about it, and a 3,000-page PDF of scanned images and a 30-page one of dense tables differ by orders of magnitude. The count caps (`NLQ_MAX_PDF_BYTES`, `NLQ_MAX_PDF_PAGES`, `NLQ_MAX_DOCUMENT_ROWS`) do not replace it; they refuse what is known to be too large before the work starts. Checked between units of work (a page, every 1,000 Word paragraphs, a section, a batch of rows), so it bounds a long document rather than interrupting one slow page. A ceiling on background ingestion, not a latency target. For PDF, Word and Excel, which parse in a child process (`NLQ_DOCUMENT_PARSE_ISOLATION`), the child is also killed when this budget plus 15 seconds of start-up has passed, which does stop a parser stuck inside one page. |
| `NLQ_DOCUMENT_PARSE_ISOLATION` | No | `true` | Parse PDF, Word and Excel documents in a child process, which is killed at the extraction deadline and runs under `NLQ_DOCUMENT_PARSE_MEMORY_BYTES` and `NLQ_DOCUMENT_PARSE_CPU_SECONDS`. A document that runs out of memory is refused as too complex; one stopped by the deadline or the CPU limit raises the same timeout as the in-process budget. A child whose caller is gone stops itself at the extraction deadline rather than at once; on Linux and macOS it runs in a session of its own, so a signal sent to the caller's process group (a supervisor's `kill -TERM -<pgid>`, a terminal hang-up) does not reach it. Set `false` only where starting a process per document is not possible; the in-process limits still apply then. |
| `NLQ_DOCUMENT_PARSE_MEMORY_BYTES` | No | `2147483648` (2 GiB) | Memory limit for the child that parses a document: an address-space limit (`RLIMIT_AS`) on Linux and macOS, which counts virtual memory, the interpreter and its libraries included, and a Job Object process memory limit on Windows, which counts committed memory. A lower hard limit already set on the process (`ulimit -v`) is kept. A container's memory limit is not one: it is enforced by the kernel, which kills the child when the container as a whole reaches it. That is reported as an extraction timeout, which a caller may retry, since the document may not be the cause. Raise it together with `NLQ_MAX_DOCUMENT_EXPANDED_BYTES`. |
| `NLQ_DOCUMENT_PARSE_CPU_SECONDS` | No | `180` | CPU-time limit for that child: `RLIMIT_CPU` on Linux and macOS, the Job Object's user-mode CPU-time limit on Windows (the child and what it starts). Above `NLQ_MAX_EXTRACTION_SECONDS`, which normally ends a slow parse first. |
| `NLQ_MAX_PDF_BYTES` | No | `104857600` (100 MiB) | Largest PDF file, on disk, that is ingested. A PDF is not a zip, so the expansion limits never see it; its size is known before anything is parsed, so an oversized file is refused before the parser opens it. Raises `DocumentTooComplexError`. |
| `NLQ_MAX_PDF_PAGES` | No | `2000` | Most pages in one PDF that are extracted. Read from the page count the document declares before any page is extracted, and counted again before extraction since a malformed file can misstate it. A longer document is refused with `DocumentTooComplexError`; split it and ingest the parts. |
| `NLQ_MAX_CONFLUENCE_PAGES` | No | `5000` | Most pages one `doc-sync-confluence` run reads from a space. A sync stores what it read as the whole space and removes the rest, so a space over the cap is refused, with nothing stored, rather than read in part. |
| `NLQ_MAX_CONFLUENCE_BYTES` | No | `209715200` (200 MiB) | Most bytes of page body (HTML storage format, UTF-8) one `doc-sync-confluence` run reads. Counted as each page body arrives, so a space of a few very large pages is bounded too. Refused rather than truncated, as above. |
| `NLQ_MAX_NOTION_BLOCKS` | No | `10000` | Most blocks read from one Notion page by `doc-sync-notion`, counting every block, text or not. A page over the cap is refused, with nothing stored, rather than stored in part. |
| `NLQ_MAX_NOTION_BYTES` | No | `52428800` (50 MiB) | Most bytes of block text (UTF-8) read from one Notion page by `doc-sync-notion`. The block cap bounds how many blocks are read, not how much text each carries, so a page of a few thousand very large blocks would pass it; this counts the text kept from each block as it is read. A page over the cap is refused, with nothing stored, rather than stored in part. |
| `NLQ_CACHE_SIGNING_KEY` | Where more than one process shares a cache | Generated | Key that signs semantic-cache entries; an entry that does not verify is treated as a miss. Unset, a key is generated and written to `cache_signing_key` under `NLQ_STATE_DIR`, or held in memory for the life of the process if that cannot be written. Every process that shares a cache must use the same key, so a deployment with more than one (separate containers, say) must set this or `NLQ_CACHE_SIGNING_KEY_FILE`. A key shorter than 32 bytes, from any source, stops the MCP server at startup; elsewhere it disables the cache (every lookup misses) and is logged as an error. Generate one with `openssl rand -hex 32` |
| `NLQ_CACHE_SIGNING_KEY_FILE` | No | — | Path to a file holding the signing key, for mounted secrets. Used when `NLQ_CACHE_SIGNING_KEY` is not set; a file that cannot be read, or is empty, is an error rather than a fallback |
| `NLQ_CACHE_ENCRYPTION` | No | `false` | Encrypt cached answers and the text of document chunks with AES-256-GCM before storing them in Qdrant. Governs writing only: text stored before it was turned on is still read, and encrypted text is decrypted whenever a key is available, with the setting on or off. Vectors, questions, SQL, document names and chunk metadata are stored as before. See [qdrant-setup.md](qdrant-setup.md#encrypting-what-qdrant-stores) |
| `NLQ_CACHE_ENCRYPTION_KEY` | Where more than one process writes to the same Qdrant | Generated | Key for `NLQ_CACHE_ENCRYPTION`. Use a random one, such as the output of `openssl rand -hex 32`: the AES key is derived from it with HKDF, which does nothing to slow guessing, so a short or chosen key can be guessed offline against a copy of Qdrant's data. A configured key under 32 characters is warned about. Unset, a key is generated on the first encrypted write and kept in `cache_encryption_key` under `NLQ_STATE_DIR`, created at mode `0600`. If it cannot be written the write fails, rather than encrypting with a key that would be lost on restart. Every process that shares a Qdrant must use the same key. Encrypted text that cannot be decrypted is a cache miss or a skipped chunk, with one warning per process for each of the two |
| `NLQ_CACHE_ENCRYPTION_KEY_FILE` | No | — | Path to a file holding that key, for mounted secrets. Used when `NLQ_CACHE_ENCRYPTION_KEY` is not set; a file that cannot be read, or is empty, is an error rather than a fallback |
| `CONNECTOR_STATEMENT_TIMEOUT_SECONDS` | No | `120` | Statement timeout for a query that does not pass its own, so a slow query fails instead of hanging. `0` disables it. How each connector enforces it, and which cannot, is in [database-hardening.md](database-hardening.md#what-each-connector-enforces) |
| `CONNECTOR_MAX_FETCH_ROWS` | No | `10000` | Most rows a connector materialises from one query — a bound on memory, not the number of rows an answer shows |
| `CONNECTOR_MAX_RESULT_BYTES` | No | `67108864` (64 MiB) | Approximate memory ceiling for one result set; whichever of this and `CONNECTOR_MAX_FETCH_ROWS` binds first stops the read |
| `CONNECTOR_CACHE_ENABLED` | No | `true` | Reuse open connectors across queries instead of building one per query. `false` builds one per query |
| `CONNECTOR_CACHE_TTL_SECONDS` | No | `900` | How long a cached connector is reused before it is rebuilt |
| `CONNECTOR_CACHE_MAX_ENTRIES` | No | `32` | How many connectors the cache keeps open; the least recently used is evicted first |
| `CONNECTOR_POOL_SIZE` | No | `5` | Connections each PostgreSQL connector keeps open |
| `CONNECTOR_MAX_OVERFLOW` | No | `5` | Extra connections a PostgreSQL connector may open under burst and then discard. Each cached connector can hold up to `CONNECTOR_POOL_SIZE` + `CONNECTOR_MAX_OVERFLOW` connections; check that against the database's own connection limit before raising either |
| `REDSHIFT_SOCKET_TIMEOUT_SECONDS` | No | `CONNECTOR_STATEMENT_TIMEOUT_SECONDS` + 30, at least 60 (`150` by default; `0` when the statement timeout is `0`) | Socket timeout for a Redshift connection, which bounds every read, not just the connect. Keep it above the statement timeout; see [connectors.md](connectors.md#amazon-redshift) |
| `SNOWFLAKE_QUEUE_TIMEOUT_SECONDS` | No | `30` | How much of a Snowflake query's statement timeout it may spend waiting in the warehouse queue before Snowflake cancels it, capped at half the timeout; the rest bounds how long it runs, so the two together stay within `CONNECTOR_STATEMENT_TIMEOUT_SECONDS`. Raise it for a warehouse that queues under load or resumes from suspension. |
| `BIGQUERY_MAX_BYTES_BILLED` | No | unset (no cap) | Most bytes one BigQuery query job may bill, set on every job that answers a question as `maximum_bytes_billed`. BigQuery fails a job that would bill more before it runs, and the answer reports the error. Schema and query-history reads are not capped. A value that is not a whole number greater than zero makes every BigQuery query fail with an error naming the variable, rather than run without the cap; other commands, `health` and `doctor` included, still run; see [connectors.md](connectors.md#bigquery-cost-cap) |
| `MCP_TRANSPORT` | No | `stdio` (the Docker image: `sse`) | Transport for `python -m nlqueries.mcp_server` and the Docker image: `stdio`, `sse` or `streamable-http`. `nlqueries mcp-server start` takes `--sse` instead |
| `MCP_HOST` | No | `127.0.0.1` (the Docker image: `0.0.0.0`) | Bind address for a networked transport, for `python -m nlqueries.mcp_server` and the Docker image. A wildcard address is refused unless `NLQ_ALLOW_INSECURE_BIND=1`; `mcp-server start` takes `--host`. `docker-compose.yml` fixes it to `0.0.0.0`, whatever `.env` says, and publishes the port on `BIND_ADDR` |
| `MCP_PORT` | No | `8000` (the Docker image: `8080`) | Port for a networked transport, for `python -m nlqueries.mcp_server` and the Docker image; `mcp-server start` takes `--port`. In `docker-compose.yml` it is the host port the server is published on, and the container keeps 8080 |
| `NLQ_MCP_MAX_QUESTION_CHARS` | No | `4000` | Longest `question` an MCP tool accepts, in characters. A longer one is refused before the tool runs. Not a positive whole number: the default, with a warning. Applies on every MCP transport; see [mcp-authentication.md](mcp-authentication.md#input-size) |
| `NLQ_MCP_MAX_SQL_CHARS` | No | `8000` | Longest SQL an MCP tool accepts (`submit_feedback`'s `generated_sql` and `corrected_sql`), in characters, with the same rules |
| `NLQ_MCP_*`, `NLQ_ALLOW_UNAUTHENTICATED_MCP`, `NLQ_ALLOW_INSECURE_BIND` | For a networked MCP transport | — | Authentication, grants, limits and the bind override for the MCP server; a networked transport will not start without authentication. See [mcp-authentication.md](mcp-authentication.md) |

### Provider and model must agree

`LLM_PROVIDER` selects a client and `LLM_MODEL` names a model, and the two can
disagree. Where they do and the request would leave for the wrong vendor, it is
refused before anything is sent.

`AnthropicClient` does not inspect the model id, so an Anthropic provider with
`LLM_MODEL=deepseek/deepseek-chat` would transmit the system prompt, the schema
and the question to api.anthropic.com and only then report the model missing —
the data has gone by the time the error arrives. That combination now raises at
the first LLM call, naming both settings. The same applies to a `bedrock/`
model reached through any other provider.

`anthropic/claude-sonnet-5-5` is accepted and treated as `claude-sonnet-5-5`.
The prefixed spelling is the form LiteLLM documents and a reasonable thing to
write; passing it through unchanged had the same shape of failure, since no
Anthropic model id contains a slash.

Two cases are deliberately not refused:

- **A bare model name.** `deepseek-chat` names no provider in its own text, and
  deciding which one it belongs to needs LiteLLM's model registry — DeepSeek's
  keys are spelled bare there, Mistral's prefixed. This path does not consult
  it. If you are running a non-Anthropic model, either give it its provider
  prefix or set `LLM_PROVIDER=litellm`.
- **A gateway.** When `api_base` or `ANTHROPIC_BASE_URL` points the client
  somewhere other than api.anthropic.com, the check is skipped: the premise no
  longer holds and your gateway may expect prefixed ids.

A gateway URL, from `ANTHROPIC_BASE_URL` or an `api_base` override for either
client, that uses `http://` to a host other than `localhost` or an IP address
that is loopback (`127.0.0.0/8`, `::1`) logs one warning per endpoint per
process. The request still goes out: an in-cluster gateway on plain http is a
legitimate setup. Hostnames are not resolved, so a name that points at
127.0.0.1 still warns. The warning shows only the scheme, host and port, never
credentials in the URL.

### Amazon Bedrock

Bedrock is reached through LiteLLM, so there is no separate provider to install.
Set `LLM_MODEL` to a `bedrock/` model id and NLQueries routes there:

```bash
LLM_MODEL=bedrock/us.anthropic.claude-sonnet-4-6
AWS_REGION=us-east-1
```

**The region never fails loudly, so set it deliberately.** LiteLLM takes the
first of these that it finds: a region passed in code, one embedded in a model
ARN, `AWS_REGION_NAME`, `AWS_REGION`, then whatever a `boto3.Session()` resolves
(`AWS_DEFAULT_REGION`, or a profile in `~/.aws/config`). If none of them
produce a region it does **not** raise — it falls back to a built-in
`us-west-2`. A deployment that forgot the region therefore calls a real region
it never chose, where the model is very likely not enabled, and the error says
the model was not found. `AWS_REGION` above is honoured; `AWS_REGION_NAME` beats
it if both are set.

Authentication is boto3's, not an API key. NLQueries passes no AWS credentials of
its own, so the ordinary chain applies: `AWS_ACCESS_KEY_ID` /
`AWS_SECRET_ACCESS_KEY` (and `AWS_SESSION_TOKEN` for temporary credentials) in
the environment, a shared `~/.aws/config` profile, or — on EC2, ECS or EKS — the
instance profile, task role, or IRSA role of the host. On a machine that already
has an AWS role, no credential settings are needed at all.

Three things to check when it does not work, because each fails differently:

- **The model must be enabled** for your account in that region, under Bedrock →
  Model access in the AWS console. Until it is, every call returns AccessDenied
  however correct the credentials are.
- **The region must match** where the model is enabled. A model id from one
  region simply does not exist in another.
- **Newer Anthropic models require a cross-region inference profile**, which is
  the `us.` or `eu.` prefix inside the id — `bedrock/us.anthropic.claude-…`, not
  `bedrock/anthropic.claude-…`. The bare id is rejected as not found.

The IAM role needs `bedrock:InvokeModel` and
`bedrock:InvokeModelWithResponseStream` on the model or inference-profile ARN.

The CLI understands this too: `health`, `process-history --annotate` and
`export-kb --describe-columns` treat a Bedrock configuration as a credential and
do not ask for an API key. `health` still makes a real call, so it reports
whether the role can actually invoke the model rather than only whether one is
configured.

`LLM_PROVIDER=bedrock` also resolves to LiteLLM, but it does **not** replace the
model id: naming the provider does not choose a model, and the default
`LLM_MODEL` is an Anthropic id that LiteLLM would route to Anthropic. So
`LLM_PROVIDER=bedrock` requires an `LLM_MODEL` beginning `bedrock/`, and without
one the first LLM call fails with a message saying so. It fails there rather
than at startup deliberately — `connect`, `extract-schema` and the diagnostics
you would run to find the problem never touch an LLM, and should still work
while you are fixing it. Setting only the model works on its own and is the
shorter path.

**The model decides, wherever it is set.** A `bedrock/` id selects Bedrock even
when nothing names a provider — before the request is built, not at the API.
That matters because the Anthropic client does not reject a `bedrock/` model id:
it would send the system prompt, the schema and the question to
`api.anthropic.com` and only then report that the model does not exist. For a
deployment that chose Bedrock to keep traffic inside its AWS account, the data
would already have left. Explicitly naming a provider *other* than Bedrock
alongside a `bedrock/` model is refused as the contradiction it is, whether it
was named in `LLM_PROVIDER` or by a host application.

**Set `LLM_MODEL_FAST` too, or unset it.** Short auxiliary calls — intent
classification and follow-up resolution — use the fast tier, and they carry the
user's question and the conversation history. On a Bedrock deployment a
`LLM_MODEL_FAST` that is not a `bedrock/` id is refused, because it would
otherwise be routed to whichever provider it does name, sending exactly that
content outside your AWS account while the main model kept working. A `.env`
carrying `LLM_MODEL_FAST=claude-haiku-4-5-20251001` from a previous Anthropic
setup is the usual way to arrive here. Leaving it unset is fine: it then follows
`LLM_MODEL`.

The same ordering applies to key-based detection: the `bedrock/` prefix is
checked before `ANTHROPIC_API_KEY`, so a Bedrock deployment that still has an
Anthropic key in its environment for something else keeps going to Bedrock.

---

**Windows note:** `~` in default paths resolves to `C:\Users\<YourUsername>` in PowerShell. To set a variable for the current session use `$env:VAR = "value"`; to persist it, use **System Properties → Environment Variables** or add it to your PowerShell profile.

---

## Audit events

Two events go to the `nlqueries.audit` logger, and to `NLQ_AUDIT_LOG` when it is set: `authorization`, one per decision the MCP guard makes (see [What gets recorded](mcp-authentication.md#what-gets-recorded)), and `query_execution`, one per statement sent to a database for an agent. Each is a JSON object with sorted keys, rendered on one line: control characters and non-ASCII text are escaped, so no field can break a record across lines. The log message is the event's words (`authorization allowed`, `query executed`) followed by that JSON, and the record is attached as `extra={"audit": ...}` as well.

Every event carries:

| Field | Meaning |
|---|---|
| `schema_version` | `1`. Raised when a field is renamed, removed or changes meaning; adding a field does not raise it, so ignore fields you do not know. |
| `event` | `authorization` or `query_execution` |
| `request_id` | 32 hex characters. One per MCP tool call, shared by its decision and the statements it ran; one per CLI invocation. A caller that uses the package as a library gets a new one per event. |
| `surface` | `mcp`, `cli` or `library` |
| `at` | When it happened, ISO 8601 in UTC |

A `query_execution` event adds:

| Field | Meaning |
|---|---|
| `principal`, `source` | The caller's subject and how it was established (`oidc`, `static-token`, `local`, `anonymous`), or `null` when not known: the library binds none |
| `agent_id` | The agent the statement ran for |
| `connector_type` | The connector's registered type, such as `postgres` or `sqlite` |
| `dialect` | The SQL grammar of that database, as the SQL policy names it (`tsql` for SQL Server); `null` when it cannot be told |
| `sql_sha256` | SHA-256 of the statement's UTF-8 bytes, the same digest the SQL policy records for a decision |
| `sql_bytes` | The statement's length in UTF-8 bytes |
| `executed_sql_sha256`, `executed_sql_bytes` | Present only when a connector wrapper rewrote the statement before it reached the database (a row filter's predicate, say) and reported the text it sent as `QueryResult.executed_sql`: the digest and length of that text. Absent when the statement ran as asked. |
| `policy_version` | The SQL policy's version |
| `policy_decision` | `allow` or `deny`: what the SQL policy decides about this statement for this dialect. It is recorded, not enforced, at this point; `null` when there is no dialect to decide for |
| `outcome` | `ok`; `error` when the database returned an error; `exception` when the call raised, a refusal to execute included |
| `row_count`, `truncated` | As the connector reported them; `null` when the call raised |
| `duration_ms` | Wall-clock time of the call |
| `error_class` | The exception's class name when the call raised, else `null`. Never its message, which can quote the statement. |

An event never carries the question, the SQL text, the rows or a credential. Recording never fails a query: if an execution event cannot be built or published, that is logged on `nlqueries.audit.sink` with its traceback at `WARNING` the first time a cause (the exception's type and the line that raised it) is seen, and at `DEBUG` after that. Every statement run through the connector an agent's query opens is recorded, whether it comes from a fresh question, a cache replay or literal grounding's value lookups. Statements the CLI runs on a connector it opens itself (the `health` check's `SELECT 1`, `export-kb`'s table sampling) are not.

---

## What leaves this machine

Every LLM call sends text to the provider set by `LLM_PROVIDER`. What it carries:

| Data | Sent | Notes |
|---|---|---|
| The question | On every question | With the last few conversation turns when the question reads as a follow-up |
| Schema | On every SQL question | Table and column names, types, keys, and descriptions, each description on one line and capped at 300 characters; business rules and glossary definitions on one line and capped at 1,000 (a rule or definition cut at the cap is logged once, as a warning naming its start) |
| Sample values | On every SQL question | Up to five stored values per column (all of them where the knowledge base holds the complete list), each quoted; a value over 80 characters is left out, not cut, and a complete list that loses one is shown as samples. Never for a column whose name marks it as personal data |
| Result rows | On the hybrid path only | Up to 20 rows, personal-data columns withheld by the name the query returns them under, so a renamed one (`SELECT email AS contact`) is sent (the row count is still sent when every column is withheld); with `NLQ_SEND_RESULT_ROWS_TO_LLM=false`, the column names and the row count only, which is the control that holds whatever a column is called. While values are withheld, a failure is reported without its error text. When no query ran, the generated statement instead. A SQL-only answer sends no rows |
| Document excerpts | On document and hybrid questions | Each excerpt in a delimited block with its source label. Only the block markers inside it are neutralised; the rest of the text, `&`, `<` and `>` included, is sent as written |

Descriptions, sample values, glossary entries, example queries and excerpts are framed in the prompt as reference data that never changes its instructions. Up to 1,000 characters of a database error go back to the model when it corrects a failed statement, and literal grounding can send nearby stored values to its repair step (see `NLQ_LITERAL_GROUNDING_REPAIR`).

Document and hybrid answers are not written to the semantic cache, so an answer built from a document is never replayed after that document has changed. Entries written by an earlier version expire with the TTL.

---

## Docker Compose services and volumes

Running `docker compose up` from the repository root starts:

| Service | Port | Purpose |
|---|---|---|
| `qdrant` | 6333 (REST), 6334 (gRPC) | Vector store for embeddings, semantic cache, document search |
| `nlqueries-core` | 8080 | MCP server (requires `NLQ_MCP_STATIC_TOKEN`, sent as a bearer token) + CLI engine |

Both services also require `QDRANT_API_KEY` in `.env`: Qdrant is started with it as its API key and the app uses it to connect, and `docker compose up` refuses to start while it or `NLQ_MCP_STATIC_TOKEN` is unset. The ports are published on `127.0.0.1` unless `BIND_ADDR` says otherwise. `QDRANT_HTTP_PORT`, `QDRANT_GRPC_PORT` and `MCP_PORT` in `.env` move the host side of each mapping; inside the compose network the services keep the ports above.

| Volume | Persists |
|---|---|
| `qdrant-data` | Qdrant collections across restarts |
| `nlqueries-data` | Knowledge bases, connector config, capsules, and feedback (mounted at `/data/nlqueries` in the container) |

Run CLI commands inside the container with `docker exec -it nlqueries-core nlqueries <command>` — when connecting to a database on your host machine, use `--host host.docker.internal`.
