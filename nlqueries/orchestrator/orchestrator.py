"""
nlqueries.orchestrator.orchestrator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Single-agent orchestrator: loads a knowledge base, assembles the LLM prompt,
streams natural-language reasoning, then yields a validated SQL final chunk.

Phase 3B: a **single** ``astream()`` call handles both reasoning and SQL
generation.  The LLM is instructed to emit reasoning prose first, then wrap
the SQL in ``<sql>…</sql>`` sentinels.  Tokens before the opening tag are
yielded immediately (low TTFT); content inside the tags is accumulated and
passed to :func:`~nlqueries.orchestrator.sql_generation.validate_and_repair`.

Public API
----------
``Orchestrator``
    Call ``handle_question(question, agent_id, dialect)`` to get an async
    token stream ending in a structured JSON SQL chunk.
"""

from __future__ import annotations

import asyncio
import datetime
import decimal
import json
import re
import time
from collections.abc import AsyncGenerator
from typing import Any

import yaml

from nlqueries import config
from nlqueries.llm import get_llm_client
from nlqueries.orchestrator.prompt_assembly import assemble_prompt_async
from nlqueries.orchestrator.sql_generation import _extract_sql, validate_and_repair
from nlqueries.telemetry import get_tracer, query_counter, query_latency

# Module-level KB cache: path -> (mtime, parsed_dict).
# Invalidated automatically when the file's mtime changes (e.g. after export-kb).
_kb_cache: dict[str, tuple[float, dict[str, Any]]] = {}

# Sentinel tags injected into the system prompt (must match _SQL_FORMAT_RULES
# in prompt_assembly.py).
_OPEN_TAG = "<sql>"
_CLOSE_TAG = "</sql>"
# How many chars to hold back before flushing pre-tag content, to catch markers
# split across token boundaries (len("<sql>") - 1 == 4).
_HOLD = len(_OPEN_TAG) - 1
_MAX_RESULT_ROWS = 200  # cap rows returned to MCP / CLI callers


def _json_default(obj: Any) -> Any:
    """Coerce DB-driver types that json.dumps can't handle."""
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    if isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class Orchestrator:
    """Single-agent orchestrator for NLQueries.

    Each invocation of ``handle_question`` independently:

    1. Loads the agent's YAML knowledge base (memoised by mtime).
    2. Assembles an :class:`~nlqueries.orchestrator.prompt_assembly.AssembledPrompt`
       — a stable cacheable ``static_system`` block plus a per-question
       ``dynamic_context`` block.
    3. Makes **one** ``astream()`` call.  Reasoning tokens are yielded
       immediately; SQL content (inside ``<sql>…</sql>``) is collected silently
       and then validated / repaired.
    4. Yields a structured JSON final chunk with the validated SQL.
    """

    async def handle_question(
        self,
        question: str,
        agent_id: str,
        dialect: str = "postgres",
        *,
        question_vector: list[float] | None = None,
        timeout_seconds: float | None = None,
    ) -> AsyncGenerator[str, None]:
        """Translate *question* into a reasoning stream followed by a SQL chunk.

        Yields tokens from the LLM's natural-language reasoning response first,
        then yields a single JSON string::

            {"type": "sql", "sql": "...", "is_valid": true,
             "validation_error": null, "dialect": "postgres", "attempt_count": 1}

        Args:
            question: The natural-language question from the user.
            agent_id: Identifier of the agent whose knowledge base to use.
                      Must match a file under ``config.KB_PATH``.
            dialect:  SQL dialect for generation and validation.
                      One of ``"postgres"``, ``"snowflake"``, ``"bigquery"``.
                      Defaults to ``"postgres"``.
            timeout_seconds: Forwarded to ``connector.execute_query`` (Task
                      26.5 — Sprint 26) so the database itself aborts a
                      runaway query rather than it running orphaned after
                      the caller has given up. Does not bound the LLM call
                      above — see the module-level note in ``sync_runner``.

        Yields:
            String tokens from the LLM reasoning response, then a final JSON
            chunk with the validated SQL result.

        Raises:
            FileNotFoundError: When no knowledge base file exists for
                               *agent_id*.
        """
        tracer = get_tracer()
        start_ms = time.perf_counter() * 1000
        with tracer.start_as_current_span("orchestrator.handle_question") as span:
            span.set_attribute("agent_id", agent_id)
            span.set_attribute("dialect", dialect)

            kb = self._load_knowledge_base(agent_id)
            collection = f"agent_{agent_id}_schema"
            prompt = await assemble_prompt_async(
                question,
                kb,
                top_k_capsules=5,
                collection=collection,
                vector=question_vector,
            )

            llm = get_llm_client()
            # Pass cache_control blocks when the provider supports them.
            system = prompt.system_blocks(cache=llm.supports_prompt_caching)

            # ------------------------------------------------------------------
            # Single streaming call — split on <sql>…</sql> sentinels.
            # Tokens before <sql> are yielded to the caller immediately (TTFT).
            # Content inside the markers is collected silently for validation.
            # all_tokens is kept for the fallback extraction path.
            # ------------------------------------------------------------------

            pending: str = ""  # chars buffered to detect cross-token markers
            sql_buf: str = ""  # SQL content accumulated inside <sql>…</sql>
            in_sql: bool = False
            found_sql: bool = False
            all_tokens: list[str] = []

            async for token in llm.astream(system, prompt.user_content()):
                all_tokens.append(token)

                if in_sql:
                    sql_buf += token
                    close_idx = sql_buf.find(_CLOSE_TAG)
                    if close_idx != -1:
                        sql_buf = sql_buf[:close_idx]
                        in_sql = False
                        found_sql = True
                    continue  # never yield SQL tokens

                # Not yet inside <sql> — buffer and check for opening tag.
                pending += token
                open_idx = pending.find(_OPEN_TAG)
                if open_idx != -1:
                    # Yield everything before the opening tag.
                    before = pending[:open_idx]
                    if before:
                        yield before
                    # Everything after <sql> goes into the SQL buffer.
                    after_tag = pending[open_idx + len(_OPEN_TAG) :]
                    if after_tag.startswith("\n"):
                        after_tag = after_tag[1:]
                    close_idx = after_tag.find(_CLOSE_TAG)
                    if close_idx != -1:
                        # Both tags arrived in the same accumulated buffer.
                        sql_buf = after_tag[:close_idx]
                        in_sql = False
                        found_sql = True
                    else:
                        sql_buf = after_tag
                        in_sql = True
                    pending = ""
                else:
                    # No opening tag yet — flush safe portion (all but last
                    # _HOLD chars), holding back enough to detect a split tag.
                    safe_len = max(0, len(pending) - _HOLD)
                    if safe_len:
                        yield pending[:safe_len]
                        pending = pending[safe_len:]

            # Flush any remaining pre-SQL content.
            if not in_sql and pending:
                yield pending

            # Fallback: no <sql>…</sql> markers in the response — extract via
            # regex / keyword scan from the full concatenated output.
            if not found_sql:
                sql_buf = _extract_sql("".join(all_tokens))

            # ------------------------------------------------------------------
            # Validate (and repair if needed) the extracted SQL.
            # validate_and_repair() reuses the same `system` blocks so Anthropic
            # prompt-cache tokens are credited on the repair call too.
            # ------------------------------------------------------------------
            result = await validate_and_repair(
                sql_buf.strip(),
                kb,
                dialect,
                llm,
                system,
            )

            span.set_attribute("sql_valid", result.is_valid)
            span.set_attribute("attempt_count", result.attempt_count)
            span.set_attribute("intent_type", "sql")

            # Execute the validated SQL and capture rows for the response.
            sql_table: dict[str, Any] = {}
            if result.is_valid:
                from nlqueries.connectors.loader import open_connector_for_agent  # noqa: PLC0415

                connector = None
                try:
                    connector = await asyncio.to_thread(open_connector_for_agent, agent_id)
                    if connector is not None:
                        qr = await asyncio.to_thread(
                            connector.execute_query, result.sql, timeout_seconds
                        )
                        sql_table = {
                            "columns": qr.columns,
                            "rows": qr.rows[:_MAX_RESULT_ROWS],
                            "row_count": qr.row_count,
                            "execution_time_ms": qr.execution_time_ms,
                            "error": qr.error,
                        }
                        span.set_attribute("row_count", qr.row_count)
                except Exception as exc:  # noqa: BLE001
                    sql_table = {"error": str(exc)}

            elapsed_ms = time.perf_counter() * 1000 - start_ms
            query_counter.add(1, {"dialect": dialect, "agent_type": "sql"})
            query_latency.record(elapsed_ms, {"dialect": dialect, "agent_type": "sql"})

            yield json.dumps(
                {
                    "type": "sql",
                    "sql": result.sql,
                    "is_valid": result.is_valid,
                    "validation_error": result.validation_error,
                    "dialect": result.dialect,
                    "attempt_count": result.attempt_count,
                    "sql_table": sql_table or None,
                },
                default=_json_default,
            )

    def _load_knowledge_base(self, agent_id: str) -> dict[str, Any]:
        """Load and return the YAML knowledge base for *agent_id*.

        Sanitises *agent_id* to a safe filename stem (colons and slashes
        become underscores) then reads
        ``{config.KB_PATH}/{safe_agent_id}.yaml``.

        The result is memoized in a module-level cache keyed by file path.
        On each call the file's mtime is checked; if unchanged the cached
        dict is returned without a disk read or YAML parse.  Callers must
        treat the returned dict as read-only.

        Raises:
            FileNotFoundError: When the knowledge base file does not exist.
        """
        safe_id = re.sub(r"[^\w.-]", "_", agent_id)
        kb_file = config.KB_PATH / f"{safe_id}.yaml"
        if not kb_file.exists():
            raise FileNotFoundError(
                f"No knowledge base found for agent '{agent_id}'. "
                f"Run 'nlqueries export-kb {agent_id}' first."
            )
        mtime = kb_file.stat().st_mtime
        key = str(kb_file)
        cached = _kb_cache.get(key)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        kb: dict[str, Any] = yaml.safe_load(kb_file.read_text(encoding="utf-8")) or {}
        _kb_cache[key] = (mtime, kb)
        return kb
