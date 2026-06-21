"""
nlqueries.orchestrator.orchestrator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Single-agent orchestrator: loads a knowledge base, assembles the LLM prompt,
streams natural-language reasoning, then yields a validated SQL final chunk.

Public API
----------
``Orchestrator``
    Call ``handle_question(question, agent_id, dialect)`` to get an async
    token stream ending in a structured JSON SQL chunk.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import AsyncGenerator
from typing import Any

import yaml

from nlqueries import config
from nlqueries.llm import get_llm_client
from nlqueries.orchestrator.prompt_assembly import assemble_prompt
from nlqueries.orchestrator.sql_generation import generate_sql
from nlqueries.telemetry import get_tracer, query_counter, query_latency


class Orchestrator:
    """Single-agent orchestrator for NLQueries.

    Each invocation of ``handle_question`` independently:

    1. Loads the agent's YAML knowledge base.
    2. Streams a natural-language reasoning response from the LLM.
    3. Calls :func:`generate_sql` to produce a validated SQL statement and
       yields it as a structured JSON final chunk.

    Multi-agent routing is out of scope for v1 and will be added in Phase 2
    (Sprint 12).
    """

    async def handle_question(
        self,
        question: str,
        agent_id: str,
        dialect: str = "postgres",
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
            system_prompt, user_prompt = assemble_prompt(
                question,
                kb,
                top_k_capsules=5,
                collection=collection,
            )

            # Step 1: stream natural-language reasoning -----------------------
            llm = get_llm_client()
            for token in llm.stream(system_prompt, user_prompt):
                yield token

            # Step 2: generate validated SQL, yield as structured final chunk -
            result = generate_sql(question, kb, dialect)
            span.set_attribute("sql_valid", result.is_valid)
            span.set_attribute("attempt_count", result.attempt_count)
            span.set_attribute("intent_type", "sql")

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
                }
            )

    def _load_knowledge_base(self, agent_id: str) -> dict[str, Any]:
        """Load and return the YAML knowledge base for *agent_id*.

        Sanitises *agent_id* to a safe filename stem (colons and slashes
        become underscores) then reads
        ``{config.KB_PATH}/{safe_agent_id}.yaml``.

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
        return yaml.safe_load(kb_file.read_text(encoding="utf-8")) or {}
