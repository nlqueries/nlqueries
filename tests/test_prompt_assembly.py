"""Tests for nlqueries.orchestrator.prompt_assembly.assemble_prompt.

Key design invariants verified here:
- ``assemble_prompt()`` returns ``AssembledPrompt``, not a tuple.
- ``static_system`` contains the FULL schema (all tables, deterministic order),
  business context, and SQL format rules — no per-question content.
- ``dynamic_context`` contains Qdrant-retrieved capsules and relevant-table hints.
- ``user_question`` is the exact question string.
- Schema is NEVER filtered from ``static_system`` by Qdrant results; filtering
  is replaced by a "most relevant tables" hint in ``dynamic_context``.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import patch

from nlqueries.orchestrator.prompt_assembly import (
    AssembledPrompt,
    assemble_prompt,
    assemble_prompt_async,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_TABLE = {
    "name": "orders",
    "description": "Customer purchase records",
    "row_count": 5000,
    "columns": [
        {"name": "id", "type": "INTEGER", "description": "Primary key"},
        {"name": "customer_id", "type": "INTEGER", "description": "FK to customers"},
        {"name": "total", "type": "DECIMAL", "description": "Order total"},
        {"name": "created_at", "type": "TIMESTAMP", "description": ""},
    ],
}

_TABLE2 = {
    "name": "customers",
    "description": "Registered customers",
    "row_count": 1200,
    "columns": [
        {"name": "id", "type": "INTEGER", "description": "Primary key"},
        {"name": "email", "type": "VARCHAR", "description": "Customer email"},
    ],
}

_CAPSULES = [
    {
        "intent": "Count orders per customer",
        "template": "SELECT customer_id, COUNT(*) FROM orders GROUP BY customer_id",
    },
    {
        "intent": "Total revenue last month",
        "template": "SELECT SUM(total) FROM orders WHERE created_at >= :start_date",
    },
    {
        "intent": "Find customer by email",
        "template": "SELECT * FROM customers WHERE email = :email_varchar",
    },
    {
        "intent": "Recent orders",
        "template": "SELECT * FROM orders ORDER BY created_at DESC LIMIT :limit_int",
    },
    {
        "intent": "High-value orders",
        "template": "SELECT * FROM orders WHERE total > :threshold_decimal",
    },
    {"intent": "All customers", "template": "SELECT * FROM customers"},
]


def _make_kb(
    tables: list[dict[str, Any]] | None = None,
    capsules: list[dict[str, Any]] | None = None,
    glossary: list[Any] | None = None,
    rules: list[Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": {"tables": tables if tables is not None else [_TABLE]},
        "business_context": {
            "glossary": glossary or [],
            "rules": rules or [],
        },
        "query_capsules": capsules if capsules is not None else _CAPSULES[:3],
    }


def _full_text(prompt: AssembledPrompt) -> str:
    """Return the concatenated text of all prompt fields for content checks."""
    return "\n".join([prompt.static_system, prompt.dynamic_context, prompt.user_question])


# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------


def test_assemble_prompt_returns_assembled_prompt() -> None:
    prompt = assemble_prompt("How many orders?", _make_kb())
    assert isinstance(prompt, AssembledPrompt)


def test_user_question_equals_input() -> None:
    question = "What is the total revenue?"
    prompt = assemble_prompt(question, _make_kb())
    assert prompt.user_question == question


def test_user_content_returns_question() -> None:
    """user_content() must return the bare question — no dynamic context duplication."""
    question = "How many orders?"
    prompt = assemble_prompt(question, _make_kb())
    assert prompt.user_content() == question


def test_static_system_is_non_empty() -> None:
    prompt = assemble_prompt("Show all customers", _make_kb())
    assert prompt.static_system.strip()


def test_system_blocks_returns_list() -> None:
    prompt = assemble_prompt("question", _make_kb())
    blocks = prompt.system_blocks(cache=False)
    assert isinstance(blocks, list)
    assert len(blocks) >= 1
    assert all(isinstance(b, dict) for b in blocks)


def test_system_blocks_with_cache_attaches_cache_control() -> None:
    prompt = assemble_prompt("question", _make_kb())
    blocks = prompt.system_blocks(cache=True)
    # First block (static_system) must have cache_control
    assert blocks[0].get("cache_control") == {"type": "ephemeral"}


def test_system_blocks_without_cache_has_no_cache_control() -> None:
    prompt = assemble_prompt("question", _make_kb())
    blocks = prompt.system_blocks(cache=False)
    for block in blocks:
        assert "cache_control" not in block


# ---------------------------------------------------------------------------
# static_system — schema section (full schema, deterministic, cacheable)
# ---------------------------------------------------------------------------


def test_static_system_includes_table_name() -> None:
    prompt = assemble_prompt("show orders", _make_kb())
    assert "orders" in prompt.static_system


def test_static_system_includes_table_description() -> None:
    prompt = assemble_prompt("show orders", _make_kb())
    assert "Customer purchase records" in prompt.static_system


def test_static_system_includes_column_names() -> None:
    prompt = assemble_prompt("show orders", _make_kb())
    assert "customer_id" in prompt.static_system
    assert "total" in prompt.static_system


def test_static_system_includes_column_types() -> None:
    prompt = assemble_prompt("show orders", _make_kb())
    assert "DECIMAL" in prompt.static_system


def test_static_system_includes_column_description_when_present() -> None:
    with patch("nlqueries.config.SCHEMA_FORMAT", "verbose"):
        prompt = assemble_prompt("show orders", _make_kb())
    assert "Primary key" in prompt.static_system


def test_static_system_includes_row_count() -> None:
    with patch("nlqueries.config.SCHEMA_FORMAT", "verbose"):
        prompt = assemble_prompt("show orders", _make_kb())
    assert "5,000" in prompt.static_system


def test_static_system_handles_empty_schema() -> None:
    kb = _make_kb(tables=[])
    prompt = assemble_prompt("any question", kb)
    assert "Database Schema" not in prompt.static_system


def test_static_system_always_includes_all_tables() -> None:
    """ALL tables must appear in static_system regardless of Qdrant results.

    This is required for prompt-cache stability — the static_system must be
    byte-identical across questions for the same agent.
    """
    kb = _make_kb(tables=[_TABLE, _TABLE2])
    mock_hits = [{"table_name": "orders", "type": "table", "score": 0.9}]
    with (
        patch("nlqueries.config.SCHEMA_FORMAT", "verbose"),
        patch("nlqueries.embeddings.qdrant_store.search_schema", return_value=mock_hits),
        patch("nlqueries.embeddings.qdrant_store.search", return_value=[]),
        patch("nlqueries.embeddings.embedder.embed_text", return_value=[0.1] * 384),
    ):
        prompt = assemble_prompt("show orders", kb, collection="col")

    assert "### Table: orders" in prompt.static_system
    assert "### Table: customers" in prompt.static_system


# ---------------------------------------------------------------------------
# dynamic_context — capsule section (per-question, Qdrant or fallback)
# ---------------------------------------------------------------------------


def test_dynamic_context_includes_capsule_intent_from_kb() -> None:
    """When Qdrant is unavailable, KB capsules appear in dynamic_context."""
    prompt = assemble_prompt("count orders", _make_kb())
    assert "Count orders per customer" in prompt.dynamic_context


def test_dynamic_context_includes_capsule_template_from_kb() -> None:
    prompt = assemble_prompt("count orders", _make_kb())
    assert "SELECT customer_id, COUNT(*)" in prompt.dynamic_context


def test_top_k_capsules_limits_included_count() -> None:
    kb = _make_kb(capsules=_CAPSULES)  # 6 capsules total
    prompt = assemble_prompt("question", kb, top_k_capsules=3)
    assert prompt.dynamic_context.count("Intent:") == 3


def test_top_k_capsules_default_is_five() -> None:
    kb = _make_kb(capsules=_CAPSULES)  # 6 capsules
    prompt = assemble_prompt("question", kb)
    assert prompt.dynamic_context.count("Intent:") == 5


def test_top_k_capsules_one() -> None:
    kb = _make_kb(capsules=_CAPSULES)
    prompt = assemble_prompt("question", kb, top_k_capsules=1)
    assert prompt.dynamic_context.count("Intent:") == 1


def test_dynamic_context_empty_when_no_capsules() -> None:
    kb = _make_kb(capsules=[])
    prompt = assemble_prompt("question", kb)
    assert "Example Queries" not in prompt.dynamic_context


# ---------------------------------------------------------------------------
# static_system — business context section
# ---------------------------------------------------------------------------


def test_static_system_includes_glossary_term() -> None:
    kb = _make_kb(glossary=[{"term": "ARR", "definition": "Annual Recurring Revenue"}])
    prompt = assemble_prompt("what is ARR?", kb)
    assert "ARR" in prompt.static_system
    assert "Annual Recurring Revenue" in prompt.static_system


def test_static_system_includes_business_rules() -> None:
    kb = _make_kb(rules=["Never show deleted records", "Always filter by active status"])
    prompt = assemble_prompt("show data", kb)
    assert "Never show deleted records" in prompt.static_system


def test_static_system_omits_business_context_when_empty() -> None:
    kb = _make_kb(glossary=[], rules=[])
    prompt = assemble_prompt("show data", kb)
    assert "Business Context" not in prompt.static_system


# ---------------------------------------------------------------------------
# static_system — instructions always present
# ---------------------------------------------------------------------------


def test_static_system_always_includes_instructions() -> None:
    prompt = assemble_prompt("any question", _make_kb())
    assert "## Instructions" in prompt.static_system
    assert "SELECT" in prompt.static_system


def test_static_system_includes_sql_sentinel_instructions() -> None:
    """LLM must be instructed to use <sql>...</sql> markers (Phase 3B)."""
    prompt = assemble_prompt("question", _make_kb())
    assert "<sql>" in prompt.static_system
    assert "</sql>" in prompt.static_system


# ---------------------------------------------------------------------------
# Qdrant collection — search calls
# ---------------------------------------------------------------------------


def test_assemble_prompt_calls_search_schema_when_collection_given() -> None:
    kb = _make_kb(tables=[_TABLE, _TABLE2])
    mock_hits = [{"table_name": "orders", "type": "table", "score": 0.95}]

    with (
        patch("nlqueries.embeddings.qdrant_store.search_schema", return_value=mock_hits) as mock_ss,
        patch("nlqueries.embeddings.qdrant_store.search", return_value=[]),
        patch("nlqueries.embeddings.embedder.embed_text", return_value=[0.1] * 384),
    ):
        assemble_prompt("how many orders?", kb, collection="my_collection")

    mock_ss.assert_called_once()
    call_args = mock_ss.call_args
    assert call_args.args[0] == "my_collection"
    assert call_args.args[1] == "how many orders?"
    assert call_args.kwargs.get("top_k") == 10


def test_assemble_prompt_calls_search_when_collection_given() -> None:
    kb = _make_kb(capsules=_CAPSULES)

    with (
        patch("nlqueries.embeddings.qdrant_store.search_schema", return_value=[]),
        patch("nlqueries.embeddings.qdrant_store.search", return_value=[]) as mock_s,
        patch("nlqueries.embeddings.embedder.embed_text", return_value=[0.1] * 384),
    ):
        assemble_prompt("count orders", kb, top_k_capsules=3, collection="my_col")

    mock_s.assert_called_once()
    call_args = mock_s.call_args
    assert call_args.args[0] == "my_col"
    assert call_args.args[1] == "count orders"
    assert call_args.kwargs.get("top_k") == 3


def test_assemble_prompt_uses_qdrant_capsules_when_returned() -> None:
    from nlqueries.processing.parameterizer import QueryCapsule

    qdrant_capsule = QueryCapsule(
        template_sql="SELECT COUNT(*) FROM orders",
        placeholders=[],
        tables=["orders"],
        columns=[],
        frequency=10,
        auto_description="qdrant fallback",
        intent="Qdrant-found capsule",
    )

    kb = _make_kb(capsules=_CAPSULES)
    with (
        patch("nlqueries.embeddings.qdrant_store.search_schema", return_value=[]),
        patch("nlqueries.embeddings.qdrant_store.search", return_value=[qdrant_capsule]),
        patch("nlqueries.embeddings.embedder.embed_text", return_value=[0.1] * 384),
    ):
        prompt = assemble_prompt("count orders", kb, collection="col")

    assert "Qdrant-found capsule" in prompt.dynamic_context


def test_assemble_prompt_falls_back_to_kb_when_qdrant_raises() -> None:
    kb = _make_kb(capsules=_CAPSULES[:2])

    with (
        patch(
            "nlqueries.embeddings.qdrant_store.search_schema",
            side_effect=RuntimeError("down"),
        ),
        patch(
            "nlqueries.embeddings.qdrant_store.search",
            side_effect=RuntimeError("down"),
        ),
    ):
        prompt = assemble_prompt("question", kb, collection="col")

    assert "Count orders per customer" in prompt.dynamic_context


def test_assemble_prompt_does_not_call_qdrant_without_collection() -> None:
    kb = _make_kb()

    with patch(
        "nlqueries.embeddings.qdrant_store.search_schema",
        side_effect=RuntimeError("should not call"),
    ) as mock_ss:
        assemble_prompt("question", kb)

    mock_ss.assert_not_called()


def test_qdrant_schema_hit_adds_relevant_tables_hint_to_dynamic_context() -> None:
    """Qdrant schema hits add a hint in dynamic_context; static_system keeps ALL tables."""
    kb = _make_kb(tables=[_TABLE, _TABLE2])
    mock_hits = [{"table_name": "orders", "type": "table", "score": 0.9}]
    with (
        patch("nlqueries.config.SCHEMA_FORMAT", "verbose"),
        patch("nlqueries.embeddings.qdrant_store.search_schema", return_value=mock_hits),
        patch("nlqueries.embeddings.qdrant_store.search", return_value=[]),
        patch("nlqueries.embeddings.embedder.embed_text", return_value=[0.1] * 384),
    ):
        prompt = assemble_prompt("show orders", kb, collection="col")

    # dynamic_context carries the relevance hint
    assert "orders" in prompt.dynamic_context
    # static_system always has both tables (not filtered)
    assert "### Table: orders" in prompt.static_system
    assert "### Table: customers" in prompt.static_system


def test_qdrant_schema_empty_hit_does_not_add_hint() -> None:
    kb = _make_kb(tables=[_TABLE, _TABLE2])

    with (
        patch("nlqueries.config.SCHEMA_FORMAT", "verbose"),
        patch("nlqueries.embeddings.qdrant_store.search_schema", return_value=[]),
        patch("nlqueries.embeddings.qdrant_store.search", return_value=[]),
        patch("nlqueries.embeddings.embedder.embed_text", return_value=[0.1] * 384),
    ):
        prompt = assemble_prompt("show something", kb, collection="col")

    # Both tables in static_system regardless
    assert "### Table: orders" in prompt.static_system
    assert "### Table: customers" in prompt.static_system


# ---------------------------------------------------------------------------
# Single-embed optimisation (vector passed to both searches)
# ---------------------------------------------------------------------------


def test_single_embed_call_passed_to_both_searches() -> None:
    """embed_text is called once; the pre-computed vector is reused for both searches."""
    kb = _make_kb()
    call_log: list[str] = []

    def fake_embed(text: str) -> list[float]:
        call_log.append(text)
        return [0.42] * 384

    with (
        patch("nlqueries.embeddings.embedder.embed_text", side_effect=fake_embed),
        patch("nlqueries.embeddings.qdrant_store.search_schema", return_value=[]) as mock_ss,
        patch("nlqueries.embeddings.qdrant_store.search", return_value=[]) as mock_s,
    ):
        assemble_prompt("my question", kb, collection="col")

    # embed_text called exactly once
    assert len(call_log) == 1
    # Both searches received the pre-computed vector
    assert mock_ss.call_args.kwargs.get("vector") == [0.42] * 384
    assert mock_s.call_args.kwargs.get("vector") == [0.42] * 384


def test_precomputed_vector_skips_embed_text_call() -> None:
    """When vector= is supplied, embed_text must NOT be called at all."""
    kb = _make_kb()
    precomputed = [0.99] * 384

    with (
        patch(
            "nlqueries.embeddings.embedder.embed_text",
            side_effect=AssertionError("embed_text must not be called when vector= is provided"),
        ),
        patch("nlqueries.embeddings.qdrant_store.search_schema", return_value=[]) as mock_ss,
        patch("nlqueries.embeddings.qdrant_store.search", return_value=[]) as mock_s,
    ):
        assemble_prompt("my question", kb, collection="col", vector=precomputed)

    assert mock_ss.call_args.kwargs.get("vector") == precomputed
    assert mock_s.call_args.kwargs.get("vector") == precomputed


def test_precomputed_vector_none_triggers_embed() -> None:
    """Without vector=, embed_text is still called (backward-compat)."""
    kb = _make_kb()

    with (
        patch("nlqueries.embeddings.embedder.embed_text", return_value=[0.1] * 384) as mock_embed,
        patch("nlqueries.embeddings.qdrant_store.search_schema", return_value=[]),
        patch("nlqueries.embeddings.qdrant_store.search", return_value=[]),
    ):
        assemble_prompt("my question", kb, collection="col")

    mock_embed.assert_called_once()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_knowledge_base_does_not_raise() -> None:
    prompt = assemble_prompt("question", {})
    assert isinstance(prompt, AssembledPrompt)
    assert prompt.user_question == "question"


def test_question_passed_through_unchanged() -> None:
    q = "How many customers signed up in 2024?"
    prompt = assemble_prompt(q, _make_kb())
    assert prompt.user_question == q


# ---------------------------------------------------------------------------
# assemble_prompt_async — concurrent variant (Phase 6C)
# ---------------------------------------------------------------------------


def test_assemble_prompt_async_matches_sync_output() -> None:
    """assemble_prompt_async must produce byte-identical output to assemble_prompt."""
    from nlqueries.processing.parameterizer import QueryCapsule

    kb = _make_kb(tables=[_TABLE, _TABLE2], capsules=_CAPSULES)
    schema_hits = [{"table_name": "orders", "type": "table", "score": 0.9}]
    verified_hits = [{"question": "count orders", "sql": "SELECT COUNT(*) FROM orders"}]
    capsule_hits = [
        QueryCapsule(
            template_sql="SELECT * FROM orders",
            placeholders=[],
            tables=["orders"],
            columns=[],
            frequency=3,
            auto_description="",
            intent="show orders",
        )
    ]

    with (
        patch("nlqueries.embeddings.qdrant_store.search_schema", return_value=schema_hits),
        patch("nlqueries.embeddings.qdrant_store.search", return_value=capsule_hits),
        patch(
            "nlqueries.orchestrator.prompt_assembly._search_verified",
            return_value=verified_hits,
        ),
        patch("nlqueries.embeddings.embedder.embed_text", return_value=[0.1] * 384),
    ):
        sync_prompt = assemble_prompt("show orders", kb, collection="col")
        async_prompt = asyncio.run(assemble_prompt_async("show orders", kb, collection="col"))

    assert async_prompt.static_system == sync_prompt.static_system
    assert async_prompt.dynamic_context == sync_prompt.dynamic_context
    assert async_prompt.user_question == sync_prompt.user_question


def test_assemble_prompt_async_runs_searches_concurrently() -> None:
    """The three dynamic-context searches must overlap, not run one after another."""
    kb = _make_kb()
    delay = 0.15

    def slow_search_schema(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        time.sleep(delay)
        return []

    def slow_search_verified(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        time.sleep(delay)
        return []

    def slow_search(*args: Any, **kwargs: Any) -> list[Any]:
        time.sleep(delay)
        return []

    with (
        patch("nlqueries.embeddings.qdrant_store.search_schema", side_effect=slow_search_schema),
        patch("nlqueries.embeddings.qdrant_store.search", side_effect=slow_search),
        patch(
            "nlqueries.orchestrator.prompt_assembly._search_verified",
            side_effect=slow_search_verified,
        ),
        patch("nlqueries.embeddings.embedder.embed_text", return_value=[0.1] * 384),
    ):
        start = time.perf_counter()
        asyncio.run(assemble_prompt_async("question", kb, collection="col"))
        elapsed = time.perf_counter() - start

    # Sequential would take >= 3 * delay (~0.45s); concurrent should take ~1 * delay.
    assert elapsed < 2 * delay


def test_assemble_prompt_async_one_search_failing_does_not_lose_others() -> None:
    """If one search raises, the other two results must still be used."""
    kb = _make_kb()
    schema_hits = [{"table_name": "orders", "type": "table", "score": 0.9}]

    with (
        patch("nlqueries.embeddings.qdrant_store.search_schema", return_value=schema_hits),
        patch(
            "nlqueries.embeddings.qdrant_store.search",
            side_effect=RuntimeError("qdrant unavailable"),
        ),
        patch(
            "nlqueries.orchestrator.prompt_assembly._search_verified",
            return_value=[],
        ),
        patch("nlqueries.embeddings.embedder.embed_text", return_value=[0.1] * 384),
    ):
        prompt = asyncio.run(assemble_prompt_async("question", kb, collection="col"))

    assert "orders" in prompt.dynamic_context


def test_assemble_prompt_async_truncates_capsules_after_verified_hits() -> None:
    """Capsule count must still be bounded by top_k - len(verified_hits), same as sync."""
    from nlqueries.processing.parameterizer import QueryCapsule

    kb = _make_kb()
    verified_hits = [
        {"question": "q1", "sql": "SELECT 1"},
        {"question": "q2", "sql": "SELECT 2"},
    ]
    capsule_hits = [
        QueryCapsule(
            template_sql=f"SELECT {i}",
            placeholders=[],
            tables=[],
            columns=[],
            frequency=1,
            auto_description="",
            intent=f"capsule {i}",
        )
        for i in range(5)
    ]

    with (
        patch("nlqueries.embeddings.qdrant_store.search_schema", return_value=[]),
        patch("nlqueries.embeddings.qdrant_store.search", return_value=capsule_hits),
        patch(
            "nlqueries.orchestrator.prompt_assembly._search_verified",
            return_value=verified_hits,
        ),
        patch("nlqueries.embeddings.embedder.embed_text", return_value=[0.1] * 384),
    ):
        prompt = asyncio.run(
            assemble_prompt_async("question", kb, top_k_capsules=3, collection="col")
        )

    # top_k=3, 2 verified hits -> only 1 capsule should remain, for 3 "Intent:" lines total.
    assert prompt.dynamic_context.count("Intent:") == 3


def test_assemble_prompt_async_single_embed_reused_across_all_three_searches() -> None:
    """embed_text is called once; all three concurrent searches reuse that vector."""
    kb = _make_kb()
    call_log: list[str] = []

    def fake_embed(text: str) -> list[float]:
        call_log.append(text)
        return [0.42] * 384

    with (
        patch("nlqueries.embeddings.embedder.embed_text", side_effect=fake_embed),
        patch("nlqueries.embeddings.qdrant_store.search_schema", return_value=[]) as mock_ss,
        patch("nlqueries.embeddings.qdrant_store.search", return_value=[]) as mock_s,
        patch("nlqueries.orchestrator.prompt_assembly._search_verified", return_value=[]) as mock_v,
    ):
        asyncio.run(assemble_prompt_async("my question", kb, collection="col"))

    assert len(call_log) == 1
    assert mock_ss.call_args.kwargs.get("vector") == [0.42] * 384
    assert mock_s.call_args.kwargs.get("vector") == [0.42] * 384
    assert mock_v.call_args.kwargs.get("vector") == [0.42] * 384


# ---------------------------------------------------------------------------
# extra_dynamic_context seam (enterprise Nexus injection point)
# ---------------------------------------------------------------------------

_NEXUS_SECTION = "## Join Paths (Nexus)\norders.customer_id = customers.id"


def test_extra_dynamic_context_appended() -> None:
    prompt = assemble_prompt("q", _make_kb(), extra_dynamic_context=_NEXUS_SECTION)
    assert _NEXUS_SECTION in prompt.dynamic_context


def test_extra_dynamic_context_none_is_noop() -> None:
    baseline = assemble_prompt("q", _make_kb())
    assert baseline.dynamic_context == assemble_prompt("q", _make_kb()).dynamic_context
    assert "Join Paths" not in baseline.dynamic_context


def test_extra_dynamic_context_does_not_touch_static_prefix() -> None:
    # The cached static prefix (schema) must be identical with/without extra —
    # otherwise injection would bust the Anthropic prompt cache.
    without = assemble_prompt("q", _make_kb())
    with_extra = assemble_prompt("q", _make_kb(), extra_dynamic_context=_NEXUS_SECTION)
    assert without.static_system == with_extra.static_system


def test_extra_dynamic_context_appears_in_system_blocks() -> None:
    prompt = assemble_prompt("q", _make_kb(), extra_dynamic_context=_NEXUS_SECTION)
    blocks = prompt.system_blocks(cache=True)
    assert any(_NEXUS_SECTION in b["text"] for b in blocks)
    # And it never lands in the cached (cache_control-bearing) static block.
    cached = [b for b in blocks if b.get("cache_control")]
    assert all(_NEXUS_SECTION not in b["text"] for b in cached)


def test_extra_dynamic_context_async_appended() -> None:
    prompt = asyncio.run(
        assemble_prompt_async("q", _make_kb(), extra_dynamic_context=_NEXUS_SECTION)
    )
    assert _NEXUS_SECTION in prompt.dynamic_context
