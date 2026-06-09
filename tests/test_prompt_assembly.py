"""Tests for nlqueries.orchestrator.prompt_assembly.assemble_prompt."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from nlqueries.orchestrator.prompt_assembly import assemble_prompt

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


# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------


def test_assemble_prompt_returns_tuple_of_two_strings() -> None:
    system, user = assemble_prompt("How many orders?", _make_kb())
    assert isinstance(system, str)
    assert isinstance(user, str)


def test_user_prompt_equals_question() -> None:
    question = "What is the total revenue?"
    _, user = assemble_prompt(question, _make_kb())
    assert user == question


def test_system_prompt_is_non_empty() -> None:
    system, _ = assemble_prompt("Show all customers", _make_kb())
    assert system.strip()


# ---------------------------------------------------------------------------
# Schema section
# ---------------------------------------------------------------------------


def test_system_prompt_includes_table_name() -> None:
    system, _ = assemble_prompt("show orders", _make_kb())
    assert "orders" in system


def test_system_prompt_includes_table_description() -> None:
    system, _ = assemble_prompt("show orders", _make_kb())
    assert "Customer purchase records" in system


def test_system_prompt_includes_column_names() -> None:
    system, _ = assemble_prompt("show orders", _make_kb())
    assert "customer_id" in system
    assert "total" in system


def test_system_prompt_includes_column_types() -> None:
    system, _ = assemble_prompt("show orders", _make_kb())
    assert "DECIMAL" in system


def test_system_prompt_includes_column_description_when_present() -> None:
    system, _ = assemble_prompt("show orders", _make_kb())
    assert "Primary key" in system


def test_system_prompt_includes_row_count() -> None:
    system, _ = assemble_prompt("show orders", _make_kb())
    assert "5,000" in system


def test_system_prompt_handles_empty_schema() -> None:
    kb = _make_kb(tables=[])
    system, _ = assemble_prompt("any question", kb)
    assert "Database Schema" not in system


# ---------------------------------------------------------------------------
# Capsule section — top-k selection
# ---------------------------------------------------------------------------


def test_system_prompt_includes_capsule_intent() -> None:
    system, _ = assemble_prompt("count orders", _make_kb())
    assert "Count orders per customer" in system


def test_system_prompt_includes_capsule_template() -> None:
    system, _ = assemble_prompt("count orders", _make_kb())
    assert "SELECT customer_id, COUNT(*)" in system


def test_top_k_capsules_limits_included_count() -> None:
    kb = _make_kb(capsules=_CAPSULES)  # 6 capsules total
    system, _ = assemble_prompt("question", kb, top_k_capsules=3)
    assert system.count("Intent:") == 3


def test_top_k_capsules_default_is_five() -> None:
    kb = _make_kb(capsules=_CAPSULES)  # 6 capsules
    system, _ = assemble_prompt("question", kb)
    assert system.count("Intent:") == 5


def test_top_k_capsules_one() -> None:
    kb = _make_kb(capsules=_CAPSULES)
    system, _ = assemble_prompt("question", kb, top_k_capsules=1)
    assert system.count("Intent:") == 1


def test_system_prompt_handles_empty_capsules() -> None:
    kb = _make_kb(capsules=[])
    system, _ = assemble_prompt("question", kb)
    assert "Example Queries" not in system


# ---------------------------------------------------------------------------
# Business context section
# ---------------------------------------------------------------------------


def test_system_prompt_includes_glossary_term() -> None:
    kb = _make_kb(glossary=[{"term": "ARR", "definition": "Annual Recurring Revenue"}])
    system, _ = assemble_prompt("what is ARR?", kb)
    assert "ARR" in system
    assert "Annual Recurring Revenue" in system


def test_system_prompt_includes_business_rules() -> None:
    kb = _make_kb(rules=["Never show deleted records", "Always filter by active status"])
    system, _ = assemble_prompt("show data", kb)
    assert "Never show deleted records" in system


def test_system_prompt_omits_business_context_when_empty() -> None:
    kb = _make_kb(glossary=[], rules=[])
    system, _ = assemble_prompt("show data", kb)
    assert "Business Context" not in system


# ---------------------------------------------------------------------------
# Instructions section always present
# ---------------------------------------------------------------------------


def test_system_prompt_always_includes_instructions() -> None:
    system, _ = assemble_prompt("any question", _make_kb())
    assert "## Instructions" in system
    assert "SELECT" in system


# ---------------------------------------------------------------------------
# Qdrant collection — schema search
# ---------------------------------------------------------------------------


def test_assemble_prompt_calls_search_schema_when_collection_given() -> None:
    kb = _make_kb(tables=[_TABLE, _TABLE2])
    mock_hits = [{"table_name": "orders", "type": "table", "score": 0.95}]

    with (
        patch("nlqueries.embeddings.qdrant_store.search_schema", return_value=mock_hits) as mock_ss,
        patch("nlqueries.embeddings.qdrant_store.search", return_value=[]),
    ):
        assemble_prompt("how many orders?", kb, collection="my_collection")

    mock_ss.assert_called_once_with("my_collection", "how many orders?", top_k=10)


def test_assemble_prompt_calls_search_when_collection_given() -> None:
    kb = _make_kb(capsules=_CAPSULES)

    with (
        patch("nlqueries.embeddings.qdrant_store.search_schema", return_value=[]),
        patch("nlqueries.embeddings.qdrant_store.search", return_value=[]) as mock_s,
    ):
        assemble_prompt("count orders", kb, top_k_capsules=3, collection="my_col")

    mock_s.assert_called_once_with("my_col", "count orders", top_k=3)


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
    ):
        system, _ = assemble_prompt("count orders", kb, collection="col")

    assert "Qdrant-found capsule" in system


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
        system, _ = assemble_prompt("question", kb, collection="col")

    assert "Count orders per customer" in system


def test_assemble_prompt_does_not_call_qdrant_without_collection() -> None:
    kb = _make_kb()

    with patch(
        "nlqueries.embeddings.qdrant_store.search_schema",
        side_effect=RuntimeError("should not call"),
    ) as mock_ss:
        assemble_prompt("question", kb)

    mock_ss.assert_not_called()


# ---------------------------------------------------------------------------
# Qdrant schema filtering narrows tables
# ---------------------------------------------------------------------------


def test_qdrant_schema_hit_filters_to_relevant_tables() -> None:
    kb = _make_kb(tables=[_TABLE, _TABLE2])

    # Only "orders" is returned by schema search → "customers" should be excluded
    mock_hits = [{"table_name": "orders", "type": "table", "score": 0.9}]
    with (
        patch("nlqueries.embeddings.qdrant_store.search_schema", return_value=mock_hits),
        patch("nlqueries.embeddings.qdrant_store.search", return_value=[]),
    ):
        system, _ = assemble_prompt("show orders", kb, collection="col")

    assert "### Table: orders" in system
    assert "### Table: customers" not in system


def test_qdrant_schema_empty_hit_falls_back_to_all_tables() -> None:
    kb = _make_kb(tables=[_TABLE, _TABLE2])

    with (
        patch("nlqueries.embeddings.qdrant_store.search_schema", return_value=[]),
        patch("nlqueries.embeddings.qdrant_store.search", return_value=[]),
    ):
        system, _ = assemble_prompt("show something", kb, collection="col")

    assert "### Table: orders" in system
    assert "### Table: customers" in system


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_knowledge_base_does_not_raise() -> None:
    system, user = assemble_prompt("question", {})
    assert isinstance(system, str)
    assert user == "question"


def test_question_passed_through_unchanged() -> None:
    q = "How many customers signed up in 2024?"
    _, user = assemble_prompt(q, _make_kb())
    assert user == q
