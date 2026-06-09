"""Tests for nlqueries.orchestrator.sql_generation."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from nlqueries.orchestrator.sql_generation import (
    SQLGenerationResult,
    _extract_sql,
    _validate_sql,
    generate_sql,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_kb(tables: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "schema": {
            "tables": tables
            or [
                {
                    "name": "orders",
                    "description": "Purchase records",
                    "columns": [
                        {"name": "id", "type": "INTEGER"},
                        {"name": "customer_id", "type": "INTEGER"},
                        {"name": "total", "type": "DECIMAL"},
                        {"name": "created_at", "type": "TIMESTAMP"},
                    ],
                },
                {
                    "name": "customers",
                    "description": "Customer records",
                    "columns": [
                        {"name": "id", "type": "INTEGER"},
                        {"name": "email", "type": "VARCHAR"},
                    ],
                },
            ]
        },
        "business_context": {"glossary": [], "rules": []},
        "query_capsules": [
            {"intent": "Count orders", "template": "SELECT COUNT(*) FROM orders"},
        ],
    }


def _make_mock_llm(*responses: str) -> MagicMock:
    """Return a mock LLMClient whose complete() returns *responses* in sequence."""
    mock = MagicMock()
    mock.complete.side_effect = list(responses)
    return mock


# ---------------------------------------------------------------------------
# SQLGenerationResult dataclass
# ---------------------------------------------------------------------------


def test_sql_generation_result_stores_all_fields() -> None:
    result = SQLGenerationResult(
        sql="SELECT * FROM orders",
        is_valid=True,
        validation_error=None,
        dialect="postgres",
        attempt_count=1,
    )
    assert result.sql == "SELECT * FROM orders"
    assert result.is_valid is True
    assert result.validation_error is None
    assert result.dialect == "postgres"
    assert result.attempt_count == 1


def test_sql_generation_result_invalid_has_error() -> None:
    result = SQLGenerationResult(
        sql="DELETE FROM orders",
        is_valid=False,
        validation_error="Only SELECT statements are allowed; got Delete",
        dialect="postgres",
        attempt_count=2,
    )
    assert result.is_valid is False
    assert result.validation_error is not None


# ---------------------------------------------------------------------------
# _extract_sql — formatting cleanup
# ---------------------------------------------------------------------------


def test_extract_sql_plain_select() -> None:
    assert _extract_sql("SELECT * FROM orders") == "SELECT * FROM orders"


def test_extract_sql_with_sql_markdown_fence() -> None:
    raw = "```sql\nSELECT * FROM orders\n```"
    assert _extract_sql(raw) == "SELECT * FROM orders"


def test_extract_sql_with_generic_code_fence() -> None:
    raw = "```\nSELECT * FROM orders\n```"
    assert _extract_sql(raw) == "SELECT * FROM orders"


def test_extract_sql_strips_leading_prose() -> None:
    raw = "Here is the SQL to answer your question:\nSELECT id FROM orders WHERE id > 5"
    sql = _extract_sql(raw)
    assert sql.startswith("SELECT")


def test_extract_sql_strips_whitespace() -> None:
    assert _extract_sql("  SELECT 1  ") == "SELECT 1"


def test_extract_sql_handles_with_clause() -> None:
    raw = "WITH cte AS (SELECT id FROM orders) SELECT * FROM cte"
    sql = _extract_sql(raw)
    assert sql.startswith("WITH")


def test_extract_sql_multiline_fence() -> None:
    raw = "```sql\nSELECT\n  id,\n  total\nFROM orders\n```"
    sql = _extract_sql(raw)
    assert "SELECT" in sql
    assert "orders" in sql
    assert "```" not in sql


# ---------------------------------------------------------------------------
# _validate_sql — individual checks
# ---------------------------------------------------------------------------


def test_validate_sql_valid_select_returns_none() -> None:
    assert _validate_sql("SELECT * FROM orders", _make_kb(), "postgres") is None


def test_validate_sql_join_across_known_tables_returns_none() -> None:
    sql = "SELECT o.id, c.email FROM orders o JOIN customers c ON o.customer_id = c.id"
    assert _validate_sql(sql, _make_kb(), "postgres") is None


def test_validate_sql_empty_string_rejected() -> None:
    error = _validate_sql("", _make_kb(), "postgres")
    assert error is not None
    assert "empty" in error.lower()


def test_validate_sql_non_select_delete_rejected() -> None:
    error = _validate_sql("DELETE FROM orders", _make_kb(), "postgres")
    assert error is not None
    assert "SELECT" in error or "Delete" in error


def test_validate_sql_non_select_insert_rejected() -> None:
    error = _validate_sql("INSERT INTO orders (id, total) VALUES (1, 99.0)", _make_kb(), "postgres")
    assert error is not None


def test_validate_sql_non_select_update_rejected() -> None:
    error = _validate_sql("UPDATE orders SET total = 0 WHERE id = 1", _make_kb(), "postgres")
    assert error is not None


def test_validate_sql_unknown_table_rejected() -> None:
    error = _validate_sql("SELECT * FROM ghost_table", _make_kb(), "postgres")
    assert error is not None
    assert "ghost_table" in error


def test_validate_sql_error_message_names_unknown_tables() -> None:
    sql = "SELECT * FROM unknown_one JOIN unknown_two ON 1=1"
    error = _validate_sql(sql, _make_kb(), "postgres")
    assert error is not None
    assert "unknown_one" in error or "unknown_two" in error


def test_validate_sql_cte_alias_not_flagged_as_unknown() -> None:
    sql = "WITH recent AS (SELECT * FROM orders WHERE id > 5) SELECT * FROM recent"
    assert _validate_sql(sql, _make_kb(), "postgres") is None


def test_validate_sql_subquery_table_checked() -> None:
    sql = "SELECT * FROM (SELECT id FROM orders) AS sub"
    assert _validate_sql(sql, _make_kb(), "postgres") is None


def test_validate_sql_empty_schema_skips_table_check() -> None:
    kb = {"schema": {"tables": []}, "business_context": {}, "query_capsules": []}
    assert _validate_sql("SELECT * FROM anything", kb, "postgres") is None


def test_validate_sql_with_clause_known_table() -> None:
    sql = "WITH cte AS (SELECT id FROM orders) SELECT count(*) FROM cte"
    assert _validate_sql(sql, _make_kb(), "postgres") is None


def test_validate_sql_dialect_snowflake_valid() -> None:
    assert _validate_sql("SELECT * FROM orders", _make_kb(), "snowflake") is None


def test_validate_sql_dialect_bigquery_valid() -> None:
    assert _validate_sql("SELECT * FROM orders", _make_kb(), "bigquery") is None


# ---------------------------------------------------------------------------
# generate_sql — integration with mocked LLM
# ---------------------------------------------------------------------------


def test_generate_sql_valid_on_first_attempt() -> None:
    mock_llm = _make_mock_llm("SELECT * FROM orders")
    with patch("nlqueries.orchestrator.sql_generation.get_llm_client", return_value=mock_llm):
        result = generate_sql("show all orders", _make_kb(), "postgres")

    assert result.is_valid is True
    assert result.attempt_count == 1
    assert result.validation_error is None
    assert "SELECT" in result.sql
    assert mock_llm.complete.call_count == 1


def test_generate_sql_strips_markdown_fences() -> None:
    mock_llm = _make_mock_llm("```sql\nSELECT * FROM orders\n```")
    with patch("nlqueries.orchestrator.sql_generation.get_llm_client", return_value=mock_llm):
        result = generate_sql("show orders", _make_kb(), "postgres")

    assert result.is_valid is True
    assert "```" not in result.sql


def test_generate_sql_invalid_table_triggers_retry() -> None:
    mock_llm = _make_mock_llm(
        "SELECT * FROM ghost_table",  # attempt 1 — unknown table
        "SELECT * FROM orders",  # attempt 2 — valid
    )
    with patch("nlqueries.orchestrator.sql_generation.get_llm_client", return_value=mock_llm):
        result = generate_sql("show all orders", _make_kb(), "postgres")

    assert mock_llm.complete.call_count == 2
    assert result.attempt_count == 2


def test_generate_sql_retry_produces_valid_result() -> None:
    mock_llm = _make_mock_llm(
        "SELECT * FROM ghost_table",
        "SELECT id, total FROM orders WHERE total > 100",
    )
    with patch("nlqueries.orchestrator.sql_generation.get_llm_client", return_value=mock_llm):
        result = generate_sql("high value orders", _make_kb(), "postgres")

    assert result.is_valid is True
    assert "orders" in result.sql
    assert result.attempt_count == 2


def test_generate_sql_non_select_triggers_retry() -> None:
    mock_llm = _make_mock_llm(
        "DELETE FROM orders",  # attempt 1 — not SELECT
        "SELECT COUNT(*) FROM orders",  # attempt 2 — valid
    )
    with patch("nlqueries.orchestrator.sql_generation.get_llm_client", return_value=mock_llm):
        result = generate_sql("count orders", _make_kb(), "postgres")

    assert mock_llm.complete.call_count == 2
    assert result.attempt_count == 2
    assert result.is_valid is True


def test_generate_sql_non_select_rejected_with_error() -> None:
    mock_llm = _make_mock_llm(
        "DELETE FROM orders",
        "DELETE FROM orders",  # second attempt also bad
    )
    with patch("nlqueries.orchestrator.sql_generation.get_llm_client", return_value=mock_llm):
        result = generate_sql("delete orders", _make_kb(), "postgres")

    assert result.is_valid is False
    assert result.validation_error is not None
    assert result.attempt_count == 2


def test_generate_sql_both_attempts_fail_returns_result() -> None:
    mock_llm = _make_mock_llm(
        "SELECT * FROM ghost_one",
        "SELECT * FROM ghost_two",
    )
    with patch("nlqueries.orchestrator.sql_generation.get_llm_client", return_value=mock_llm):
        result = generate_sql("show something", _make_kb(), "postgres")

    assert result.is_valid is False
    assert result.validation_error is not None
    assert result.attempt_count == 2


def test_generate_sql_no_retry_when_first_valid() -> None:
    mock_llm = _make_mock_llm("SELECT COUNT(*) FROM customers")
    with patch("nlqueries.orchestrator.sql_generation.get_llm_client", return_value=mock_llm):
        result = generate_sql("count customers", _make_kb(), "postgres")

    assert mock_llm.complete.call_count == 1
    assert result.attempt_count == 1


def test_generate_sql_dialect_stored_in_result() -> None:
    mock_llm = _make_mock_llm("SELECT * FROM orders")
    with patch("nlqueries.orchestrator.sql_generation.get_llm_client", return_value=mock_llm):
        result = generate_sql("show orders", _make_kb(), "snowflake")

    assert result.dialect == "snowflake"


def test_generate_sql_retry_includes_error_in_correction_prompt() -> None:
    """The error message from attempt 1 must appear in the attempt 2 user prompt."""
    mock_llm = _make_mock_llm(
        "SELECT * FROM ghost_table",
        "SELECT * FROM orders",
    )
    with patch("nlqueries.orchestrator.sql_generation.get_llm_client", return_value=mock_llm):
        generate_sql("question", _make_kb(), "postgres")

    correction_user: str = mock_llm.complete.call_args_list[1].args[1]
    assert "ghost_table" in correction_user or "error" in correction_user.lower()


def test_generate_sql_system_prompt_contains_schema() -> None:
    """The system prompt passed to the LLM must include the table name."""
    mock_llm = _make_mock_llm("SELECT * FROM orders")
    with patch("nlqueries.orchestrator.sql_generation.get_llm_client", return_value=mock_llm):
        generate_sql("show orders", _make_kb(), "postgres")

    system_prompt: str = mock_llm.complete.call_args_list[0].args[0]
    assert "orders" in system_prompt


def test_generate_sql_system_prompt_mentions_dialect() -> None:
    mock_llm = _make_mock_llm("SELECT * FROM orders")
    with patch("nlqueries.orchestrator.sql_generation.get_llm_client", return_value=mock_llm):
        generate_sql("show orders", _make_kb(), "bigquery")

    system_prompt: str = mock_llm.complete.call_args_list[0].args[0]
    assert "bigquery" in system_prompt.lower()


def test_generate_sql_uses_llm_complete_not_stream() -> None:
    """generate_sql must call complete(), not stream()."""
    mock_llm = _make_mock_llm("SELECT * FROM orders")
    with patch("nlqueries.orchestrator.sql_generation.get_llm_client", return_value=mock_llm):
        generate_sql("show orders", _make_kb(), "postgres")

    mock_llm.complete.assert_called()
    mock_llm.stream.assert_not_called()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_generate_sql_empty_response_triggers_retry() -> None:
    mock_llm = _make_mock_llm(
        "",  # attempt 1 — empty
        "SELECT * FROM orders",  # attempt 2 — valid
    )
    with patch("nlqueries.orchestrator.sql_generation.get_llm_client", return_value=mock_llm):
        result = generate_sql("show orders", _make_kb(), "postgres")

    assert result.attempt_count == 2


@pytest.mark.parametrize("dialect", ["postgres", "snowflake", "bigquery"])
def test_generate_sql_all_dialects_accepted(dialect: str) -> None:
    mock_llm = _make_mock_llm("SELECT * FROM orders")
    with patch("nlqueries.orchestrator.sql_generation.get_llm_client", return_value=mock_llm):
        result = generate_sql("show orders", _make_kb(), dialect)

    assert result.dialect == dialect
