"""Tests for nlqueries.orchestrator.sql_generation."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nlqueries.orchestrator.sql_generation import (
    SQLGenerationResult,
    _extract_sql,
    _kb_to_sqlglot_schema,
    _try_mechanical_repair,
    _validate_columns,
    _validate_sql,
    generate_sql,
    validate_and_repair,
)

from tests.test_prompt_assembly import ANSWER_RULE_PHRASES

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


@pytest.mark.parametrize("dialect", ["mssql", "postgresql", "mariadb"])
def test_validate_sql_accepts_a_connector_dialect_spelling(dialect: str) -> None:
    """Spellings a connector uses and sqlglot rejects. The policy maps them;
    the schema check's re-parse used to pass them raw, raised ValueError, and
    so failed every correction after a database error for these dialects."""
    assert _validate_sql("SELECT * FROM orders", _make_kb(), dialect) is None
    error = _validate_sql("SELECT * FROM nonexistent_table", _make_kb(), dialect)
    assert error is not None and "nonexistent_table" in error


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


# ---------------------------------------------------------------------------
# Phase 4B: _kb_to_sqlglot_schema
# ---------------------------------------------------------------------------


def test_kb_to_sqlglot_schema_maps_tables_and_columns() -> None:
    schema = _kb_to_sqlglot_schema(_make_kb())
    assert "orders" in schema
    assert schema["orders"]["id"] == "INTEGER"
    assert schema["orders"]["total"] == "DECIMAL"


def test_kb_to_sqlglot_schema_includes_all_tables() -> None:
    schema = _kb_to_sqlglot_schema(_make_kb())
    assert "orders" in schema
    assert "customers" in schema


def test_kb_to_sqlglot_schema_empty_kb_returns_empty() -> None:
    schema = _kb_to_sqlglot_schema({})
    assert schema == {}


def test_kb_to_sqlglot_schema_skips_tables_without_name() -> None:
    kb = {"schema": {"tables": [{"columns": [{"name": "id", "type": "INT"}]}]}}
    schema = _kb_to_sqlglot_schema(kb)
    assert schema == {}


# ---------------------------------------------------------------------------
# Phase 4B: _validate_columns
# ---------------------------------------------------------------------------


def test_validate_columns_valid_explicit_column_returns_none() -> None:
    import sqlglot as sg

    stmt = sg.parse_one("SELECT id FROM orders", dialect="postgres")
    assert _validate_columns(stmt, _make_kb(), "postgres") is None


def test_validate_columns_select_star_returns_none() -> None:
    import sqlglot as sg

    stmt = sg.parse_one("SELECT * FROM orders", dialect="postgres")
    assert _validate_columns(stmt, _make_kb(), "postgres") is None


def test_validate_columns_unknown_column_returns_error() -> None:
    import sqlglot as sg

    stmt = sg.parse_one("SELECT ghost_col FROM orders", dialect="postgres")
    error = _validate_columns(stmt, _make_kb(), "postgres")
    assert error is not None
    assert "ghost_col" in error or "Column" in error


def test_validate_columns_empty_schema_returns_none() -> None:
    """No schema → column validation skipped (best-effort permissive)."""
    import sqlglot as sg

    stmt = sg.parse_one("SELECT anything FROM orders", dialect="postgres")
    kb = {"schema": {"tables": []}, "business_context": {}, "query_capsules": []}
    assert _validate_columns(stmt, kb, "postgres") is None


def test_validate_columns_join_valid_columns_returns_none() -> None:
    import sqlglot as sg

    sql = "SELECT o.id, c.email FROM orders o JOIN customers c ON o.customer_id = c.id"
    stmt = sg.parse_one(sql, dialect="postgres")
    assert _validate_columns(stmt, _make_kb(), "postgres") is None


# A column read through a `SELECT *` could not be resolved while the stars were
# left unexpanded, so valid SQL was refused as naming a column that does not
# exist. Each shape here was refused before.
_THROUGH_A_STAR = {
    "a table": "SELECT total FROM (SELECT * FROM orders) o",
    "a join": (
        "SELECT email FROM (SELECT * FROM orders JOIN customers "
        "ON customers.id = orders.customer_id) j"
    ),
    "a qualified star": (
        "SELECT email FROM (SELECT customers.* FROM orders JOIN customers "
        "ON customers.id = orders.customer_id) j"
    ),
    "a subquery": "SELECT label FROM (SELECT * FROM (SELECT 'big' AS label, id FROM orders) a) b",
    "a union of subqueries": (
        "SELECT label, id FROM ("
        "SELECT * FROM (SELECT 'largest' AS label, id FROM orders ORDER BY total DESC LIMIT 1) a "
        "UNION ALL "
        "SELECT * FROM (SELECT 'newest' AS label, id FROM orders "
        "ORDER BY created_at DESC LIMIT 1) b)"
    ),
    "a CTE": (
        "WITH big AS (SELECT 'big' AS label FROM orders) SELECT label FROM (SELECT * FROM big) s"
    ),
}


@pytest.mark.parametrize("dialect", ["postgres", "sqlite"])
@pytest.mark.parametrize("shape", list(_THROUGH_A_STAR))
def test_validate_columns_accepts_a_column_read_through_a_star(shape: str, dialect: str) -> None:
    import sqlglot as sg

    stmt = sg.parse_one(_THROUGH_A_STAR[shape], dialect=dialect)
    assert _validate_columns(stmt, _make_kb(), dialect) is None


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT ghost_col FROM (SELECT * FROM orders) o",
        "SELECT ghost_col FROM (SELECT * FROM (SELECT 'big' AS label FROM orders) a) b",
    ],
)
def test_validate_columns_still_rejects_an_unknown_column_read_through_a_star(sql: str) -> None:
    """Expanding the stars must not let a real mistake through."""
    import sqlglot as sg

    error = _validate_columns(sg.parse_one(sql, dialect="postgres"), _make_kb(), "postgres")
    assert error is not None
    assert "ghost_col" in error


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT id FROM (SELECT * FROM orders JOIN customers "
        "ON customers.id = orders.customer_id) j",
        "WITH j AS (SELECT * FROM orders JOIN customers ON customers.id = orders.customer_id) "
        "SELECT id FROM j",
    ],
)
def test_validate_columns_refuses_a_name_a_joined_star_repeats(sql: str) -> None:
    """Both tables have an `id`; read through the star it is ambiguous, which
    PostgreSQL refuses. The first check refused it before stars were expanded."""
    import sqlglot as sg

    error = _validate_columns(sg.parse_one(sql, dialect="postgres"), _make_kb(), "postgres")
    assert error is not None
    assert "'id' is ambiguous" in error


@pytest.mark.parametrize(
    "sql",
    [
        # An outer star over the joined star generates both `id` columns
        # itself; the statement names `total`, never `id`.
        "SELECT * FROM (SELECT * FROM orders JOIN customers "
        "ON customers.id = orders.customer_id) j WHERE total > 100",
        "WITH j AS (SELECT * FROM orders JOIN customers ON customers.id = orders.customer_id) "
        "SELECT * FROM j WHERE total > 100",
        # The alias `t` is the joined star only inside EXISTS; the outer `id`
        # reads the other `t`, which has one.
        "SELECT id FROM (SELECT * FROM orders) t WHERE EXISTS (SELECT 1 FROM (SELECT * "
        "FROM orders JOIN customers ON customers.id = orders.customer_id) t)",
        # Two derived tables in one scope: `a.id` reads `a`, which has one
        # `id`; only `b` repeats it. The bare `id` inside `a` is written too,
        # so the name alone does not settle it: the column's own source does.
        # The bare `email`, only in `b`, is what fails the first check.
        "SELECT a.id, email FROM (SELECT * FROM orders WHERE id > 0) a JOIN (SELECT * FROM "
        "orders JOIN customers ON customers.id = orders.customer_id) b ON a.total = b.total",
    ],
)
def test_validate_columns_accepts_a_repeated_name_the_statement_never_reads(sql: str) -> None:
    import sqlglot as sg

    assert _validate_columns(sg.parse_one(sql, dialect="postgres"), _make_kb(), "postgres") is None


def test_validate_sql_accepts_a_union_of_star_subqueries() -> None:
    """The whole check, not only the column step, for the shape that was refused."""
    assert _validate_sql(_THROUGH_A_STAR["a union of subqueries"], _make_kb(), "sqlite") is None


# ---------------------------------------------------------------------------
# Phase 4B: _validate_sql includes column check
# ---------------------------------------------------------------------------


def test_validate_sql_rejects_unknown_column() -> None:
    error = _validate_sql("SELECT ghost_col FROM orders", _make_kb(), "postgres")
    assert error is not None
    assert "Column" in error or "ghost_col" in error


def test_validate_sql_accepts_valid_explicit_column() -> None:
    assert _validate_sql("SELECT id, total FROM orders", _make_kb(), "postgres") is None


# ---------------------------------------------------------------------------
# A string of several statements is not repaired by dropping the rest
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM orders; DELETE FROM orders",
        # Refused too, though both read: which of two answers the user wanted
        # is not something a repair can know.
        "SELECT * FROM orders; SELECT 1",
        "SELECT * FROM orders;\n-- note\nDROP TABLE orders",
    ],
)
def test_mechanical_repair_never_drops_a_statement(sql: str) -> None:
    """It used to keep the first statement and call the result valid, so a
    write after a read vanished without an error or a warning. Transpiling
    did the same: it returns one string per statement and only the first was
    looked at."""
    repaired, err = _try_mechanical_repair(sql, _make_kb(), "postgres")
    assert err is not None
    assert repaired == sql


def _repairing_llm(*replies: str) -> MagicMock:
    llm = MagicMock()
    queue = list(replies)

    async def _acomplete(*a: Any, **kw: Any) -> str:
        return queue.pop(0)

    llm.acomplete = _acomplete
    llm.supports_prompt_caching = False
    return llm


def _repair_with_provenance(sql: str, *replies: str) -> tuple[SQLGenerationResult, list[str]]:
    from nlqueries.orchestrator.provenance import Provenance, use_provenance

    collector = Provenance()

    async def _run() -> SQLGenerationResult:
        with use_provenance(collector):
            return await validate_and_repair(sql, _make_kb(), "postgres", _repairing_llm(*replies))

    return asyncio.run(_run()), collector.validator


def test_a_trailing_write_the_model_cannot_repair_stays_invalid() -> None:
    result, _ = _repair_with_provenance(
        "SELECT * FROM orders; DELETE FROM orders",
        "SELECT * FROM orders; DELETE FROM orders",
    )

    assert result.is_valid is False
    assert result.validation_error is not None
    assert "one statement" in result.validation_error


def test_a_trailing_write_repaired_by_the_model_says_so() -> None:
    """Decision: the model's repair of a refused statement is accepted, and the
    refusal is kept in provenance, so a reader can see what it replaced."""
    result, warnings = _repair_with_provenance(
        "SELECT * FROM orders; DELETE FROM orders",
        "SELECT * FROM orders",
    )

    assert result.is_valid is True
    assert result.sql.strip().rstrip(";") == "SELECT * FROM orders"
    assert result.attempt_count == 2, "the model was asked to correct it"
    assert len(warnings) == 1
    assert warnings[0].startswith("repaired after: Refused by SQL policy")
    assert "one statement" in warnings[0]


def test_a_write_inside_a_cte_repaired_by_the_model_says_so() -> None:
    result, warnings = _repair_with_provenance(
        "WITH gone AS (DELETE FROM orders RETURNING id) SELECT count(*) FROM gone",
        "SELECT count(*) FROM orders",
    )

    assert result.is_valid is True
    assert result.attempt_count == 2
    assert len(warnings) == 1
    assert warnings[0].startswith("repaired after: Refused by SQL policy")


def test_a_repaired_unknown_table_records_no_policy_warning() -> None:
    """Only a refusal is worth recording: an unknown table repaired is the
    ordinary correction loop, and a warning on every one would be noise."""
    result, warnings = _repair_with_provenance(
        "SELECT * FROM ghost_table",
        "SELECT * FROM orders",
    )

    assert result.is_valid is True
    assert warnings == []


def test_mechanical_repair_single_statement_unchanged() -> None:
    sql = "SELECT * FROM orders"
    repaired, err = _try_mechanical_repair(sql, _make_kb(), "postgres")
    assert err is None
    assert "orders" in repaired


# ---------------------------------------------------------------------------
# Phase 4B: EXPLAIN gate via validate_and_repair
# ---------------------------------------------------------------------------


def _make_async_llm_for_repair(repair_sql: str = "SELECT * FROM orders") -> MagicMock:
    """Return a mock LLMClient whose acomplete() returns a valid SQL string."""
    mock = MagicMock()

    async def _acomplete(*args: Any, **kwargs: Any) -> str:
        return repair_sql

    mock.acomplete = _acomplete
    mock.supports_prompt_caching = False
    return mock


def test_explain_gate_skipped_when_no_connector() -> None:
    """EXPLAIN gate must be a no-op when connector=None."""

    async def _run() -> SQLGenerationResult:
        llm = _make_async_llm_for_repair()
        return await validate_and_repair(
            "SELECT * FROM orders",
            _make_kb(),
            "postgres",
            llm,
            explain_check=True,  # enabled but no connector
        )

    result = asyncio.run(_run())
    assert result.is_valid is True


def test_explain_gate_skipped_when_explain_check_false() -> None:
    """EXPLAIN gate must be a no-op when explain_check=False."""
    connector = MagicMock()
    connector.execute.side_effect = RuntimeError("should not be called")

    async def _run() -> SQLGenerationResult:
        llm = _make_async_llm_for_repair()
        return await validate_and_repair(
            "SELECT * FROM orders",
            _make_kb(),
            "postgres",
            llm,
            connector=connector,
            explain_check=False,
        )

    result = asyncio.run(_run())
    assert result.is_valid is True
    connector.execute.assert_not_called()


def test_explain_gate_marks_invalid_on_db_error() -> None:
    """When EXPLAIN raises, result is marked invalid with the error message."""
    connector = MagicMock()
    connector.execute.side_effect = RuntimeError("syntax error at position 5")

    async def _run() -> SQLGenerationResult:
        llm = _make_async_llm_for_repair()
        return await validate_and_repair(
            "SELECT * FROM orders",
            _make_kb(),
            "postgres",
            llm,
            connector=connector,
            explain_check=True,
        )

    result = asyncio.run(_run())
    assert result.is_valid is False
    assert result.validation_error is not None
    assert "EXPLAIN failed" in result.validation_error


def test_explain_gate_passes_on_successful_explain() -> None:
    """When EXPLAIN succeeds, result remains valid."""
    connector = MagicMock()
    connector.execute.return_value = None  # success

    async def _run() -> SQLGenerationResult:
        llm = _make_async_llm_for_repair()
        return await validate_and_repair(
            "SELECT * FROM orders",
            _make_kb(),
            "postgres",
            llm,
            connector=connector,
            explain_check=True,
        )

    result = asyncio.run(_run())
    assert result.is_valid is True
    connector.execute.assert_called_once()
    assert "EXPLAIN" in connector.execute.call_args[0][0]


def test_validate_numeric_clauses_detects_quoted_limit() -> None:
    """LIMIT '10' (string literal) should be flagged as invalid."""
    from nlqueries.orchestrator.sql_generation import _validate_sql

    sql = "SELECT id FROM orders LIMIT '10'"
    err = _validate_sql(sql, _make_kb(), "postgres")
    assert err is not None
    assert "LIMIT" in err


def test_mechanical_repair_fixes_quoted_limit() -> None:
    """Mechanical repair must convert LIMIT '10' → LIMIT 10 without an LLM call."""
    from nlqueries.orchestrator.sql_generation import _try_mechanical_repair

    sql = "SELECT id FROM orders LIMIT '10'"
    repaired, error = _try_mechanical_repair(sql, _make_kb(), "postgres")
    assert error is None
    assert "'10'" not in repaired


def test_validate_and_repair_fixes_quoted_limit() -> None:
    """End-to-end: LIMIT '10' is auto-corrected without an LLM call (attempt_count stays 1)."""

    async def _run() -> SQLGenerationResult:
        mock_llm = MagicMock()
        mock_llm.supports_prompt_caching = False
        return await validate_and_repair(
            "SELECT id FROM orders LIMIT '10'",
            _make_kb(),
            "postgres",
            mock_llm,
        )

    result = asyncio.run(_run())
    assert result.is_valid is True
    assert "'10'" not in result.sql
    assert result.attempt_count == 1  # mechanical repair, no LLM call


def test_explain_gate_not_run_on_already_invalid_result() -> None:
    """EXPLAIN must not run when static validation already failed."""
    connector = MagicMock()
    connector.execute.side_effect = RuntimeError("should not be called")

    async def _run() -> SQLGenerationResult:
        mock_llm = MagicMock()

        async def _acomplete(*a: Any, **kw: Any) -> str:
            return "SELECT * FROM ghost_table"  # repair also returns invalid SQL

        mock_llm.acomplete = _acomplete
        mock_llm.supports_prompt_caching = False
        return await validate_and_repair(
            "SELECT * FROM ghost_table",
            _make_kb(),
            "postgres",
            mock_llm,
            connector=connector,
            explain_check=True,
        )

    result = asyncio.run(_run())
    assert result.is_valid is False
    connector.execute.assert_not_called()


# ---------------------------------------------------------------------------
# Prose-as-SQL: the model answers in English and it reaches the SQL path
#
# Both cases below were taken from a production log during the W-0 load
# baseline, not invented. They are the reason this section exists.
# ---------------------------------------------------------------------------

# "I can't help with that." — the English preposition, mid-sentence.
PROSE_WITH = (
    "I'm sorry, I can't help with that.\n\n"
    "If you have a specific question based on the tables provided in the "
    "schema, please ask, and I can help generate a valid SQL query for you."
)

# A clarification request. The apostrophe in "I'll" opens a string literal that
# never closes, so this fails in the TOKENIZER, before parsing begins.
PROSE_APOSTROPHE = (
    "I need more detail (for example which titles, ratings, or professions), "
    "please let me know and I'll help."
)


def test_validate_sql_catches_tokenizer_errors_not_just_parse_errors() -> None:
    """A tokenizer failure must be a validation error, never an exception.

    ``TokenError`` is a *sibling* of ``ParseError`` under ``SqlglotError``, not a
    subclass, so ``except ParseError`` did not catch it. It escaped
    ``_validate_sql`` and surfaced to the user as a 500 from the chat socket —
    the exact outcome this function exists to prevent.
    """
    error = _validate_sql(PROSE_APOSTROPHE, _make_kb(), "postgres")

    assert error is not None
    # Wording moved when the SQL policy took over this check; the property
    # asserted is that the tokenizer failure is reported rather than raised.
    assert "parsed" in error.lower()


def test_validate_sql_reports_rather_than_raises_on_arbitrary_prose() -> None:
    """No English input may raise out of the validator, whatever it contains."""
    for text in (PROSE_WITH, PROSE_APOSTROPHE, "Sorry — I don't know.", "¯\_(ツ)_/¯"):
        assert _validate_sql(text, _make_kb(), "postgres") is not None


def test_extract_sql_does_not_treat_the_english_word_with_as_a_cte() -> None:
    """`help with that.` is not a CTE.

    The keyword scan used a substring search, so this prose was returned as the
    statement ``with that.\n\nIf you have a specific question…`` — which is how
    an English sentence came to be handed on as the model's SQL.
    """
    extracted = _extract_sql(PROSE_WITH)

    assert not extracted.lower().startswith("with that")


def test_extract_sql_still_finds_a_real_cte() -> None:
    """The fix must not cost us actual CTE extraction."""
    text = (
        "Here is the query you asked for:\n\n"
        "WITH recent AS (SELECT id FROM orders) SELECT * FROM recent"
    )

    assert _extract_sql(text).upper().startswith("WITH RECENT AS")


def test_extract_sql_finds_a_cte_with_a_column_list_and_recursive() -> None:
    """`WITH RECURSIVE name (cols) AS` is still a CTE head."""
    text = "Sure. WITH RECURSIVE tree (id, parent) AS (SELECT 1, NULL) SELECT * FROM tree"

    assert _extract_sql(text).upper().startswith("WITH RECURSIVE TREE")


def test_extract_sql_prefers_a_real_select_over_a_prose_with() -> None:
    """Prose containing "with" must not shadow the SELECT that follows it."""
    text = "I can help with that. SELECT id FROM orders"

    assert _extract_sql(text).upper().startswith("SELECT ID FROM ORDERS")


def test_prose_never_becomes_valid_sql_end_to_end() -> None:
    """The whole point: prose in, a validation error out — not an exception,
    and never a statement the caller would be willing to execute."""
    for prose in (PROSE_WITH, PROSE_APOSTROPHE):
        result = _validate_sql(_extract_sql(prose), _make_kb(), "postgres")
        assert result is not None


# ---------------------------------------------------------------------------
# The answer rules: in this prompt too, because it drives the repair step
# ---------------------------------------------------------------------------


def test_the_sql_system_prompt_carries_the_answer_rules() -> None:
    from nlqueries.orchestrator.prompt_assembly import _ANSWER_RULES
    from nlqueries.orchestrator.sql_generation import _build_sql_system_prompt

    system = _build_sql_system_prompt(_make_kb(), "postgres")

    assert _ANSWER_RULES and _ANSWER_RULES in system


@pytest.mark.parametrize("phrase", ANSWER_RULE_PHRASES)
def test_the_sql_system_prompt_carries_each_answer_rule(phrase: str) -> None:
    """The phrases the cached block is checked for, so both prompts are held to
    one list."""
    from nlqueries.orchestrator.sql_generation import _build_sql_system_prompt

    assert phrase in _build_sql_system_prompt(_make_kb(), "postgres")


def test_the_rule_is_shared_with_prompt_assembly_not_restated() -> None:
    """One sentence in two places is how the two prompts come to disagree."""
    import inspect

    import nlqueries.orchestrator.sql_generation as sql_generation

    assert "SELECT only the columns the question asks for" not in inspect.getsource(sql_generation)


def test_generate_sql_sends_the_projection_rule() -> None:
    mock_llm = _make_mock_llm("SELECT id FROM orders")
    with patch("nlqueries.orchestrator.sql_generation.get_llm_client", return_value=mock_llm):
        generate_sql("show orders", _make_kb(), "postgres")

    system_prompt: str = mock_llm.complete.call_args_list[0].args[0]
    assert "SELECT only the columns the question asks for" in system_prompt


def test_the_repair_step_sends_the_projection_rule() -> None:
    """A repaired statement must not put the extra columns back."""
    seen: list[Any] = []

    async def _run() -> SQLGenerationResult:
        mock_llm = MagicMock()

        async def _acomplete(system: Any, user: str, **kw: Any) -> str:
            seen.append(system)
            return "<sql>SELECT id FROM orders</sql>"

        mock_llm.acomplete = _acomplete
        mock_llm.supports_prompt_caching = False
        return await validate_and_repair(
            "SELECT * FROM ghost_table", _make_kb(), "postgres", mock_llm
        )

    with patch("nlqueries.config.SELF_CONSISTENCY", "off"):
        result = asyncio.run(_run())

    assert result.attempt_count == 2, "the LLM repair must have run"
    assert len(seen) == 1
    assert "SELECT only the columns the question asks for" in seen[0]


# ---------------------------------------------------------------------------
# A statement the output budget cut off
#
# It ends mid-clause, or, cut at a clause boundary, parses and answers another
# question. Either way it needs writing whole, with room to write it: the
# correction tier's share of the budget cut a long statement off a second time.
# ---------------------------------------------------------------------------


def _recording_llm(reply: str, *, cut_off: bool = False) -> tuple[MagicMock, list[dict[str, Any]]]:
    """An LLM whose repair returns *reply*, recording each call; *cut_off* makes
    it finish the way a client does when the budget ran out."""
    from nlqueries.llm.client import note_finish

    calls: list[dict[str, Any]] = []
    llm = MagicMock()
    llm.supports_prompt_caching = False

    async def _acomplete(system: Any, user: str, **kw: Any) -> str:
        calls.append({"user": user, **kw})
        note_finish("max_tokens" if cut_off else "end_turn", reply)
        return reply

    llm.acomplete = _acomplete
    return llm, calls


def _repair(sql: str, llm: MagicMock, *, cut_off: bool) -> SQLGenerationResult:
    async def _run() -> SQLGenerationResult:
        return await validate_and_repair(sql, _make_kb(), "postgres", llm, cut_off=cut_off)

    with patch("nlqueries.config.SELF_CONSISTENCY", "all"):
        return asyncio.run(_run())


def test_a_cut_off_statement_that_parses_is_written_again_not_accepted() -> None:
    """`SELECT id, total FROM orders` is valid, and is not what was asked: the
    WHERE clause was still to come."""
    from nlqueries.llm import output_budget

    llm, calls = _recording_llm("<sql>SELECT id, total FROM orders WHERE total > 100</sql>")
    with patch(
        "nlqueries.orchestrator.candidates.generate_candidates",
        side_effect=AssertionError("candidates are written on the smaller budget"),
    ):
        result = _repair("SELECT id, total FROM orders", llm, cut_off=True)

    assert result.is_valid
    assert "WHERE total > 100" in result.sql
    assert result.attempt_count == 2
    assert len(calls) == 1
    assert calls[0]["max_tokens"] == output_budget("answer")
    assert "ran out of room before the statement was complete" in calls[0]["user"]


def test_an_invalid_statement_not_cut_off_keeps_the_correction_budget() -> None:
    from nlqueries.llm import output_budget

    llm, calls = _recording_llm("<sql>SELECT id FROM orders</sql>")
    with patch("nlqueries.config.SELF_CONSISTENCY", "off"):

        async def _run() -> SQLGenerationResult:
            return await validate_and_repair(
                "SELECT * FROM ghost_table", _make_kb(), "postgres", llm
            )

        result = asyncio.run(_run())

    assert result.is_valid
    assert calls[0]["max_tokens"] == output_budget("correction")
    assert "ran out of room" not in calls[0]["user"]


@pytest.mark.parametrize(
    "reply",
    [
        "<sql>SELECT id FROM orders WHERE total >",  # does not parse
        "<sql>SELECT id FROM orders",  # parses, and is not complete
    ],
)
def test_a_repair_cut_off_as_well_says_so(reply: str) -> None:
    llm, _ = _recording_llm(reply, cut_off=True)
    result = _repair("SELECT id FROM orders WHERE", llm, cut_off=True)

    assert not result.is_valid
    assert result.validation_error is not None
    assert "LLM_MAX_OUTPUT_TOKENS" in result.validation_error


def test_a_repair_cut_off_after_its_statement_closed_is_kept() -> None:
    """Out of room in what followed </sql>; the statement itself is whole."""
    llm, _ = _recording_llm(
        "<sql>SELECT id FROM orders WHERE total > 100</sql> This lists", cut_off=True
    )
    result = _repair("SELECT id FROM orders WHERE", llm, cut_off=True)

    assert result.is_valid
    assert "WHERE total > 100" in result.sql


def test_a_fallback_repair_cut_off_says_so_too() -> None:
    """Self-consistency that returns no candidate falls back to one reply; that
    reply is checked for a cut-off like any other."""
    llm, _ = _recording_llm("<sql>SELECT id FROM orders", cut_off=True)
    with patch(
        "nlqueries.orchestrator.candidates.generate_candidates", new=AsyncMock(return_value=[])
    ):
        result = _repair("SELECT * FROM ghost_table", llm, cut_off=False)

    assert not result.is_valid
    assert result.validation_error is not None
    assert "LLM_MAX_OUTPUT_TOKENS" in result.validation_error
