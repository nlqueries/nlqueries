"""
Tests for nlqueries.processing.parameterizer (Task 3.2.1).

Required scenarios (per spec):
  - Date range filter         → DATE placeholders
  - Status string filter      → VARCHAR placeholder
  - Integer ID lookup         → INT placeholder
  - Multi-table join          → both tables extracted
  - Subquery                  → handled without error

Additional coverage:
  - Float literal             → DECIMAL placeholder
  - Timestamp string          → TIMESTAMP placeholder
  - Schema type lookup        → SchemaSpec overrides default VARCHAR
  - Duplicate column name     → disambiguated with _2 suffix
  - IN list                   → each value parameterized under the same column
  - Unknown column context    → param_N naming
  - No-literal query          → empty placeholders, correct tables/columns
  - auto_description format   → "Query on … filtering by …"
  - frequency propagated      → from cluster.execution_count
  - intent starts empty       → filled later by LLM annotator
  - parameterize_clusters     → one capsule per cluster
"""

from __future__ import annotations

import pytest
from nlqueries.connectors.base import ColumnSpec, SchemaSpec, TableSpec
from nlqueries.processing.parameterizer import (
    Placeholder,
    QueryCapsule,
    parameterize_cluster,
    parameterize_clusters,
)
from nlqueries.processing.query_clusterer import QueryCluster

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cluster(sql: str, frequency: int = 10) -> QueryCluster:
    """Build a minimal QueryCluster for parameterizer tests."""
    return QueryCluster(
        representative_sql=sql,
        fingerprint=sql,
        execution_count=frequency,
        member_count=1,
        tables_referenced=[],
    )


def _schema(*cols: tuple[str, str]) -> SchemaSpec:
    """Build a one-table SchemaSpec with the given (column_name, type) pairs."""
    columns = [
        ColumnSpec(
            name=name,
            type=col_type,
            nullable=True,
            is_primary_key=False,
            is_foreign_key=False,
            references=None,
            description=None,
        )
        for name, col_type in cols
    ]
    return SchemaSpec(
        database="test_db",
        extracted_at="2024-01-01T00:00:00",
        tables=[
            TableSpec(
                name="test_table",
                schema="public",
                row_count=None,
                columns=columns,
                description=None,
            )
        ],
    )


def _placeholder_map(capsule: QueryCapsule) -> dict[str, str]:
    """Return {placeholder_name: type} for easy assertions."""
    return {p.name: p.type for p in capsule.placeholders}


# ---------------------------------------------------------------------------
# date range filter  (required by spec)
# ---------------------------------------------------------------------------


def test_date_range_filter_placeholder_types() -> None:
    sql = "SELECT * FROM events WHERE created_at BETWEEN '2024-01-01' AND '2024-12-31'"
    capsule = parameterize_cluster(_cluster(sql))
    assert len(capsule.placeholders) == 2
    assert all(p.type == "DATE" for p in capsule.placeholders)


def test_date_range_filter_placeholder_names() -> None:
    sql = "SELECT * FROM events WHERE created_at BETWEEN '2024-01-01' AND '2024-12-31'"
    capsule = parameterize_cluster(_cluster(sql))
    ph = _placeholder_map(capsule)
    assert "created_at" in ph
    assert "created_at_2" in ph


def test_date_range_filter_template_contains_placeholders() -> None:
    sql = "SELECT * FROM events WHERE created_at BETWEEN '2024-01-01' AND '2024-12-31'"
    capsule = parameterize_cluster(_cluster(sql))
    assert "[created_at:DATE]" in capsule.template_sql
    assert "[created_at_2:DATE]" in capsule.template_sql


# ---------------------------------------------------------------------------
# status string filter  (required by spec)
# ---------------------------------------------------------------------------


def test_status_string_filter_defaults_to_varchar() -> None:
    sql = "SELECT id FROM orders WHERE status = 'pending'"
    capsule = parameterize_cluster(_cluster(sql))
    ph = _placeholder_map(capsule)
    assert ph.get("status") == "VARCHAR"


def test_status_string_filter_template_contains_placeholder() -> None:
    sql = "SELECT id FROM orders WHERE status = 'pending'"
    capsule = parameterize_cluster(_cluster(sql))
    assert "[status:VARCHAR]" in capsule.template_sql


# ---------------------------------------------------------------------------
# integer ID lookup  (required by spec)
# ---------------------------------------------------------------------------


def test_integer_id_lookup_type_is_int() -> None:
    sql = "SELECT * FROM users WHERE id = 42"
    capsule = parameterize_cluster(_cluster(sql))
    ph = _placeholder_map(capsule)
    assert ph.get("id") == "INT"


def test_integer_id_lookup_template_contains_placeholder() -> None:
    sql = "SELECT * FROM users WHERE id = 42"
    capsule = parameterize_cluster(_cluster(sql))
    assert "[id:INT]" in capsule.template_sql
    # Original literal must not appear verbatim
    assert "42" not in capsule.template_sql


# ---------------------------------------------------------------------------
# multi-table join  (required by spec)
# ---------------------------------------------------------------------------


def test_multi_table_join_tables_extracted() -> None:
    sql = (
        "SELECT o.id, u.name FROM orders o"
        " JOIN users u ON o.user_id = u.id WHERE o.status = 'active'"
    )
    capsule = parameterize_cluster(_cluster(sql))
    assert "orders" in capsule.tables
    assert "users" in capsule.tables


def test_multi_table_join_columns_extracted() -> None:
    sql = (
        "SELECT o.id, u.name FROM orders o"
        " JOIN users u ON o.user_id = u.id WHERE o.status = 'active'"
    )
    capsule = parameterize_cluster(_cluster(sql))
    assert "id" in capsule.columns or "status" in capsule.columns  # at least one column found


def test_multi_table_join_placeholder_for_string_filter() -> None:
    sql = (
        "SELECT o.id, u.name FROM orders o"
        " JOIN users u ON o.user_id = u.id WHERE o.status = 'active'"
    )
    capsule = parameterize_cluster(_cluster(sql))
    ph = _placeholder_map(capsule)
    assert ph.get("status") == "VARCHAR"


# ---------------------------------------------------------------------------
# subquery  (required by spec)
# ---------------------------------------------------------------------------


def test_subquery_does_not_raise() -> None:
    sql = "SELECT id FROM users WHERE id IN (SELECT user_id FROM orders WHERE total > 100)"
    capsule = parameterize_cluster(_cluster(sql))
    assert isinstance(capsule, QueryCapsule)


def test_subquery_has_placeholders() -> None:
    sql = "SELECT id FROM users WHERE id IN (SELECT user_id FROM orders WHERE total > 100)"
    capsule = parameterize_cluster(_cluster(sql))
    # 100 is an integer literal — at minimum one INT placeholder
    assert any(p.type == "INT" for p in capsule.placeholders)


def test_subquery_tables_include_both() -> None:
    sql = "SELECT id FROM users WHERE id IN (SELECT user_id FROM orders WHERE total > 100)"
    capsule = parameterize_cluster(_cluster(sql))
    assert "users" in capsule.tables
    assert "orders" in capsule.tables


# ---------------------------------------------------------------------------
# Float literal → DECIMAL
# ---------------------------------------------------------------------------


def test_float_literal_becomes_decimal() -> None:
    sql = "SELECT * FROM products WHERE price > 9.99"
    capsule = parameterize_cluster(_cluster(sql))
    ph = _placeholder_map(capsule)
    assert ph.get("price") == "DECIMAL"


# ---------------------------------------------------------------------------
# Timestamp string → TIMESTAMP
# ---------------------------------------------------------------------------


def test_timestamp_string_becomes_timestamp() -> None:
    sql = "SELECT * FROM events WHERE created_at > '2024-01-01 12:00:00'"
    capsule = parameterize_cluster(_cluster(sql))
    ph = _placeholder_map(capsule)
    assert ph.get("created_at") == "TIMESTAMP"


def test_timestamp_with_t_separator() -> None:
    sql = "SELECT * FROM logs WHERE ts > '2024-06-01T08:30:00'"
    capsule = parameterize_cluster(_cluster(sql))
    ph = _placeholder_map(capsule)
    assert ph.get("ts") == "TIMESTAMP"


# ---------------------------------------------------------------------------
# Schema type lookup
# ---------------------------------------------------------------------------


def test_schema_type_overrides_default_varchar() -> None:
    sql = "SELECT * FROM test_table WHERE status = 'active'"
    # Declare 'status' as TEXT in the schema (should still map to TEXT, not VARCHAR)
    schema = _schema(("status", "TEXT"))
    capsule = parameterize_cluster(_cluster(sql), schema=schema)
    ph = _placeholder_map(capsule)
    assert ph.get("status") == "TEXT"


def test_schema_type_does_not_override_date_detection() -> None:
    # Even if schema says VARCHAR, an ISO date string should → DATE
    sql = "SELECT * FROM test_table WHERE dt = '2024-03-15'"
    schema = _schema(("dt", "VARCHAR"))
    capsule = parameterize_cluster(_cluster(sql), schema=schema)
    ph = _placeholder_map(capsule)
    assert ph.get("dt") == "DATE"


# ---------------------------------------------------------------------------
# IN list — each value uses the same column name with deduplication
# ---------------------------------------------------------------------------


def test_in_list_all_placeholders_reference_column() -> None:
    sql = "SELECT * FROM orders WHERE status IN ('pending', 'active', 'shipped')"
    capsule = parameterize_cluster(_cluster(sql))
    ph = _placeholder_map(capsule)
    assert len(capsule.placeholders) == 3
    # First occurrence: "status", rest: "status_2", "status_3"
    assert "status" in ph
    assert "status_2" in ph
    assert "status_3" in ph
    assert all(v == "VARCHAR" for v in ph.values())


# ---------------------------------------------------------------------------
# Unknown column context → param_N naming
# ---------------------------------------------------------------------------


def test_literal_in_limit_gets_param_name() -> None:
    sql = "SELECT id FROM users LIMIT 10"
    capsule = parameterize_cluster(_cluster(sql))
    assert len(capsule.placeholders) == 1
    assert capsule.placeholders[0].name.startswith("param_")
    assert capsule.placeholders[0].type == "INT"


# ---------------------------------------------------------------------------
# No-literal query
# ---------------------------------------------------------------------------


def test_no_literal_query_has_empty_placeholders() -> None:
    sql = "SELECT id, name FROM users ORDER BY name"
    capsule = parameterize_cluster(_cluster(sql))
    assert capsule.placeholders == []
    assert capsule.template_sql  # non-empty


def test_no_literal_query_tables_extracted() -> None:
    sql = "SELECT id, name FROM users ORDER BY name"
    capsule = parameterize_cluster(_cluster(sql))
    assert capsule.tables == ["users"]


# ---------------------------------------------------------------------------
# auto_description
# ---------------------------------------------------------------------------


def test_auto_description_includes_table() -> None:
    sql = "SELECT id FROM users WHERE status = 'active'"
    capsule = parameterize_cluster(_cluster(sql))
    assert "users" in capsule.auto_description


def test_auto_description_includes_filter_column() -> None:
    sql = "SELECT id FROM users WHERE status = 'active'"
    capsule = parameterize_cluster(_cluster(sql))
    assert "status" in capsule.auto_description


def test_auto_description_no_where_clause() -> None:
    sql = "SELECT id, name FROM customers"
    capsule = parameterize_cluster(_cluster(sql))
    assert capsule.auto_description == "Query on customers"


def test_auto_description_multiple_tables() -> None:
    sql = "SELECT o.id FROM orders o JOIN users u ON o.user_id = u.id WHERE o.total > 100"
    capsule = parameterize_cluster(_cluster(sql))
    assert "orders" in capsule.auto_description
    assert "users" in capsule.auto_description


# ---------------------------------------------------------------------------
# QueryCapsule fields
# ---------------------------------------------------------------------------


def test_frequency_propagated_from_cluster() -> None:
    sql = "SELECT id FROM users WHERE active = true"
    capsule = parameterize_cluster(_cluster(sql, frequency=42))
    assert capsule.frequency == 42


def test_intent_starts_empty() -> None:
    sql = "SELECT id FROM users WHERE active = true"
    capsule = parameterize_cluster(_cluster(sql))
    assert capsule.intent == ""


def test_output_is_query_capsule_instance() -> None:
    sql = "SELECT id FROM users WHERE status = 'active'"
    capsule = parameterize_cluster(_cluster(sql))
    assert isinstance(capsule, QueryCapsule)
    assert isinstance(capsule.placeholders, list)
    assert all(isinstance(p, Placeholder) for p in capsule.placeholders)


# ---------------------------------------------------------------------------
# parameterize_clusters — one capsule per cluster
# ---------------------------------------------------------------------------


def test_parameterize_clusters_length_matches() -> None:
    clusters = [
        _cluster("SELECT id FROM users WHERE id = 1"),
        _cluster("SELECT id FROM orders WHERE status = 'active'"),
        _cluster("SELECT * FROM products WHERE price > 10.0"),
    ]
    capsules = parameterize_clusters(clusters)
    assert len(capsules) == 3


def test_parameterize_clusters_empty_input() -> None:
    assert parameterize_clusters([]) == []


def test_parameterize_clusters_preserves_order() -> None:
    clusters = [
        _cluster("SELECT id FROM users WHERE id = 1", frequency=30),
        _cluster("SELECT id FROM orders WHERE id = 2", frequency=10),
    ]
    capsules = parameterize_clusters(clusters)
    assert capsules[0].frequency == 30
    assert capsules[1].frequency == 10


# ---------------------------------------------------------------------------
# Integration: real filter + cluster → parameterize
# ---------------------------------------------------------------------------


def test_integration_filter_cluster_parameterize() -> None:
    """End-to-end: raw QueryRecords → QueryCapsules with typed placeholders."""
    from nlqueries.connectors.base import QueryRecord
    from nlqueries.processing.query_clusterer import cluster_queries
    from nlqueries.processing.query_filter import filter_and_deduplicate

    records = [
        QueryRecord(
            sql=f"SELECT * FROM orders WHERE status = 'status_{i}' AND total > {i * 10}",
            execution_count=i + 1,
            avg_duration_ms=None,
            last_executed=None,
        )
        for i in range(1, 6)
    ]

    normalized = filter_and_deduplicate(records)
    clusters = cluster_queries(normalized)
    capsules = parameterize_clusters(clusters)

    assert len(capsules) >= 1
    for cap in capsules:
        assert cap.template_sql
        assert cap.frequency > 0
        # All original literals should be replaced
        for i in range(1, 6):
            assert f"status_{i}" not in cap.template_sql
        assert any(p.type == "VARCHAR" for p in cap.placeholders)
        assert any(p.type in ("INT", "DECIMAL") for p in cap.placeholders)


@pytest.mark.parametrize(
    "sql, expected_types",
    [
        ("SELECT * FROM t WHERE id = 1", {"id": "INT"}),
        ("SELECT * FROM t WHERE price = 3.14", {"price": "DECIMAL"}),
        ("SELECT * FROM t WHERE dt = '2024-06-01'", {"dt": "DATE"}),
        ("SELECT * FROM t WHERE ts = '2024-06-01 10:00:00'", {"ts": "TIMESTAMP"}),
        ("SELECT * FROM t WHERE name = 'alice'", {"name": "VARCHAR"}),
    ],
)
def test_literal_type_inference_parametrized(sql: str, expected_types: dict[str, str]) -> None:
    capsule = parameterize_cluster(_cluster(sql))
    ph = _placeholder_map(capsule)
    for col, expected_type in expected_types.items():
        assert ph.get(col) == expected_type, f"{col}: expected {expected_type}, got {ph.get(col)}"


# ---------------------------------------------------------------------------
# skip_string_literals and numeric-literal quoting (semantic-cache fixes)
# ---------------------------------------------------------------------------


def test_skip_string_literals_leaves_varchar_unchanged() -> None:
    from nlqueries.processing.parameterizer import _parameterize_sql

    sql = "SELECT * FROM movies WHERE title_type = 'movie' AND release_year = 2020"
    template, placeholders = _parameterize_sql(sql, {}, skip_string_literals=True)
    ph = {p.name: p.type for p in placeholders}
    # VARCHAR literal should be left as-is (not parameterized)
    assert "title_type" not in ph
    assert "'movie'" in template
    # INT literal should still be parameterized
    assert "release_year" in ph
    assert ph["release_year"] == "INT"


def test_int_placeholder_uses_no_quotes_in_template() -> None:
    from nlqueries.processing.parameterizer import _parameterize_sql

    sql = "SELECT * FROM orders WHERE id = 42"
    template, placeholders = _parameterize_sql(sql, {})
    assert len(placeholders) == 1
    assert placeholders[0].type == "INT"
    # The placeholder must NOT be wrapped in single quotes in the template.
    # This ensures binding produces `id = 42` (integer) not `id = '42'` (string).
    assert "[id:INT]" in template
    assert "'[id:INT]'" not in template


def test_limit_placeholder_uses_no_quotes_in_template() -> None:
    from nlqueries.processing.parameterizer import _parameterize_sql

    sql = "SELECT id FROM users LIMIT 10"
    template, placeholders = _parameterize_sql(sql, {})
    assert len(placeholders) == 1
    assert placeholders[0].type == "INT"
    assert "[param_1:INT]" in template
    assert "'[param_1:INT]'" not in template
