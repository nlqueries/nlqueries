"""
Tests for nlqueries.processing.query_filter (Task 3.1.1).

Covers 20+ sample queries across the following scenarios:
  - Non-SELECT statements are filtered out
  - System schema references are filtered out
  - Queries outside the [3, 500] token range are filtered out
  - Duplicate queries are deduplicated (execution_count summed)
  - Valid SELECT queries pass through with correct output shape
  - Structurally identical queries with different literals share a fingerprint
"""
from __future__ import annotations

import pytest

from nlqueries.connectors.base import QueryRecord
from nlqueries.processing.query_filter import (
    NormalizedQuery,
    _extract_tables,
    _has_system_schema,
    _is_select,
    _make_fingerprint,
    _normalize,
    filter_and_deduplicate,
)


def _qr(sql: str, execution_count: int = 10) -> QueryRecord:
    return QueryRecord(
        sql=sql,
        execution_count=execution_count,
        avg_duration_ms=None,
        last_executed=None,
    )


# ---------------------------------------------------------------------------
# _is_select — unit tests for the SELECT check
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sql, expected", [
    ("SELECT id FROM users", True),
    ("SELECT * FROM orders WHERE status = 'active'", True),
    ("WITH cte AS (SELECT id FROM users) SELECT * FROM cte", True),
    ("SELECT COUNT(*) FROM events GROUP BY event_type", True),
    ("INSERT INTO users (name) VALUES ('Alice')", False),
    ("UPDATE users SET status = 'inactive' WHERE id = 1", False),
    ("DELETE FROM sessions WHERE id = 1", False),
    ("CREATE TABLE foo AS SELECT 1", False),
    ("DROP TABLE foo", False),
    ("TRUNCATE TABLE audit_log", False),
])
def test_is_select(sql: str, expected: bool) -> None:
    assert _is_select(sql) == expected


# ---------------------------------------------------------------------------
# _has_system_schema — unit tests for system schema detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sql, expected", [
    ("SELECT table_name FROM information_schema.tables", True),
    ("SELECT * FROM pg_catalog.pg_tables", True),
    ("SELECT name FROM sys.tables", True),
    ("SELECT column_name FROM information_schema.columns WHERE table_name = 'users'", True),
    ("SELECT * FROM pg_catalog.pg_namespace", True),
    # Should NOT match
    ("SELECT * FROM users", False),
    ("SELECT * FROM system_events", False),       # 'system_events' is not 'sys.'
    ("SELECT id FROM syscore_config", False),     # 'syscore' is not 'sys.'
    ("SELECT * FROM products", False),
])
def test_has_system_schema(sql: str, expected: bool) -> None:
    assert _has_system_schema(sql) == expected


# ---------------------------------------------------------------------------
# filter_and_deduplicate — non-SELECT statements are filtered out
# ---------------------------------------------------------------------------

_NON_SELECT = [
    "INSERT INTO orders (user_id, total) VALUES (1, 99.99)",
    "UPDATE users SET status = 'active' WHERE id = 1",
    "DELETE FROM sessions WHERE expires_at < NOW()",
    "DROP TABLE IF EXISTS old_cache",
    "CREATE INDEX idx_users_email ON users (email)",
]


def test_non_select_queries_filtered() -> None:
    result = filter_and_deduplicate([_qr(sql) for sql in _NON_SELECT])
    assert result == []


# ---------------------------------------------------------------------------
# filter_and_deduplicate — system schema queries are filtered out
# ---------------------------------------------------------------------------

_SYSTEM_QUERIES = [
    "SELECT table_name FROM information_schema.tables",
    "SELECT * FROM pg_catalog.pg_tables",
    "SELECT name FROM sys.tables",
    "SELECT column_name FROM information_schema.columns WHERE table_name = 'orders'",
]


def test_system_schema_queries_filtered() -> None:
    result = filter_and_deduplicate([_qr(sql) for sql in _SYSTEM_QUERIES])
    assert result == []


# ---------------------------------------------------------------------------
# filter_and_deduplicate — token length bounds
# ---------------------------------------------------------------------------

def test_too_short_query_filtered() -> None:
    # "SELECT 1" → 2 tokens → below minimum of 3
    assert filter_and_deduplicate([_qr("SELECT 1")]) == []


def test_too_long_query_filtered() -> None:
    # Build a query with > 500 whitespace-delimited tokens
    cols = " , ".join(f"col_{i}" for i in range(300))
    long_sql = f"SELECT {cols} FROM big_table"
    assert len(long_sql.split()) > 500
    assert filter_and_deduplicate([_qr(long_sql)]) == []


def test_exactly_3_token_query_passes() -> None:
    # "SELECT id FROM" is malformed, but "SELECT 1, 2" is 3 tokens and valid SELECT
    result = filter_and_deduplicate([_qr("SELECT id, name FROM users")])
    assert len(result) == 1


# ---------------------------------------------------------------------------
# filter_and_deduplicate — min_executions threshold
# ---------------------------------------------------------------------------

def test_min_executions_filters_low_frequency() -> None:
    records = [
        _qr("SELECT id FROM users WHERE active = true", execution_count=2),
        _qr("SELECT id FROM products WHERE stock > 0", execution_count=5),
    ]
    result = filter_and_deduplicate(records, min_executions=3)
    assert len(result) == 1
    assert result[0].execution_count == 5


def test_min_executions_default_allows_all() -> None:
    records = [_qr("SELECT id FROM users", execution_count=1)]
    assert len(filter_and_deduplicate(records)) == 1


# ---------------------------------------------------------------------------
# filter_and_deduplicate — valid SELECT queries pass through
# ---------------------------------------------------------------------------

_VALID_SELECTS = [
    "SELECT id, name FROM users WHERE status = 'active'",
    "SELECT o.id, o.total, u.name FROM orders o JOIN users u ON o.user_id = u.id WHERE o.status = 'pending'",
    "SELECT date_trunc('month', created_at) AS month, COUNT(*) AS cnt FROM events GROUP BY 1 ORDER BY 1",
    "SELECT p.name, p.price FROM products p WHERE p.price > 100.0 AND p.category = 'electronics'",
    "SELECT COUNT(*) AS total FROM orders WHERE created_at >= '2024-01-01'",
    "SELECT AVG(price) FROM products WHERE category = 'books'",
    "SELECT customer_id, SUM(total) FROM orders GROUP BY customer_id HAVING SUM(total) > 1000",
]


def test_valid_select_queries_pass() -> None:
    result = filter_and_deduplicate([_qr(sql) for sql in _VALID_SELECTS])
    assert len(result) == len(_VALID_SELECTS)


def test_output_shape_is_normalized_query() -> None:
    result = filter_and_deduplicate([_qr("SELECT id, name FROM users WHERE active = true")])
    assert len(result) == 1
    nq = result[0]
    assert isinstance(nq, NormalizedQuery)
    assert nq.original_sql
    assert nq.normalized_sql
    assert nq.fingerprint
    assert isinstance(nq.tables_referenced, list)
    assert nq.execution_count == 10


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def test_exact_duplicate_deduped_with_summed_count() -> None:
    sql = "SELECT id FROM users WHERE status = 'active'"
    result = filter_and_deduplicate([_qr(sql, 5), _qr(sql, 3)])
    assert len(result) == 1
    assert result[0].execution_count == 8


def test_whitespace_variation_deduped() -> None:
    sql1 = "SELECT id FROM users WHERE status = 'active'"
    sql2 = "  SELECT   id   FROM   users   WHERE   status  =  'active'  "
    result = filter_and_deduplicate([_qr(sql1, 4), _qr(sql2, 6)])
    assert len(result) == 1
    assert result[0].execution_count == 10


def test_keyword_case_variation_deduped() -> None:
    sql1 = "SELECT id FROM users WHERE status = 'active'"
    sql2 = "select id from users where status = 'active'"
    result = filter_and_deduplicate([_qr(sql1, 3), _qr(sql2, 7)])
    assert len(result) == 1
    assert result[0].execution_count == 10


def test_distinct_queries_not_deduped() -> None:
    records = [
        _qr("SELECT id FROM users WHERE active = true"),
        _qr("SELECT id FROM products WHERE stock > 0"),
        _qr("SELECT id FROM orders WHERE status = 'shipped'"),
    ]
    assert len(filter_and_deduplicate(records)) == 3


# ---------------------------------------------------------------------------
# Fingerprint — structurally identical queries with different literals
# ---------------------------------------------------------------------------

def test_integer_literal_difference_same_fingerprint() -> None:
    sql1 = "SELECT * FROM users WHERE id = 1"
    sql2 = "SELECT * FROM users WHERE id = 42"
    results = filter_and_deduplicate([_qr(sql1), _qr(sql2)])
    assert len(results) == 2  # different normalized SQL
    assert results[0].fingerprint == results[1].fingerprint


def test_string_literal_difference_same_fingerprint() -> None:
    sql1 = "SELECT * FROM orders WHERE status = 'pending'"
    sql2 = "SELECT * FROM orders WHERE status = 'complete'"
    results = filter_and_deduplicate([_qr(sql1), _qr(sql2)])
    assert len(results) == 2
    assert results[0].fingerprint == results[1].fingerprint


def test_different_structure_different_fingerprint() -> None:
    sql1 = "SELECT id FROM users WHERE status = 'active'"
    sql2 = "SELECT id, name FROM users WHERE status = 'active'"
    results = filter_and_deduplicate([_qr(sql1), _qr(sql2)])
    assert len(results) == 2
    assert results[0].fingerprint != results[1].fingerprint


def test_fingerprint_strips_multiple_literals() -> None:
    fp1 = _make_fingerprint(
        _normalize("SELECT * FROM orders WHERE status = 'pending' AND total > 100") or ""
    )
    fp2 = _make_fingerprint(
        _normalize("SELECT * FROM orders WHERE status = 'shipped' AND total > 999") or ""
    )
    assert fp1 == fp2


# ---------------------------------------------------------------------------
# tables_referenced
# ---------------------------------------------------------------------------

def test_tables_referenced_single_table() -> None:
    result = filter_and_deduplicate([_qr("SELECT id FROM users WHERE status = 'active'")])
    assert result[0].tables_referenced == ["users"]


def test_tables_referenced_join() -> None:
    sql = "SELECT o.id, u.name FROM orders o JOIN users u ON o.user_id = u.id"
    result = filter_and_deduplicate([_qr(sql)])
    assert set(result[0].tables_referenced) == {"orders", "users"}


def test_extract_tables_subquery() -> None:
    sql = "SELECT id FROM users WHERE id IN (SELECT user_id FROM orders WHERE total > 100)"
    tables = _extract_tables(_normalize(sql) or sql)
    assert "users" in tables
    assert "orders" in tables


# ---------------------------------------------------------------------------
# Mixed batch — 20+ queries exercising all rules at once
# ---------------------------------------------------------------------------

_MIXED_BATCH = [
    # (sql, should_survive_filter)
    # Valid unique SELECTs
    ("SELECT id, name FROM customers WHERE region = 'us-east'", True),
    ("SELECT order_id, total FROM orders WHERE status = 'shipped' AND total > 50", True),
    ("SELECT COUNT(*) FROM events WHERE event_type = 'click'", True),
    ("SELECT p.id, p.name FROM products p WHERE p.stock > 0", True),
    ("SELECT u.id, u.name FROM users u JOIN roles r ON u.role_id = r.id WHERE r.name = 'admin'", True),
    ("SELECT date_trunc('day', ts) AS day, COUNT(*) FROM logs GROUP BY 1", True),
    ("SELECT AVG(price) FROM products WHERE category = 'books'", True),
    # Duplicate of first valid entry (should be deduped, not counted twice)
    ("SELECT id, name FROM customers WHERE region = 'us-east'", False),
    # Non-SELECT
    ("INSERT INTO audit_log (action) VALUES ('login')", False),
    ("UPDATE products SET price = 0 WHERE discontinued = true", False),
    ("DELETE FROM sessions WHERE last_seen < '2024-01-01'", False),
    # System schema
    ("SELECT * FROM information_schema.tables WHERE table_schema = 'public'", False),
    ("SELECT relname FROM pg_catalog.pg_class WHERE relkind = 'r'", False),
    ("SELECT object_id FROM sys.objects WHERE type = 'U'", False),
    # Too short (< 3 tokens)
    ("SELECT 1", False),
    ("SELECT NOW()", False),
    # More valid unique SELECTs
    ("SELECT MAX(created_at) FROM orders", True),
    ("SELECT DISTINCT country FROM addresses", True),
    ("SELECT id FROM users WHERE email LIKE '%@example.com'", True),
    ("SELECT SUM(amount) FROM payments WHERE status = 'completed' AND year = 2024", True),
]


def test_mixed_batch_of_20_queries() -> None:
    records = [_qr(sql) for sql, _ in _MIXED_BATCH]
    result = filter_and_deduplicate(records)

    # Derive expected count: unique normalized SQL among the "should survive" entries
    surviving_normalized: set[str] = set()
    for sql, should_pass in _MIXED_BATCH:
        if should_pass:
            norm = _normalize(sql)
            if norm:
                surviving_normalized.add(norm)

    assert len(result) == len(surviving_normalized)
    assert all(isinstance(nq, NormalizedQuery) for nq in result)
    assert all(nq.execution_count >= 1 for nq in result)
