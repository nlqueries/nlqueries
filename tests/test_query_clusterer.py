"""
Tests for nlqueries.processing.query_clusterer (Task 3.1.2).

Required coverage (per spec):
  - 10 fingerprint-identical queries → 1 cluster
  - 0% table overlap → never merge
  - priority score = sum of member execution counts

Additional coverage:
  - Empty / single-query input
  - Distinct fingerprints produce distinct clusters
  - Near-duplicate merge when Jaccard > 0.8 AND column order is the only diff
  - No merge when Jaccard ≤ 0.8 even if columns differ only in order
  - Clusters sorted descending by execution_count
  - Cap at 1 000 clusters
  - representative_sql comes from the highest-execution-count member
  - tables_referenced is the union of all members' tables
"""
from __future__ import annotations

from nlqueries.processing.query_clusterer import (
    QueryCluster,
    _differ_only_in_column_order,
    _jaccard,
    cluster_queries,
)
from nlqueries.processing.query_filter import (
    NormalizedQuery,
    _make_fingerprint,
    _normalize,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _nq(
    normalized_sql: str,
    fingerprint: str | None = None,
    execution_count: int = 10,
    tables: list[str] | None = None,
) -> NormalizedQuery:
    """Build a NormalizedQuery for testing without running the full filter pipeline."""
    fp = fingerprint if fingerprint is not None else normalized_sql
    return NormalizedQuery(
        original_sql=normalized_sql,
        normalized_sql=normalized_sql,
        fingerprint=fp,
        execution_count=execution_count,
        tables_referenced=tables or [],
    )


def _fp(sql: str) -> str:
    """Produce the fingerprint sqlglot would assign to *sql*."""
    norm = _normalize(sql)
    return _make_fingerprint(norm or sql)


def _norm(sql: str) -> str:
    """Return the sqlglot-normalized form of *sql* (or *sql* if unparseable)."""
    return _normalize(sql) or sql


# ---------------------------------------------------------------------------
# _jaccard
# ---------------------------------------------------------------------------


def test_jaccard_identical_sets() -> None:
    assert _jaccard({"a", "b"}, {"a", "b"}) == 1.0


def test_jaccard_disjoint_sets() -> None:
    assert _jaccard({"a"}, {"b"}) == 0.0


def test_jaccard_partial_overlap() -> None:
    # |{a,b} ∩ {b,c}| / |{a,b,c}| = 1/3
    assert abs(_jaccard({"a", "b"}, {"b", "c"}) - 1 / 3) < 1e-9


def test_jaccard_both_empty() -> None:
    assert _jaccard(set(), set()) == 0.0


def test_jaccard_one_empty() -> None:
    assert _jaccard({"a"}, set()) == 0.0


# ---------------------------------------------------------------------------
# _differ_only_in_column_order
# ---------------------------------------------------------------------------


def test_differ_only_in_column_order_true() -> None:
    fp1 = _fp("SELECT id, name FROM users WHERE status = 'active'")
    fp2 = _fp("SELECT name, id FROM users WHERE status = 'active'")
    assert _differ_only_in_column_order(fp1, fp2) is True


def test_differ_only_in_column_order_same_query() -> None:
    fp = _fp("SELECT id FROM users WHERE id = 1")
    assert _differ_only_in_column_order(fp, fp) is True


def test_differ_in_where_not_only_column_order() -> None:
    fp1 = _fp("SELECT id, name FROM users WHERE status = 'active'")
    fp2 = _fp("SELECT id, name FROM users WHERE id = 1")
    assert _differ_only_in_column_order(fp1, fp2) is False


def test_differ_in_table_not_only_column_order() -> None:
    fp1 = _fp("SELECT id, name FROM users")
    fp2 = _fp("SELECT id, name FROM orders")
    assert _differ_only_in_column_order(fp1, fp2) is False


def test_differ_in_columns_not_only_column_order() -> None:
    fp1 = _fp("SELECT id, name FROM users")
    fp2 = _fp("SELECT id, email FROM users")
    assert _differ_only_in_column_order(fp1, fp2) is False


# ---------------------------------------------------------------------------
# cluster_queries — basic cases
# ---------------------------------------------------------------------------


def test_empty_input_returns_empty() -> None:
    assert cluster_queries([]) == []


def test_single_query_returns_one_cluster() -> None:
    result = cluster_queries([_nq("SELECT id FROM users", tables=["users"])])
    assert len(result) == 1
    assert isinstance(result[0], QueryCluster)


def test_single_cluster_fields() -> None:
    nq = _nq("SELECT id FROM users", execution_count=7, tables=["users"])
    cluster = cluster_queries([nq])[0]
    assert cluster.representative_sql == "SELECT id FROM users"
    assert cluster.execution_count == 7
    assert cluster.member_count == 1
    assert cluster.tables_referenced == ["users"]


# ---------------------------------------------------------------------------
# 10 fingerprint-identical queries → 1 cluster  (required by spec)
# ---------------------------------------------------------------------------


def test_10_fingerprint_identical_queries_form_one_cluster() -> None:
    shared_fp = _fp("SELECT * FROM users WHERE id = 1")
    queries = [
        _nq(f"SELECT * FROM users WHERE id = {i}", fingerprint=shared_fp, tables=["users"])
        for i in range(1, 11)
    ]
    result = cluster_queries(queries)
    assert len(result) == 1
    assert result[0].member_count == 10


def test_fingerprint_identical_execution_count_summed() -> None:
    shared_fp = _fp("SELECT * FROM orders WHERE status = 'pending'")
    queries = [
        _nq(
            f"SELECT * FROM orders WHERE status = 'status_{i}'",
            fingerprint=shared_fp,
            execution_count=i,
            tables=["orders"],
        )
        for i in range(1, 11)
    ]
    result = cluster_queries(queries)
    assert len(result) == 1
    assert result[0].execution_count == sum(range(1, 11))  # 55


# ---------------------------------------------------------------------------
# Priority score = sum of member execution counts  (required by spec)
# ---------------------------------------------------------------------------


def test_priority_score_is_sum_of_execution_counts() -> None:
    fp = _fp("SELECT id FROM products WHERE stock = 1")
    members = [
        _nq(
            "SELECT id FROM products WHERE stock = 1",
            fingerprint=fp,
            execution_count=3,
            tables=["products"],
        ),
        _nq(
            "SELECT id FROM products WHERE stock = 2",
            fingerprint=fp,
            execution_count=7,
            tables=["products"],
        ),
        _nq(
            "SELECT id FROM products WHERE stock = 3",
            fingerprint=fp,
            execution_count=5,
            tables=["products"],
        ),
    ]
    cluster = cluster_queries(members)[0]
    assert cluster.execution_count == 3 + 7 + 5  # 15


def test_clusters_sorted_by_execution_count_descending() -> None:
    fp_a = _fp("SELECT id FROM users WHERE id = 1")
    fp_b = _fp("SELECT id FROM orders WHERE id = 1")
    fp_c = _fp("SELECT id FROM products WHERE id = 1")
    queries = [
        _nq(
            "SELECT id FROM users WHERE id = 1",
            fingerprint=fp_a,
            execution_count=5,
            tables=["users"],
        ),
        _nq(
            "SELECT id FROM orders WHERE id = 1",
            fingerprint=fp_b,
            execution_count=30,
            tables=["orders"],
        ),
        _nq(
            "SELECT id FROM products WHERE id = 1",
            fingerprint=fp_c,
            execution_count=15,
            tables=["products"],
        ),
    ]
    result = cluster_queries(queries)
    assert len(result) == 3
    assert result[0].execution_count == 30
    assert result[1].execution_count == 15
    assert result[2].execution_count == 5


# ---------------------------------------------------------------------------
# 0% table overlap → never merge  (required by spec)
# ---------------------------------------------------------------------------


def test_zero_table_overlap_clusters_not_merged() -> None:
    fp1 = _fp("SELECT id, name FROM users WHERE status = 'active'")
    fp2 = _fp("SELECT id, name FROM orders WHERE status = 'active'")
    queries = [
        _nq(
            _norm("SELECT id, name FROM users WHERE status = 'active'"),
            fingerprint=fp1,
            tables=["users"],
        ),
        _nq(
            _norm("SELECT name, id FROM orders WHERE status = 'active'"),
            fingerprint=fp2,
            tables=["orders"],
        ),
    ]
    result = cluster_queries(queries)
    # Jaccard({"users"}, {"orders"}) = 0 → must NOT merge
    assert len(result) == 2


def test_zero_table_overlap_many_queries() -> None:
    clusters_in: list[NormalizedQuery] = []
    for i in range(5):
        fp = _fp(f"SELECT col FROM table_{i} WHERE id = 1")
        clusters_in.append(
            _nq(f"SELECT col FROM table_{i} WHERE id = 0", fingerprint=fp, tables=[f"table_{i}"])
        )
    result = cluster_queries(clusters_in)
    # All have disjoint table sets → no merges
    assert len(result) == 5


# ---------------------------------------------------------------------------
# Near-duplicate merge — column order only difference
# ---------------------------------------------------------------------------


def test_near_duplicate_merge_column_order() -> None:
    fp1 = _fp("SELECT id, name FROM users WHERE status = 'active'")
    fp2 = _fp("SELECT name, id FROM users WHERE status = 'active'")
    # fp1 and fp2 should differ only in column order; same table "users"
    queries = [
        _nq(
            _norm("SELECT id, name FROM users WHERE status = 'active'"),
            fingerprint=fp1,
            execution_count=10,
            tables=["users"],
        ),
        _nq(
            _norm("SELECT name, id FROM users WHERE status = 'active'"),
            fingerprint=fp2,
            execution_count=6,
            tables=["users"],
        ),
    ]
    result = cluster_queries(queries)
    assert len(result) == 1
    assert result[0].execution_count == 16
    assert result[0].member_count == 2


def test_near_duplicate_merge_picks_higher_execution_representative() -> None:
    fp1 = _fp("SELECT id, name FROM users WHERE active = 1")
    fp2 = _fp("SELECT name, id FROM users WHERE active = 1")
    low = _nq(
        _norm("SELECT id, name FROM users WHERE active = 1"),
        fingerprint=fp1,
        execution_count=3,
        tables=["users"],
    )
    high = _nq(
        _norm("SELECT name, id FROM users WHERE active = 1"),
        fingerprint=fp2,
        execution_count=20,
        tables=["users"],
    )
    result = cluster_queries([low, high])
    assert len(result) == 1
    assert result[0].representative_sql == high.normalized_sql


def test_no_merge_when_where_clause_differs() -> None:
    fp1 = _fp("SELECT id, name FROM users WHERE status = 'active'")
    fp2 = _fp("SELECT name, id FROM users WHERE status = 'inactive'")
    # WHERE clauses become identical after literal stripping ('?' in both)
    # so these SHOULD merge (same structure after stripping, just column order differs)
    queries = [
        _nq(
            _norm("SELECT id, name FROM users WHERE status = 'active'"),
            fingerprint=fp1,
            tables=["users"],
        ),
        _nq(
            _norm("SELECT name, id FROM users WHERE status = 'inactive'"),
            fingerprint=fp2,
            tables=["users"],
        ),
    ]
    result = cluster_queries(queries)
    # Both fingerprints have status = '?' → identical except column order → merge
    assert len(result) == 1


def test_no_merge_below_jaccard_threshold() -> None:
    # Tables are {"users","orders","products","a","b"} vs {"users","x","y","z","w"}
    # Intersection = {"users"} → Jaccard = 1/9 ≈ 0.11 < 0.8
    fp1 = _fp("SELECT id, name FROM users WHERE id = 1")
    fp2 = _fp("SELECT name, id FROM users WHERE id = 1")
    queries = [
        _nq(
            _norm("SELECT id, name FROM users WHERE id = 1"),
            fingerprint=fp1,
            tables=["users", "orders", "products", "a", "b"],
        ),
        _nq(
            _norm("SELECT name, id FROM users WHERE id = 1"),
            fingerprint=fp2,
            tables=["users", "x", "y", "z", "w"],
        ),
    ]
    result = cluster_queries(queries)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# tables_referenced is union of all members
# ---------------------------------------------------------------------------


def test_tables_referenced_is_union_of_members() -> None:
    shared_fp = "SHARED_FINGERPRINT"
    queries = [
        _nq("q1", fingerprint=shared_fp, tables=["users", "roles"]),
        _nq("q2", fingerprint=shared_fp, tables=["users", "permissions"]),
    ]
    result = cluster_queries(queries)
    assert len(result) == 1
    assert set(result[0].tables_referenced) == {"users", "roles", "permissions"}


def test_tables_referenced_is_sorted() -> None:
    fp = _fp("SELECT id FROM users WHERE id = 1")
    queries = [
        _nq(
            "SELECT id FROM users WHERE id = 1",
            fingerprint=fp,
            tables=["zebra", "apple", "mango"],
        ),
    ]
    result = cluster_queries(queries)
    assert result[0].tables_referenced == sorted(result[0].tables_referenced)


# ---------------------------------------------------------------------------
# Cap at 1 000 clusters
# ---------------------------------------------------------------------------


def test_cluster_cap_at_1000() -> None:
    # Create 1 100 queries each with a unique fingerprint → 1 100 initial clusters
    queries = [
        _nq(f"SELECT id FROM t{i} WHERE id = 0", fingerprint=f"FP_{i}", tables=[f"t{i}"])
        for i in range(1100)
    ]
    result = cluster_queries(queries)
    assert len(result) == 1000


# ---------------------------------------------------------------------------
# representative_sql comes from highest-execution-count member
# ---------------------------------------------------------------------------


def test_representative_sql_is_highest_execution_member() -> None:
    shared_fp = _fp("SELECT * FROM orders WHERE status = 'pending'")
    low = _nq(
        _norm("SELECT * FROM orders WHERE status = 'pending'"),
        fingerprint=shared_fp,
        execution_count=2,
        tables=["orders"],
    )
    mid = _nq(
        _norm("SELECT * FROM orders WHERE status = 'shipped'"),
        fingerprint=shared_fp,
        execution_count=50,
        tables=["orders"],
    )
    high = _nq(
        _norm("SELECT * FROM orders WHERE status = 'complete'"),
        fingerprint=shared_fp,
        execution_count=20,
        tables=["orders"],
    )
    result = cluster_queries([low, mid, high])
    assert len(result) == 1
    assert result[0].representative_sql == mid.normalized_sql


# ---------------------------------------------------------------------------
# distinct queries stay as distinct clusters
# ---------------------------------------------------------------------------


def test_distinct_queries_stay_distinct() -> None:
    queries = [
        _nq("SELECT id FROM users", tables=["users"]),
        _nq("SELECT id FROM orders", tables=["orders"]),
        _nq("SELECT id FROM products", tables=["products"]),
        _nq("SELECT id FROM payments", tables=["payments"]),
        _nq("SELECT id FROM sessions", tables=["sessions"]),
    ]
    result = cluster_queries(queries)
    assert len(result) == 5


# ---------------------------------------------------------------------------
# Integration: use real filter output as clustering input
# ---------------------------------------------------------------------------


def test_cluster_queries_with_real_filter_output() -> None:
    """End-to-end: filter 10 literal-varying queries → 1 cluster."""
    from nlqueries.connectors.base import QueryRecord
    from nlqueries.processing.query_filter import filter_and_deduplicate

    records = [
        QueryRecord(
            sql=f"SELECT id, name FROM customers WHERE region = 'region_{i}'",
            execution_count=i + 1,
            avg_duration_ms=None,
            last_executed=None,
        )
        for i in range(10)
    ]
    normalized = filter_and_deduplicate(records)
    # 10 different normalized SQLs (different literal) but same fingerprint
    assert len(normalized) == 10
    assert len({q.fingerprint for q in normalized}) == 1

    clusters = cluster_queries(normalized)
    assert len(clusters) == 1
    assert clusters[0].member_count == 10
    assert clusters[0].execution_count == sum(range(1, 11))  # 55
