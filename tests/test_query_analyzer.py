"""Tests for nlqueries.analysis.query_analyzer (Task 21.2).

All connector.execute_query and LLMClient calls are mocked — no live database
or LLM calls are made.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

from nlqueries.analysis.query_analyzer import (
    SLOW_QUERY_THRESHOLD_MS,
    QueryAnalysis,
    QueryPlanEntry,
    analyze_query,
)
from nlqueries.connectors.base import QueryResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_connector(explain_result: QueryResult | None = None) -> MagicMock:
    """Return a mock DatabaseConnector whose execute_query() returns explain_result."""
    connector = MagicMock()
    if explain_result is not None:
        connector.execute_query.return_value = explain_result
    return connector


def _make_explain_result(plan_json: list[dict[str, Any]]) -> QueryResult:
    """Build a QueryResult that looks like a Postgres EXPLAIN (ANALYZE, FORMAT JSON) response."""
    return QueryResult(
        columns=["QUERY PLAN"],
        rows=[[json.dumps(plan_json)]],
        row_count=1,
        execution_time_ms=1.0,
        error=None,
    )


def _seq_scan_plan(relation: str = "orders", plan_rows: int = 50_000) -> list[dict[str, Any]]:
    """Minimal EXPLAIN JSON with a Seq Scan on a large table."""
    return [
        {
            "Plan": {
                "Node Type": "Seq Scan",
                "Relation Name": relation,
                "Plan Rows": plan_rows,
                "Actual Rows": plan_rows,
                "Actual Total Time": 2500.0,
                "Total Cost": 9999.0,
                "Actual Loops": 1,
                "Filter": "status = 'active'",
            },
            "Planning Time": 1.0,
            "Execution Time": 2500.0,
        }
    ]


# ---------------------------------------------------------------------------
# test_fast_query_is_not_slow
# ---------------------------------------------------------------------------


class TestFastQueryIsNotSlow:
    def test_below_threshold_is_not_slow(self) -> None:
        """A query that finishes below SLOW_QUERY_THRESHOLD_MS must have is_slow=False."""
        connector = _make_connector(
            explain_result=QueryResult(
                columns=["QUERY PLAN"],
                rows=[],
                row_count=0,
                execution_time_ms=0.5,
                error="no rows",  # causes plan to be skipped
            )
        )

        result = analyze_query(
            sql="SELECT COUNT(*) FROM orders",
            dialect="postgres",
            execution_time_ms=500,
            connector=connector,
        )

        assert isinstance(result, QueryAnalysis)
        assert result.is_slow is False
        assert result.execution_time_ms == 500

    def test_exactly_at_threshold_minus_one_is_not_slow(self) -> None:
        """execution_time_ms = SLOW_QUERY_THRESHOLD_MS - 1 should be fast."""
        connector = _make_connector(
            explain_result=QueryResult(
                columns=[], rows=[], row_count=0, execution_time_ms=0.0, error="skip"
            )
        )
        result = analyze_query(
            sql="SELECT 1",
            dialect="postgres",
            execution_time_ms=SLOW_QUERY_THRESHOLD_MS - 1,
            connector=connector,
        )
        assert result.is_slow is False


# ---------------------------------------------------------------------------
# test_slow_query_flagged
# ---------------------------------------------------------------------------


class TestSlowQueryFlagged:
    def test_above_threshold_is_slow(self) -> None:
        """A query that takes 3000 ms must have is_slow=True."""
        connector = _make_connector(
            explain_result=QueryResult(
                columns=[], rows=[], row_count=0, execution_time_ms=0.0, error="skip"
            )
        )

        result = analyze_query(
            sql="SELECT * FROM large_table",
            dialect="postgres",
            execution_time_ms=3000,
            connector=connector,
        )

        assert result.is_slow is True
        assert result.execution_time_ms == 3000

    def test_exactly_at_threshold_is_slow(self) -> None:
        """execution_time_ms == SLOW_QUERY_THRESHOLD_MS is considered slow (>= boundary)."""
        connector = _make_connector(
            explain_result=QueryResult(
                columns=[], rows=[], row_count=0, execution_time_ms=0.0, error="skip"
            )
        )
        result = analyze_query(
            sql="SELECT 1",
            dialect="postgres",
            execution_time_ms=SLOW_QUERY_THRESHOLD_MS,
            connector=connector,
        )
        assert result.is_slow is True


# ---------------------------------------------------------------------------
# test_seq_scan_warning_detected
# ---------------------------------------------------------------------------


class TestSeqScanWarningDetected:
    def test_seq_scan_on_large_table_produces_warning(self) -> None:
        """A Seq Scan node with Plan Rows > 10 000 must appear in warnings."""
        explain_result = _make_explain_result(_seq_scan_plan(relation="orders", plan_rows=50_000))
        connector = _make_connector(explain_result=explain_result)

        result = analyze_query(
            sql="SELECT * FROM orders WHERE status = 'active'",
            dialect="postgres",
            execution_time_ms=2500,
            connector=connector,
        )

        assert result.plan is not None
        assert len(result.plan) >= 1
        assert result.plan[0].node_type == "Seq Scan"
        assert result.plan[0].estimated_rows == 50_000

        assert any("orders" in w for w in result.warnings)
        assert any("Sequential scan" in w for w in result.warnings)

    def test_seq_scan_on_small_table_no_warning(self) -> None:
        """A Seq Scan on a table with <= 10 000 rows must not produce a warning."""
        explain_result = _make_explain_result(_seq_scan_plan(relation="config", plan_rows=100))
        connector = _make_connector(explain_result=explain_result)

        result = analyze_query(
            sql="SELECT * FROM config",
            dialect="postgres",
            execution_time_ms=50,
            connector=connector,
        )

        assert result.warnings == []

    def test_plan_entries_populated_from_explain(self) -> None:
        """QueryPlanEntry fields are correctly parsed from the EXPLAIN JSON."""
        explain_result = _make_explain_result(_seq_scan_plan(relation="users", plan_rows=20_000))
        connector = _make_connector(explain_result=explain_result)

        result = analyze_query(
            sql="SELECT * FROM users",
            dialect="postgres",
            execution_time_ms=100,
            connector=connector,
        )

        assert result.plan is not None
        entry = result.plan[0]
        assert isinstance(entry, QueryPlanEntry)
        assert entry.estimated_rows == 20_000
        assert entry.cost is not None
        assert entry.actual_time_ms is not None


# ---------------------------------------------------------------------------
# test_recommendation_generated_only_when_flag_true
# ---------------------------------------------------------------------------


class TestRecommendationGeneratedOnlyWhenFlagTrue:
    def test_no_recommendation_when_flag_false(self) -> None:
        """With generate_recommendation=False, LLM is not called and recommendation is None."""
        connector = _make_connector(
            explain_result=QueryResult(
                columns=[], rows=[], row_count=0, execution_time_ms=0.0, error="skip"
            )
        )

        mock_llm = MagicMock()
        mock_llm.complete.return_value = "Add an index."

        with patch("nlqueries.analysis.query_analyzer._generate_recommendation") as mock_gen:
            result = analyze_query(
                sql="SELECT * FROM big_table",
                dialect="postgres",
                execution_time_ms=5000,
                connector=connector,
                generate_recommendation=False,
            )
            mock_gen.assert_not_called()

        assert result.recommendation is None

    def test_recommendation_generated_for_slow_query_with_flag_true(self) -> None:
        """With generate_recommendation=True and a slow query, LLM is called."""
        connector = _make_connector(
            explain_result=QueryResult(
                columns=[], rows=[], row_count=0, execution_time_ms=0.0, error="skip"
            )
        )
        expected_text = "Add an index on the status column to avoid the sequential scan."

        with patch(
            "nlqueries.analysis.query_analyzer._generate_recommendation",
            return_value=expected_text,
        ) as mock_gen:
            result = analyze_query(
                sql="SELECT * FROM orders WHERE status = 'pending'",
                dialect="postgres",
                execution_time_ms=4000,
                connector=connector,
                generate_recommendation=True,
            )
            mock_gen.assert_called_once()

        assert result.recommendation == expected_text
        assert result.is_slow is True

    def test_no_recommendation_for_fast_query_even_with_flag_true(self) -> None:
        """Even with flag=True, a fast query gets no recommendation."""
        connector = _make_connector(
            explain_result=QueryResult(
                columns=[], rows=[], row_count=0, execution_time_ms=0.0, error="skip"
            )
        )

        with patch("nlqueries.analysis.query_analyzer._generate_recommendation") as mock_gen:
            result = analyze_query(
                sql="SELECT 1",
                dialect="postgres",
                execution_time_ms=100,
                connector=connector,
                generate_recommendation=True,
            )
            mock_gen.assert_not_called()

        assert result.recommendation is None

    def test_llm_complete_called_with_sql_and_time(self) -> None:
        """The real _generate_recommendation path calls llm.complete with the SQL and timing."""
        connector = _make_connector(
            explain_result=QueryResult(
                columns=[], rows=[], row_count=0, execution_time_ms=0.0, error="skip"
            )
        )

        mock_llm = MagicMock()
        mock_llm.complete.return_value = "Create an index on the join column."

        with patch("nlqueries.llm.get_llm_client", return_value=mock_llm):
            result = analyze_query(
                sql="SELECT * FROM orders JOIN customers ON orders.cid = customers.id",
                dialect="postgres",
                execution_time_ms=3500,
                connector=connector,
                generate_recommendation=True,
            )

        mock_llm.complete.assert_called_once()
        call_args = mock_llm.complete.call_args
        user_prompt: str = call_args.args[1] if call_args.args else call_args.kwargs["user"]
        assert "3500" in user_prompt
        assert result.recommendation == "Create an index on the join column."


# ---------------------------------------------------------------------------
# test_plan_is_none_for_bigquery_dialect
# ---------------------------------------------------------------------------


class TestPlanIsNoneForBigqueryDialect:
    def test_bigquery_plan_is_none(self) -> None:
        """For BigQuery dialect, plan must be None and execute_query must not be called."""
        connector = _make_connector()

        result = analyze_query(
            sql="SELECT COUNT(*) FROM `project.dataset.orders`",
            dialect="bigquery",
            execution_time_ms=3000,
            connector=connector,
        )

        assert result.plan is None
        connector.execute_query.assert_not_called()

    def test_snowflake_plan_is_none(self) -> None:
        """For Snowflake dialect, plan must be None and execute_query must not be called."""
        connector = _make_connector()

        result = analyze_query(
            sql="SELECT COUNT(*) FROM orders",
            dialect="snowflake",
            execution_time_ms=3000,
            connector=connector,
        )

        assert result.plan is None
        connector.execute_query.assert_not_called()

    def test_bigquery_warnings_empty(self) -> None:
        """No warnings can be generated without a plan."""
        connector = _make_connector()

        result = analyze_query(
            sql="SELECT * FROM orders",
            dialect="bigquery",
            execution_time_ms=500,
            connector=connector,
        )

        assert result.warnings == []
