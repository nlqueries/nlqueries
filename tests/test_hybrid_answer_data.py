"""A hybrid answer must show the data it is answering from.

`_extract_sql_query_result` read only the `sql` key of the SQL sub-agent's
final chunk and built a one-column, one-row table holding the statement text.
The sub-agent had already executed that statement and reported the rows in the
same chunk, under `sql_table` — so a hybrid answer asserted things about data
it discarded, and could not report truncation at all, because `truncated` and
`truncation_reason` live on the frame it threw away.

The statement-only table is still right when nothing ran, which is why these
tests pin both directions rather than only the new one.
"""

from __future__ import annotations

import json
from typing import Any

from nlqueries.connectors.base import QueryResult
from nlqueries.orchestrator.multi_agent_orchestrator import _extract_sql_query_result
from nlqueries.orchestrator.orchestrator import _MAX_RESULT_ROWS, sql_table_chunk


def _chunk(**overrides: Any) -> list[str]:
    payload: dict[str, Any] = {
        "type": "sql",
        "sql": "SELECT region, revenue FROM sales",
        "is_valid": True,
        "execution_mode": "execute",
        "validation_error": None,
        "dialect": "postgres",
        "attempt_count": 1,
        "sql_table": None,
    }
    payload.update(overrides)
    return [json.dumps(payload)]


def _table(**overrides: Any) -> dict[str, Any]:
    table: dict[str, Any] = {
        "columns": ["region", "revenue"],
        "rows": [["EMEA", 120], ["APAC", 80]],
        "row_count": 2,
        "execution_time_ms": 12.5,
        "error": None,
        "truncated": False,
        "truncation_reason": None,
    }
    table.update(overrides)
    return table


def test_the_executed_rows_are_what_the_answer_carries() -> None:
    result = _extract_sql_query_result(_chunk(sql_table=_table()))
    assert result is not None
    assert result.columns == ["region", "revenue"]
    assert result.rows == [["EMEA", 120], ["APAC", 80]]
    assert result.row_count == 2
    assert result.execution_time_ms == 12.5
    # The statement text must not be masquerading as the data.
    assert result.columns != ["sql_query"]


def test_truncation_survives_into_the_hybrid_answer() -> None:
    """The flags that could not previously exist on this path at all."""
    result = _extract_sql_query_result(
        _chunk(sql_table=_table(row_count=99_999, truncated=True, truncation_reason="row_budget"))
    )
    assert result is not None
    assert result.truncated is True
    assert result.truncation_reason == "row_budget"
    assert result.row_count == 99_999, "the true total, not the number of rows carried"


def test_a_complete_result_is_not_reported_as_truncated() -> None:
    """Canary for the above: the flags must track the frame, not default to set."""
    result = _extract_sql_query_result(_chunk(sql_table=_table()))
    assert result is not None
    assert result.truncated is False
    assert result.truncation_reason is None


def test_the_statement_is_still_returned_when_nothing_executed() -> None:
    """Generate-only, or an execution the policy refused: `sql_table` is empty
    and the synthesis prompt is better with the statement than with nothing."""
    result = _extract_sql_query_result(_chunk(execution_mode="generate_only"))
    assert result is not None
    assert result.columns == ["sql_query"]
    assert result.rows == [["SELECT region, revenue FROM sales"]]
    assert result.error is None


def test_an_execution_failure_is_not_reported_as_nothing_having_run() -> None:
    """`sql_table` is `{"error": ...}` with no columns when the connector raised.
    Falling back to the statement-only table is right; losing the error is not."""
    result = _extract_sql_query_result(
        _chunk(sql_table={"error": 'relation "sales" does not exist'})
    )
    assert result is not None
    assert result.columns == ["sql_query"]
    assert result.error == 'relation "sales" does not exist'


def test_a_validation_failure_still_reports_its_reason() -> None:
    result = _extract_sql_query_result(
        _chunk(is_valid=False, validation_error="statement is not a SELECT")
    )
    assert result is not None
    assert result.error == "statement is not a SELECT"


def test_a_chunk_that_is_not_sql_or_not_json_yields_nothing() -> None:
    assert _extract_sql_query_result([]) is None
    assert _extract_sql_query_result(["{not json"]) is None
    assert _extract_sql_query_result([json.dumps({"type": "answer", "text": "hi"})]) is None
    assert _extract_sql_query_result([json.dumps({"type": "sql", "sql": ""})]) is None
    assert _extract_sql_query_result([json.dumps(["not", "a", "mapping"])]) is None


def test_the_whole_chain_stays_bounded_and_keeps_the_reason() -> None:
    """The seam, not the unit: sub-agent -> extraction -> what hybrid emits.

    Carrying the executed result instead of a one-cell table raises a fair
    question about size, and the hybrid branch re-emits with ``cap=False``. It
    stays bounded anyway, because the rows it receives were already capped by
    the sub-agent's own ``sql_table_chunk`` -- ``cap=False`` suppresses that
    builder's cap, and there is nothing left to cap.

    The same hop is what makes truncation reportable on this path at all, which
    it was not before: ``row_count`` stays the true total while the rows are the
    capped set, and the reason names which cap bit.
    """
    answer = QueryResult(
        columns=["a"],
        rows=[[i] for i in range(_MAX_RESULT_ROWS * 25)],
        row_count=_MAX_RESULT_ROWS * 25,
        execution_time_ms=1.0,
        error=None,
    )
    emitted = sql_table_chunk(answer)  # what the SQL sub-agent puts on the wire
    carried = _extract_sql_query_result([json.dumps(_chunk_payload(emitted))])
    assert carried is not None

    final = sql_table_chunk(carried, cap=False)  # what the hybrid branch emits
    assert len(final["rows"]) == _MAX_RESULT_ROWS, "the hybrid payload must stay bounded"
    assert final["row_count"] == _MAX_RESULT_ROWS * 25, "row_count is the true total"
    assert final["truncated"] is True
    assert final["truncation_reason"] == "orchestrator_row_cap"


def _chunk_payload(sql_table: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "sql",
        "sql": "SELECT a FROM t",
        "is_valid": True,
        "execution_mode": "execute",
        "sql_table": sql_table,
    }
