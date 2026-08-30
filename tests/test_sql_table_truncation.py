"""The sql_table frame must say when it is not the whole answer.

`sql_table` is built where a query's rows leave the orchestrator, and it did two
things wrong at once. It dropped `QueryResult.truncated` and
`truncation_reason`, so a connector that stopped at its row or byte budget
reported nothing downstream — every caller saw the dataclass default. And it
*applied* a cap of its own, returning at most `_MAX_RESULT_ROWS` rows beside the
full `row_count`, without recording that either.

That combination is the failure `QueryResult.truncated`'s own comment names:
"silently returning the first N rows of a larger answer is a wrong answer, not a
partial one". A caller reading the frame saw a short list, a large count, and
nothing saying which was the answer — and once a layer above starts reporting
`truncated`, the silence becomes a positive claim of completeness.
"""

from __future__ import annotations

from typing import Any

import pytest
from nlqueries.connectors.base import QueryResult
from nlqueries.orchestrator.orchestrator import (
    _MAX_RESULT_ROWS,
    ORCHESTRATOR_ROW_CAP,
    sql_table_chunk,
)


def _result(n_rows: int, **kw: Any) -> QueryResult:
    return QueryResult(
        columns=["n"],
        rows=[[i] for i in range(n_rows)],
        row_count=kw.pop("row_count", n_rows),
        execution_time_ms=1.0,
        error=None,
        **kw,
    )


def test_a_short_complete_result_is_not_marked_truncated() -> None:
    """Canary. Without it every assertion below is satisfied by a frame that
    reports truncation unconditionally."""
    chunk = sql_table_chunk(_result(3))
    assert chunk["rows"] == [[0], [1], [2]]
    assert chunk["truncated"] is False
    assert chunk["truncation_reason"] is None


def test_the_row_cap_is_reported_as_truncation() -> None:
    """The bug that made the silence dangerous: the frame shortens the rows
    itself, and said nothing. A caller saw 200 rows beside a row_count of 500."""
    chunk = sql_table_chunk(_result(_MAX_RESULT_ROWS + 300))
    assert len(chunk["rows"]) == _MAX_RESULT_ROWS
    assert chunk["row_count"] == _MAX_RESULT_ROWS + 300
    assert chunk["truncated"] is True
    assert chunk["truncation_reason"] == ORCHESTRATOR_ROW_CAP


def test_exactly_the_cap_is_not_truncation() -> None:
    """Off-by-one guard: a result that fits exactly is complete, and reporting
    it as shortened would send callers hunting for rows that do not exist."""
    chunk = sql_table_chunk(_result(_MAX_RESULT_ROWS))
    assert len(chunk["rows"]) == _MAX_RESULT_ROWS
    assert chunk["truncated"] is False
    assert chunk["truncation_reason"] is None


def test_a_connector_budget_is_carried_through() -> None:
    """The half that was simply dropped. The connector sets both fields; nothing
    read them, so they never left this frame."""
    qr = _result(5, row_count=10_000, truncated=True, truncation_reason="byte_budget")
    chunk = sql_table_chunk(qr)
    assert chunk["truncated"] is True
    assert chunk["truncation_reason"] == "byte_budget"


def test_the_cap_wins_when_both_apply() -> None:
    """They are not mutually exclusive — the connector's budget is far above the
    cap, so a large result trips both. `truncation_reason` names the constraint
    that shortened what the caller is actually holding: told `row_budget`, they
    would narrow the query and still receive exactly 200 rows."""
    qr = _result(
        _MAX_RESULT_ROWS + 1, row_count=99_999, truncated=True, truncation_reason="row_budget"
    )
    chunk = sql_table_chunk(qr)
    assert chunk["truncated"] is True
    assert chunk["truncation_reason"] == ORCHESTRATOR_ROW_CAP


def test_the_uncapped_caller_still_reports_the_connectors_truncation() -> None:
    """The hybrid branch passes cap=False because it has never applied the row
    cap. It must still forward what the connector said."""
    qr = _result(500, row_count=99_999, truncated=True, truncation_reason="row_budget")
    chunk = sql_table_chunk(qr, cap=False)
    assert len(chunk["rows"]) == 500, "cap=False must not shorten the rows"
    assert chunk["truncated"] is True
    assert chunk["truncation_reason"] == "row_budget"


def test_an_uncapped_frame_never_invents_the_cap_reason() -> None:
    """Canary for the above: with cap=False a long result is complete as far as
    this frame is concerned, and must not be labelled with the cap it did not
    apply."""
    chunk = sql_table_chunk(_result(_MAX_RESULT_ROWS + 300), cap=False)
    assert len(chunk["rows"]) == _MAX_RESULT_ROWS + 300
    assert chunk["truncated"] is False
    assert chunk["truncation_reason"] is None


@pytest.mark.parametrize("cap", [True, False])
def test_the_frame_always_carries_both_keys(cap: bool) -> None:
    """`sync_runner._parse_final_chunk` reads these with `.get`, so a missing key
    is indistinguishable from a false one. Absent them, the layer above reports
    completeness it was never told about."""
    chunk = sql_table_chunk(_result(1), cap=cap)
    assert "truncated" in chunk
    assert "truncation_reason" in chunk
