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

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock, patch

from nlqueries.connectors.base import QueryResult
from nlqueries.orchestrator.intent_classifier import (
    IntentClassificationResult,
    IntentType,
)
from nlqueries.orchestrator.multi_agent_orchestrator import _extract_sql_query_result
from nlqueries.orchestrator.orchestrator import _MAX_RESULT_ROWS, sql_table_chunk


def _payload(**overrides: Any) -> dict[str, Any]:
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
    return payload


def _chunk(**overrides: Any) -> list[str]:
    return [json.dumps(_payload(**overrides))]


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


# ---------------------------------------------------------------------------
# The consequence downstream of the extraction: what may be cached.
# ---------------------------------------------------------------------------


def _drive_hybrid(sql_tokens: list[str]) -> MagicMock:
    """Run one hybrid turn with *sql_tokens* and return the cache mock.

    The hybrid entry stores the merged answer alone, and a hit replays it for
    the whole TTL with no re-execution -- unlike the SQL branch, which re-runs
    its stored statement precisely because stored prose cannot carry fresh rows.
    """
    from nlqueries.orchestrator.multi_agent_orchestrator import (
        MultiAgentOrchestrator,
        drain_background_tasks,
    )
    from nlqueries.orchestrator.result_merger import HybridQueryResult

    doc_tokens = [
        "The policy states...",
        json.dumps({"type": "citations", "citations": []}),
    ]

    def _gen_factory(tokens: list[str]):  # type: ignore[no-untyped-def]
        async def _gen(*_a: object, **_k: object):  # type: ignore[no-untyped-def]
            for t in tokens:
                yield t

        return _gen

    sql_instance = MagicMock()
    sql_instance.handle_question = _gen_factory(sql_tokens)
    doc_instance = MagicMock()
    doc_instance.handle_question = _gen_factory(doc_tokens)

    cache = MagicMock()
    cache.get.return_value = None  # force a miss so the write path is reached

    merged = HybridQueryResult(
        sql_answer="rows",
        sql_table=None,
        document_answer="excerpts",
        citations=[],
        merged_answer="EMEA billed 120 and APAC 80.",
    )

    async def _run() -> None:
        with (
            patch(
                "nlqueries.orchestrator.multi_agent_orchestrator.SemanticCache",
                return_value=cache,
            ),
            patch(
                "nlqueries.orchestrator.multi_agent_orchestrator.aclassify_intent",
                return_value=IntentClassificationResult(
                    intent=IntentType.hybrid, confidence=0.95, reasoning="Mock."
                ),
            ),
            patch(
                "nlqueries.orchestrator.multi_agent_orchestrator.aresolve_followup",
                return_value=MagicMock(resolved="Who bought what?"),
            ),
            patch(
                "nlqueries.orchestrator.multi_agent_orchestrator.Orchestrator",
                return_value=sql_instance,
            ),
            patch(
                "nlqueries.orchestrator.multi_agent_orchestrator.DocumentOrchestrator",
                return_value=doc_instance,
            ),
            patch(
                "nlqueries.orchestrator.multi_agent_orchestrator.merge_results",
                return_value=merged,
            ),
        ):
            orch = MultiAgentOrchestrator()
            async for _ in orch.handle_question(
                "Who bought what?", "agent1", available_types=["sql", "document", "hybrid"]
            ):
                pass
            await drain_background_tasks()

    asyncio.run(_run())
    return cache


def test_a_hybrid_answer_built_from_live_rows_is_not_cached() -> None:
    """The regression this change would otherwise introduce.

    A hybrid entry holds the merged prose and nothing else, and a hit replays it
    verbatim for the whole TTL with no re-execution. That was harmless while the
    prose was synthesised from the statement text; now that it is synthesised
    from executed rows, a semantically similar question hours later would be
    answered with figures from the first run, stated as current.
    """
    cache = _drive_hybrid(["Answering.", json.dumps(_payload(sql_table=_table()))])
    cache.put.assert_not_called()


def test_a_hybrid_answer_that_executed_nothing_is_still_cached() -> None:
    """The other half, or the fix would be "stop caching hybrid answers".

    Generate-only mode, a refused execution and a failed validation all produce
    prose synthesised from the statement text alone. It carries no figures, so
    it cannot go stale, and those are the answers the cache was helping.
    """
    cache = _drive_hybrid(["Answering.", json.dumps(_payload(sql_table=None))])
    cache.put.assert_called_once()


def test_an_execution_error_does_not_count_as_live_data() -> None:
    """`sql_table` is `{"error": ...}` with no columns when the connector raised.

    No figures reached the prose, so the entry is cacheable by the same
    reasoning -- and this is the branch that is easiest to get wrong, since the
    table is present but holds nothing.
    """
    cache = _drive_hybrid(
        ["Answering.", json.dumps(_payload(sql_table={"error": "connection refused"}))]
    )
    cache.put.assert_called_once()
