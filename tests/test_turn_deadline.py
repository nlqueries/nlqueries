"""The caller's turn deadline reaches the one correction after a database error.

``timeout_seconds`` bounds a statement. A caller that bounds the whole turn
with its own wall clock -- enterprise's synchronous ``POST /query`` wraps
``run_query`` in ``asyncio.wait_for`` -- could otherwise have core start a
correction it cannot finish: the database's error, which the caller could have
returned, becomes a bare timeout, and the corrected statement runs on with no
one waiting. The deadline is passed down and the correction starts only when
at least as much time remains as the failed attempt took.

These pin the rule and, separately, that the value actually arrives: a
parameter every layer accepts and one layer forgets to forward is a guard that
silently never runs.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any
from unittest.mock import MagicMock, patch

from nlqueries.connectors.base import QueryResult
from nlqueries.orchestrator.orchestrator import _retry_after_error

_FINAL = json.dumps(
    {
        "type": "sql",
        "sql": "SELECT 1",
        "is_valid": True,
        "validation_error": None,
        "dialect": "postgres",
        "attempt_count": 1,
        "sql_table": None,
    }
)


def _failed() -> QueryResult:
    return QueryResult(
        columns=[], rows=[], row_count=0, execution_time_ms=5.0, error="syntax error"
    )


def _capturing(sink: dict[str, Any]) -> Any:
    async def _gen(*args: Any, **kwargs: Any) -> Any:
        sink.update(kwargs)
        yield _FINAL

    return _gen


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------


def test_a_correction_needs_as_much_time_left_as_the_attempt_took() -> None:
    now = time.monotonic()
    # Ten seconds in, five left: a correction the same size would overrun.
    assert _retry_after_error(_failed(), 30.0, started=now - 10, deadline=now + 5) is False
    # Ten seconds in, twenty left: it fits.
    assert _retry_after_error(_failed(), 30.0, started=now - 10, deadline=now + 20) is True


def test_a_passed_deadline_never_corrects() -> None:
    now = time.monotonic()
    assert _retry_after_error(_failed(), 30.0, started=now, deadline=now - 1) is False


def test_no_deadline_leaves_the_rule_as_it_was() -> None:
    """Chat has no wall clock around the turn and passes none."""
    now = time.monotonic()
    assert _retry_after_error(_failed(), 30.0, started=now - 1_000) is True
    assert _retry_after_error(_failed(), 30.0) is True


def test_the_deadline_does_not_override_the_other_gates() -> None:
    """Plenty of turn left does not make a refusal or a timeout retryable."""
    now = time.monotonic()
    refused = QueryResult(
        columns=[],
        rows=[],
        row_count=0,
        execution_time_ms=5.0,
        error="permission denied for table orders",
    )
    slow = QueryResult(
        columns=[], rows=[], row_count=0, execution_time_ms=29_000.0, error="canceled"
    )
    assert _retry_after_error(refused, 30.0, started=now, deadline=now + 600) is False
    assert _retry_after_error(slow, 30.0, started=now, deadline=now + 600) is False


# ---------------------------------------------------------------------------
# The value arrives
# ---------------------------------------------------------------------------


def test_run_query_forwards_the_deadline() -> None:
    from nlqueries.orchestrator.sync_runner import run_query

    sink: dict[str, Any] = {}
    instance = MagicMock()
    instance.handle_question = _capturing(sink)
    with patch("nlqueries.orchestrator.sync_runner.MultiAgentOrchestrator", return_value=instance):
        asyncio.run(run_query("q", "agent1", deadline=1234.5))
    assert sink["deadline"] == 1234.5


def test_the_sql_route_forwards_the_deadline() -> None:
    from nlqueries.orchestrator.multi_agent_orchestrator import MultiAgentOrchestrator

    sink: dict[str, Any] = {}
    sql_instance = MagicMock()
    sql_instance.handle_question = _capturing(sink)

    async def run() -> None:
        with (
            patch(
                "nlqueries.orchestrator.multi_agent_orchestrator.Orchestrator",
                return_value=sql_instance,
            ),
            patch("nlqueries.orchestrator.multi_agent_orchestrator.DocumentOrchestrator"),
            patch("nlqueries.orchestrator.multi_agent_orchestrator.SemanticCache") as cache,
            patch("nlqueries.embeddings.embedder.embed_text", return_value=[0.0] * 384),
        ):
            cache.return_value.get.return_value = None
            async for _ in MultiAgentOrchestrator().handle_question(
                "q", "agent1", available_types=["sql"], deadline=1234.5
            ):
                pass

    asyncio.run(run())
    assert sink["deadline"] == 1234.5


def test_the_hybrid_route_forwards_the_deadline() -> None:
    from nlqueries.orchestrator.multi_agent_orchestrator import _run_hybrid

    sql_sink: dict[str, Any] = {}
    sql_instance = MagicMock()
    sql_instance.handle_question = _capturing(sql_sink)
    doc_instance = MagicMock()
    doc_instance.handle_question = _capturing({})

    with (
        patch(
            "nlqueries.orchestrator.multi_agent_orchestrator.Orchestrator",
            return_value=sql_instance,
        ),
        patch(
            "nlqueries.orchestrator.multi_agent_orchestrator.DocumentOrchestrator",
            return_value=doc_instance,
        ),
    ):
        asyncio.run(_run_hybrid("q", "agent1", "postgres", deadline=1234.5))
    assert sql_sink["deadline"] == 1234.5


# ---------------------------------------------------------------------------
# The corrected statement is bounded by the deadline too
# ---------------------------------------------------------------------------


def test_without_a_deadline_the_statement_timeout_is_unchanged() -> None:
    from nlqueries.orchestrator.orchestrator import _correction_budget

    assert _correction_budget(27.0, None) == (True, 27.0)
    assert _correction_budget(None, None) == (True, None)


def test_the_correction_timeout_is_clamped_to_the_time_left() -> None:
    """The failed attempt's time is almost all generation, so it says nothing
    about how long the corrected statement will run; given the full statement
    timeout it could outlast the caller's clock."""
    from nlqueries.orchestrator.orchestrator import _correction_budget

    runnable, timeout = _correction_budget(27.0, time.monotonic() + 11.0)
    assert runnable is True
    assert timeout is not None and 10.0 < timeout <= 11.0

    # A statement timeout shorter than what is left still wins.
    assert _correction_budget(5.0, time.monotonic() + 60.0) == (True, 5.0)


def test_no_statement_timeout_leaves_the_deadline_as_the_bound() -> None:
    """Zero is the connectors' "no timeout"; the deadline must still apply."""
    from nlqueries.orchestrator.orchestrator import _correction_budget

    runnable, timeout = _correction_budget(0.0, time.monotonic() + 8.0)
    assert runnable is True
    assert timeout is not None and 7.0 < timeout <= 8.0


def test_under_a_second_left_the_correction_is_not_run() -> None:
    from nlqueries.orchestrator.orchestrator import _correction_budget

    assert _correction_budget(27.0, time.monotonic() + 0.5) == (False, None)
    assert _correction_budget(27.0, time.monotonic() - 1.0) == (False, None)
