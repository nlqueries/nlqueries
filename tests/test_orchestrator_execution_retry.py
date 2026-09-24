"""One correction after the database rejects a generated statement.

Static validation parses with sqlglot and checks tables against the knowledge
base, so a statement the generic grammar accepts and the engine refuses reaches
the database. On nlq-test that was ``... ORDER BY total DESC LIMIT 1 UNION ALL
SELECT ...``: sqlglot parsed it, Snowflake rejected it, and the user saw only
the error. The database's message is handed back to the model once.

What these pin, beyond "it retries": the frame always reports the statement
that ran last beside that statement's own result; the correction is validated
like any other model output before it may run; and the cases that are not the
model's to fix -- a timeout, a raised exception such as a row filter refusing
the statement -- are not retried.
"""

from __future__ import annotations

import asyncio
import json
import re
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import yaml
from nlqueries.connectors.base import QueryResult
from nlqueries.execution import ExecutionPolicy
from nlqueries.orchestrator.orchestrator import Orchestrator, _retry_after_error
from nlqueries.orchestrator.sql_generation import SQLGenerationResult

FIRST_SQL = (
    "SELECT id FROM orders ORDER BY total DESC LIMIT 1 "
    "UNION ALL SELECT id FROM orders ORDER BY total ASC LIMIT 1"
)
DB_ERROR = "SQL compilation error: syntax error line 1 at position 49 unexpected 'UNION'."
CORRECTED_SQL = (
    "(SELECT id FROM orders ORDER BY total DESC LIMIT 1) "
    "UNION ALL (SELECT id FROM orders ORDER BY total ASC LIMIT 1)"
)


def _kb() -> dict[str, Any]:
    return {
        "schema": {
            "tables": [
                {
                    "name": "orders",
                    "description": "Purchase records",
                    "row_count": 100,
                    "columns": [
                        {"name": "id", "type": "INTEGER", "description": ""},
                        {"name": "total", "type": "DECIMAL", "description": ""},
                    ],
                }
            ]
        },
        "business_context": {"glossary": [], "rules": []},
        "query_capsules": [],
    }


class _LLM:
    """Streams nothing for generation; answers the correction call with *reply*."""

    supports_prompt_caching = False

    def __init__(self, reply: str | Exception = "") -> None:
        self._reply = reply
        self.stream_systems: list[Any] = []
        self.complete_calls: list[tuple[Any, str]] = []

    async def astream(self, system: Any, user: str) -> Any:
        self.stream_systems.append(system)
        for t in ():
            yield t

    async def acomplete(self, system: Any, user: str, **_kw: Any) -> str:
        self.complete_calls.append((system, user))
        if isinstance(self._reply, Exception):
            raise self._reply
        return self._reply


def _ok(rows: list[list[Any]], ms: float = 5.0) -> QueryResult:
    return QueryResult(
        columns=["id"], rows=rows, row_count=len(rows), execution_time_ms=ms, error=None
    )


def _failed(error: str = DB_ERROR, ms: float = 5.0) -> QueryResult:
    return QueryResult(columns=[], rows=[], row_count=0, execution_time_ms=ms, error=error)


#: What the last `_run` recorded outside the frame: the tracing span and the
#: provenance hook, for the tests that check observability.
_OBSERVED: dict[str, MagicMock] = {}


def _run(
    llm: _LLM,
    results: list[QueryResult | Exception],
    *,
    timeout_seconds: float | None = None,
    default_timeout: float = 120.0,
) -> tuple[dict[str, Any], MagicMock]:
    """Drive one question through the fresh path; return the frame and the connector."""
    connector = MagicMock()
    connector.execute_query.side_effect = results
    first = SQLGenerationResult(
        sql=FIRST_SQL, is_valid=True, validation_error=None, dialect="snowflake", attempt_count=1
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        kb_path = Path(tmpdir)
        safe = re.sub(r"[^\w.-]", "_", "agent1")
        (kb_path / f"{safe}.yaml").write_text(yaml.dump(_kb()), encoding="utf-8")
        with (
            patch("nlqueries.orchestrator.orchestrator.config") as cfg,
            patch("nlqueries.orchestrator.orchestrator.get_llm_client", return_value=llm),
            patch(
                "nlqueries.orchestrator.orchestrator.validate_and_repair",
                new=AsyncMock(return_value=first),
            ),
            patch("nlqueries.connectors.loader.open_connector_for_agent", return_value=connector),
            patch("nlqueries.embeddings.qdrant_store.search_schema", return_value=[]),
            patch("nlqueries.embeddings.qdrant_store.search", return_value=[]),
            patch("nlqueries.orchestrator.orchestrator.get_tracer") as tracer,
            patch("nlqueries.orchestrator.orchestrator.record_validator_warning") as warned,
        ):
            _OBSERVED["span"] = (
                tracer.return_value.start_as_current_span.return_value.__enter__.return_value
            )
            _OBSERVED["warned"] = warned
            cfg.KB_PATH = kb_path
            cfg.CONNECTOR_STATEMENT_TIMEOUT_SECONDS = default_timeout

            async def collect() -> list[str]:
                out: list[str] = []
                async for tok in Orchestrator().handle_question(
                    "top and bottom order",
                    "agent1",
                    dialect="snowflake",
                    timeout_seconds=timeout_seconds,
                    execution=ExecutionPolicy.execute_read_only(),
                ):
                    out.append(tok)
                return out

            tokens = asyncio.run(collect())
    return json.loads(tokens[-1]), connector


def _executed(connector: MagicMock) -> list[str]:
    return [c.args[0] for c in connector.execute_query.call_args_list]


# ---------------------------------------------------------------------------
# The correction
# ---------------------------------------------------------------------------


def test_a_rejected_statement_is_corrected_and_the_correction_runs() -> None:
    llm = _LLM(f"<sql>{CORRECTED_SQL}</sql>")
    frame, connector = _run(llm, [_failed(), _ok([[7], [3]])])

    assert _executed(connector) == [FIRST_SQL, CORRECTED_SQL]
    assert frame["sql"] == CORRECTED_SQL, "the frame must name the statement behind the rows"
    assert frame["sql_table"]["rows"] == [[7], [3]]
    assert frame["sql_table"]["error"] is None
    assert frame["attempt_count"] == 2


def test_the_model_is_given_the_database_error_and_the_failed_statement() -> None:
    llm = _LLM(f"<sql>{CORRECTED_SQL}</sql>")
    _run(llm, [_failed(), _ok([[7]])])

    assert len(llm.complete_calls) == 1
    system, user = llm.complete_calls[0]
    assert DB_ERROR in user
    assert FIRST_SQL in user
    # The same system prompt the statement was generated with, so the cached
    # prefix is reused rather than a second, cold prompt being built.
    assert system == llm.stream_systems[0]


def test_the_model_is_given_the_question() -> None:
    """The system prompt holds the schema, not the question -- that travels in
    the user turn. Without it, an "invalid identifier" correction picks a
    column with nothing tying it to what was asked."""
    llm = _LLM(f"<sql>{CORRECTED_SQL}</sql>")
    _run(llm, [_failed(), _ok([[7]])])

    system, user = llm.complete_calls[0]
    assert "top and bottom order" in user
    assert "top and bottom order" not in json.dumps(system), (
        "premise: the question is not already in the system prompt"
    )


def test_a_long_database_error_is_cut_off() -> None:
    """Driver text can quote database values; only a bounded amount reaches
    the prompt."""
    from nlqueries.orchestrator.sql_generation import _DB_ERROR_MAX_CHARS

    marker = "TAIL-THAT-MUST-NOT-ARRIVE"
    long_error = "x" * (_DB_ERROR_MAX_CHARS + 50) + marker
    llm = _LLM(f"<sql>{CORRECTED_SQL}</sql>")
    _run(llm, [_failed(long_error), _ok([[7]])])

    _, user = llm.complete_calls[0]
    assert marker not in user
    assert "x" * _DB_ERROR_MAX_CHARS + " [truncated]" in user


def test_one_correction_only_and_the_frame_reports_what_ran_last() -> None:
    """A correction that also fails is reported as itself: its SQL, its error."""
    second_error = "SQL compilation error: invalid identifier 'TOTAL_SPEND'"
    llm = _LLM(f"<sql>{CORRECTED_SQL}</sql>")
    frame, connector = _run(llm, [_failed(), _failed(second_error)])

    assert connector.execute_query.call_count == 2, "a second correction was attempted"
    assert len(llm.complete_calls) == 1
    assert frame["sql"] == CORRECTED_SQL
    assert frame["sql_table"]["error"] == second_error


def test_a_statement_that_succeeds_is_not_touched() -> None:
    llm = _LLM("<sql>SELECT 1</sql>")
    frame, connector = _run(llm, [_ok([[1]])])

    assert _executed(connector) == [FIRST_SQL]
    assert llm.complete_calls == []
    assert frame["sql"] == FIRST_SQL
    assert frame["attempt_count"] == 1


# ---------------------------------------------------------------------------
# The correction is model output, and is validated as such
# ---------------------------------------------------------------------------


def test_a_correction_the_policy_refuses_never_runs() -> None:
    llm = _LLM("<sql>DELETE FROM orders</sql>")
    frame, connector = _run(llm, [_failed()])

    assert _executed(connector) == [FIRST_SQL], "a refused correction reached the database"
    assert frame["sql"] == FIRST_SQL
    assert frame["sql_table"]["error"] == DB_ERROR


def test_a_correction_naming_an_unknown_table_never_runs() -> None:
    llm = _LLM("<sql>SELECT id FROM customers</sql>")
    frame, connector = _run(llm, [_failed()])

    assert _executed(connector) == [FIRST_SQL]
    assert frame["sql"] == FIRST_SQL
    assert frame["sql_table"]["error"] == DB_ERROR


def test_the_same_statement_back_is_not_run_again() -> None:
    llm = _LLM(f"<sql>{FIRST_SQL}</sql>")
    frame, connector = _run(llm, [_failed()])

    assert connector.execute_query.call_count == 1
    assert frame["sql_table"]["error"] == DB_ERROR


def test_a_failed_correction_call_keeps_the_database_error() -> None:
    """The user needs the database's message; the repair call's failure must
    not replace it."""
    llm = _LLM(RuntimeError("provider unavailable"))
    frame, connector = _run(llm, [_failed()])

    assert connector.execute_query.call_count == 1
    assert frame["sql"] == FIRST_SQL
    assert frame["sql_table"]["error"] == DB_ERROR


# ---------------------------------------------------------------------------
# Not the model's to fix
# ---------------------------------------------------------------------------


def test_a_raised_exception_is_not_retried() -> None:
    """Enterprise's row filter raises when it cannot rewrite a statement. Asking
    the model to rewrite until the filter accepts it is not a correction."""
    llm = _LLM(f"<sql>{CORRECTED_SQL}</sql>")
    frame, connector = _run(llm, [ValueError("row filter could not be applied")])

    assert connector.execute_query.call_count == 1
    assert llm.complete_calls == []
    assert frame["sql_table"]["error"] == "row filter could not be applied"


def test_an_attempt_that_used_the_timeout_is_not_run_again() -> None:
    llm = _LLM(f"<sql>{CORRECTED_SQL}</sql>")
    frame, connector = _run(llm, [_failed("canceled", ms=9_500.0)], timeout_seconds=10.0)

    assert connector.execute_query.call_count == 1
    assert llm.complete_calls == []
    assert frame["sql_table"]["error"] == "canceled"


def test_the_timeout_falls_back_to_the_connector_default() -> None:
    """With no per-request timeout, the connectors use
    CONNECTOR_STATEMENT_TIMEOUT_SECONDS; the retry rule reads the same."""
    llm = _LLM(f"<sql>{CORRECTED_SQL}</sql>")
    _, connector = _run(llm, [_failed("canceled", ms=19_000.0)], default_timeout=20.0)
    assert connector.execute_query.call_count == 1

    llm = _LLM(f"<sql>{CORRECTED_SQL}</sql>")
    _, connector = _run(llm, [_failed(ms=19_000.0), _ok([[1]])], default_timeout=120.0)
    assert connector.execute_query.call_count == 2, "a fast failure under a long default"


def test_the_share_boundary() -> None:
    """At 90% of the budget an attempt counts as timed out; below it, not."""
    assert _retry_after_error(_failed(ms=8_999.0), 10.0) is True
    assert _retry_after_error(_failed(ms=9_000.0), 10.0) is False


def test_no_timeout_means_any_failure_is_worth_a_correction() -> None:
    """Zero is the connectors' "no timeout", so elapsed time proves nothing."""
    assert _retry_after_error(_failed(ms=10_000_000.0), 0) is True


def test_a_result_without_an_error_is_never_retried() -> None:
    assert _retry_after_error(_ok([[1]]), 10.0) is False


# ---------------------------------------------------------------------------
# A refusal is not a defect in the SQL
# ---------------------------------------------------------------------------

REFUSALS = [
    "permission denied for table orders",
    "The SELECT permission was denied on the object 'orders', database 'db', schema 'dbo'.",
    "SQL access control error: Insufficient privileges to operate on table 'ORDERS'",
    "SQL compilation error: Object 'ORDERS' does not exist or not authorized.",
    "Access Denied: Table p:d.orders: User does not have permission to query table p:d.orders",
]


def test_a_permission_refusal_is_not_corrected() -> None:
    """The knowledge base can list what the query role was never granted, so a
    statement passes every check and is still refused. The only "fix" is a
    different table or column -- an answer to a different question."""
    for refusal in REFUSALS:
        llm = _LLM("<sql>SELECT id FROM orders</sql>")
        frame, connector = _run(llm, [_failed(refusal)])
        assert connector.execute_query.call_count == 1, refusal
        assert llm.complete_calls == [], refusal
        assert frame["sql_table"]["error"] == refusal


def test_a_column_error_is_still_corrected() -> None:
    """Negative control for the refusal pattern: an ordinary defect must not
    match it, or the retry would never fire on the errors it exists for."""
    for error in (
        "SQL compilation error: invalid identifier 'TOTAL_SPEND'",
        'column "total_spend" does not exist',
        DB_ERROR,
    ):
        assert _retry_after_error(_failed(error), 10.0) is True, error


def test_the_prompt_forbids_routing_around_a_refusal() -> None:
    """The backstop for a driver whose refusal wording the pattern misses."""
    llm = _LLM(f"<sql>{CORRECTED_SQL}</sql>")
    _run(llm, [_failed(), _ok([[7]])])

    _, user = llm.complete_calls[0]
    assert "Do not replace a table or column with a different one" in user
    assert "return the same statement unchanged" in user


# ---------------------------------------------------------------------------
# What an operator can see
# ---------------------------------------------------------------------------


def test_the_span_reports_the_attempt_that_ran() -> None:
    """attempt_count is set on the span before execution; after a correction
    runs it is re-set, so the span agrees with the frame."""
    llm = _LLM(f"<sql>{CORRECTED_SQL}</sql>")
    frame, _ = _run(llm, [_failed(), _ok([[7]])])

    counts = [
        c.args[1]
        for c in _OBSERVED["span"].set_attribute.call_args_list
        if c.args[0] == "attempt_count"
    ]
    assert counts[-1] == frame["attempt_count"] == 2


def test_a_refused_correction_is_recorded() -> None:
    llm = _LLM("<sql>DELETE FROM orders</sql>")
    _run(llm, [_failed()])

    warnings = [c.args[0] for c in _OBSERVED["warned"].call_args_list]
    assert any("Refused by SQL policy" in w for w in warnings), warnings
