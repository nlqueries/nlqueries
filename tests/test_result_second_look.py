"""A second look at what a statement returned, and one correction if it looks wrong.

A statement can run cleanly and still miss the question: a join repeats every
row, a ranking is headed by a row with no value. The orchestrator reads the
result (`result_checks`), asks the model once to correct a statement whose
result shows a sign it can name with confidence, and keeps the correction only
if the correction's own result is clean. What still looks wrong reaches the
caller as notes. These pin that, and that a result with nothing wrong is
reported exactly as before.
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
from nlqueries.orchestrator.orchestrator import Orchestrator
from nlqueries.orchestrator.sql_generation import SQLGenerationResult

FIRST_SQL = "SELECT c.name FROM customers c JOIN orders o ON o.customer_id = c.id"
CORRECTED_SQL = "SELECT DISTINCT c.name FROM customers c JOIN orders o ON o.customer_id = c.id"
QUESTION = "which customers have ordered"


def _kb() -> dict[str, Any]:
    return {
        "schema": {
            "tables": [
                {
                    "name": "customers",
                    "row_count": 10,
                    "columns": [
                        {"name": "id", "type": "INTEGER"},
                        {"name": "name", "type": "TEXT"},
                    ],
                },
                {
                    "name": "orders",
                    "row_count": 100,
                    "columns": [
                        {"name": "id", "type": "INTEGER"},
                        {"name": "customer_id", "type": "INTEGER"},
                        {"name": "status", "type": "TEXT"},
                    ],
                },
            ]
        },
        "business_context": {"glossary": [], "rules": []},
        "query_capsules": [],
    }


class _LLM:
    """Streams nothing for generation; answers the correction with *reply*."""

    supports_prompt_caching = False

    def __init__(self, reply: str | Exception = "") -> None:
        self._reply = reply
        self.corrections: list[str] = []

    async def astream(self, system: Any, user: str) -> Any:
        for t in ():
            yield t

    async def acomplete(self, system: Any, user: str, **_kw: Any) -> str:
        self.corrections.append(user)
        if isinstance(self._reply, Exception):
            raise self._reply
        return self._reply


def _rows(*names: str) -> QueryResult:
    rows = [[n] for n in names]
    return QueryResult(
        columns=["name"], rows=rows, row_count=len(rows), execution_time_ms=1.0, error=None
    )


def _run(
    llm: _LLM, results: list[QueryResult | Exception], first_sql: str = FIRST_SQL
) -> tuple[dict[str, Any], MagicMock]:
    connector = MagicMock()
    connector.execute_query.side_effect = results
    first = SQLGenerationResult(
        sql=first_sql, is_valid=True, validation_error=None, dialect="postgres", attempt_count=1
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
            patch("nlqueries.embeddings.embedder.embed_text", return_value=[0.0] * 384),
            patch("nlqueries.orchestrator.prompt_assembly._search_verified", return_value=[]),
        ):
            cfg.KB_PATH = kb_path
            cfg.CONNECTOR_STATEMENT_TIMEOUT_SECONDS = 120.0

            async def collect() -> list[str]:
                return [
                    tok
                    async for tok in Orchestrator().handle_question(
                        QUESTION,
                        "agent1",
                        dialect="postgres",
                        execution=ExecutionPolicy.execute_read_only(),
                    )
                ]

            tokens = asyncio.run(collect())
    return json.loads(tokens[-1]), connector


def _executed(connector: MagicMock) -> list[str]:
    return [c.args[0] for c in connector.execute_query.call_args_list]


def test_repeated_rows_are_corrected_and_the_correction_reported() -> None:
    llm = _LLM(f"<sql>{CORRECTED_SQL}</sql>")
    frame, connector = _run(llm, [_rows("Ann", "Bo", "Ann"), _rows("Ann", "Bo")])

    assert _executed(connector) == [FIRST_SQL, CORRECTED_SQL]
    assert frame["sql"] == CORRECTED_SQL, "the frame names the statement behind the rows"
    assert frame["sql_table"]["rows"] == [["Ann"], ["Bo"]]
    assert frame["attempt_count"] == 2
    # A correction for repeats cannot fail the repeat check, and the repeats
    # may have been real records, so both counts are told.
    (note,) = frame["result_notes"]
    assert note.startswith("The statement first returned 3 rows, some of them identical; these 2 ")
    assert "the first result was right" in note


def test_a_kept_correction_for_another_doubt_carries_no_such_note() -> None:
    first = "SELECT o.status, o.id FROM orders o ORDER BY o.status LIMIT 1"
    corrected = (
        "SELECT o.status, o.id FROM orders o WHERE o.status IS NOT NULL ORDER BY o.status LIMIT 1"
    )

    def _ranked(status: str | None) -> QueryResult:
        return QueryResult(
            columns=["status", "id"],
            rows=[[status, 7]],
            row_count=1,
            execution_time_ms=1.0,
            error=None,
        )

    llm = _LLM(f"<sql>{corrected}</sql>")
    frame, connector = _run(llm, [_ranked(None), _ranked("open")], first_sql=first)

    assert _executed(connector) == [first, corrected]
    assert frame["sql"] == corrected
    assert "result_notes" not in frame


def test_the_model_is_told_the_question_the_statement_and_what_looks_wrong() -> None:
    llm = _LLM(f"<sql>{CORRECTED_SQL}</sql>")
    _run(llm, [_rows("Ann", "Ann"), _rows("Ann")])

    (prompt,) = llm.corrections
    assert QUESTION in prompt
    assert FIRST_SQL in prompt
    assert "repeat another row exactly" in prompt
    assert "return the same statement unchanged" in prompt


def test_a_statement_the_model_keeps_is_not_run_again_and_the_doubt_is_noted() -> None:
    """Two customers can share a name; the model may say the rows are right."""
    llm = _LLM(f"<sql>{FIRST_SQL}</sql>")
    frame, connector = _run(llm, [_rows("Ann", "Ann")])

    assert _executed(connector) == [FIRST_SQL]
    assert frame["sql"] == FIRST_SQL
    assert frame["attempt_count"] == 1
    assert len(frame["result_notes"]) == 1
    assert "repeat another row exactly" in frame["result_notes"][0]


def test_a_correction_whose_rows_still_look_wrong_is_not_kept() -> None:
    """Trading one doubt for another is not an improvement."""
    other = "SELECT c.name FROM customers c JOIN orders o ON o.customer_id = c.id WHERE o.id > 0"
    llm = _LLM(f"<sql>{other}</sql>")
    frame, connector = _run(llm, [_rows("Ann", "Ann"), _rows("Bo", "Bo")])

    assert _executed(connector) == [FIRST_SQL, other]
    assert frame["sql"] == FIRST_SQL
    assert frame["sql_table"]["rows"] == [["Ann"], ["Ann"]]
    assert frame["result_notes"]


def test_a_correction_the_validator_refuses_is_not_run() -> None:
    llm = _LLM("<sql>SELECT name FROM ghost_table</sql>")
    frame, connector = _run(llm, [_rows("Ann", "Ann")])

    assert _executed(connector) == [FIRST_SQL]
    assert frame["sql"] == FIRST_SQL
    assert frame["result_notes"]


def test_a_correction_the_database_rejects_leaves_the_rows_that_ran() -> None:
    failed = QueryResult(columns=[], rows=[], row_count=0, execution_time_ms=1.0, error="boom")
    llm = _LLM(f"<sql>{CORRECTED_SQL}</sql>")
    frame, _ = _run(llm, [_rows("Ann", "Ann"), failed])

    assert frame["sql"] == FIRST_SQL
    assert frame["sql_table"]["error"] is None
    assert frame["sql_table"]["rows"] == [["Ann"], ["Ann"]]


def test_a_failed_correction_call_leaves_the_rows_that_ran() -> None:
    frame, connector = _run(_LLM(RuntimeError("provider down")), [_rows("Ann", "Ann")])

    assert _executed(connector) == [FIRST_SQL]
    assert frame["sql_table"]["rows"] == [["Ann"], ["Ann"]]
    assert frame["result_notes"]


def test_a_doubt_a_rule_cannot_judge_is_only_noted() -> None:
    """No rows is often the answer: no correction is asked for."""
    llm = _LLM("<sql>SELECT 1</sql>")
    sql = "SELECT id FROM orders WHERE status = 'lost'"
    frame, connector = _run(llm, [_rows()], first_sql=sql)

    assert llm.corrections == []
    assert _executed(connector) == [sql]
    assert frame["result_notes"] == [
        "No rows matched. If you expected some, a filter may be narrower than intended, "
        "for example a value spelled differently in the data."
    ]


def test_a_result_with_nothing_wrong_is_reported_as_before() -> None:
    llm = _LLM("<sql>SELECT 1</sql>")
    frame, connector = _run(llm, [_rows("Ann", "Bo")])

    assert llm.corrections == []
    assert _executed(connector) == [FIRST_SQL]
    assert "result_notes" not in frame


def test_a_check_that_raises_leaves_the_rows_as_they_ran() -> None:
    """The checks are advisory; one that fails must not turn rows into an error."""
    llm = _LLM("<sql>SELECT 1</sql>")
    with patch("nlqueries.orchestrator.orchestrator.check_result", side_effect=ValueError("boom")):
        frame, connector = _run(llm, [_rows("Ann", "Ann")])

    assert llm.corrections == []
    assert frame["sql_table"]["error"] is None
    assert frame["sql_table"]["rows"] == [["Ann"], ["Ann"]]
    assert "result_notes" not in frame


def test_a_check_of_the_correction_that_raises_keeps_the_first_result() -> None:
    from nlqueries.orchestrator import result_checks

    real = result_checks.check_result
    calls = []

    def first_only(sql: str, result: QueryResult, dialect: str) -> Any:
        calls.append(sql)
        if len(calls) > 1:
            raise ValueError("boom")
        return real(sql, result, dialect)

    llm = _LLM(f"<sql>{CORRECTED_SQL}</sql>")
    with patch("nlqueries.orchestrator.orchestrator.check_result", side_effect=first_only):
        frame, _ = _run(llm, [_rows("Ann", "Ann"), _rows("Ann")])

    assert frame["sql"] == FIRST_SQL
    assert frame["sql_table"]["rows"] == [["Ann"], ["Ann"]]
    assert frame["result_notes"]


# ---------------------------------------------------------------------------
# A correction for repeats keeps every row the first result had
# ---------------------------------------------------------------------------


def test_a_correction_for_repeats_that_leaves_out_an_empty_value_is_not_kept() -> None:
    # The customer with no name is a row the question listed; filtering it out
    # answers another question.
    dropped = CORRECTED_SQL + " WHERE c.name IS NOT NULL"
    llm = _LLM(f"<sql>{dropped}</sql>")
    frame, connector = _run(llm, [_rows("Ann", None, "Ann"), _rows("Ann")])

    assert _executed(connector) == [FIRST_SQL, dropped]
    assert frame["sql"] == FIRST_SQL
    assert frame["sql_table"]["rows"] == [["Ann"], [None], ["Ann"]]
    (note,) = frame["result_notes"]
    assert "repeat another row exactly" in note, "the repeats are still told"


def test_a_correction_for_repeats_that_collapses_rows_is_not_kept() -> None:
    collapsed = "SELECT MAX(c.name) AS name FROM customers c JOIN orders o ON o.customer_id = c.id"
    llm = _LLM(f"<sql>{collapsed}</sql>")
    frame, _connector = _run(llm, [_rows("Ann", "Bo", "Ann"), _rows("Bo")])
    assert frame["sql"] == FIRST_SQL
    assert frame["sql_table"]["rows"] == [["Ann"], ["Bo"], ["Ann"]]


def test_a_correction_for_repeats_that_changes_the_columns_is_not_kept() -> None:
    widened = "SELECT DISTINCT c.name, c.id FROM customers c JOIN orders o ON o.customer_id = c.id"
    named = QueryResult(
        columns=["name", "id"],
        rows=[["Ann", 1], ["Bo", 2]],
        row_count=2,
        execution_time_ms=1.0,
        error=None,
    )
    llm = _LLM(f"<sql>{widened}</sql>")
    frame, _connector = _run(llm, [_rows("Ann", "Bo", "Ann"), named])
    assert frame["sql"] == FIRST_SQL


def test_a_correction_for_repeats_that_reaches_further_under_a_limit_is_kept() -> None:
    # Three rows asked for, one of them a repeat: without it the limit reaches
    # one more, and every row the first had is still there.
    first, corrected = FIRST_SQL + " LIMIT 3", CORRECTED_SQL + " LIMIT 3"
    llm = _LLM(f"<sql>{corrected}</sql>")
    frame, _connector = _run(
        llm, [_rows("Ann", "Bo", "Ann"), _rows("Ann", "Bo", "Cy")], first_sql=first
    )
    assert frame["sql"] == corrected
    assert frame["sql_table"]["rows"] == [["Ann"], ["Bo"], ["Cy"]]


def test_a_correction_for_repeats_and_another_doubt_may_change_the_rows() -> None:
    # Headed by a row with no name as well as repeating: the correction for the
    # ranking rightly leaves that row out, so only a repeats-only correction is
    # held to keeping every row.
    first = FIRST_SQL + " ORDER BY c.name LIMIT 3"
    corrected = CORRECTED_SQL + " WHERE c.name IS NOT NULL ORDER BY c.name LIMIT 3"
    llm = _LLM(f"<sql>{corrected}</sql>")
    frame, _connector = _run(llm, [_rows(None, "Ann", "Ann"), _rows("Ann")], first_sql=first)
    assert frame["sql"] == corrected
