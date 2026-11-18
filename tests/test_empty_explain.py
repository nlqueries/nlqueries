"""
tests.test_empty_explain
~~~~~~~~~~~~~~~~~~~~~~~~
Which condition left a result empty (owner feedback F20, phase 1): each
condition of the statement's WHERE left out in turn, the rows counted through
the request's own connector, and each condition without which rows come back
named with the count.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import yaml
from nlqueries.connectors.sqlite import SQLiteConnector
from nlqueries.execution import ExecutionPolicy
from nlqueries.orchestrator.empty_explain import (
    COUNT_TIMEOUT_SECONDS,
    MAX_CONDITIONS,
    count_statements,
    explain_empty_result,
)
from nlqueries.orchestrator.result_checks import EMPTY_RESULT_NOTE
from nlqueries.orchestrator.sql_generation import SQLGenerationResult
from nlqueries.query_settings import QuerySettings, use_query_settings

#: 23 late flights from SFO, none of them a night service.
LATE_NIGHT = (
    "SELECT f.id, f.dep FROM flights f "
    "WHERE f.origin = 'SFO' AND f.dep > 2000 AND f.night = 'YES' ORDER BY f.dep LIMIT 10"
)


class _Recording(SQLiteConnector):
    """A SQLite connector that keeps each statement it is given, and the timeout."""

    def __init__(self) -> None:
        super().__init__()
        self.statements: list[tuple[str, float | None]] = []

    def _execute_query(
        self, sql: str, timeout_seconds: float | None = None, max_rows: int | None = None
    ) -> Any:
        self.statements.append((sql, timeout_seconds))
        return super()._execute_query(sql, timeout_seconds, max_rows)


@pytest.fixture
def connector(tmp_path: Path) -> _Recording:
    path = tmp_path / "travel.db"
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE flights (id INTEGER, origin TEXT, dep INTEGER, night TEXT, email TEXT)"
        )
        rows = [(i, "SFO", 2030 + i, "NO", f"p{i}@example.com") for i in range(23)]
        rows += [(100 + i, "BOS", 900 + i, "YES", f"q{i}@example.com") for i in range(5)]
        conn.executemany("INSERT INTO flights VALUES (?, ?, ?, ?, ?)", rows)
    conn_ = _Recording()
    conn_.connect({"database": str(path)})
    # Counts are read as the statement is: with permission to read.
    conn_.bind_execution_policy(ExecutionPolicy.execute_read_only())
    return conn_


def _explain(
    sql: str,
    connector: _Recording,
    dialect: str = "sqlite",
    budget: tuple[bool, float | None] = (True, None),
) -> list[str]:
    return asyncio.run(
        explain_empty_result(sql, connector=connector, dialect=dialect, budget=lambda: budget)
    )


# --- The counts ---------------------------------------------------------------------


def test_each_condition_is_left_out_in_turn_without_ordering_or_limit() -> None:
    built = count_statements(LATE_NIGHT, "postgres")

    assert [c for c, _ in built] == ["f.origin = 'SFO'", "f.dep > 2000", "f.night = 'YES'"]
    assert built[2][1] == (
        "SELECT COUNT(*) FROM (SELECT 1 AS c1 FROM flights AS f "
        "WHERE f.origin = 'SFO' AND f.dep > 2000) AS nlq_counted"
    )
    assert all("ORDER" not in s and "LIMIT" not in s for _, s in built)


def test_a_grouped_statement_counts_its_groups_with_each_column_named() -> None:
    sql = (
        "SELECT TOP 5 origin, COUNT(*) FROM flights "
        "WHERE dep > 2000 AND night = 'YES' GROUP BY origin ORDER BY 2 DESC"
    )
    (_, first), _ = count_statements(sql, "mssql")

    # SQL Server refuses a derived table with an unnamed column, or an ORDER BY.
    assert first == (
        "SELECT COUNT(*) FROM (SELECT origin AS c1, COUNT(*) AS c2 FROM flights "
        "WHERE night = 'YES' GROUP BY origin) AS nlq_counted"
    )


def test_a_with_clause_moves_to_the_outer_statement() -> None:
    sql = "WITH late AS (SELECT * FROM flights WHERE dep > 2000) SELECT id FROM late WHERE a = 1"

    ((_, statement),) = count_statements(sql, "mysql")

    assert statement.startswith("WITH late AS (")
    assert "FROM (SELECT 1 AS c1 FROM late) AS nlq_counted" in statement


def test_personal_data_and_subqueries_are_never_left_out_or_named() -> None:
    sql = (
        "SELECT id FROM flights WHERE email = 'a@example.com' "
        "AND origin IN (SELECT code FROM airports) AND night = 'YES'"
    )

    assert [c for c, _ in count_statements(sql, "sqlite")] == ["night = 'YES'"]


def test_at_most_the_first_conditions_are_tried() -> None:
    sql = "SELECT id FROM t WHERE " + " AND ".join(f"c{i} = {i}" for i in range(7))

    assert [c for c, _ in count_statements(sql, "sqlite")] == [
        f"c{i} = {i}" for i in range(MAX_CONDITIONS)
    ]


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT id FROM flights",
        "SELECT id FROM flights WHERE a = 1 UNION SELECT id FROM flights WHERE b = 2",
        "SELECT DISTINCT * FROM flights WHERE a = 1",
        "DELETE FROM flights WHERE a = 1",
        "not sql at all (",
    ],
)
def test_nothing_is_built_where_there_is_no_condition_to_leave_out(sql: str) -> None:
    assert count_statements(sql, "postgres") == []


# --- The note -----------------------------------------------------------------------


def test_the_condition_without_which_rows_come_back_is_named(connector: _Recording) -> None:
    notes = _explain(LATE_NIGHT, connector)

    assert notes == ["No rows match every condition. Without f.night = 'YES', 23 rows match."]
    # Each count went through the connector the request holds, so the row
    # filters bound to it count only what the user can see; one per condition.
    assert len(connector.statements) == 3
    assert all(s.startswith("SELECT COUNT(*) FROM (") for s, _ in connector.statements)


def test_each_condition_that_is_enough_on_its_own_is_named(connector: _Recording) -> None:
    sql = "SELECT id FROM flights WHERE origin = 'BOS' AND night = 'NO'"

    (note,) = _explain(sql, connector)

    assert note == (
        "No rows match every condition. Without origin = 'BOS', 23 rows match. "
        "Without night = 'NO', 5 rows match."
    )


def test_no_note_when_no_one_condition_is_enough(connector: _Recording) -> None:
    sql = "SELECT id FROM flights WHERE origin = 'LAX' AND night = 'MAYBE'"

    assert _explain(sql, connector) == []


def test_a_condition_on_personal_data_is_not_named(connector: _Recording) -> None:
    sql = "SELECT id FROM flights WHERE email = 'nobody@example.com' AND origin = 'SFO'"

    assert _explain(sql, connector) == []
    assert not any("email" in c for c, _ in count_statements(sql, "sqlite"))


@pytest.mark.parametrize("dialect", ["bigquery", "snowflake", "mssql", "tsql"])
def test_engines_that_bill_by_data_read_or_ignore_the_timeout_are_not_counted(
    connector: _Recording, dialect: str
) -> None:
    assert _explain(LATE_NIGHT, connector, dialect=dialect) == []
    assert connector.statements == []


def test_each_count_is_held_to_its_own_timeout_and_the_time_left(connector: _Recording) -> None:
    _explain(LATE_NIGHT, connector, budget=(True, 60.0))
    _explain(LATE_NIGHT, connector, budget=(True, 1.5))

    timeouts = [t for _, t in connector.statements]
    assert timeouts == [COUNT_TIMEOUT_SECONDS] * 3 + [1.5] * 3


def test_a_count_is_never_run_without_a_timeout(connector: _Recording) -> None:
    """Zero or less is "no timeout" to the connectors: a caller that turned the
    statement timeout off still gets counts held to theirs."""
    _explain(LATE_NIGHT, connector, budget=(True, 0.0))
    _explain(LATE_NIGHT, connector, budget=(True, -1.0))

    assert [t for _, t in connector.statements] == [COUNT_TIMEOUT_SECONDS] * 6


def test_with_no_time_left_nothing_is_counted(connector: _Recording) -> None:
    assert _explain(LATE_NIGHT, connector, budget=(False, None)) == []
    assert connector.statements == []


def test_a_count_that_fails_is_skipped_and_said(
    connector: _Recording, caplog: pytest.LogCaptureFixture
) -> None:
    sql = "SELECT id FROM flights WHERE no_such_column = 1 AND night = 'MAYBE'"

    with caplog.at_level("WARNING", logger="nlqueries.orchestrator.empty_explain"):
        assert _explain(sql, connector) == []

    assert "Counting rows without one condition returned an error." in caplog.messages


def test_a_grouped_statement_keeps_the_aliases_it_groups_and_filters_by(
    connector: _Recording,
) -> None:
    sql = (
        "SELECT origin AS o, COUNT(*) AS n FROM flights "
        "WHERE night = 'YES' AND dep > 2000 GROUP BY o HAVING n > 1"
    )

    (note,) = _explain(sql, connector)

    # Each count ran: one SFO group without the night condition, one BOS group
    # without the time.
    assert note == (
        "No rows match every condition. Without night = 'YES', 1 row matches. "
        "Without dep > 2000, 1 row matches."
    )


def test_a_column_with_no_alias_gets_a_name_no_other_column_has() -> None:
    sql = "SELECT a AS c2, b, COUNT(*) FROM t WHERE x = 1 GROUP BY a, b"

    ((_, statement),) = count_statements(sql, "postgres")

    assert "SELECT a AS c2, b AS _c2, COUNT(*) AS c3 FROM t GROUP BY a, b" in statement


# --- Through the orchestrator -------------------------------------------------------


class _QuietLLM:
    supports_prompt_caching = False

    async def astream(self, system: Any, user: str) -> Any:
        for t in ():
            yield t

    async def acomplete(self, system: Any, user: str, **_kw: Any) -> str:
        return ""


def _answer(
    sql: str, connector: _Recording, tmp_path: Path, settings: QuerySettings
) -> dict[str, Any]:
    from nlqueries.orchestrator.orchestrator import Orchestrator

    first = SQLGenerationResult(
        sql=sql, is_valid=True, validation_error=None, dialect="sqlite", attempt_count=1
    )
    kb = {
        "schema": {
            "tables": [
                {
                    "name": "flights",
                    "row_count": 28,
                    "columns": [
                        {"name": n, "type": t}
                        for n, t in (
                            ("id", "INTEGER"),
                            ("origin", "TEXT"),
                            ("dep", "INTEGER"),
                            ("night", "TEXT"),
                            ("email", "TEXT"),
                        )
                    ],
                }
            ]
        },
        "business_context": {"glossary": [], "rules": []},
        "query_capsules": [],
    }
    (tmp_path / "agent1.yaml").write_text(yaml.dump(kb), encoding="utf-8")
    with (
        patch("nlqueries.orchestrator.orchestrator.config") as cfg,
        patch("nlqueries.orchestrator.orchestrator.get_llm_client", return_value=_QuietLLM()),
        patch(
            "nlqueries.orchestrator.orchestrator.validate_and_repair",
            new=AsyncMock(return_value=first),
        ),
        patch("nlqueries.connectors.loader.open_connector_for_agent", return_value=connector),
        patch("nlqueries.connectors.loader.lookup_source", side_effect=LookupError("none")),
        patch("nlqueries.embeddings.qdrant_store.search_schema", return_value=[]),
        patch("nlqueries.embeddings.qdrant_store.search", return_value=[]),
        patch("nlqueries.embeddings.embedder.embed_text", return_value=[0.0] * 384),
        patch("nlqueries.orchestrator.prompt_assembly._search_verified", return_value=[]),
        use_query_settings(settings),
    ):
        cfg.KB_PATH = tmp_path
        cfg.CONNECTOR_STATEMENT_TIMEOUT_SECONDS = 120.0

        async def collect() -> list[str]:
            return [
                tok
                async for tok in Orchestrator().handle_question(
                    "late night flights from san francisco",
                    "agent1",
                    dialect="sqlite",
                    execution=ExecutionPolicy.execute_read_only(),
                )
            ]

        tokens = asyncio.run(collect())
    frame: dict[str, Any] = json.loads(tokens[-1])
    return frame


def test_the_answer_names_the_condition_in_place_of_the_generic_note(
    connector: _Recording, tmp_path: Path
) -> None:
    frame = _answer(LATE_NIGHT, connector, tmp_path, QuerySettings(empty_result_explain=True))

    assert frame["sql_table"]["rows"] == []
    assert frame["result_notes"] == [
        "No rows match every condition. Without f.night = 'YES', 23 rows match."
    ]


def test_off_by_default_the_answer_keeps_the_generic_note(
    connector: _Recording, tmp_path: Path
) -> None:
    frame = _answer(LATE_NIGHT, connector, tmp_path, QuerySettings())

    assert frame["result_notes"] == [EMPTY_RESULT_NOTE]
    assert not any(s.startswith("SELECT COUNT(*) FROM (") for s, _ in connector.statements)


def test_rows_mean_nothing_is_counted(connector: _Recording, tmp_path: Path) -> None:
    sql = "SELECT id FROM flights WHERE origin = 'SFO' AND dep > 2000"

    frame = _answer(sql, connector, tmp_path, QuerySettings(empty_result_explain=True))

    assert len(frame["sql_table"]["rows"]) == 23
    assert not any(s.startswith("SELECT COUNT(*) FROM (") for s, _ in connector.statements)


def test_an_empty_result_already_explained_is_not_counted(
    connector: _Recording, tmp_path: Path
) -> None:
    """When the value a statement names is not stored, that note says why."""
    from nlqueries.orchestrator.empty_result import EmptyResultFollowup

    said = "No night is 'YES'. Stored values it may mean: 'Y'."

    async def followed_up(result: Any, qr: Any, **_kw: Any) -> EmptyResultFollowup:
        return EmptyResultFollowup(result=result, qr=qr, notes=[said])

    with patch(
        "nlqueries.orchestrator.orchestrator.follow_up_empty_result", side_effect=followed_up
    ):
        frame = _answer(LATE_NIGHT, connector, tmp_path, QuerySettings(empty_result_explain=True))

    assert frame["result_notes"] == [said]
    assert not any(s.startswith("SELECT COUNT(*) FROM (") for s, _ in connector.statements)


def test_an_answer_with_the_note_is_not_cached(connector: _Recording, tmp_path: Path) -> None:
    """The counts change with the data: the note is not to be replayed."""
    from nlqueries.orchestrator.multi_agent_orchestrator import _empty_with_notes

    explained = _answer(LATE_NIGHT, connector, tmp_path, QuerySettings(empty_result_explain=True))
    generic = _answer(LATE_NIGHT, connector, tmp_path, QuerySettings())

    assert _empty_with_notes([json.dumps(explained)])
    assert not _empty_with_notes([json.dumps(generic)])
