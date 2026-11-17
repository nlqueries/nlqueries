"""
tests.test_empty_result_followup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A statement that returns no rows because a value it names is not stored, and
what is done about it (owner feedback F19): the stored values it may mean,
offered as choices, run with when there is exactly one, or named in the note.
Also the lookups' tables, qualified with the knowledge base's schema when the
draft leaves it out.
"""

from __future__ import annotations

import asyncio
import re
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from nlqueries.connectors.base import PermittedConnector
from nlqueries.connectors.loader import LookupSource
from nlqueries.connectors.sqlite import SQLiteConnector
from nlqueries.execution import ExecutionPolicy
from nlqueries.orchestrator import literal_grounding
from nlqueries.orchestrator.empty_result import follow_up_empty_result
from nlqueries.orchestrator.literal_grounding import empty_result_literals, ground_literals
from nlqueries.orchestrator.match_notes import choice_key, use_chosen_values
from nlqueries.orchestrator.sql_generation import SQLGenerationResult
from nlqueries.query_settings import QuerySettings, use_query_settings

CITIES = ["DALLAS", "FORT WORTH", "BOSTON", "PITTSBURGH", "DENVER"] + [
    f"CITY {i}" for i in range(30)
]


class _Recording(SQLiteConnector):
    def __init__(self, statements: list[str]) -> None:
        super().__init__()
        self.statements = statements

    def _execute_query(
        self, sql: str, timeout_seconds: float | None = None, max_rows: int | None = None
    ) -> Any:
        self.statements.append(sql)
        return super()._execute_query(sql, timeout_seconds, max_rows)


@pytest.fixture(autouse=True)
def _fresh_cache() -> Any:
    literal_grounding._cache.clear()
    yield
    literal_grounding._cache.clear()


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "travel.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE city (city_code TEXT, city_name TEXT)")
        conn.executemany("INSERT INTO city VALUES (?, ?)", [(c[:3], c) for c in CITIES])
        conn.execute("CREATE TABLE airport (airport_code TEXT, airport_name TEXT)")
        conn.execute("INSERT INTO airport VALUES ('DFW', 'DALLAS/FORT WORTH INTERNATIONAL')")
        conn.execute("CREATE TABLE customer (email TEXT, city_name TEXT)")
        conn.execute("INSERT INTO customer VALUES ('a@example.com', 'BOSTON')")
    return path


@pytest.fixture
def statements() -> list[str]:
    return []


@pytest.fixture
def connector(db: Path, statements: list[str]) -> _Recording:
    conn = _Recording(statements)
    conn.connect({"database": str(db)})
    # The statement itself runs as a query does: with permission to read.
    conn.bind_execution_policy(ExecutionPolicy.execute_read_only())
    return conn


@pytest.fixture
def source(connector: _Recording) -> LookupSource:
    return LookupSource(
        key="travel",
        open=lambda: PermittedConnector(connector, ExecutionPolicy.execute_read_only()),
    )


def _kb(schema: str = "") -> dict[str, Any]:
    text = {"type": "TEXT"}
    return {
        "schema": {
            "tables": [
                {
                    "name": "city",
                    "schema": schema,
                    "row_count": len(CITIES),
                    "columns": [{"name": "city_code", **text}, {"name": "city_name", **text}],
                },
                {
                    "name": "airport",
                    "schema": schema,
                    "columns": [{"name": "airport_code", **text}, {"name": "airport_name", **text}],
                },
                {
                    "name": "customer",
                    "schema": schema,
                    "columns": [{"name": "email", **text}, {"name": "city_name", **text}],
                },
            ]
        }
    }


def _run(sql: str, connector: _Recording) -> Any:
    return connector.execute_query(sql, 5)


def _follow_up(
    sql: str,
    connector: _Recording,
    source: LookupSource,
    *,
    may_rerun: bool = True,
    kb: dict[str, Any] | None = None,
) -> Any:
    qr = _run(sql, connector)
    result = SQLGenerationResult(
        sql=sql, is_valid=True, validation_error=None, dialect="sqlite", attempt_count=1
    )
    return asyncio.run(
        follow_up_empty_result(
            result,
            qr,
            connector=connector,
            kb=kb or _kb(),
            dialect="sqlite",
            lookups=source,
            budget=lambda: (may_rerun, 5),
        )
    )


DFW = "SELECT city_code FROM city WHERE city_name = 'DALLAS FORT WORTH'"


def test_a_literal_that_matches_nothing_gets_its_whole_word_parts(source: LookupSource) -> None:
    found = asyncio.run(empty_result_literals(DFW, _kb(), "sqlite", source))

    assert [(f.column, f.literal, f.candidates) for f in found] == [
        ("city_name", "DALLAS FORT WORTH", ("DALLAS", "FORT WORTH"))
    ]


def test_ask_offers_the_parts_as_choices(connector: _Recording, source: LookupSource) -> None:
    with use_query_settings(QuerySettings(ambiguity="ask")):
        followup = _follow_up(DFW, connector, source)

    assert followup is not None
    (choice,) = followup.choices
    assert choice.options == ("DALLAS", "FORT WORTH")
    assert choice.key == choice_key("city", "city_name", "DALLAS FORT WORTH")
    assert followup.notes == [
        "No city_name is 'DALLAS FORT WORTH'. Did you mean: 'DALLAS', 'FORT WORTH'?"
    ]
    assert followup.result.sql == DFW, "nothing runs until the user has chosen"


def test_pick_does_not_guess_between_several(connector: _Recording, source: LookupSource) -> None:
    with use_query_settings(QuerySettings(ambiguity="pick")):
        followup = _follow_up(DFW, connector, source)

    assert followup is not None and followup.choices == []
    assert followup.notes == [
        "No city_name is 'DALLAS FORT WORTH'. Stored values it may mean: 'DALLAS', 'FORT WORTH'."
    ]
    assert followup.qr.rows == []


def test_pick_runs_again_with_the_one_near_spelling(
    connector: _Recording, source: LookupSource
) -> None:
    sql = "SELECT city_code FROM city WHERE city_name = 'PITSBURGH'"
    with use_query_settings(QuerySettings(ambiguity="pick")):
        followup = _follow_up(sql, connector, source)

    assert followup is not None
    assert "'PITTSBURGH'" in followup.result.sql
    assert [list(r) for r in followup.qr.rows] == [["PIT"]]
    assert followup.notes == [
        "No city_name is 'PITSBURGH'; the answer uses 'PITTSBURGH', the stored value it "
        "is closest to."
    ]


def test_without_time_left_the_one_candidate_is_named_not_run(
    connector: _Recording, source: LookupSource
) -> None:
    sql = "SELECT city_code FROM city WHERE city_name = 'PITSBURGH'"
    followup = _follow_up(sql, connector, source, may_rerun=False)
    assert followup is not None and followup.qr.rows == []
    assert followup.notes == [
        "No city_name is 'PITSBURGH'. Stored values it may mean: 'PITTSBURGH'."
    ]


def test_a_stored_value_that_still_returns_nothing_is_left_alone(
    connector: _Recording, source: LookupSource
) -> None:
    sql = "SELECT city_code FROM city WHERE city_name = 'DALLAS' AND city_code = 'BOS'"
    assert _follow_up(sql, connector, source) is None


def test_a_personal_data_column_offers_nothing(connector: _Recording, source: LookupSource) -> None:
    sql = "SELECT city_name FROM customer WHERE email = 'b@example.com'"
    assert _follow_up(sql, connector, source) is None


def test_a_literal_with_nothing_near_is_named_in_the_note(
    connector: _Recording, source: LookupSource
) -> None:
    sql = "SELECT city_code FROM city WHERE city_name = 'ATLANTIS'"
    followup = _follow_up(sql, connector, source)
    assert followup is not None
    assert followup.notes == [
        "No rows matched: no stored city_name is 'ATLANTIS', and no stored value is close to it."
    ]


def test_rows_mean_nothing_is_looked_up(
    connector: _Recording, source: LookupSource, statements: list[str]
) -> None:
    sql = "SELECT city_code FROM city WHERE city_name = 'BOSTON'"
    qr = _run(sql, connector)
    statements.clear()
    followup = asyncio.run(
        follow_up_empty_result(
            SQLGenerationResult(
                sql=sql, is_valid=True, validation_error=None, dialect="sqlite", attempt_count=1
            ),
            qr,
            connector=connector,
            kb=_kb(),
            dialect="sqlite",
            lookups=source,
            budget=lambda: (True, 5),
        )
    )
    assert followup is None and statements == []


def test_the_value_chosen_for_an_unmatched_literal_is_used(source: LookupSource) -> None:
    key = choice_key("city", "city_name", "DALLAS FORT WORTH")
    with use_chosen_values({key: "FORT WORTH"}):
        grounded = asyncio.run(ground_literals(DFW, _kb(), "sqlite", source))

    assert "'FORT WORTH'" in grounded.sql
    assert grounded.match_notes[0][0] == "'DALLAS FORT WORTH' was taken as 'FORT WORTH', as chosen."


class _SearchPathWithout(SQLiteConnector):
    """A database whose search path leaves schema `travel` out: a statement
    naming a bare `city` fails, as on Postgres, and one naming `travel.city`
    reads the table."""

    def __init__(self, statements: list[str]) -> None:
        super().__init__()
        self.statements = statements

    def _execute_query(
        self, sql: str, timeout_seconds: float | None = None, max_rows: int | None = None
    ) -> Any:
        self.statements.append(sql)
        flat = sql.replace('"', "")
        if "travel." not in flat.lower():
            raise RuntimeError('relation "city" does not exist')
        return super()._execute_query(
            re.sub(r'"?travel"?\.', "", sql, flags=re.IGNORECASE), timeout_seconds, max_rows
        )


def test_lookups_read_the_schema_the_knowledge_base_records(
    db: Path, statements: list[str]
) -> None:
    """The draft names `city`; the table lives in schema `travel`. Where the
    search path leaves `travel` out, every lookup of the bare name failed and
    grounding was skipped (F19 addendum). The lookups now name `travel.city`,
    so the literal is grounded."""
    conn = _SearchPathWithout(statements)
    conn.connect({"database": str(db)})
    source = LookupSource(
        key="multi", open=lambda: PermittedConnector(conn, ExecutionPolicy.execute_read_only())
    )

    grounded = asyncio.run(
        ground_literals(
            "SELECT city_code FROM city WHERE city_name = 'boston'", _kb("travel"), "sqlite", source
        )
    )

    assert "'BOSTON'" in grounded.sql
    assert statements and all("travel.city" in s.lower().replace('"', "") for s in statements)


def test_a_table_two_schemas_hold_is_left_as_written(
    source: LookupSource, statements: list[str]
) -> None:
    kb = _kb("travel")
    city = dict(kb["schema"]["tables"][0], schema="archive")
    kb["schema"]["tables"].append(city)
    asyncio.run(
        ground_literals("SELECT 1 FROM city WHERE city_name = 'boston'", kb, "sqlite", source)
    )
    assert statements and not any("travel.city" in s.lower() for s in statements)


# --- Through the orchestrator ------------------------------------------------


class _QuietLLM:
    supports_prompt_caching = False

    async def astream(self, system: Any, user: str) -> Any:
        for t in ():
            yield t

    async def acomplete(self, system: Any, user: str, **_kw: Any) -> str:
        return ""


def _answer(
    sql: str, connector: _Recording, source: LookupSource, tmp_path: Path, ambiguity: str
) -> dict[str, Any]:
    import json
    from unittest.mock import AsyncMock, patch

    import yaml
    from nlqueries.orchestrator.orchestrator import Orchestrator

    first = SQLGenerationResult(
        sql=sql, is_valid=True, validation_error=None, dialect="sqlite", attempt_count=1
    )
    (tmp_path / "agent1.yaml").write_text(
        yaml.dump(
            {**_kb(), "business_context": {"glossary": [], "rules": []}, "query_capsules": []}
        ),
        encoding="utf-8",
    )
    with (
        patch("nlqueries.orchestrator.orchestrator.config") as cfg,
        patch("nlqueries.orchestrator.orchestrator.get_llm_client", return_value=_QuietLLM()),
        patch(
            "nlqueries.orchestrator.orchestrator.validate_and_repair",
            new=AsyncMock(return_value=first),
        ),
        patch("nlqueries.connectors.loader.open_connector_for_agent", return_value=connector),
        patch("nlqueries.connectors.loader.lookup_source", return_value=source),
        patch("nlqueries.embeddings.qdrant_store.search_schema", return_value=[]),
        patch("nlqueries.embeddings.qdrant_store.search", return_value=[]),
        patch("nlqueries.embeddings.embedder.embed_text", return_value=[0.0] * 384),
        patch("nlqueries.orchestrator.prompt_assembly._search_verified", return_value=[]),
        use_query_settings(QuerySettings(ambiguity=ambiguity)),
    ):
        cfg.KB_PATH = tmp_path
        cfg.CONNECTOR_STATEMENT_TIMEOUT_SECONDS = 120.0

        async def collect() -> list[str]:
            return [
                tok
                async for tok in Orchestrator().handle_question(
                    "flights to dallas fort worth",
                    "agent1",
                    dialect="sqlite",
                    execution=ExecutionPolicy.execute_read_only(),
                )
            ]

        tokens = asyncio.run(collect())
    return json.loads(tokens[-1])


def test_the_answer_offers_choices_in_place_of_the_generic_note(
    connector: _Recording, source: LookupSource, tmp_path: Path
) -> None:
    frame = _answer(DFW, connector, source, tmp_path, "ask")

    assert [c["options"] for c in frame["choices"]] == [["DALLAS", "FORT WORTH"]]
    assert frame["result_notes"] == [
        "No city_name is 'DALLAS FORT WORTH'. Did you mean: 'DALLAS', 'FORT WORTH'?"
    ]


def test_the_answer_runs_with_the_one_stored_spelling(
    connector: _Recording, source: LookupSource, tmp_path: Path
) -> None:
    frame = _answer(
        "SELECT city_code FROM city WHERE city_name = 'PITSBURGH'",
        connector,
        source,
        tmp_path,
        "pick",
    )

    assert "'PITTSBURGH'" in frame["sql"]
    assert frame["sql_table"]["rows"] == [["PIT"]]
    assert frame["result_notes"] == [
        "No city_name is 'PITSBURGH'; the answer uses 'PITTSBURGH', the stored value it "
        "is closest to."
    ]
    assert "choices" not in frame


def test_a_lone_whole_word_part_is_named_not_run(tmp_path: Path, statements: list[str]) -> None:
    """'NOT ACTIVE' contains 'ACTIVE', its opposite: run with it, the answer is
    the reverse of the question (from #413's review)."""
    path = tmp_path / "accounts.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE account (id TEXT, status TEXT)")
        conn.executemany(
            "INSERT INTO account VALUES (?, ?)",
            [("1", "ACTIVE"), ("2", "INACTIVE")] + [(str(i), f"STATE {i}") for i in range(30)],
        )
    conn = _Recording(statements)
    conn.connect({"database": str(path)})
    conn.bind_execution_policy(ExecutionPolicy.execute_read_only())
    src = LookupSource(
        key="acct", open=lambda: PermittedConnector(conn, ExecutionPolicy.execute_read_only())
    )
    kb = {
        "schema": {
            "tables": [
                {
                    "name": "account",
                    "columns": [{"name": "id", "type": "TEXT"}, {"name": "status", "type": "TEXT"}],
                }
            ]
        }
    }
    sql = "SELECT id FROM account WHERE status = 'NOT ACTIVE'"

    followup = _follow_up(sql, conn, src, kb=kb)

    assert followup is not None and followup.qr.rows == []
    assert followup.result.sql == sql, "not run with its opposite"
    assert followup.notes == ["No status is 'NOT ACTIVE'. Stored values it may mean: 'ACTIVE'."]


def test_the_rerun_budget_is_asked_after_the_lookups(
    connector: _Recording, source: LookupSource, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Asked before them, a rerun could start after the caller's deadline had
    passed: the lookups can spend seconds (from #413's review)."""
    from nlqueries.orchestrator import empty_result

    events: list[str] = []
    real = empty_result.empty_result_literals

    async def recording(*args: Any) -> Any:
        found = await real(*args)
        events.append("lookups")
        return found

    monkeypatch.setattr(empty_result, "empty_result_literals", recording)

    def budget() -> tuple[bool, float | None]:
        events.append("budget")
        return True, 5

    sql = "SELECT city_code FROM city WHERE city_name = 'PITSBURGH'"
    asyncio.run(
        follow_up_empty_result(
            SQLGenerationResult(
                sql=sql, is_valid=True, validation_error=None, dialect="sqlite", attempt_count=1
            ),
            _run(sql, connector),
            connector=connector,
            kb=_kb(),
            dialect="sqlite",
            lookups=source,
            budget=budget,
        )
    )

    assert events == ["lookups", "budget"]
