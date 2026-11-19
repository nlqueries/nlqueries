"""A value index read before the database, through ``LookupSource.values``.

Core supplies no index; a layer above it adds one by wrapping ``lookup_source``.
These check what grounding does with what an index returns: that it answers
without the database, that a stale one is checked once, that a partial one is
not read as the whole column, and that a column it does not cover, or an index
that fails, is looked up as without one. Against a real SQLite file through the
real connector, as ``test_literal_grounding`` is.
"""

from __future__ import annotations

import asyncio
import copy
import sqlite3
import time
from pathlib import Path
from typing import Any

import pytest
from nlqueries.connectors.base import PermittedConnector
from nlqueries.connectors.loader import IndexedValues, LookupSource
from nlqueries.connectors.sqlite import SQLiteConnector
from nlqueries.execution import ExecutionPolicy
from nlqueries.orchestrator import literal_grounding
from nlqueries.orchestrator.literal_grounding import (
    GroundingResult,
    empty_result_literals,
    ground_literals,
)
from nlqueries.query_settings import QuerySettings, use_query_settings

KB: dict[str, Any] = {
    "schema": {
        "tables": [
            {
                "name": "schools",
                "schema": "main",
                "columns": [
                    {"name": "id", "type": "INTEGER", "is_primary_key": True},
                    {"name": "status", "type": "TEXT"},
                    {"name": "city", "type": "TEXT"},
                    {"name": "district_code", "type": "TEXT", "is_foreign_key": True},
                    {"name": "email", "type": "TEXT"},
                ],
            }
        ]
    }
}

ROWS = [
    (1, "Legal", "São Paulo", "A1", "a@example.org"),
    (2, "Active", "Dallas", "B2", "b@example.org"),
    (3, "Closed", "Fort Worth", "C3", "c@example.org"),
]


@pytest.fixture(autouse=True)
def _fresh_cache() -> Any:
    literal_grounding._cache.clear()
    yield
    literal_grounding._cache.clear()


class _Recording(SQLiteConnector):
    def __init__(self, statements: list[str]) -> None:
        super().__init__()
        self.statements = statements

    def _execute_query(
        self, sql: str, timeout_seconds: float | None = None, max_rows: int | None = None
    ) -> Any:
        self.statements.append(sql)
        return super()._execute_query(sql, timeout_seconds, max_rows)


class _Index:
    """A value index: fixed answers per column, recording every call."""

    def __init__(self, answers: dict[str, IndexedValues | None] | None = None) -> None:
        self.answers = answers or {}
        self.calls: list[tuple[str | None, str | None, str, str, str]] = []
        self.error: Exception | None = None
        self.delay = 0.0

    def __call__(
        self, catalog: str | None, schema: str | None, table: str, column: str, literal: str
    ) -> IndexedValues | None:
        self.calls.append((catalog, schema, table, column, literal))
        if self.delay:
            time.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return self.answers.get(column)


class _Source:
    def __init__(self, db: Path, index: _Index | None) -> None:
        self.statements: list[str] = []
        self.opened: list[PermittedConnector] = []
        self.inner = _Recording(self.statements)
        self.inner.connect({"database": str(db)})
        self.index = index
        self.source = LookupSource(key=str(db), open=self._open, values=index)

    def _open(self) -> PermittedConnector:
        wrapper = PermittedConnector(self.inner, ExecutionPolicy.execute_read_only())
        self.opened.append(wrapper)
        return wrapper


def _db(path: Path, rows: list[tuple[Any, ...]] = ROWS) -> Path:
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE schools (id INTEGER PRIMARY KEY, status TEXT, city TEXT, "
            "district_code TEXT, email TEXT)"
        )
        conn.executemany("INSERT INTO schools VALUES (?, ?, ?, ?, ?)", rows)
    conn.close()
    return path


def _source(tmp_path: Path, name: str = "s.db", **answers: IndexedValues | None) -> _Source:
    return _Source(_db(tmp_path / name), _Index(answers))


def _ground(
    sql: str, src: _Source, kb: dict[str, Any] = KB, mode: str = "loose"
) -> GroundingResult:
    with use_query_settings(QuerySettings(match_mode=mode)):
        return asyncio.run(ground_literals(sql, kb, "sqlite", src.source))


def _complete(*values: str) -> IndexedValues:
    return IndexedValues(values=values, complete=True)


def _partial(*values: str, current: bool = True) -> IndexedValues:
    return IndexedValues(values=values, complete=False, current=current)


# --- The index answers without the database ------------------------------------


def test_a_literal_the_index_holds_runs_as_written_and_opens_no_connector(
    tmp_path: Path,
) -> None:
    src = _source(tmp_path, status=_complete("Legal", "Active", "Closed"))
    sql = "SELECT id FROM schools WHERE status = 'Legal'"
    result = _ground(sql, src)
    assert result.sql == sql
    assert result.substitutions == []
    assert src.opened == []
    assert src.statements == []


def test_a_literal_in_the_wrong_case_takes_the_index_s_spelling_without_a_query(
    tmp_path: Path,
) -> None:
    src = _source(tmp_path, status=_complete("LEGAL", "Active", "Closed"))
    result = _ground("SELECT id FROM schools WHERE status = 'legal'", src)
    assert "'LEGAL'" in result.sql
    assert result.substitutions == [("schools", "status", "legal", "LEGAL")]
    assert src.statements == []


def test_the_index_is_asked_with_the_schema_the_knowledge_base_records(tmp_path: Path) -> None:
    src = _source(tmp_path, status=_complete("Legal"))
    _ground("SELECT id FROM schools WHERE status = 'legal'", src)
    assert src.index is not None
    assert src.index.calls == [(None, "main", "schools", "status", "legal")]


def test_the_index_is_asked_with_the_catalog_a_three_part_name_gives(tmp_path: Path) -> None:
    """Two tables of one schema and name in different databases are told apart."""
    src = _source(tmp_path, status=_complete("Legal"))
    _ground("SELECT id FROM warehouse.main.schools WHERE status = 'legal'", src)
    assert src.index is not None
    assert src.index.calls == [("warehouse", "main", "schools", "status", "legal")]


def test_the_index_answers_under_a_row_scope_whose_lookups_are_not_cached(
    tmp_path: Path,
) -> None:
    """An index is built through the request's filters, so it answers where the
    connector declares no cache scope. The database here spells the value
    'Active'; the index's 'ACTIVE' is what is taken."""
    src = _source(tmp_path, status=_complete("Legal", "ACTIVE"))

    class _Scoped:
        cache_scope = None

        def __init__(self, inner: Any) -> None:
            self._inner = inner

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

    src.source = LookupSource(key="k", open=lambda: _Scoped(src._open()), values=src.index)
    result = _ground("SELECT id FROM schools WHERE status = 'active'", src)
    assert result.substitutions == [("schools", "status", "active", "ACTIVE")]


def test_accents_alone_differ_and_the_index_s_value_is_taken(tmp_path: Path) -> None:
    src = _source(tmp_path, city=_complete("São Paulo", "Dallas", "Fort Worth"))
    result = _ground("SELECT id FROM schools WHERE city = 'sao paulo'", src)
    assert result.substitutions == [("schools", "city", "sao paulo", "São Paulo")]


def test_accents_alone_differ_in_a_stored_complete_list_too(tmp_path: Path) -> None:
    """The same pass runs over the knowledge base's complete list."""
    src = _Source(_db(tmp_path / "s.db"), None)
    kb = copy.deepcopy(KB)
    city = kb["schema"]["tables"][0]["columns"][2]
    city["samples"], city["values_complete"] = ["São Paulo", "Dallas", "Fort Worth"], True
    result = _ground("SELECT id FROM schools WHERE city = 'Sao Paulo'", src, kb)
    assert result.substitutions == [("schools", "city", "Sao Paulo", "São Paulo")]


# --- A partial answer is not the whole column ----------------------------------


def test_a_complete_index_lists_every_value_in_its_note(tmp_path: Path) -> None:
    src = _source(tmp_path, status=_complete("Legal", "Active", "Closed"))
    result = _ground("SELECT id FROM schools WHERE status = 'Pending'", src)
    assert result.notes == [
        "No row has status = 'Pending'; the column holds only these values: "
        "'Legal', 'Active', 'Closed'."
    ]


def test_a_partial_index_does_not_claim_its_values_are_the_whole_column(
    tmp_path: Path,
) -> None:
    src = _source(tmp_path, status=_partial("Legally closed", "Active"))
    result = _ground("SELECT id FROM schools WHERE status = 'Legal'", src)
    assert len(result.notes) == 1
    assert "holds only these values" not in result.notes[0]
    assert result.notes[0].startswith("No row has status = 'Legal'; nearby values: ")
    assert "'Legally closed'" in result.notes[0]


def test_one_value_containing_the_literal_is_not_taken_from_a_partial_index(
    tmp_path: Path,
) -> None:
    """Asked to match by containment, one containing value is the match only
    where it is the only one the column holds, which a partial index cannot say."""
    complete = _ground(
        "SELECT id FROM schools WHERE status = 'Lega'",
        _source(tmp_path, status=_complete("Legal", "Active")),
        mode="contains",
    )
    assert complete.substitutions == [("schools", "status", "Lega", "Legal")]
    partial = _ground(
        "SELECT id FROM schools WHERE status = 'Lega'",
        _source(tmp_path, "p.db", status=_partial("Legal", "Active")),
        mode="contains",
    )
    assert partial.substitutions == []


# --- A stale index is checked once --------------------------------------------


def test_an_index_not_current_checks_the_database_once_before_a_miss(tmp_path: Path) -> None:
    """'Closed' was stored after the build: the check finds it, so it is not
    read as a miss and not replaced by a near value."""
    src = _source(tmp_path, status=_partial("Legal", "Active", "Close", current=False))
    sql = "SELECT id FROM schools WHERE status = 'Closed'"
    result = _ground(sql, src)
    assert result.sql == sql
    assert result.notes == []
    assert len(src.statements) == 1
    assert "status = 'Closed'" in src.statements[0]


def test_a_current_index_runs_no_check(tmp_path: Path) -> None:
    src = _source(tmp_path, status=_partial("Legal", "Active", "Close"))
    _ground("SELECT id FROM schools WHERE status = 'Closed'", src)
    assert src.statements == []


def test_a_stale_index_whose_check_fails_does_not_fail_the_question(tmp_path: Path) -> None:
    """The check's query errors (here the table is missing): grounding skips
    the literal as it does any lookup that fails, and an empty result's
    follow-up finds nothing to offer, rather than either raising."""
    empty = tmp_path / "empty.db"
    sqlite3.connect(empty).close()
    src = _Source(empty, _Index({"status": _partial("Legal", current=False)}))
    sql = "SELECT id FROM schools WHERE status = 'Closed'"
    assert _ground(sql, src).sql == sql
    assert asyncio.run(empty_result_literals(sql, KB, "sqlite", src.source)) == []
    assert src.statements


def test_a_table_too_large_to_scan_runs_no_check_even_when_the_index_is_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nlqueries import config

    monkeypatch.setattr(config, "LITERAL_GROUNDING_MAX_ROWS", 10)
    kb = copy.deepcopy(KB)
    kb["schema"]["tables"][0]["row_count"] = 1_000
    src = _source(tmp_path, status=_partial("LEGAL", current=False))
    result = _ground("SELECT id FROM schools WHERE status = 'legal'", src, kb)
    assert result.substitutions == [("schools", "status", "legal", "LEGAL")]
    assert src.statements == []


# --- Without an answer from the index, the database as before -----------------


def test_a_column_the_index_does_not_cover_is_looked_up_in_the_database(
    tmp_path: Path,
) -> None:
    src = _source(tmp_path, status=_complete("Legal"))
    result = _ground("SELECT id FROM schools WHERE city = 'dallas'", src)
    assert result.substitutions == [("schools", "city", "dallas", "Dallas")]
    assert src.statements


def test_an_index_that_fails_leaves_the_column_to_the_database(tmp_path: Path) -> None:
    src = _source(tmp_path, status=_complete("Wrong"))
    assert src.index is not None
    src.index.error = RuntimeError("index down")
    result = _ground("SELECT id FROM schools WHERE status = 'legal'", src)
    assert result.substitutions == [("schools", "status", "legal", "Legal")]
    assert src.statements


def test_an_index_slower_than_the_deadline_is_not_waited_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(literal_grounding, "_TOTAL_BUDGET_S", 0.3)
    src = _source(tmp_path, status=_complete("Legal"))
    assert src.index is not None
    src.index.delay = 2.0
    sql = "SELECT id FROM schools WHERE status = 'legal'"
    started = time.monotonic()
    result = _ground(sql, src)
    assert time.monotonic() - started < 1.5
    assert result.sql == sql


def test_keys_and_personal_data_never_reach_the_index(tmp_path: Path) -> None:
    src = _source(tmp_path, district_code=_complete("A1"), email=_complete("x"))
    _ground("SELECT id FROM schools WHERE district_code = 'a1' AND email = 'a@example.org'", src)
    assert src.index is not None
    assert src.index.calls == []


def test_what_the_index_says_is_not_cached_past_a_rebuild(tmp_path: Path) -> None:
    src = _source(tmp_path, status=_complete("Legal "))
    sql = "SELECT id FROM schools WHERE status = 'legal'"
    assert _ground(sql, src).substitutions == [("schools", "status", "legal", "Legal ")]
    assert src.index is not None
    src.index.answers["status"] = _complete("LEGAL")
    assert _ground(sql, src).substitutions == [("schools", "status", "legal", "LEGAL")]


# --- After an empty result ----------------------------------------------------


def test_an_empty_result_offers_the_index_s_values_for_the_literal_s_words(
    tmp_path: Path,
) -> None:
    src = _source(tmp_path, city=_partial("Dallas", "Fort Worth", "Dalhart"))
    found = asyncio.run(
        empty_result_literals(
            "SELECT id FROM schools WHERE city = 'Dallas Fort Worth'", KB, "sqlite", src.source
        )
    )
    assert [(f.column, f.literal) for f in found] == [("city", "Dallas Fort Worth")]
    assert found[0].candidates[:2] == ("Dallas", "Fort Worth")
    assert src.statements == []
