"""String literals in a WHERE clause, matched to what the column stores.

A model writes ``status = 'legal'`` where the data has ``'Legal'``, or copies
``' = '`` from a hint where the column holds ``'='``: valid SQL that returns no
rows. Grounding checks each such literal against the database before the
statement runs. These run against a real SQLite file through the real
connector, read-only, as the orchestrator would.
"""

from __future__ import annotations

import asyncio
import faulthandler
import json
import logging
import sqlite3
import sys
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from nlqueries.connectors.base import PermittedConnector, QueryResult
from nlqueries.connectors.loader import LookupSource, lookup_source
from nlqueries.connectors.sqlite import SQLiteConnector
from nlqueries.execution import ExecutionPolicy
from nlqueries.orchestrator import literal_grounding
from nlqueries.orchestrator.literal_grounding import GroundingResult, _fold, ground_literals
from nlqueries.orchestrator.provenance import Provenance, use_provenance
from nlqueries.orchestrator.sql_generation import SQLGenerationResult, validate_and_repair
from sqlglot import exp

KB: dict[str, Any] = {
    "schema": {
        "tables": [
            {
                "name": "schools",
                "columns": [
                    {"name": "id", "type": "INTEGER", "is_primary_key": True},
                    {"name": "status", "type": "TEXT"},
                    {"name": "funding", "type": "VARCHAR(40)"},
                    {"name": "soc", "type": "TEXT"},
                    {"name": "district_code", "type": "TEXT", "is_foreign_key": True},
                    {"name": "enrolment", "type": "INTEGER"},
                ],
            }
        ]
    }
}

ROWS = [
    (1, "Legal", "Directly funded", "Youth Authority Facilities", "A1", 120),
    (2, "Legal", "Locally funded", "Juvenile Court Schools", "B2", 80),
    (3, "Active", "=", "Youth Authority Facilities", "C3", 40),
    (4, "ACTIVE", "Directly funded", "Special Education Schools", "D4", 300),
]


@pytest.fixture(autouse=True)
def _fresh_cache() -> Any:
    """The cache lives for the process; each test starts without it."""
    literal_grounding._cache.clear()
    yield
    literal_grounding._cache.clear()


def _schools_db(path: Path, rows: list[tuple[Any, ...]] = ROWS) -> Path:
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE schools (id INTEGER PRIMARY KEY, status TEXT, funding TEXT, "
            "soc TEXT, district_code TEXT, enrolment INTEGER)"
        )
        conn.executemany("INSERT INTO schools VALUES (?, ?, ?, ?, ?, ?)", rows)
    conn.close()
    return path


class _Recording(SQLiteConnector):
    """A real SQLite connector that records every statement it runs."""

    def __init__(self, statements: list[str] | None = None) -> None:
        super().__init__()
        self.statements: list[str] = statements if statements is not None else []
        self.closed = False

    def _execute_query(
        self, sql: str, timeout_seconds: float | None = None, max_rows: int | None = None
    ) -> Any:
        self.statements.append(sql)
        return super()._execute_query(sql, timeout_seconds, max_rows)

    def close(self) -> None:
        self.closed = True
        super().close()


class _Source:
    """A lookup source on a SQLite file, as the loader builds one: core's wrapper
    around one connector, recording what it hands out and the statements run."""

    def __init__(self, db: Path, key: str | None = None) -> None:
        self.db = db
        self.statements: list[str] = []
        self.opened: list[PermittedConnector] = []
        self.inner = _Recording(self.statements)
        self.inner.connect({"database": str(db)})
        self.source = LookupSource(key=key or str(db), open=self._open)

    def _open(self) -> PermittedConnector:
        wrapper = PermittedConnector(self.inner, ExecutionPolicy.execute_read_only())
        self.opened.append(wrapper)
        return wrapper


def _entry(db: Path | str) -> dict[str, str]:
    """A connectors-file entry for a SQLite file."""
    return {"db_type": "sqlite", "url": f"sqlite:///{Path(db).as_posix()}"}


@pytest.fixture
def src(tmp_path: Path) -> _Source:
    return _Source(_schools_db(tmp_path / "schools.db"))


def _ground(
    sql: str, source: _Source | LookupSource | None, kb: dict[str, Any] = KB
) -> GroundingResult:
    lookups = source.source if isinstance(source, _Source) else source
    return asyncio.run(ground_literals(sql, kb, "sqlite", lookups))


# --- Substitutions ---------------------------------------------------------------


def test_a_literal_in_the_wrong_case_takes_the_stored_spelling(src: _Source) -> None:
    result = _ground("SELECT COUNT(*) FROM schools WHERE status = 'legal'", src)

    assert result.sql == "SELECT COUNT(*) FROM schools WHERE status = 'Legal'"
    assert result.substitutions == [("schools", "status", "legal", "Legal")]
    assert result.notes == []


def test_a_literal_with_extra_spaces_takes_the_stored_value(src: _Source) -> None:
    """Copied from a hint as ' = ' where the column holds '='."""
    result = _ground("SELECT id FROM schools WHERE funding = ' = '", src)

    assert result.sql == "SELECT id FROM schools WHERE funding = '='"


def test_a_column_qualified_by_an_alias_is_resolved_to_its_table(src: _Source) -> None:
    result = _ground("SELECT s.id FROM schools AS s WHERE s.funding = 'Directly Funded'", src)

    assert "s.funding = 'Directly funded'" in result.sql


def test_each_value_of_an_in_list_is_grounded(src: _Source) -> None:
    result = _ground("SELECT id FROM schools WHERE status IN ('legal', 'Active')", src)

    assert result.sql == "SELECT id FROM schools WHERE status IN ('Legal', 'Active')"


def test_a_like_without_a_wildcard_is_grounded_and_one_with_is_not(src: _Source) -> None:
    plain = _ground("SELECT id FROM schools WHERE status LIKE 'legal'", src)
    wild = _ground("SELECT id FROM schools WHERE status LIKE 'leg%'", src)

    assert plain.sql == "SELECT id FROM schools WHERE status LIKE 'Legal'"
    assert wild.sql == "SELECT id FROM schools WHERE status LIKE 'leg%'"


def test_a_value_that_exists_as_written_is_left_after_one_lookup(src: _Source) -> None:
    sql = "SELECT id FROM schools WHERE status = 'Legal'"

    result = _ground(sql, src)

    assert result.sql == sql and result.substitutions == [] and result.notes == []
    assert len(src.statements) == 1


def test_a_render_that_changes_any_other_literal_is_not_used(
    src: _Source, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Grounding re-renders the statement it changed. A render that also
    changed a literal grounding did not replace would change what the
    statement asks, and the flagged positions, counted on the tree, would not
    be shown to hold for it."""
    real = exp.Select.sql

    def _render(self: exp.Select, *args: Any, **kwargs: Any) -> str:
        return str(real(self, *args, **kwargs)).replace("'kept'", "'altered'")

    monkeypatch.setattr(exp.Select, "sql", _render)
    sql = "SELECT id, 'kept' AS tag FROM schools WHERE status = 'legal'"

    result = _ground(sql, src)

    assert result.sql == sql
    assert result.substitutions == []


# --- How the loose pass is written --------------------------------------------------


def test_on_sqlite_the_loose_pass_compares_with_nocase(src: _Source) -> None:
    result = _ground("SELECT id FROM schools WHERE status = 'legal'", src)

    loose = src.statements[1]
    assert "COLLATE NOCASE" in loose and "LOWER(" not in loose, loose
    assert result.sql == "SELECT id FROM schools WHERE status = 'Legal'"


def test_the_loose_pass_ignores_spaces_around_the_stored_value_too(tmp_path: Path) -> None:
    """Answered by the loose pass itself: no containing query is needed."""
    src = _posts(tmp_path, "  Spaced Title  ")

    result = _ground("SELECT id FROM posts WHERE title = 'spaced title'", src, POSTS_KB)

    assert result.sql == "SELECT id FROM posts WHERE title = '  Spaced Title  '"
    assert not any(" LIKE " in s for s in src.statements), src.statements


def test_on_sqlite_the_containing_search_leaves_the_column_as_stored(src: _Source) -> None:
    """SQLite's LIKE ignores ASCII case already; no LOWER on every row."""
    result = _ground("SELECT id FROM schools WHERE soc = 'youth authority school'", src)

    likes = [s for s in src.statements if " LIKE " in s]
    assert likes and all("LOWER(" not in s for s in likes), likes
    assert "soc LIKE '%youth authority school%'" in likes[0]
    # Still case-insensitive: the stored value differs in case from the search.
    assert result.notes and "'Youth Authority Facilities'" in result.notes[0]


def test_on_postgres_the_loose_pass_lowercases_both_sides() -> None:
    statements: list[str] = []
    empty = QueryResult(columns=["x"], rows=[], row_count=0, execution_time_ms=0.0, error=None)
    connector = MagicMock()

    def _execute_query(sql: str, *args: Any, **kwargs: Any) -> QueryResult:
        statements.append(sql)
        return empty

    connector.execute_query.side_effect = _execute_query
    source = LookupSource(key="postgres", open=lambda: connector)

    asyncio.run(
        ground_literals("SELECT id FROM schools WHERE status = 'legal'", KB, "postgres", source)
    )

    loose = statements[1]
    assert "LOWER(TRIM(status)) = LOWER(TRIM('legal'))" in loose, loose
    assert "COLLATE" not in loose
    likes = [s for s in statements if " LIKE " in s]
    assert likes and all("LOWER(status) LIKE '%legal%'" in s for s in likes), likes


# --- Punctuation at either end -----------------------------------------------------

POSTS_KB: dict[str, Any] = {
    "schema": {
        "tables": [
            {
                "name": "posts",
                "columns": [
                    {"name": "id", "type": "INTEGER", "is_primary_key": True},
                    {"name": "title", "type": "TEXT"},
                ],
            }
        ]
    }
}


def _posts(tmp_path: Path, *titles: str) -> _Source:
    db = tmp_path / "posts.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE posts (id INTEGER PRIMARY KEY, title TEXT)")
        conn.executemany("INSERT INTO posts (title) VALUES (?)", [(t,) for t in titles])
    conn.close()
    return _Source(db)


@pytest.mark.parametrize(
    ("value", "folded"),
    [
        (
            "Open source tools for visualizing multi-dimensional data?",
            "open source tools for visualizing multi-dimensional data",
        ),
        ("  (Hello,   World!) ", "hello, world"),
        ("'quoted'", "quoted"),
        ("[a] {b}", "a] {b"),
        ("tab\there.", "tab here"),
        ("a.b", "a.b"),
        ("?", ""),
        ("-", ""),
    ],
)
def test_fold_lowercases_collapses_spaces_and_strips_punctuation_only_at_the_ends(
    value: str, folded: str
) -> None:
    assert _fold(value) == folded


def test_a_literal_missing_the_stored_question_mark_takes_the_stored_value(
    tmp_path: Path,
) -> None:
    stored = "Open source tools for visualizing multi-dimensional data?"
    src = _posts(tmp_path, stored, "Help understand kNN for multi-dimensional data")

    result = _ground(
        "SELECT id FROM posts WHERE title = "
        "'Open source tools for visualizing multi-dimensional data'",
        src,
        POSTS_KB,
    )

    assert result.sql == f"SELECT id FROM posts WHERE title = '{stored}'"
    assert result.substitutions == [
        ("posts", "title", "Open source tools for visualizing multi-dimensional data", stored)
    ]


def test_a_literal_with_a_question_mark_the_stored_value_lacks_takes_the_stored_value(
    tmp_path: Path,
) -> None:
    """The other direction: the literal carries the punctuation, so a search for
    it as written would never find the stored value."""
    src = _posts(tmp_path, "Bayesian inference", "Bayesian inference in practice")

    result = _ground("SELECT id FROM posts WHERE title = 'Bayesian inference?'", src, POSTS_KB)

    assert result.sql == "SELECT id FROM posts WHERE title = 'Bayesian inference'"
    assert result.substitutions == [("posts", "title", "Bayesian inference?", "Bayesian inference")]


def test_a_literal_differing_in_case_and_end_punctuation_is_grounded(tmp_path: Path) -> None:
    src = _posts(tmp_path, "(The Art of Statistics)", "The Art of Statistics, Revisited")

    result = _ground("SELECT id FROM posts WHERE title = 'the art of statistics'", src, POSTS_KB)

    assert result.sql == "SELECT id FROM posts WHERE title = '(The Art of Statistics)'"


def test_several_values_equal_once_folded_are_left_alone_with_a_note(tmp_path: Path) -> None:
    src = _posts(tmp_path, "Why use R?", "Why use R!", "Why use Rust?")
    sql = "SELECT id FROM posts WHERE title = 'why use r'"

    result = _ground(sql, src, POSTS_KB)

    assert result.sql == sql and result.substitutions == []
    assert len(result.notes) == 1
    assert "several stored values" in result.notes[0]
    assert "'Why use R?'" in result.notes[0] and "'Why use R!'" in result.notes[0]
    assert "Rust" not in result.notes[0]


def test_a_literal_under_three_characters_once_folded_is_not_folded(tmp_path: Path) -> None:
    """'ab' folds to two characters: matched exactly or not at all, so the stored
    'ab.' is offered in a note, not substituted."""
    src = _posts(tmp_path, "ab.", "abc")
    sql = "SELECT id FROM posts WHERE title = 'ab'"

    result = _ground(sql, src, POSTS_KB)

    assert result.sql == sql and result.substitutions == []
    assert result.notes and "'ab.'" in result.notes[0]


def test_a_long_stored_value_is_cut_in_the_note(tmp_path: Path) -> None:
    """Notes go into repair prompts: a free-text value is cut, and the cut marked."""
    long_title = "data science " + "x" * 300
    src = _posts(tmp_path, long_title)

    result = _ground("SELECT id FROM posts WHERE title = 'data science'", src, POSTS_KB)
    note = result.notes[0] if result.notes else ""

    assert f"'{long_title[:100]}' [cut]" in note, note
    assert long_title[:101] not in note


def test_several_long_matches_are_cut_in_the_note(tmp_path: Path) -> None:
    """A 100-character literal matches two stored values that add end punctuation."""
    literal = "word " * 19 + "words"
    src = _posts(tmp_path, literal + "?", literal + "!")

    result = _ground(f"SELECT id FROM posts WHERE title = '{literal}'", src, POSTS_KB)

    assert len(literal) == 100
    assert result.notes and "several stored values" in result.notes[0]
    assert result.notes[0].count("[cut]") == 2, result.notes[0]


def test_with_no_match_the_containing_values_are_fetched_once_for_the_note(
    tmp_path: Path,
) -> None:
    src = _posts(tmp_path, "Statistics for data science, part one", "Data science in R")
    sql = "SELECT id FROM posts WHERE title = 'data science'"

    result = _ground(sql, src, POSTS_KB)

    assert result.sql == sql and result.substitutions == []
    assert result.notes and "'Data science in R'" in result.notes[0]
    assert sum(" LIKE " in s for s in src.statements) == 1, src.statements


# --- Left alone, with a note ------------------------------------------------------


def test_an_ambiguous_match_is_left_alone_with_a_note(src: _Source) -> None:
    """'active' matches both 'Active' and 'ACTIVE': not ours to pick."""
    sql = "SELECT id FROM schools WHERE status = 'active'"

    result = _ground(sql, src)

    assert result.sql == sql
    assert result.substitutions == []
    assert len(result.notes) == 1
    assert "status = 'active'" in result.notes[0]
    assert "'Active'" in result.notes[0] and "'ACTIVE'" in result.notes[0]


def test_a_missing_value_is_left_alone_with_the_nearest_stored_values(src: _Source) -> None:
    sql = "SELECT id FROM schools WHERE soc = 'Youth Authority School'"

    result = _ground(sql, src)

    assert result.sql == sql
    assert result.notes and "No row has soc = 'Youth Authority School'" in result.notes[0]
    assert "'Youth Authority Facilities'" in result.notes[0]


# --- What is not looked up ----------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        # Keys: an identifier's "nearest value" is not a fix.
        "SELECT status FROM schools WHERE district_code = 'a1'",
        # Not a text column.
        "SELECT status FROM schools WHERE enrolment = '120'",
        # A column the knowledge base does not know.
        "SELECT status FROM schools WHERE nickname = 'x'",
        # Free text, not a stored category.
        "SELECT status FROM schools WHERE soc = '" + "x" * 101 + "'",
    ],
)
def test_comparisons_that_are_not_grounded_open_and_run_nothing(src: _Source, sql: str) -> None:
    result = _ground(sql, src)

    assert result.sql == sql
    assert src.opened == [] and src.statements == []


def test_nothing_happens_without_a_source() -> None:
    """An agent with no registered connector gets no source, and is never grounded."""
    sql = "SELECT id FROM schools WHERE status = 'legal'"

    assert _ground(sql, None).sql == sql


def test_a_failing_lookup_leaves_the_statement_alone() -> None:
    broken = MagicMock()
    broken.execute_query.side_effect = RuntimeError("connection reset")
    sql = "SELECT id FROM schools WHERE status = 'legal'"

    result = _ground(sql, LookupSource(key="broken", open=lambda: broken))

    assert result.sql == sql and result.substitutions == [] and result.notes == []


def test_a_source_that_cannot_connect_is_tried_once() -> None:
    """No connector, so nothing else to try: one attempt, not one per literal."""
    attempts: list[int] = []

    def _refuse() -> Any:
        attempts.append(1)
        raise ConnectionError("refused")

    sql = "SELECT id FROM schools WHERE status = 'legal' OR funding = 'x' OR soc = 'y'"

    result = _ground(sql, LookupSource(key="down", open=_refuse))

    assert result.sql == sql
    assert len(attempts) == 1


def test_a_connector_that_cannot_be_opened_is_tried_once() -> None:
    """The loader returns None when it cannot connect: one attempt, no lookups."""
    attempts: list[int] = []

    def _none() -> None:
        attempts.append(1)

    sql = "SELECT id FROM schools WHERE status = 'legal' OR funding = 'x' OR soc = 'y'"

    result = _ground(sql, LookupSource(key="none", open=_none))

    assert result.sql == sql
    assert len(attempts) == 1


# --- Personal data ----------------------------------------------------------------


def _people(tmp_path: Path, column: str) -> tuple[_Source, dict[str, Any]]:
    """A table of two people's addresses, its column named *column*."""
    db = tmp_path / "people.db"
    with sqlite3.connect(db) as conn:
        conn.execute(f"CREATE TABLE people (id INTEGER PRIMARY KEY, {column} TEXT)")
        conn.executemany(
            "INSERT INTO people VALUES (?, ?)", [(1, "john.doe@acme.com"), (2, "jane.roe@acme.com")]
        )
    conn.close()
    kb = {
        "schema": {
            "tables": [
                {
                    "name": "people",
                    "columns": [
                        {"name": "id", "type": "INTEGER", "is_primary_key": True},
                        {"name": column, "type": "TEXT"},
                    ],
                }
            ]
        }
    }
    return _Source(db), kb


@pytest.mark.parametrize(
    "column", ["email", "contact_phone", "home_address", "api_token", "password_hash"]
)
def test_a_column_named_as_personal_data_is_never_looked_up(tmp_path: Path, column: str) -> None:
    """The refusal export-kb makes: a lookup would copy other people's values
    into a note, the repair prompt and provenance."""
    src, kb = _people(tmp_path, column)
    sql = f"SELECT id FROM people WHERE {column} = 'John.Doe@acme.com'"

    result = _ground(sql, src, kb)

    assert result.sql == sql and result.notes == [] and result.substitutions == []
    assert src.opened == [] and src.statements == []


def test_the_same_column_under_an_ordinary_name_is_grounded(tmp_path: Path) -> None:
    src, kb = _people(tmp_path, "contact")

    result = _ground("SELECT id FROM people WHERE contact = 'John.Doe@acme.com'", src, kb)

    assert result.sql == "SELECT id FROM people WHERE contact = 'john.doe@acme.com'"


# --- Bounds, the cache, and closing --------------------------------------------------


def test_at_most_eight_literals_are_looked_up_per_statement(src: _Source) -> None:
    values = ", ".join(f"'Legal{i}'" for i in range(12))

    _ground(f"SELECT id FROM schools WHERE status IN ({values})", src)

    looked_up = {f"Legal{i}" for i in range(12) if any(f"'Legal{i}'" in s for s in src.statements)}
    assert len(looked_up) == 8


def test_the_step_stops_when_its_time_is_spent(
    src: _Source, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(literal_grounding, "_TOTAL_BUDGET_S", 0.0)

    result = _ground("SELECT id FROM schools WHERE status = 'legal'", src)

    assert src.opened == [] and src.statements == []
    assert result.sql == "SELECT id FROM schools WHERE status = 'legal'"


def test_a_repeated_literal_is_answered_from_the_cache(src: _Source) -> None:
    sql = "SELECT id FROM schools WHERE status = 'legal'"

    _ground(sql, src)
    asked = len(src.statements)
    again = _ground(sql, src)

    assert asked > 0 and len(src.statements) == asked
    assert again.sql == "SELECT id FROM schools WHERE status = 'Legal'"


def test_the_cache_is_keyed_on_the_database_not_the_source_object(tmp_path: Path) -> None:
    """The loader builds a new source per request. Keyed on the object, the cache
    would die with each request and no lookup would ever be answered twice."""
    db = _schools_db(tmp_path / "schools.db")
    first, second = _Source(db, key="db-1"), _Source(db, key="db-1")
    sql = "SELECT id FROM schools WHERE status = 'legal'"

    _ground(sql, first)
    result = _ground(sql, second)

    assert first.statements and second.statements == []
    assert result.sql == "SELECT id FROM schools WHERE status = 'Legal'"


def test_two_databases_never_share_an_answer(src: _Source, tmp_path: Path) -> None:
    other_db = tmp_path / "other.db"
    with sqlite3.connect(other_db) as conn:
        conn.execute("CREATE TABLE schools (id INTEGER, status TEXT)")
        conn.execute("INSERT INTO schools VALUES (1, 'LEGAL')")
    conn.close()
    sql = "SELECT id FROM schools WHERE status = 'legal'"

    assert "'Legal'" in _ground(sql, src).sql
    assert "'LEGAL'" in _ground(sql, _Source(other_db)).sql


def test_an_outcome_past_its_time_is_looked_up_again(
    src: _Source, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A value added to the column is seen once the outcome has expired."""
    monkeypatch.setattr(literal_grounding, "_CACHE_TTL_S", -1.0)
    sql = "SELECT id FROM schools WHERE status = 'legal'"

    _ground(sql, src)
    asked = len(src.statements)
    _ground(sql, src)

    assert asked > 0 and len(src.statements) == 2 * asked


def test_the_cache_drops_the_least_recently_used_outcome_first(
    src: _Source, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(literal_grounding, "_MAX_CACHED", 2)
    first, second, third = (
        "SELECT id FROM schools WHERE status = 'legal'",
        "SELECT id FROM schools WHERE funding = ' = '",
        "SELECT id FROM schools WHERE soc = 'youth authority facilities'",
    )
    for sql in (first, second, first, third):  # `first` used again: `second` is oldest
        _ground(sql, src)
    asked = len(src.statements)

    _ground(first, src)
    _ground(third, src)
    assert len(src.statements) == asked, "the two most recent were kept"
    _ground(second, src)
    assert len(src.statements) > asked, "the least recently used was dropped"


def test_each_call_releases_what_it_opened_and_the_pooled_connector_stays_open(
    src: _Source, monkeypatch: pytest.MonkeyPatch
) -> None:
    released: list[PermittedConnector] = []
    monkeypatch.setattr(PermittedConnector, "close", lambda self: released.append(self))

    _ground("SELECT id FROM schools WHERE status = 'legal'", src)
    _ground("SELECT id FROM schools WHERE funding = ' = '", src)

    assert len(src.opened) == 2
    assert released == src.opened
    assert not src.inner.closed


# --- Row scope ------------------------------------------------------------------------


class _RowFiltered:
    """A per-request wrapper that restricts rows, forwarding everything else to
    what it wraps, as the enterprise layer's row filter does."""

    def __init__(self, inner: Any, scope: str | None = None) -> None:
        self._inner = inner
        if scope is not None:
            self.cache_scope = scope

    def execute_query(self, sql: str, *args: Any, **kwargs: Any) -> Any:
        return self._inner.execute_query(sql, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _filtered(src: _Source, scope: str | None) -> LookupSource:
    return LookupSource(key=src.source.key, open=lambda: _RowFiltered(src._open(), scope))


def test_a_wrapper_that_declares_no_scope_is_not_cached(src: _Source) -> None:
    """A value one user may see is not one every user may: without a declared
    scope, every request looks its values up again."""
    sql = "SELECT id FROM schools WHERE status = 'legal'"

    _ground(sql, _filtered(src, None))
    asked = len(src.statements)
    again = _ground(sql, _filtered(src, None))

    assert asked > 0 and len(src.statements) == 2 * asked
    assert again.sql == "SELECT id FROM schools WHERE status = 'Legal'"


def test_a_subclass_of_the_core_wrapper_that_declares_no_scope_is_not_cached(
    src: _Source,
) -> None:
    """A row filter written as a subclass of PermittedConnector inherits its
    delegation; it must not inherit sharing as well."""

    class _FilteringSubclass(PermittedConnector):
        pass

    source = LookupSource(
        key=src.source.key,
        open=lambda: _FilteringSubclass(src.inner, ExecutionPolicy.execute_read_only()),
    )
    sql = "SELECT id FROM schools WHERE status = 'legal'"

    _ground(sql, source)
    asked = len(src.statements)
    _ground(sql, source)

    assert asked > 0 and len(src.statements) == 2 * asked


def test_exactly_the_core_wrapper_is_shared(src: _Source) -> None:
    source = LookupSource(
        key=src.source.key,
        open=lambda: PermittedConnector(src.inner, ExecutionPolicy.execute_read_only()),
    )
    sql = "SELECT id FROM schools WHERE status = 'legal'"

    _ground(sql, source)
    asked = len(src.statements)
    _ground(sql, source)

    assert asked > 0 and len(src.statements) == asked


def test_outcomes_are_shared_only_within_a_declared_scope(src: _Source) -> None:
    sql = "SELECT id FROM schools WHERE status = 'legal'"

    _ground(sql, _filtered(src, "tenant-a"))
    asked = len(src.statements)
    _ground(sql, _filtered(src, "tenant-a"))
    assert len(src.statements) == asked, "same scope: answered from the cache"
    _ground(sql, _filtered(src, "tenant-b"))
    assert len(src.statements) == 2 * asked, "another scope: looked up again"
    _ground(sql, src)
    assert len(src.statements) == 3 * asked, "the unrestricted wrapper: its own entry"


def test_the_lookups_run_in_the_callers_context(src: _Source) -> None:
    """A wrapper reads request-bound state from a ContextVar when it is opened;
    the lookups' own threads must see the caller's."""
    import contextvars

    bound: contextvars.ContextVar[str | None] = contextvars.ContextVar("bound", default=None)
    seen: list[str | None] = []

    def _open() -> Any:
        seen.append(bound.get())
        return src._open()

    async def run() -> GroundingResult:
        bound.set("request-1")
        return await ground_literals(
            "SELECT id FROM schools WHERE status = 'legal'",
            KB,
            "sqlite",
            LookupSource(key="ctx", open=_open),
        )

    result = asyncio.run(run())

    assert seen == ["request-1"]
    assert result.sql == "SELECT id FROM schools WHERE status = 'Legal'"


# --- Stuck lookups -------------------------------------------------------------------


class _Stuck:
    """A connector whose lookups block until released."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.closed = threading.Event()

    def execute_query(self, sql: str, *args: Any, **kwargs: Any) -> QueryResult:
        self.started.set()
        self.release.wait(30)
        return QueryResult(columns=["x"], rows=[], row_count=0, execution_time_ms=0.0, error=None)

    def close(self) -> None:
        self.closed.set()


def test_a_stuck_lookup_is_abandoned_at_the_deadline_and_closed_when_it_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(literal_grounding, "_TOTAL_BUDGET_S", 0.5)
    stuck = _Stuck()
    sql = "SELECT id FROM schools WHERE status = 'legal'"

    started = time.monotonic()
    result = _ground(sql, LookupSource(key="stuck", open=lambda: stuck))
    elapsed = time.monotonic() - started

    try:
        assert stuck.started.is_set(), "the lookup ran"
        assert result.sql == sql
        assert elapsed < 5, f"waited {elapsed:.1f}s for a lookup it should have abandoned"
        # Never closed under a statement still running on it ...
        assert not stuck.closed.is_set()
        # ... and on a daemon thread, which the interpreter does not wait for at exit.
        lookups = [t for t in threading.enumerate() if t.name == "literal-grounding"]
        assert lookups and all(t.daemon for t in lookups)
    finally:
        stuck.release.set()
    # ... and closed once that statement returns.
    assert stuck.closed.wait(5)


def test_after_an_abandoned_lookup_nothing_more_runs_on_its_connector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even with time left on the clock, as when a timer fires a little early:
    the abandoned statement still holds the connector."""
    monkeypatch.setattr(literal_grounding, "_TOTAL_BUDGET_S", 10.0)
    calls: list[str] = []
    stuck = _Stuck()
    original = stuck.execute_query

    def _counted(sql: str, *args: Any, **kwargs: Any) -> QueryResult:
        calls.append(sql)
        return original(sql, *args, **kwargs)

    stuck.execute_query = _counted  # type: ignore[method-assign]

    waits: list[float] = []

    async def _times_out(awaitable: Any, timeout: float) -> Any:
        waits.append(timeout)
        if len(waits) == 1:  # opening the connector
            return await awaitable
        stuck.started.wait(5)  # the statement is running on the connector
        raise TimeoutError

    monkeypatch.setattr(literal_grounding.asyncio, "wait_for", _times_out)
    sql = "SELECT id FROM schools WHERE status = 'legal' OR funding = 'x'"
    try:
        result = _ground(sql, LookupSource(key="stuck", open=lambda: stuck))

        assert result.sql == sql
        assert len(calls) == 1, calls
    finally:
        stuck.release.set()


def test_the_cache_lock_is_not_held_while_a_lookup_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(literal_grounding, "_TOTAL_BUDGET_S", 10.0)
    stuck = _Stuck()
    source = LookupSource(key="stuck", open=lambda: stuck)
    worker = threading.Thread(
        target=_ground, args=("SELECT id FROM schools WHERE status = 'legal'", source)
    )
    worker.start()
    try:
        assert stuck.started.wait(5)
        acquired = literal_grounding._cache_lock.acquire(timeout=1)
        assert acquired, "the cache lock was held across a database call"
        literal_grounding._cache_lock.release()
    finally:
        stuck.release.set()
        worker.join(10)


def test_four_concurrent_run_query_calls_on_their_own_loops_all_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Four threads, each with its own event loop, calling run_query at once
    against one SQLite agent, each with literals that need grounding: what an
    evaluation harness with four workers does.

    Before the SQLite connector took a lock this deadlocked the process. The
    pooled SQLite connector is one `sqlite3` connection with a Python
    authorizer: one thread held the connection's mutex waiting for the GIL to
    call the authorizer, the other held the GIL waiting for the mutex.

    A deadlock on the GIL freezes every thread, this one included, so no join
    timeout could report it; faulthandler's watchdog runs without the GIL and
    ends the run with every thread's stack instead of hanging it.
    """
    from nlqueries.orchestrator import sync_runner
    from nlqueries.orchestrator.orchestrator import Orchestrator

    # Enough rows that the lookups take long enough to overlap.
    statuses = ["Legal", "Active", "Closed", "Merged", "Pending", "Pilot", "Charter", "Annex"]
    rows = [
        (i, statuses[i % len(statuses)], "Directly funded", "Youth", f"D{i}", i)
        for i in range(1, 20_001)
    ]
    db = _schools_db(tmp_path / "schools.db", rows)
    connectors_file = tmp_path / "connectors.yaml"
    connectors_file.write_text(yaml.safe_dump({"agent1": _entry(db)}), encoding="utf-8")
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    (kb_dir / "agent1.yaml").write_text(yaml.dump(KB), encoding="utf-8")
    monkeypatch.setattr("nlqueries.config.CONNECTORS_FILE", connectors_file)

    in_wrong_case = ", ".join(f"'{s.lower()}'" for s in statuses)
    generated = f"SELECT id FROM schools WHERE status IN ({in_wrong_case})"
    expected = "SELECT id FROM schools WHERE status IN ({})".format(
        ", ".join(f"'{s}'" for s in statuses)
    )

    llm = MagicMock()
    llm.supports_prompt_caching = False

    async def _astream(system: Any, user: str) -> Any:
        yield f"<sql>{generated}</sql>"

    llm.astream = _astream

    class _SqlOnly:
        """The SQL sub-agent alone: routing and the semantic cache are not what
        this is about, and would reach for Qdrant."""

        async def handle_question(self, question: str, agent_id: str, **kwargs: Any) -> Any:
            async for token in Orchestrator().handle_question(
                question, agent_id, dialect=kwargs["dialect"], execution=kwargs["execution"]
            ):
                if token.startswith("{") and '"type": "sql"' in token:
                    # Labelled by the multi-agent layer this stands in for.
                    token = json.dumps({**json.loads(token), "agent_type": "sql"})
                yield token

    results: dict[int, Any] = {}
    errors: dict[int, BaseException] = {}
    barrier = threading.Barrier(4)

    def _worker(n: int) -> None:
        loop = asyncio.new_event_loop()
        try:
            barrier.wait(10)
            results[n] = loop.run_until_complete(
                sync_runner.run_query(
                    "which schools are legal",
                    "agent1",
                    dialect="sqlite",
                    history=[],
                    explain=True,
                    execution=ExecutionPolicy.generate_only(),
                )
            )
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors[n] = exc
        finally:
            loop.close()

    with (
        patch("nlqueries.orchestrator.orchestrator.config") as cfg,
        patch("nlqueries.orchestrator.orchestrator.get_llm_client", return_value=llm),
        patch.object(sync_runner, "MultiAgentOrchestrator", _SqlOnly),
        patch("nlqueries.embeddings.qdrant_store.search_schema", return_value=[]),
        patch("nlqueries.embeddings.qdrant_store.search", return_value=[]),
        patch("nlqueries.embeddings.embedder.embed_text", return_value=[0.0] * 384),
        patch("nlqueries.orchestrator.prompt_assembly._search_verified", return_value=[]),
    ):
        cfg.KB_PATH = kb_dir
        cfg.LITERAL_GROUNDING = True
        cfg.CONNECTOR_STATEMENT_TIMEOUT_SECONDS = 30.0
        faulthandler.dump_traceback_later(60, exit=True, file=sys.__stderr__)
        try:
            threads = [threading.Thread(target=_worker, args=(n,), daemon=True) for n in range(4)]
            started = time.monotonic()
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(max(0.0, 30 - (time.monotonic() - started)))
            elapsed = time.monotonic() - started
        finally:
            faulthandler.cancel_dump_traceback_later()

    assert not any(thread.is_alive() for thread in threads), f"still running after {elapsed:.1f}s"
    assert errors == {}
    assert sorted(results) == [0, 1, 2, 3]
    for result in results.values():
        assert result.sql == expected
        assert len(result.provenance.literals_grounded) == len(statuses)


# --- Provenance ------------------------------------------------------------------


LEGAL_RECORD = [{"table": "schools", "column": "status", "before": "legal", "after": "Legal"}]


def _recorded(sql: str, src: _Source, llm: Any = None) -> tuple[SQLGenerationResult, Provenance]:
    collected = Provenance()
    with use_provenance(collected):
        result = _validate(sql, src, llm)
    return result, collected


def test_a_substitution_in_the_returned_statement_is_recorded(src: _Source) -> None:
    result, collected = _recorded("SELECT id FROM schools WHERE status = 'legal'", src)

    assert result.sql == "SELECT id FROM schools WHERE status = 'Legal'"
    assert collected.literals_grounded == LEGAL_RECORD
    assert collected.to_dict()["literals_grounded"] == LEGAL_RECORD


def test_grounding_alone_records_nothing(src: _Source) -> None:
    """Only the caller knows whether it keeps the grounded statement."""
    collected = Provenance()
    with use_provenance(collected):
        result = _ground("SELECT id FROM schools WHERE status = 'legal'", src)

    assert result.substitutions and collected.literals_grounded == []


def test_the_notes_pass_before_an_llm_repair_records_nothing(
    src: _Source, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The invalid statement is grounded for its notes and then discarded."""
    monkeypatch.setattr("nlqueries.config.SELF_CONSISTENCY", "off")
    llm = MagicMock()
    llm.acomplete = AsyncMock(return_value="<sql>SELECT id FROM schools</sql>")
    invalid = "SELECT id FROM schools JOIN ghost ON 1 = 1 WHERE status = 'legal'"

    result, collected = _recorded(invalid, src, llm)

    assert result.sql == "SELECT id FROM schools"
    assert collected.literals_grounded == []


def test_a_grounded_statement_reverted_for_validity_records_nothing(
    src: _Source, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nlqueries.orchestrator import sql_generation

    real = sql_generation._validate_sql

    def _refuses_the_grounded_one(sql: str, *args: Any, **kwargs: Any) -> str | None:
        return "re-render rejected" if "'Legal'" in sql else real(sql, *args, **kwargs)

    monkeypatch.setattr(sql_generation, "_validate_sql", _refuses_the_grounded_one)

    result, collected = _recorded("SELECT id FROM schools WHERE status = 'legal'", src)

    assert result.sql == "SELECT id FROM schools WHERE status = 'legal'"
    assert collected.literals_grounded == []


def test_a_reverted_statement_records_nothing_even_where_the_value_appears(
    src: _Source, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stored value is already in the statement, so only knowing the
    grounded statement was not kept rules the substitution out."""
    from nlqueries.orchestrator import sql_generation

    real = sql_generation._validate_sql
    grounded = "status = 'Legal' OR status = 'Legal'"

    def _refuses_the_grounded_one(sql: str, *args: Any, **kwargs: Any) -> str | None:
        return "re-render rejected" if grounded in sql else real(sql, *args, **kwargs)

    monkeypatch.setattr(sql_generation, "_validate_sql", _refuses_the_grounded_one)
    sql = "SELECT id FROM schools WHERE status = 'legal' OR status = 'Legal'"

    result, collected = _recorded(sql, src)

    assert result.sql == sql
    assert collected.literals_grounded == []


def test_a_repair_that_changes_a_grounded_literal_is_rejected(src: _Source) -> None:
    """Grounded to 'Legal'; the repair, correcting the flagged soc literal, also
    changed that one, which grounding did not flag. Rejected, so the
    substitution stands and is recorded."""
    sql = "SELECT id FROM schools WHERE status = 'legal' AND soc = 'Youth Authority School'"
    llm = MagicMock()
    llm.acomplete = AsyncMock(
        return_value=(
            "<sql>SELECT id FROM schools WHERE status = 'Active' "
            "AND soc = 'Youth Authority Facilities'</sql>"
        )
    )

    result, collected = _recorded(sql, src, llm)

    assert result.sql == (
        "SELECT id FROM schools WHERE status = 'Legal' AND soc = 'Youth Authority School'"
    )
    assert collected.literals_grounded == LEGAL_RECORD
    assert collected.literal_repair is not None
    assert (
        collected.literal_repair["reason"] == "rejected: changed a literal grounding did not flag"
    )


def _validate(sql: str, source: _Source | None, llm: Any = None) -> SQLGenerationResult:
    return asyncio.run(
        validate_and_repair(
            sql,
            KB,
            "sqlite",
            llm or MagicMock(),
            lookups=source.source if source is not None else None,
        )
    )


def test_a_valid_statement_is_grounded_before_it_is_returned(src: _Source) -> None:
    """A wrong literal is valid SQL, so this is the path it takes."""
    result = _validate("SELECT id FROM schools WHERE status = 'legal'", src)

    assert result.is_valid
    assert result.sql == "SELECT id FROM schools WHERE status = 'Legal'"


def test_validate_and_repair_does_not_ground_without_lookups() -> None:
    result = _validate("SELECT id FROM schools WHERE status = 'legal'", None)

    assert result.sql == "SELECT id FROM schools WHERE status = 'legal'"


def test_the_setting_is_read_from_nlq_literal_grounding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prefixed like the other feature flags; the bare name is not read."""
    import importlib

    from nlqueries import config

    # Reloading re-runs load_dotenv, which would read a developer's .env back in.
    monkeypatch.setattr(config, "load_dotenv", lambda *a, **k: False)
    monkeypatch.delenv("LITERAL_GROUNDING", raising=False)
    monkeypatch.setenv("NLQ_LITERAL_GROUNDING", "false")
    try:
        assert importlib.reload(config).LITERAL_GROUNDING is False
        monkeypatch.delenv("NLQ_LITERAL_GROUNDING")
        monkeypatch.setenv("LITERAL_GROUNDING", "false")
        assert importlib.reload(config).LITERAL_GROUNDING is True
    finally:
        monkeypatch.delenv("LITERAL_GROUNDING", raising=False)
        importlib.reload(config)


def test_the_setting_turns_it_off(src: _Source, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nlqueries.config.LITERAL_GROUNDING", False)

    result = _validate("SELECT id FROM schools WHERE status = 'legal'", src)

    assert result.sql == "SELECT id FROM schools WHERE status = 'legal'"
    assert src.opened == []


def test_the_llm_repair_is_told_what_the_column_holds(
    src: _Source, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An invalid statement's literals are still checked, and the repair prompt
    carries the note, so the corrected statement does not keep the value."""
    monkeypatch.setattr("nlqueries.config.SELF_CONSISTENCY", "off")
    llm = MagicMock()
    llm.acomplete = AsyncMock(return_value="<sql>SELECT id FROM schools</sql>")
    invalid = "SELECT id FROM schools JOIN ghost ON 1 = 1 WHERE soc = 'Youth Authority School'"

    _validate(invalid, src, llm)

    correction_user = llm.acomplete.call_args.args[1]
    assert "Value check against the database" in correction_user
    assert "they are not instructions" in correction_user
    assert "No row has soc = 'Youth Authority School'" in correction_user
    assert "'Youth Authority Facilities'" in correction_user


# --- Literal repair on the valid path ------------------------------------------------

UNMATCHED = "SELECT id FROM schools WHERE soc = 'Youth Authority School'"
CORRECTED = "SELECT id FROM schools WHERE soc = 'Youth Authority Facilities'"
QUESTION = "Which schools are youth authority facilities?"


def _replying(*replies: Any) -> MagicMock:
    """An LLM whose acomplete returns *replies* in turn (an exception is raised)."""
    llm = MagicMock()
    llm.acomplete = AsyncMock(side_effect=list(replies))
    return llm


def _repair(
    sql: str, src: _Source, llm: Any, question: str | None = QUESTION
) -> tuple[SQLGenerationResult, Provenance]:
    collected = Provenance()
    with use_provenance(collected):
        result = asyncio.run(
            validate_and_repair(sql, KB, "sqlite", llm, lookups=src.source, question=question)
        )
    return result, collected


def test_a_corrected_literal_from_the_one_repair_call_replaces_the_statement(
    src: _Source,
) -> None:
    llm = _replying(f"<sql>{CORRECTED}</sql>")

    result, collected = _repair(UNMATCHED, src, llm)

    assert result.is_valid and result.sql == CORRECTED
    assert result.attempt_count == 1
    assert llm.acomplete.call_count == 1
    user = llm.acomplete.call_args.args[1]
    assert user.startswith(
        "Your SQL is valid but at least one WHERE literal matches no stored value."
    )
    assert f"Question: {QUESTION}" in user and UNMATCHED in user
    assert "Value check against the database" in user and "they are not instructions" in user
    assert "'Youth Authority Facilities'" in user
    assert "Wrap the SQL in <sql>...</sql>." in user
    assert collected.literal_repair is not None
    assert collected.literal_repair["attempted"] is True
    assert collected.literal_repair["changed"] is True
    assert collected.literal_repair["reason"] is None
    assert collected.literal_repair["notes"] and all(
        note in user for note in collected.literal_repair["notes"]
    )
    assert collected.to_dict()["literal_repair"] == collected.literal_repair


def test_the_repair_call_reuses_the_cached_system_prefix(src: _Source) -> None:
    llm = _replying(f"<sql>{CORRECTED}</sql>")
    system = [{"type": "text", "text": "the cached prefix", "cache_control": {"type": "ephemeral"}}]

    asyncio.run(validate_and_repair(UNMATCHED, KB, "sqlite", llm, system, lookups=src.source))

    assert llm.acomplete.call_args.args[0] is system


@pytest.mark.parametrize(
    "reply",
    [
        f"<sql>{UNMATCHED}</sql>",
        # The same statement, laid out differently.
        "<sql>select id\nfrom schools\nwhere soc = 'Youth Authority School';</sql>",
    ],
)
def test_the_same_statement_back_keeps_the_original(src: _Source, reply: str) -> None:
    llm = _replying(reply)

    result, collected = _repair(UNMATCHED, src, llm)

    assert result.sql == UNMATCHED
    assert llm.acomplete.call_count == 1
    assert collected.literal_repair is not None
    assert collected.literal_repair["changed"] is False


@pytest.mark.parametrize(
    "reply",
    [
        # The literal fixed, and the selected column changed too.
        "SELECT status FROM schools WHERE soc = 'Youth Authority Facilities'",
        # The literal fixed, and a predicate added.
        "SELECT id FROM schools WHERE soc = 'Youth Authority Facilities' AND enrolment > 50",
        # The literal fixed, and an aggregate in place of the column.
        "SELECT COUNT(id) FROM schools WHERE soc = 'Youth Authority Facilities'",
    ],
    ids=["column-changed", "predicate-added", "aggregate"],
)
def test_an_answer_that_changes_more_than_literals_keeps_the_original(
    src: _Source, reply: str
) -> None:
    llm = _replying(f"<sql>{reply}</sql>")

    result, collected = _repair(UNMATCHED, src, llm)

    assert result.is_valid and result.sql == UNMATCHED
    assert llm.acomplete.call_count == 1
    assert collected.literal_repair is not None
    assert collected.literal_repair["changed"] is False
    assert collected.literal_repair["reason"] == "rejected: non-literal change"


def test_a_changed_number_beside_the_literal_keeps_the_original(src: _Source) -> None:
    """Only string literals may differ: a number is not one."""
    original = "SELECT id FROM schools WHERE soc = 'Youth Authority School' AND enrolment > 50"
    llm = _replying(
        "<sql>SELECT id FROM schools WHERE soc = 'Youth Authority Facilities' "
        "AND enrolment > 60</sql>"
    )

    result, collected = _repair(original, src, llm)

    assert result.sql == original
    assert collected.literal_repair is not None
    assert collected.literal_repair["reason"] == "rejected: non-literal change"


def test_a_repair_that_changes_a_grounded_literal_holding_a_flagged_value_is_rejected(
    src: _Source,
) -> None:
    """status = 'legal' is grounded to 'Legal' and funding = 'Legal' matches
    nothing, so the statement holds 'Legal' twice and only the second is
    flagged. Flagged by position, not value: the repair may not change the
    first."""
    sql = (
        "SELECT id FROM schools WHERE status = 'legal' AND funding = 'Legal' "
        "AND soc = 'Youth Authority School'"
    )
    llm = _replying(
        "<sql>SELECT id FROM schools WHERE status = 'Active' AND funding = 'Active' "
        "AND soc = 'Youth Authority Facilities'</sql>"
    )

    result, collected = _repair(sql, src, llm)

    assert result.sql == sql.replace("'legal'", "'Legal'")
    assert collected.literals_grounded == LEGAL_RECORD
    assert collected.literal_repair is not None
    assert (
        collected.literal_repair["reason"] == "rejected: changed a literal grounding did not flag"
    )


def test_a_repair_of_the_flagged_literals_beside_a_grounded_one_holding_the_same_value(
    src: _Source,
) -> None:
    """The same statement, with the repair changing only what was flagged:
    grounding and the gate count the literals' positions the same way."""
    sql = (
        "SELECT id FROM schools WHERE status = 'legal' AND funding = 'Legal' "
        "AND soc = 'Youth Authority School'"
    )
    fixed = (
        "SELECT id FROM schools WHERE status = 'Legal' AND funding = 'Directly funded' "
        "AND soc = 'Youth Authority Facilities'"
    )
    llm = _replying(f"<sql>{fixed}</sql>")

    result, collected = _repair(sql, src, llm)

    assert result.sql == fixed
    assert collected.literals_grounded == LEGAL_RECORD
    assert collected.literal_repair is not None and collected.literal_repair["changed"] is True


def test_a_repair_that_changes_an_unchecked_literal_holding_a_flagged_value_is_rejected(
    src: _Source,
) -> None:
    """The label repeats the flagged literal but compares nothing, so grounding
    never checked it and the repair may not change it."""
    sql = (
        "SELECT id, 'Youth Authority School' AS label FROM schools "
        "WHERE soc = 'Youth Authority School'"
    )
    llm = _replying(
        "<sql>SELECT id, 'Youth Authority Facilities' AS label FROM schools "
        "WHERE soc = 'Youth Authority Facilities'</sql>"
    )

    result, collected = _repair(sql, src, llm)

    assert result.sql == sql
    assert collected.literal_repair is not None
    assert (
        collected.literal_repair["reason"] == "rejected: changed a literal grounding did not flag"
    )


def test_a_repair_that_changes_a_literal_that_matched_is_rejected(src: _Source) -> None:
    """funding = 'Directly funded' exists as written; only soc was flagged."""
    sql = (
        "SELECT id FROM schools WHERE soc = 'Youth Authority School' "
        "AND funding = 'Directly funded'"
    )
    llm = MagicMock()
    llm.acomplete = AsyncMock(
        return_value=(
            "<sql>SELECT id FROM schools WHERE soc = 'Youth Authority Facilities' "
            "AND funding = 'Locally funded'</sql>"
        )
    )

    result, collected = _recorded(sql, src, llm)

    assert result.sql == sql
    assert collected.literal_repair is not None
    assert collected.literal_repair["changed"] is False
    assert (
        collected.literal_repair["reason"] == "rejected: changed a literal grounding did not flag"
    )


def test_a_repair_may_settle_a_literal_several_values_match(src: _Source) -> None:
    """'active' matches both 'Active' and 'ACTIVE': flagged, so the repair may
    choose between them while it corrects the soc literal."""
    sql = "SELECT id FROM schools WHERE status = 'active' AND soc = 'Youth Authority School'"
    fixed = "SELECT id FROM schools WHERE status = 'ACTIVE' AND soc = 'Special Education Schools'"
    llm = MagicMock()
    llm.acomplete = AsyncMock(return_value=f"<sql>{fixed}</sql>")

    result, collected = _recorded(sql, src, llm)

    assert result.sql == fixed
    assert collected.literal_repair is not None and collected.literal_repair["changed"] is True


def test_a_repair_of_the_flagged_literal_alone_is_accepted_beside_one_that_matched(
    src: _Source,
) -> None:
    sql = (
        "SELECT id FROM schools WHERE soc = 'Youth Authority School' "
        "AND funding = 'Directly funded'"
    )
    fixed = (
        "SELECT id FROM schools WHERE soc = 'Youth Authority Facilities' "
        "AND funding = 'Directly funded'"
    )
    llm = MagicMock()
    llm.acomplete = AsyncMock(return_value=f"<sql>{fixed}</sql>")

    result, collected = _recorded(sql, src, llm)

    assert result.sql == fixed
    assert collected.literal_repair is not None and collected.literal_repair["changed"] is True


def test_an_invalid_statement_back_keeps_the_original(src: _Source) -> None:
    llm = _replying("<sql>SELECT id FROM ghost WHERE soc = 'Youth Authority Facilities'</sql>")

    result, collected = _repair(UNMATCHED, src, llm)

    assert result.is_valid and result.sql == UNMATCHED
    assert llm.acomplete.call_count == 1
    assert collected.literal_repair is not None
    assert collected.literal_repair["changed"] is False


def test_a_failed_call_keeps_the_original(src: _Source) -> None:
    llm = _replying(RuntimeError("provider down"))

    result, collected = _repair(UNMATCHED, src, llm)

    assert result.is_valid and result.sql == UNMATCHED
    assert llm.acomplete.call_count == 1
    assert collected.literal_repair is not None
    assert collected.literal_repair["changed"] is False


def test_the_corrected_statement_is_grounded_again_without_a_second_call(
    src: _Source,
) -> None:
    llm = _replying(
        "<sql>SELECT id FROM schools WHERE soc = 'youth authority facilities'</sql>",
        "<sql>SELECT 1</sql>",
    )

    result, _ = _repair(UNMATCHED, src, llm)

    assert result.sql == CORRECTED
    assert llm.acomplete.call_count == 1


def test_a_corrected_statement_still_unmatched_gets_no_second_call(src: _Source) -> None:
    """Valid and different, so it is kept; grounded again, it still matches no
    stored value, and that does not start another call."""
    still_wrong = "SELECT id FROM schools WHERE soc = 'Youth Authority Schools'"
    llm = _replying(f"<sql>{still_wrong}</sql>", f"<sql>{CORRECTED}</sql>")

    result, _ = _repair(UNMATCHED, src, llm)

    assert result.sql == still_wrong
    assert llm.acomplete.call_count == 1


def test_the_mechanical_repair_path_gets_the_repair_too(src: _Source) -> None:
    """Valid only once mechanically repaired: a string LIMIT."""
    llm = _replying(f"<sql>{CORRECTED} LIMIT 5</sql>")

    result, collected = _repair(f"{UNMATCHED} LIMIT '5'", src, llm)

    assert result.sql == f"{CORRECTED} LIMIT 5"
    assert llm.acomplete.call_count == 1
    assert collected.literal_repair is not None and collected.literal_repair["changed"]


def test_with_the_setting_off_no_repair_call_is_made(
    src: _Source, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("nlqueries.config.LITERAL_GROUNDING_REPAIR", False)
    llm = _replying(f"<sql>{CORRECTED}</sql>")

    result, collected = _repair(UNMATCHED, src, llm)

    assert result.sql == UNMATCHED
    llm.acomplete.assert_not_called()
    assert collected.literal_repair is None


@pytest.mark.parametrize(
    "sql",
    [
        # No row has it, and nothing near it: a note with no values to offer.
        "SELECT id FROM schools WHERE soc = 'zzzz qqqq'",
        # Several values match: grounding leaves it, and so does the repair.
        "SELECT id FROM schools WHERE status = 'active'",
    ],
)
def test_notes_without_nearby_values_make_no_call(src: _Source, sql: str) -> None:
    llm = _replying(f"<sql>{CORRECTED}</sql>")

    result, collected = _repair(sql, src, llm)

    assert result.sql == sql
    llm.acomplete.assert_not_called()
    assert collected.literal_repair is None


def test_every_note_goes_into_the_call(src: _Source) -> None:
    """The literal with nearby values triggers it; the call also carries the note
    for a literal with none."""
    sql = "SELECT id FROM schools WHERE soc = 'Youth Authority School' OR funding = 'zzzz qqqq'"
    llm = _replying(f"<sql>{sql}</sql>")

    _, collected = _repair(sql, src, llm)

    assert collected.literal_repair is not None
    notes = collected.literal_repair["notes"]
    assert len(notes) == 2
    assert any("'zzzz qqqq'" in note for note in notes)
    user = llm.acomplete.call_args.args[1]
    assert all(note in user for note in notes)


# --- The orchestrator: lookups, whatever the policy ----------------------------------


def _drive(
    execution: ExecutionPolicy,
    *,
    generated: str = "SELECT 1",
    source: LookupSource | None = None,
    grounding: bool = True,
    validate: AsyncMock | None = None,
) -> tuple[dict[str, Any], MagicMock, MagicMock]:
    """Drive one question through the real orchestrator; return its SQL frame,
    the `lookup_source` mock and the `open_connector_for_agent` mock."""
    import tempfile

    from nlqueries.orchestrator.orchestrator import Orchestrator

    executing = MagicMock()
    executing.execute_query.return_value = QueryResult(
        columns=["n"], rows=[[1]], row_count=1, execution_time_ms=1.0, error=None
    )
    opener = MagicMock(return_value=executing)
    finder = MagicMock(return_value=source)
    llm = MagicMock()
    llm.supports_prompt_caching = False

    async def _astream(system: Any, user: str) -> Any:
        yield f"<sql>{generated}</sql>"

    llm.astream = _astream
    extra = (
        [patch("nlqueries.orchestrator.orchestrator.validate_and_repair", new=validate)]
        if validate is not None
        else []
    )
    frames: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "agent1.yaml").write_text(yaml.dump(KB), encoding="utf-8")
        with (
            patch("nlqueries.orchestrator.orchestrator.config") as cfg,
            patch("nlqueries.orchestrator.orchestrator.get_llm_client", return_value=llm),
            patch("nlqueries.connectors.loader.open_connector_for_agent", new=opener),
            patch("nlqueries.connectors.loader.lookup_source", new=finder),
            patch("nlqueries.embeddings.qdrant_store.search_schema", return_value=[]),
            patch("nlqueries.embeddings.qdrant_store.search", return_value=[]),
            patch("nlqueries.embeddings.embedder.embed_text", return_value=[0.0] * 384),
            patch("nlqueries.orchestrator.prompt_assembly._search_verified", return_value=[]),
        ):
            for p in extra:
                p.start()
            try:
                cfg.KB_PATH = Path(tmpdir)
                cfg.LITERAL_GROUNDING = grounding
                cfg.CONNECTOR_STATEMENT_TIMEOUT_SECONDS = 30.0

                async def collect() -> None:
                    async for token in Orchestrator().handle_question(
                        "how many legal schools", "agent1", dialect="sqlite", execution=execution
                    ):
                        if token.startswith("{") and '"type": "sql"' in token:
                            frames.append(json.loads(token))

                asyncio.run(collect())
            finally:
                for p in extra:
                    p.stop()
    return frames[-1], finder, opener


def _validated() -> AsyncMock:
    return AsyncMock(
        return_value=SQLGenerationResult(
            sql="SELECT 1", is_valid=True, validation_error=None, dialect="sqlite", attempt_count=1
        )
    )


@pytest.mark.parametrize(
    "execution", [ExecutionPolicy.execute_read_only(), ExecutionPolicy.generate_only()]
)
def test_validation_gets_lookups_whatever_the_policy(execution: ExecutionPolicy) -> None:
    """Generate-only forbids running the statement, not reading the column's
    values: grounding still runs (owner, 2026-10-09). The lookups are not the
    request's connector, which opens only to execute, after validation."""
    source = LookupSource(key="k", open=MagicMock())
    validate = _validated()

    frame, finder, opener = _drive(execution, source=source, validate=validate)

    finder.assert_called_once_with("agent1")
    assert validate.call_args.kwargs["lookups"] is source
    assert "connector" not in validate.call_args.kwargs
    assert opener.call_count == (1 if execution.may_execute else 0)
    if not execution.may_execute:
        assert frame["sql_table"] is None and frame["execution_mode"] == "generate_only"


def test_validation_is_given_the_question() -> None:
    validate = _validated()

    _drive(ExecutionPolicy.generate_only(), validate=validate)

    assert validate.call_args.kwargs["question"] == "how many legal schools"


def test_an_agent_with_no_connector_gets_no_lookups() -> None:
    validate = _validated()

    _, finder, opener = _drive(ExecutionPolicy.generate_only(), validate=validate, source=None)

    finder.assert_called_once()
    assert validate.call_args.kwargs["lookups"] is None
    opener.assert_not_called()


def test_with_the_setting_off_no_source_is_looked_for() -> None:
    validate = _validated()

    _, finder, _ = _drive(ExecutionPolicy.generate_only(), validate=validate, grounding=False)

    finder.assert_not_called()
    assert validate.call_args.kwargs["lookups"] is None


def test_a_generate_only_request_is_grounded_and_its_statement_never_runs(src: _Source) -> None:
    """End to end, through the real validation and a real database: the literal
    is corrected in the SQL returned, and the only statements the database sees
    are the read-only lookups."""
    generated = "SELECT id FROM schools WHERE status = 'legal'"

    frame, _, opener = _drive(
        ExecutionPolicy.generate_only(), generated=generated, source=src.source
    )

    assert frame["sql"] == "SELECT id FROM schools WHERE status = 'Legal'"
    assert frame["sql_table"] is None
    assert src.statements, "the lookups ran"
    assert generated not in src.statements
    assert frame["sql"] not in src.statements
    opener.assert_not_called()


# --- The loader's source ----------------------------------------------------------


def _register(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entries: dict[str, Any]) -> None:
    connectors_file = tmp_path / "connectors.yaml"
    connectors_file.write_text(yaml.safe_dump(entries), encoding="utf-8")
    monkeypatch.setattr("nlqueries.config.CONNECTORS_FILE", connectors_file)


def test_an_unregistered_agent_has_no_source_and_nothing_is_logged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _register(tmp_path, monkeypatch, {"other": _entry(tmp_path / "x.db")})

    with caplog.at_level(logging.WARNING, logger="nlqueries"):
        assert lookup_source("agent1") is None

    assert caplog.records == []


def test_a_source_reads_through_the_hookable_opener_with_read_permission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Through `open_connector_for_agent` as it stands when called: a wrapper
    installed on the module after import, as the enterprise layer installs its
    row filter, is the one that runs."""
    from nlqueries.connectors import loader

    db = _schools_db(tmp_path / "schools.db")
    _register(tmp_path, monkeypatch, {"agent1": _entry(db)})
    calls: list[tuple[str, ExecutionPolicy]] = []
    original = loader.open_connector_for_agent

    def _hooked(agent_id: str, *args: Any, **kwargs: Any) -> Any:
        calls.append((agent_id, args[0] if args else kwargs["execution"]))
        return original(agent_id, *args, **kwargs)

    monkeypatch.setattr(loader, "open_connector_for_agent", _hooked)
    source = lookup_source("agent1")
    assert source is not None

    first, second = source.open(), source.open()

    assert calls == [("agent1", ExecutionPolicy.execute_read_only())] * 2
    assert isinstance(first, PermittedConnector) and isinstance(second, PermittedConnector)
    assert len(loader._cache) == 1, "one pooled connector serves both"
    assert first.execute_query("SELECT COUNT(*) FROM schools").rows[0][0] == 4


def test_the_source_key_follows_the_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stable while the entry is, so the cache is shared across requests; new when
    it changes, so a repointed agent does not inherit another database's answers."""
    _register(tmp_path, monkeypatch, {"agent1": _entry(tmp_path / "a.db")})
    first, again = lookup_source("agent1"), lookup_source("agent1")
    _register(tmp_path, monkeypatch, {"agent1": _entry(tmp_path / "b.db")})
    moved = lookup_source("agent1")

    assert first is not None and again is not None and moved is not None
    assert first.key == again.key
    assert moved.key != first.key


# --- Provenance reaches the caller's result ----------------------------------------


def test_literals_grounded_reaches_the_run_query_result(src: _Source) -> None:
    """What a caller of `run_query(explain=True)` captures: the record has to
    arrive on the result's provenance, including through `dataclasses.asdict`,
    which is how a caller serialises it."""
    import dataclasses

    from nlqueries.orchestrator.followup_resolver import ResolvedQuestion
    from nlqueries.orchestrator.sync_runner import run_query

    async def _gen(*_a: Any, **_k: Any) -> Any:
        # The real orchestrator's call, against a real database.
        validated = await validate_and_repair(
            "SELECT id FROM schools WHERE status = 'legal'",
            KB,
            "sqlite",
            MagicMock(),
            lookups=src.source,
        )
        yield "answer "
        yield json.dumps(
            {"type": "sql", "agent_type": "sql", "sql": validated.sql, "is_valid": True}
        )

    orchestrator = MagicMock()
    orchestrator.handle_question = _gen
    no_followup = ResolvedQuestion(
        original="how many legal schools",
        resolved="how many legal schools",
        is_followup=False,
        reasoning="",
    )
    with (
        patch(
            "nlqueries.orchestrator.sync_runner.MultiAgentOrchestrator", return_value=orchestrator
        ),
        patch("nlqueries.orchestrator.sync_runner.resolve_followup", return_value=no_followup),
    ):
        result = asyncio.run(run_query("how many legal schools", "agent1", explain=True))

    expected = [{"table": "schools", "column": "status", "before": "legal", "after": "Legal"}]
    assert result.provenance is not None
    assert result.provenance.literals_grounded == expected
    assert dataclasses.asdict(result.provenance)["literals_grounded"] == expected
