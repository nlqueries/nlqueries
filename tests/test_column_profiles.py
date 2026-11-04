"""What a whole column holds, stored by ``export-kb`` and shown in the prompt.

Samples show the first rows. A join key padded with spaces, a year column with
odd values, a date range that stops early, a column that is mostly empty: each
is visible only across the whole column, in one aggregate per table.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import sqlglot
from nlqueries.connectors.base import ColumnSpec, SchemaSpec, TableSpec, column_profile_sql
from nlqueries.connectors.sqlite import SQLiteConnector
from nlqueries.execution import ExecutionPolicy
from nlqueries.knowledge import column_profiles
from nlqueries.knowledge.column_profiles import (
    collect_column_profiles,
    profile_kind,
    profile_notes,
)
from nlqueries.knowledge.kb_generator import generate_knowledge_base


def _col(name: str, col_type: str = "TEXT", *, pk: bool = False, fk: bool = False) -> ColumnSpec:
    return ColumnSpec(name, col_type, True, pk, fk, "films.id" if fk else None, None)


def _schema(row_count: int | None = None) -> SchemaSpec:
    return SchemaSpec(
        database="db",
        tables=[
            TableSpec(
                "credits",
                "main",
                row_count,
                [
                    _col("film_id", "TEXT", fk=True),  # padded with a leading space
                    _col("release_year"),  # '2009' and 'c. 2009'
                    _col("released_on", "DATE"),
                    _col("tagline"),  # mostly empty
                    _col("contact_email"),  # personal data
                    _col("runtime", "INTEGER"),
                ],
                None,
            )
        ],
        extracted_at="2026-11-04T00:00:00",
    )


@pytest.fixture
def connector(tmp_path: Path) -> SQLiteConnector:
    db = tmp_path / "films.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE credits (film_id TEXT, release_year TEXT, released_on DATE, "
            "tagline TEXT, contact_email TEXT, runtime INTEGER)"
        )
        conn.executemany(
            "INSERT INTO credits VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    f" F{i}",
                    "c. 2009" if i % 25 == 0 else str(1990 + i % 30),
                    f"{1990 + i % 30}-0{1 + i % 9}-15",
                    "A tagline" if i % 10 == 0 else None,
                    f"p{i}@example.org",
                    90 + i % 60,
                )
                for i in range(1, 201)
            ],
        )
    sqlite = SQLiteConnector()
    sqlite.connect({"database": str(db)})
    sqlite.bind_execution_policy(ExecutionPolicy.execute_read_only())
    return sqlite


# ---------------------------------------------------------------------------
# The statement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dialect", "length"),
    [("postgres", "LENGTH("), ("sqlite", "LENGTH("), ("tsql", "LEN("), ("bigquery", "LENGTH(")],
)
def test_the_statement_is_one_aggregate_in_each_dialect(dialect: str, length: str) -> None:
    sql = column_profile_sql(
        "credits",
        "main",
        [("film_id", "text"), ("released_on", "date"), ("runtime", "plain")],
        dialect,
    )
    tree = sqlglot.parse_one(sql, read=dialect if dialect != "tsql" else "tsql")
    assert isinstance(tree, sqlglot.exp.Select)
    assert len(tree.expressions) == 1 + 4 + 3 + 1
    assert length in sql.upper()
    assert "TRIM(" in sql.upper()


def test_the_statement_passes_the_sql_policy() -> None:
    from nlqueries.sql_policy import evaluate

    for dialect in ("postgres", "sqlite", "snowflake", "bigquery", "mysql", "duckdb", "tsql"):
        sql = column_profile_sql("credits", "main", [("a b", "text"), ("d", "date")], dialect)
        decision = evaluate(sql, dialect)
        assert decision.allowed, (dialect, decision.summary())


# ---------------------------------------------------------------------------
# What is collected
# ---------------------------------------------------------------------------


def test_a_padded_key_is_counted(connector: Any) -> None:
    profiles = collect_column_profiles(connector, _schema(), "sqlite")
    assert profiles["credits"]["film_id"]["padded"] == 200


def test_a_text_column_records_its_shortest_and_longest_value(connector: Any) -> None:
    profiles = collect_column_profiles(connector, _schema(), "sqlite")
    assert profiles["credits"]["release_year"]["length"] == [4, 7]


def test_a_date_column_records_its_range(connector: Any) -> None:
    profiles = collect_column_profiles(connector, _schema(), "sqlite")
    assert profiles["credits"]["released_on"]["range"] == ["1990-01-15", "2019-09-15"]


def test_a_mostly_empty_column_records_its_share_of_nulls(connector: Any) -> None:
    profiles = collect_column_profiles(connector, _schema(), "sqlite")
    assert profiles["credits"]["tagline"]["nulls"] == 0.9


def test_a_column_with_nothing_to_say_has_no_profile(connector: Any) -> None:
    profiles = collect_column_profiles(connector, _schema(), "sqlite")
    assert "runtime" not in profiles["credits"]


def test_personal_data_is_not_profiled(connector: Any) -> None:
    profiles = collect_column_profiles(connector, _schema(), "sqlite")
    assert "contact_email" not in profiles["credits"]


def test_a_table_over_the_row_cap_is_not_profiled(connector: Any) -> None:
    assert collect_column_profiles(connector, _schema(row_count=500), "sqlite", max_rows=100) == {}


def test_a_wide_table_is_profiled_in_batches(connector: Any) -> None:
    """Five profiled columns, two per statement: three statements, and the
    same profiles as one."""
    counting = MagicMock(wraps=connector)
    with patch.object(column_profiles, "COLUMNS_PER_QUERY", 2):
        batched = collect_column_profiles(counting, _schema(), "sqlite")

    assert counting.execute_query.call_count == 3
    first = counting.execute_query.call_args_list[0].args[0]
    assert '"film_id"' in first and '"release_year"' in first
    assert '"released_on"' not in first, "each statement profiles its batch only"
    assert batched == collect_column_profiles(connector, _schema(), "sqlite")


def test_a_failed_statement_leaves_the_table_without_profiles() -> None:
    failing = MagicMock()
    failing.execute_query.side_effect = RuntimeError("timeout")
    assert collect_column_profiles(failing, _schema(), "sqlite") == {}


@pytest.mark.parametrize(
    ("col_type", "kind"),
    [
        ("DATE", "date"),
        ("TIMESTAMP", "date"),
        ("datetime", "date"),
        ("VARCHAR(20)", "text"),
        ("", "text"),
        ("INTEGER", "plain"),
        ("NUMERIC(10,2)", "plain"),
        # Several values, or a structure: TRIM, LENGTH and MIN reject them.
        ("VARCHAR[]", "plain"),
        ("ARRAY<STRING>", "plain"),
        ("STRUCT<a STRING>", "plain"),
        ("DATE[]", "plain"),
    ],
)
def test_the_kind_of_profile_follows_the_type(col_type: str, kind: str) -> None:
    assert profile_kind(col_type) == kind


@pytest.mark.parametrize(
    ("col_type", "kind"), [("text", "plain"), ("NTEXT", "plain"), ("nvarchar(50)", "text")]
)
def test_sql_server_legacy_text_is_only_counted(col_type: str, kind: str) -> None:
    assert profile_kind(col_type, "tsql") == kind
    assert profile_kind("text", "postgres") == "text"


def test_a_table_whose_aggregates_fail_still_has_its_nulls_counted(connector: Any) -> None:
    """A column the aggregates reject fails the statement for the whole table,
    as a BigQuery REPEATED field reported as STRING does; the retry counts
    values only, so the table keeps its share of NULLs."""
    seen: list[str] = []

    class _Refuses:
        def execute_query(self, sql: str, timeout: float) -> Any:
            seen.append(sql)
            if "TRIM(" in sql.upper():
                raise RuntimeError("TRIM over an array")
            return connector.execute_query(sql, timeout)

    profiles = collect_column_profiles(_Refuses(), _schema(), "sqlite")

    assert len(seen) == 2
    assert profiles["credits"]["tagline"] == {"nulls": 0.9}
    assert "padded" not in profiles["credits"].get("film_id", {})


# ---------------------------------------------------------------------------
# Stored, and shown
# ---------------------------------------------------------------------------


def test_the_knowledge_base_stores_each_profile(connector: Any) -> None:
    profiles = collect_column_profiles(connector, _schema(), "sqlite")
    kb = generate_knowledge_base(_schema(), [], "films", column_profiles=profiles)
    columns = {c["name"]: c for c in kb["schema"]["tables"][0]["columns"]}

    assert columns["film_id"]["profile"]["padded"] == 200
    assert "profile" not in columns["contact_email"]
    assert "profile" not in columns["runtime"]


def test_a_padded_column_is_shown_with_the_trim_to_use() -> None:
    notes = profile_notes({"name": "film_id", "profile": {"padded": 82}})
    assert notes == ["82 values have spaces around them: compare TRIM(film_id)"]


def test_a_mostly_empty_column_is_shown_as_such() -> None:
    assert profile_notes({"name": "c", "profile": {"nulls": 0.97}}) == ["97% NULL"]
    assert profile_notes({"name": "c", "profile": {"nulls": 0.05}}) == []


def test_a_length_range_is_shown_only_where_it_contradicts_the_samples() -> None:
    odd = {"name": "y", "samples": ["2009", "2015"], "profile": {"length": [4, 6]}}
    names = {"name": "n", "samples": ["Ann", "Bartholomew"], "profile": {"length": [2, 40]}}
    regular = {"name": "y", "samples": ["2009", "2015"], "profile": {"length": [4, 4]}}

    assert profile_notes(odd) == ["lengths 4-6, unlike the samples"]
    assert profile_notes(names) == []
    assert profile_notes(regular) == []


def test_a_date_range_is_shown() -> None:
    note = profile_notes({"name": "d", "profile": {"range": ["2001-01-01", "2015-12-31"]}})
    assert note == ["range 2001-01-01 to 2015-12-31"]


def test_the_prompt_schema_shows_the_notes() -> None:
    from nlqueries.orchestrator.prompt_assembly import _render_m_schema

    kb = {
        "schema": {
            "tables": [
                {
                    "name": "credits",
                    "columns": [{"name": "film_id", "type": "TEXT", "profile": {"padded": 3}}],
                }
            ]
        }
    }
    assert "3 values have spaces around them: compare TRIM(film_id)" in _render_m_schema(kb)


def test_the_mcp_schema_tool_shows_the_notes(tmp_path: Path) -> None:
    import yaml
    from nlqueries.mcp_server.server import get_agent_schema

    kb = {
        "schema": {
            "tables": [
                {
                    "name": "credits",
                    "columns": [{"name": "tagline", "type": "TEXT", "profile": {"nulls": 0.9}}],
                }
            ]
        }
    }
    (tmp_path / "films.yaml").write_text(yaml.safe_dump(kb), encoding="utf-8")
    with patch("nlqueries.mcp_server.server.config.KB_PATH", tmp_path):
        assert "90% NULL" in get_agent_schema("films")


# ---------------------------------------------------------------------------
# export-kb
# ---------------------------------------------------------------------------


def _profiled(queries: list[str]) -> bool:
    return any("TRIM(" in q.upper() for q in queries)


def test_export_kb_profiles_by_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from tests.test_kb_column_values import _export

    result, queries = _export(monkeypatch, tmp_path, 100)

    assert result.exit_code == 0, result.output
    assert _profiled(queries)
    assert "Profiled" in result.output


def test_export_kb_no_profile_runs_no_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tests.test_kb_column_values import _export

    result, queries = _export(monkeypatch, tmp_path, 100, "--no-profile")

    assert result.exit_code == 0, result.output
    assert not _profiled(queries)
    assert "Profiled" not in result.output


def test_export_kb_does_not_profile_a_table_over_values_max_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tests.test_kb_column_values import _export

    result, queries = _export(monkeypatch, tmp_path, 101, "--values-max-rows", "100")

    assert result.exit_code == 0, result.output
    assert not _profiled(queries)


def test_a_statement_that_timed_out_is_not_retried(
    connector: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A count reads the same rows, so it would time out again, and on a
    warehouse be charged again."""
    import time

    monkeypatch.setattr(column_profiles, "_VALUES_TIMEOUT_S", 0.05)
    seen: list[str] = []

    class _Slow:
        def execute_query(self, sql: str, timeout: float) -> Any:
            seen.append(sql)
            time.sleep(0.06)
            raise TimeoutError("statement timeout")

    assert collect_column_profiles(_Slow(), _schema(), "sqlite") == {}
    assert len(seen) == 1
