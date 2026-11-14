"""Column values in the knowledge base, so the prompt shows how they are spelled.

A text column with few distinct values is stored whole and marked
``values_complete``, and the prompt renders it as "values: [...]": the model
then has 'Legal' and 'Directly funded' to copy rather than a spelling to guess.
Any other column gets a few samples, rendered "samples: [...]". Keys and
personal-data columns get nothing.
"""

from __future__ import annotations

import dataclasses
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from nlqueries.connectors.base import ColumnSpec, SchemaSpec, TableSpec, column_values_sql
from nlqueries.connectors.sqlite import SQLiteConnector
from nlqueries.execution import ExecutionPolicy
from nlqueries.knowledge.kb_generator import (
    VALUES_MAX_ROWS,
    collect_column_values,
    generate_knowledge_base,
)

STATUSES = ["Active", "Closed", "Legal", "Merged", "Pending"]


def _col(name: str, col_type: str = "TEXT", *, pk: bool = False, fk: bool = False) -> ColumnSpec:
    return ColumnSpec(name, col_type, True, pk, fk, "other.id" if fk else None, None)


SCHEMA = SchemaSpec(
    database="db",
    tables=[
        TableSpec(
            "schools",
            "main",
            None,
            [
                _col("id", "INTEGER", pk=True),
                _col("district_id", "TEXT", fk=True),
                _col("status"),  # 5 distinct values
                _col("name"),  # 500 distinct values
                _col("contact_email"),  # personal data
                _col("notes"),  # long free text
                _col("category"),  # few values, but long ones
                _col("enrolment", "INTEGER"),
            ],
            None,
        )
    ],
    extracted_at="2026-10-09T00:00:00",
)


@pytest.fixture
def connector(tmp_path: Path) -> SQLiteConnector:
    db = tmp_path / "schools.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE schools (id INTEGER PRIMARY KEY, district_id TEXT, status TEXT, "
            "name TEXT, contact_email TEXT, notes TEXT, category TEXT, enrolment INTEGER)"
        )
        conn.executemany(
            "INSERT INTO schools VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    i,
                    f"D{i % 7}",
                    STATUSES[i % 5],
                    f"School number {i}",
                    f"head{i}@example.org",
                    # Distinct per row, so it is sampled, not listed: a column
                    # whose one value fits in 200 characters is listed, as it
                    # should be.
                    f"A long free-text note about school {i} and its history " * 3,
                    ["c" * 60, "d" * 60, "e" * 60, "f" * 60][i % 4],
                    100 + i,
                )
                for i in range(500)
            ],
        )
    sqlite = SQLiteConnector()
    sqlite.connect({"database": str(db)})
    sqlite.bind_execution_policy(ExecutionPolicy.execute_read_only())
    return sqlite


def test_a_column_with_few_values_gets_every_one_and_the_flag(connector: Any) -> None:
    samples, complete = collect_column_values(connector, SCHEMA, 3, "sqlite")

    assert samples["schools"]["status"] == STATUSES
    assert "status" in complete["schools"]


def test_a_column_with_many_values_keeps_three_samples(connector: Any) -> None:
    samples, complete = collect_column_values(connector, SCHEMA, 3, "sqlite")

    assert len(samples["schools"]["name"]) == 3
    assert "name" not in complete["schools"]


def test_values_too_long_to_list_are_sampled_instead(connector: Any) -> None:
    """Four values of 60 characters: few enough, but 240 characters together."""
    samples, complete = collect_column_values(connector, SCHEMA, 3, "sqlite")

    assert "category" not in complete["schools"]
    assert len(samples["schools"]["category"]) == 3


def test_a_number_column_is_sampled(connector: Any) -> None:
    samples, _ = collect_column_values(connector, SCHEMA, 3, "sqlite")

    assert samples["schools"]["enrolment"] == ["100", "101", "102"]


@pytest.mark.parametrize(
    "column",
    [
        "id",  # primary key
        "district_id",  # foreign key
        "contact_email",  # personal data: never leaves the database
        "notes",  # averages over 80 characters
    ],
)
def test_keys_personal_data_and_free_text_get_nothing(connector: Any, column: str) -> None:
    samples, complete = collect_column_values(connector, SCHEMA, 3, "sqlite")

    assert column not in samples["schools"]
    assert column not in complete["schools"]


def test_a_failing_query_leaves_the_column_without_values() -> None:
    class _Refusing:
        def execute_query(self, *a: Any, **k: Any) -> Any:
            raise RuntimeError("permission denied")

    assert collect_column_values(_Refusing(), SCHEMA, 3, "sqlite") == ({}, {})


def test_the_kb_stores_the_full_list_with_its_flag_and_caps_samples(connector: Any) -> None:
    samples, complete = collect_column_values(connector, SCHEMA, 3, "sqlite")

    kb = generate_knowledge_base(
        SCHEMA, [], "agent", column_samples=samples, column_values_complete=complete
    )
    columns = {c["name"]: c for c in kb["schema"]["tables"][0]["columns"]}

    assert columns["status"]["samples"] == STATUSES
    assert columns["status"]["values_complete"] is True
    assert len(columns["name"]["samples"]) == 3
    assert "values_complete" not in columns["name"]


def test_the_prompt_says_values_for_a_full_list_and_samples_otherwise() -> None:
    from nlqueries.orchestrator.prompt_assembly import _render_m_schema

    kb = {
        "schema": {
            "tables": [
                {
                    "name": "schools",
                    "columns": [
                        {
                            "name": "status",
                            "type": "TEXT",
                            "samples": STATUSES,
                            "values_complete": True,
                        },
                        {"name": "name", "type": "TEXT", "samples": ["A", "B", "C"]},
                    ],
                }
            ]
        }
    }

    rendered = _render_m_schema(kb)

    assert "values: ['Active', 'Closed', 'Legal', 'Merged', 'Pending']" in rendered
    assert "samples: ['A', 'B', 'C']" in rendered
    assert "samples: ['Active'" not in rendered


@pytest.mark.parametrize(
    ("dialect", "expected"),
    [
        (
            "sqlite",
            'SELECT DISTINCT "Academic Year" FROM "main"."frpm" '
            'WHERE NOT "Academic Year" IS NULL LIMIT 21',
        ),
        (
            "tsql",
            "SELECT DISTINCT TOP 21 [Academic Year] FROM [main].[frpm] "
            "WHERE NOT [Academic Year] IS NULL",
        ),
    ],
)
def test_the_values_query_quotes_the_column_and_bounds_for_the_dialect(
    dialect: str, expected: str
) -> None:
    assert column_values_sql("frpm", "main", "Academic Year", 21, dialect) == expected


# --- Large tables -------------------------------------------------------------------


def _sized(row_count: int | None) -> SchemaSpec:
    """SCHEMA with the table's size as the connector reported it."""
    table = dataclasses.replace(SCHEMA.tables[0], row_count=row_count)
    return dataclasses.replace(SCHEMA, tables=[table])


class _Counting:
    """A connector that records each query and returns nothing."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    def execute_query(self, sql: str, *args: Any, **kwargs: Any) -> Any:
        self.queries.append(sql)
        raise RuntimeError("no rows here")


def test_a_table_larger_than_the_cap_gets_no_values_and_no_queries() -> None:
    """Its values cost a sample query and a SELECT DISTINCT per text column, and
    the DISTINCT reads every row of a column with few values."""
    counting = _Counting()

    assert collect_column_values(counting, _sized(501), 3, "sqlite", 500) == ({}, {})
    assert counting.queries == []


@pytest.mark.parametrize("row_count", [500, None])
def test_a_table_at_the_cap_or_of_unknown_size_is_collected(
    connector: Any, row_count: int | None
) -> None:
    samples, complete = collect_column_values(connector, _sized(row_count), 3, "sqlite", 500)

    assert samples["schools"]["status"] == STATUSES and "status" in complete["schools"]


def test_a_cap_of_zero_collects_any_table(connector: Any) -> None:
    samples, _ = collect_column_values(connector, _sized(10**12), 3, "sqlite", 0)

    assert samples["schools"]["status"] == STATUSES


def test_the_default_cap_is_ten_million_rows_and_the_cli_s_default(connector: Any) -> None:
    from nlqueries.cli import main as cli_main

    option = next(p for p in cli_main.export_kb.params if p.name == "values_max_rows")
    counting = _Counting()

    assert VALUES_MAX_ROWS == 10_000_000 and option.default == VALUES_MAX_ROWS
    assert collect_column_values(counting, _sized(VALUES_MAX_ROWS + 1), 3, "sqlite") == ({}, {})
    assert counting.queries == []
    assert collect_column_values(connector, _sized(VALUES_MAX_ROWS), 3, "sqlite")[1]


def _export(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, row_count: int | None, *args: str
) -> tuple[Any, list[str]]:
    """Run ``export-kb`` against one table of *row_count* rows; the output and
    the queries it ran."""
    from click.testing import CliRunner
    from nlqueries.cli import main as cli_main
    from nlqueries.connectors.base import QueryResult

    queries: list[str] = []

    class _Connector(SQLiteConnector):
        def connect(self, credentials: Any) -> None:
            pass

        def extract_schema(self) -> SchemaSpec:
            column = ColumnSpec("region", "text", True, False, False, None, None)
            table = TableSpec("orders", "sales", row_count, [column], None)
            return SchemaSpec(database="db", tables=[table], extracted_at="2026-10-10T00:00:00")

        def _execute_query(
            self, sql: str, timeout_seconds: float | None = None, max_rows: int | None = None
        ) -> QueryResult:
            queries.append(sql)
            return QueryResult(["region"], [["north"]], 1, 0.0, None)

    def _no_capsules(connector_id: str) -> list[Any]:
        raise FileNotFoundError(connector_id)

    monkeypatch.setattr(cli_main, "_resolve_alias", lambda value: value)
    monkeypatch.setattr(cli_main, "_require_connector", lambda connector_id: {"db_type": "sqlite"})
    monkeypatch.setattr(cli_main, "connector_class_for", lambda db_type, cfg: _Connector)
    monkeypatch.setattr(cli_main, "credentials_for", lambda connector_id, cfg: {})
    monkeypatch.setattr("nlqueries.processing.pipeline.load_capsules", _no_capsules)
    monkeypatch.setattr("nlqueries.feedback.store.load_feedback", lambda connector_id: [])

    result = CliRunner().invoke(
        cli_main.cli, ["export-kb", "c1", "--output", str(tmp_path / "kb.yaml"), *args]
    )
    return result, queries


def test_export_kb_skips_a_table_over_values_max_rows_and_says_so(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result, queries = _export(monkeypatch, tmp_path, 101, "--values-max-rows", "100")

    assert result.exit_code == 0, result.output
    assert not any("DISTINCT" in q or "LIMIT" in q for q in queries)
    assert "No values for 1 table(s) of more than 100 rows" in result.output


def test_export_kb_collects_a_table_within_values_max_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result, queries = _export(monkeypatch, tmp_path, 100, "--values-max-rows", "100")

    assert result.exit_code == 0, result.output
    assert any("DISTINCT" in q for q in queries)
    assert "No values for" not in result.output


@pytest.mark.parametrize("args", [(), ("--no-include-samples",)])
def test_export_kb_says_when_a_table_reports_no_row_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, args: tuple[str, ...]
) -> None:
    """Neither row cap can skip such a table, and the generic connector reports
    no sizes at all; said even without values, since grounding's cap reads the
    same sizes."""
    result, _ = _export(monkeypatch, tmp_path, None, *args)
    output = " ".join(result.output.split())

    assert result.exit_code == 0, result.output
    assert "1 of 1 table(s) report no row count, so no row cap can skip them" in output


def test_export_kb_says_nothing_of_sizes_when_every_table_reports_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result, _ = _export(monkeypatch, tmp_path, 100)

    assert result.exit_code == 0, result.output
    assert "report no row count" not in " ".join(result.output.split())


# --- Topping up a short sample ------------------------------------------------------


SPARSE = SchemaSpec(
    database="db",
    tables=[
        TableSpec(
            "visits",
            "main",
            None,
            [
                _col("id", "INTEGER", pk=True),
                _col("remark"),  # NULL in the first 250 rows, then one value per row
                _col("score", "INTEGER"),  # NULL in the first 250 rows, then 7 values
                _col("ward", "INTEGER"),  # 1 in the first 100 rows, then 4 values
                _col("seats", "INTEGER"),  # one value per row
                _col("visitor_email"),  # personal data
                _col("essay"),  # NULL in the first 250 rows, then long free text
            ],
            None,
        )
    ],
    extracted_at="2026-11-14T00:00:00",
)


class _Recording:
    """*connector*, recording each statement it is sent."""

    def __init__(self, connector: Any) -> None:
        self._connector = connector
        self.queries: list[str] = []

    def execute_query(self, sql: str, *args: Any, **kwargs: Any) -> Any:
        self.queries.append(sql)
        return self._connector.execute_query(sql, *args, **kwargs)


@pytest.fixture
def sparse(tmp_path: Path) -> _Recording:
    db = tmp_path / "visits.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE visits (id INTEGER PRIMARY KEY, remark TEXT, score INTEGER, "
            "ward INTEGER, seats INTEGER, visitor_email TEXT, essay TEXT)"
        )
        conn.executemany(
            "INSERT INTO visits VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    i,
                    None if i <= 250 else f"remark {i}",
                    None if i <= 250 else i % 7,
                    1 if i <= 100 else i % 4,
                    1000 + i,
                    f"visitor{i}@example.org",
                    None if i <= 250 else f"A long account of visit {i}, written up " * 4,
                )
                for i in range(1, 301)
            ],
        )
    sqlite = SQLiteConnector()
    sqlite.connect({"database": str(db)})
    sqlite.bind_execution_policy(ExecutionPolicy.execute_read_only())
    return _Recording(sqlite)


def _asked_about(queries: list[str], column: str) -> list[str]:
    return [q for q in queries if f'"{column}"' in q]


def test_a_sparse_text_column_with_many_values_is_topped_up_at_no_cost(
    sparse: _Recording,
) -> None:
    """NULL in the first rows, so they gave it nothing; its SELECT DISTINCT had
    returned twenty-one values before finding too many to list."""
    samples, complete = collect_column_values(sparse, SPARSE, 3, "sqlite")

    assert len(samples["visits"]["remark"]) == 3
    assert all(v.startswith("remark ") for v in samples["visits"]["remark"])
    assert "remark" not in complete.get("visits", set())
    assert len(_asked_about(sparse.queries, "remark")) == 1


def test_a_sparse_number_column_is_topped_up_with_one_bounded_query(sparse: _Recording) -> None:
    samples, complete = collect_column_values(sparse, SPARSE, 3, "sqlite")

    assert len(samples["visits"]["score"]) == 3
    assert set(samples["visits"]["score"]) <= {str(n) for n in range(7)}
    assert "score" not in complete.get("visits", set())
    (query,) = _asked_about(sparse.queries, "score")
    assert query == column_values_sql("visits", "main", "score", 3, "sqlite")


def test_a_number_column_repeating_one_value_is_topped_up(sparse: _Recording) -> None:
    """The first rows hold 1 three times: one sample, where the column has four."""
    samples, _ = collect_column_values(sparse, SPARSE, 3, "sqlite")

    assert samples["visits"]["ward"][0] == "1"
    assert len(samples["visits"]["ward"]) == 3
    assert len(set(samples["visits"]["ward"])) == 3


def test_a_column_the_first_rows_fill_costs_no_query(sparse: _Recording) -> None:
    samples, _ = collect_column_values(sparse, SPARSE, 3, "sqlite")

    assert samples["visits"]["seats"] == ["1001", "1002", "1003"]
    assert _asked_about(sparse.queries, "seats") == []


def test_keys_and_personal_data_are_never_topped_up(sparse: _Recording) -> None:
    samples, _ = collect_column_values(sparse, SPARSE, 3, "sqlite")

    assert "visitor_email" not in samples["visits"] and "id" not in samples["visits"]
    assert _asked_about(sparse.queries, "visitor_email") == []
    assert _asked_about(sparse.queries, "id") == []


def test_free_text_found_by_the_top_up_is_still_dropped(sparse: _Recording) -> None:
    samples, _ = collect_column_values(sparse, SPARSE, 3, "sqlite")

    assert "essay" not in samples["visits"]


def test_known_samples_stand_in_for_the_first_rows(sparse: _Recording) -> None:
    """A caller that sampled the table already: no sample statement, its
    values first, and only what it came up short on topped up."""
    known = {"visits": {"seats": [1001, 1002, None, 1003], "ward": [3, 3]}}

    samples, _ = collect_column_values(sparse, SPARSE, 3, "sqlite", known_samples=known)

    assert not any(q.startswith("SELECT *") for q in sparse.queries)
    assert samples["visits"]["seats"] == ["1001", "1002", "1003"]
    assert _asked_about(sparse.queries, "seats") == []
    # The caller's value first: the DISTINCT returns 1 before it.
    assert samples["visits"]["ward"][0] == "3" and len(set(samples["visits"]["ward"])) == 3
    assert len(samples["visits"]["score"]) == 3  # not named: sampled with nothing found


def test_a_failed_top_up_keeps_what_the_first_rows_gave(sparse: _Recording) -> None:
    class _NoDistinct(_Recording):
        def execute_query(self, sql: str, *args: Any, **kwargs: Any) -> Any:
            if "DISTINCT" in sql:
                raise RuntimeError("permission denied")
            return super().execute_query(sql, *args, **kwargs)

    connector = _NoDistinct(sparse._connector)

    samples, complete = collect_column_values(connector, SPARSE, 3, "sqlite")

    assert samples["visits"]["ward"] == ["1"]
    assert samples["visits"]["seats"] == ["1001", "1002", "1003"]
    assert "score" not in samples["visits"] and "remark" not in samples["visits"]
    assert complete == {}


def test_no_samples_asked_for_means_no_top_up(sparse: _Recording) -> None:
    """``sample_rows=0``: the complete lists only, as a caller wanting no
    samples has always had."""
    samples, _ = collect_column_values(sparse, SPARSE, 0, "sqlite")

    assert samples == {}
    assert _asked_about(sparse.queries, "score") == []
