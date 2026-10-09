"""Column values in the knowledge base, so the prompt shows how they are spelled.

A text column with few distinct values is stored whole and marked
``values_complete``, and the prompt renders it as "values: [...]": the model
then has 'Legal' and 'Directly funded' to copy rather than a spelling to guess.
Any other column gets a few samples, rendered "samples: [...]". Keys and
personal-data columns get nothing.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest
from nlqueries.connectors.base import ColumnSpec, SchemaSpec, TableSpec, column_values_sql
from nlqueries.connectors.sqlite import SQLiteConnector
from nlqueries.execution import ExecutionPolicy
from nlqueries.knowledge.kb_generator import collect_column_values, generate_knowledge_base

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
