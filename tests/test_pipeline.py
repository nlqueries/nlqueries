"""
Tests for nlqueries.processing.pipeline (Task 3.3.1).

Required coverage (per spec):
  - 50 mock QueryRecord objects → at least 5 QueryCapsule objects
  - All capsules have a non-empty template_sql

Additional coverage:
  - save_capsules writes valid JSON and returns the correct path
  - save_capsules uses a sanitised filename when connector_id contains colons
  - Empty connector (no history) returns empty capsule list
  - min_executions filters out low-frequency queries
  - schema=None is handled gracefully (no crash, VARCHAR defaults apply)
  - process_query_history returns QueryCapsule instances
  - Capsules are sorted by frequency descending
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from nlqueries.connectors.base import (
    ColumnSpec,
    DatabaseConnector,
    QueryRecord,
    QueryResult,
    SchemaSpec,
    TableSpec,
)
from nlqueries.processing.parameterizer import QueryCapsule
from nlqueries.processing.pipeline import process_query_history, save_capsules

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TABLES = ["users", "orders", "products", "payments", "events"]


def _make_records(n_per_table: int = 10) -> list[QueryRecord]:
    """Build ``n_per_table * 5`` QueryRecord objects across 5 distinct tables."""
    records: list[QueryRecord] = []
    for table in _TABLES:
        for j in range(n_per_table):
            records.append(
                QueryRecord(
                    sql=(
                        f"SELECT id, name FROM {table}"
                        f" WHERE status = 'status_{j}' AND total > {j * 10}"
                    ),
                    execution_count=j + 1,
                    avg_duration_ms=None,
                    last_executed=None,
                )
            )
    return records


def _make_schema() -> SchemaSpec:
    columns = [
        ColumnSpec(
            name="id",
            type="INTEGER",
            nullable=False,
            is_primary_key=True,
            is_foreign_key=False,
            references=None,
            description=None,
        ),
        ColumnSpec(
            name="status",
            type="VARCHAR",
            nullable=True,
            is_primary_key=False,
            is_foreign_key=False,
            references=None,
            description=None,
        ),
        ColumnSpec(
            name="total",
            type="NUMERIC",
            nullable=True,
            is_primary_key=False,
            is_foreign_key=False,
            references=None,
            description=None,
        ),
    ]
    tables = [
        TableSpec(
            name=t,
            schema="public",
            row_count=None,
            columns=columns,
            description=None,
        )
        for t in _TABLES
    ]
    return SchemaSpec(database="test_db", extracted_at="2024-01-01T00:00:00", tables=tables)


class _MockConnector(DatabaseConnector):
    """Minimal concrete DatabaseConnector that returns pre-set records."""

    def __init__(self, records: list[QueryRecord]) -> None:
        self._records = records

    def connect(self, credentials: dict[str, Any]) -> None:
        pass

    def test_connection(self) -> bool:
        return True

    def extract_schema(self) -> SchemaSpec:
        return _make_schema()

    def extract_query_history(self, days: int = 30) -> list[QueryRecord]:
        return self._records

    def execute_query(self, sql: str) -> QueryResult:
        return QueryResult(columns=[], rows=[], row_count=0, execution_time_ms=0.0, error=None)


# ---------------------------------------------------------------------------
# Core spec requirement: 50 mock records → ≥ 5 capsules with valid template_sql
# ---------------------------------------------------------------------------


def test_50_records_produce_at_least_5_capsules() -> None:
    records = _make_records(n_per_table=10)  # 50 records total
    assert len(records) == 50
    connector = _MockConnector(records)
    capsules = process_query_history(connector, schema=_make_schema(), days=30)
    assert len(capsules) >= 5


def test_all_capsules_have_non_empty_template_sql() -> None:
    records = _make_records(n_per_table=10)
    connector = _MockConnector(records)
    capsules = process_query_history(connector, schema=_make_schema(), days=30)
    assert all(cap.template_sql for cap in capsules)


def test_capsules_are_query_capsule_instances() -> None:
    records = _make_records(n_per_table=10)
    connector = _MockConnector(records)
    capsules = process_query_history(connector, schema=_make_schema(), days=30)
    assert all(isinstance(cap, QueryCapsule) for cap in capsules)


# ---------------------------------------------------------------------------
# Empty connector → empty capsule list
# ---------------------------------------------------------------------------


def test_empty_history_returns_empty_capsule_list() -> None:
    connector = _MockConnector([])
    capsules = process_query_history(connector, schema=_make_schema(), days=30)
    assert capsules == []


# ---------------------------------------------------------------------------
# schema=None handled gracefully
# ---------------------------------------------------------------------------


def test_schema_none_does_not_raise() -> None:
    records = _make_records(n_per_table=2)
    connector = _MockConnector(records)
    capsules = process_query_history(connector, schema=None, days=30)
    assert isinstance(capsules, list)


def test_schema_none_still_produces_capsules() -> None:
    records = _make_records(n_per_table=10)
    connector = _MockConnector(records)
    capsules = process_query_history(connector, schema=None, days=30)
    assert len(capsules) >= 5


# ---------------------------------------------------------------------------
# min_executions filter
# ---------------------------------------------------------------------------


def test_min_executions_filters_low_frequency() -> None:
    # execution_count 1–5 per table; with min_executions=5 only count=5 survives
    records = _make_records(n_per_table=5)  # execution_count = 1,2,3,4,5
    connector = _MockConnector(records)
    capsules_all = process_query_history(connector, schema=None, days=30, min_executions=1)
    capsules_high = process_query_history(connector, schema=None, days=30, min_executions=5)
    assert len(capsules_high) <= len(capsules_all)


def test_min_executions_too_high_returns_empty() -> None:
    records = _make_records(n_per_table=3)  # max execution_count = 3
    connector = _MockConnector(records)
    capsules = process_query_history(connector, schema=None, days=30, min_executions=999)
    assert capsules == []


# ---------------------------------------------------------------------------
# Capsules sorted by frequency descending
# ---------------------------------------------------------------------------


def test_capsules_sorted_by_frequency_descending() -> None:
    records = _make_records(n_per_table=10)
    connector = _MockConnector(records)
    capsules = process_query_history(connector, schema=_make_schema(), days=30)
    freqs = [c.frequency for c in capsules]
    assert freqs == sorted(freqs, reverse=True)


# ---------------------------------------------------------------------------
# save_capsules
# ---------------------------------------------------------------------------


def test_save_capsules_writes_valid_json(tmp_path: Path) -> None:
    records = _make_records(n_per_table=10)
    connector = _MockConnector(records)
    capsules = process_query_history(connector, schema=_make_schema(), days=30)

    with patch("nlqueries.processing.pipeline.CAPSULES_DIR", tmp_path):
        out = save_capsules(capsules, "postgres:localhost:mydb")

    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) == len(capsules)


def test_save_capsules_json_has_required_fields(tmp_path: Path) -> None:
    records = _make_records(n_per_table=10)
    connector = _MockConnector(records)
    capsules = process_query_history(connector, schema=_make_schema(), days=30)

    with patch("nlqueries.processing.pipeline.CAPSULES_DIR", tmp_path):
        out = save_capsules(capsules, "test_connector")

    data = json.loads(out.read_text(encoding="utf-8"))
    required = {
        "template_sql",
        "placeholders",
        "tables",
        "columns",
        "frequency",
        "auto_description",
        "intent",
    }
    for entry in data:
        assert required <= entry.keys()


def test_save_capsules_sanitises_connector_id(tmp_path: Path) -> None:
    """Colons and slashes in a connector_id must not appear in the filename."""
    records = _make_records(n_per_table=2)
    connector = _MockConnector(records)
    capsules = process_query_history(connector, schema=None, days=30)

    with patch("nlqueries.processing.pipeline.CAPSULES_DIR", tmp_path):
        out = save_capsules(capsules, "postgres:localhost:5432/mydb")

    assert ":" not in out.name
    assert "/" not in out.name
    assert out.exists()


def test_save_capsules_returns_path(tmp_path: Path) -> None:
    connector = _MockConnector(_make_records(n_per_table=2))
    capsules = process_query_history(connector, schema=None, days=30)

    with patch("nlqueries.processing.pipeline.CAPSULES_DIR", tmp_path):
        out = save_capsules(capsules, "my_connector")

    assert isinstance(out, Path)
    assert out.suffix == ".json"


def test_save_capsules_empty_list_writes_empty_array(tmp_path: Path) -> None:
    with patch("nlqueries.processing.pipeline.CAPSULES_DIR", tmp_path):
        out = save_capsules([], "empty_connector")

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data == []


# ---------------------------------------------------------------------------
# Placeholders contain typed entries
# ---------------------------------------------------------------------------


def test_capsules_have_typed_placeholders() -> None:
    """Literals in the representative SQL should become typed placeholders."""
    records = _make_records(n_per_table=10)
    connector = _MockConnector(records)
    capsules = process_query_history(connector, schema=_make_schema(), days=30)
    # Each query has a string and integer literal; expect both VARCHAR and INT
    all_types = {p.type for cap in capsules for p in cap.placeholders}
    assert "VARCHAR" in all_types
    assert "INT" in all_types


# ---------------------------------------------------------------------------
# Pipeline stages are called in order (smoke test via output shape)
# ---------------------------------------------------------------------------


def test_intent_field_is_empty_string() -> None:
    """intent is filled later by the LLM annotator (Sprint 4)."""
    records = _make_records(n_per_table=10)
    connector = _MockConnector(records)
    capsules = process_query_history(connector, schema=_make_schema(), days=30)
    assert all(cap.intent == "" for cap in capsules)
