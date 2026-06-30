"""Tests for nlqueries.knowledge.kb_stats (#35)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import yaml

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _write_kb(path: Path, kb: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(kb), encoding="utf-8")


def _make_kb(
    tables: list[dict[str, Any]] | None = None,
    capsules: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "schema": {"tables": tables or []},
        "query_capsules": capsules or [],
        "business_context": {"glossary": [], "rules": []},
    }


def _make_stub_connector(
    table_count: int = 5,
    col_count_per_table: int = 4,
    fk_references: bool = True,
) -> Any:
    """Return a mock DatabaseConnector with canned extract_schema()."""
    from nlqueries.connectors.base import ColumnSpec, SchemaSpec, TableSpec

    tables = []
    for i in range(table_count):
        cols = [
            ColumnSpec(
                name=f"col_{j}",
                type="text",
                nullable=True,
                is_primary_key=(j == 0),
                is_foreign_key=(j == col_count_per_table - 1 and fk_references and i > 0),
                references="table_0.col_0"
                if (j == col_count_per_table - 1 and fk_references and i > 0)
                else None,
                description=None,
            )
            for j in range(col_count_per_table)
        ]
        tables.append(
            TableSpec(
                name=f"table_{i}", schema="public", row_count=100, columns=cols, description=None
            )
        )

    spec = SchemaSpec(database="testdb", tables=tables, extracted_at="2026-01-01T00:00:00")
    connector = MagicMock()
    connector.extract_schema.return_value = spec
    connector.get_schema_summary.return_value = (
        table_count,
        table_count * col_count_per_table,
    )
    return connector


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestComputeKBStats:
    def test_missing_kb_returns_zero_counts(self, tmp_path: Path) -> None:
        """compute_kb_stats returns zeroes when the KB file does not exist."""
        from nlqueries.knowledge.kb_stats import compute_kb_stats

        stats = compute_kb_stats("agent1", tmp_path / "missing.yaml")
        assert stats.kb_tables == 0
        assert stats.kb_columns == 0
        assert stats.kb_mtime is None

    def test_basic_schema_counts(self, tmp_path: Path) -> None:
        """Tables and columns are counted correctly from the KB YAML."""
        from nlqueries.knowledge.kb_stats import compute_kb_stats

        kb = _make_kb(
            tables=[
                {
                    "name": "users",
                    "description": "User accounts",
                    "columns": [
                        {"name": "id", "type": "int", "description": ""},
                        {"name": "email", "type": "text", "description": "User email"},
                    ],
                },
                {
                    "name": "orders",
                    "description": "",
                    "columns": [
                        {"name": "id", "type": "int", "description": ""},
                    ],
                },
            ]
        )
        kb_path = tmp_path / "kb.yaml"
        _write_kb(kb_path, kb)

        stats = compute_kb_stats("agent1", kb_path)

        assert stats.kb_tables == 2
        assert stats.kb_tables_with_desc == 1
        assert stats.kb_columns == 3
        assert stats.kb_columns_with_desc == 1
        assert stats.kb_mtime is not None

    def test_capsule_coverage(self, tmp_path: Path) -> None:
        """Query capsule counts and intent coverage are computed correctly."""
        from nlqueries.knowledge.kb_stats import compute_kb_stats

        kb = _make_kb(
            capsules=[
                {"intent": "Show all users", "template": "SELECT * FROM users", "frequency": 10},
                {"intent": "", "template": "SELECT id FROM orders", "frequency": 2},
            ]
        )
        kb_path = tmp_path / "kb.yaml"
        _write_kb(kb_path, kb)

        stats = compute_kb_stats("agent1", kb_path)

        assert stats.capsule_count == 2
        assert stats.capsule_with_intent == 1

    def test_join_counting_from_capsule_sql(self, tmp_path: Path) -> None:
        """JOIN keywords in capsule templates are counted."""
        from nlqueries.knowledge.kb_stats import compute_kb_stats

        kb = _make_kb(
            capsules=[
                {
                    "intent": "Orders with user info",
                    "template": "SELECT * FROM orders JOIN users ON orders.user_id = users.id",
                    "frequency": 5,
                },
            ]
        )
        kb_path = tmp_path / "kb.yaml"
        _write_kb(kb_path, kb)

        stats = compute_kb_stats("agent1", kb_path)
        assert stats.joins_in_capsules == 1

    def test_live_db_counts_from_connector(self, tmp_path: Path) -> None:
        """When a connector is provided, db_tables and db_columns are populated."""
        from nlqueries.knowledge.kb_stats import compute_kb_stats

        kb = _make_kb(tables=[{"name": "t1", "description": "", "columns": []}])
        kb_path = tmp_path / "kb.yaml"
        _write_kb(kb_path, kb)

        connector = _make_stub_connector(table_count=3, col_count_per_table=4)
        stats = compute_kb_stats("agent1", kb_path, connector)

        assert stats.db_tables == 3
        assert stats.db_columns == 12

    def test_fk_join_coverage(self, tmp_path: Path) -> None:
        """FK pairs are detected from the live schema and matched to capsule JOINs."""
        from nlqueries.knowledge.kb_stats import compute_kb_stats

        kb = _make_kb(
            capsules=[
                {
                    "intent": "x",
                    "template": (
                        "SELECT * FROM table_0 JOIN table_1 ON table_1.col_3 = table_0.col_0"
                    ),
                    "frequency": 1,
                }
            ]
        )
        kb_path = tmp_path / "kb.yaml"
        _write_kb(kb_path, kb)

        connector = _make_stub_connector(table_count=2, col_count_per_table=4, fk_references=True)
        stats = compute_kb_stats("agent1", kb_path, connector)

        assert stats.fk_joins is not None
        assert stats.fk_joins >= 1
        assert stats.fk_joins_seen is not None

    def test_no_connector_leaves_db_fields_none(self, tmp_path: Path) -> None:
        """Without a connector, live-DB fields remain None."""
        from nlqueries.knowledge.kb_stats import compute_kb_stats

        kb = _make_kb()
        kb_path = tmp_path / "kb.yaml"
        _write_kb(kb_path, kb)

        stats = compute_kb_stats("agent1", kb_path, connector=None)
        assert stats.db_tables is None
        assert stats.db_columns is None
        assert stats.fk_joins is None

    def test_ambiguous_columns_flagged(self, tmp_path: Path) -> None:
        """Columns with generic names (status, type, …) and no description are counted."""
        from nlqueries.knowledge.kb_stats import compute_kb_stats

        kb = _make_kb(
            tables=[
                {
                    "name": "items",
                    "description": "",
                    "columns": [
                        {"name": "status", "type": "text", "description": ""},
                        {"name": "type", "type": "text", "description": "Item type"},  # has desc
                        {"name": "price", "type": "numeric", "description": ""},
                    ],
                }
            ]
        )
        kb_path = tmp_path / "kb.yaml"
        _write_kb(kb_path, kb)

        stats = compute_kb_stats("agent1", kb_path)
        assert stats.ambiguous_columns == 1  # only "status" (type has a description)


class TestKBStatsCommand:
    """CLI integration tests for `nlqueries kb-stats`."""

    def test_missing_kb_exits_nonzero(self, tmp_path: Path) -> None:
        """kb-stats exits 1 when the KB file does not exist."""
        from unittest.mock import patch

        from click.testing import CliRunner
        from nlqueries.cli.main import cli

        runner = CliRunner()
        with (
            patch("nlqueries.cli.main.KB_PATH", tmp_path),
            patch("nlqueries.cli.main._load_connectors", return_value={}),
        ):
            result = runner.invoke(cli, ["kb-stats", "myagent"])
        assert result.exit_code == 1
        assert "export-kb" in result.output

    def test_existing_kb_exits_zero(self, tmp_path: Path) -> None:
        """kb-stats exits 0 when the KB file exists."""
        from unittest.mock import patch

        from click.testing import CliRunner
        from nlqueries.cli.main import cli

        kb = _make_kb(
            tables=[
                {
                    "name": "t",
                    "description": "Test",
                    "columns": [{"name": "id", "type": "int", "description": ""}],
                }
            ]
        )
        (tmp_path / "myagent.yaml").write_text(yaml.dump(kb), encoding="utf-8")

        runner = CliRunner()
        with (
            patch("nlqueries.cli.main.KB_PATH", tmp_path),
            patch("nlqueries.cli.main._load_connectors", return_value={}),
        ):
            result = runner.invoke(cli, ["kb-stats", "myagent"])
        assert result.exit_code == 0, result.output
        assert "Knowledge Base" in result.output

    def test_json_flag_emits_parseable_json(self, tmp_path: Path) -> None:
        """--json emits a parseable JSON object with expected top-level keys."""
        import json
        from unittest.mock import patch

        from click.testing import CliRunner
        from nlqueries.cli.main import cli

        kb = _make_kb()
        (tmp_path / "myagent.yaml").write_text(yaml.dump(kb), encoding="utf-8")

        runner = CliRunner()
        with (
            patch("nlqueries.cli.main.KB_PATH", tmp_path),
            patch("nlqueries.cli.main._load_connectors", return_value={}),
        ):
            result = runner.invoke(cli, ["kb-stats", "myagent", "--json"])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert "schema_coverage" in data
        assert "query_coverage" in data
        assert "quality_signals" in data
