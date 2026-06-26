"""CLI integration tests — process-history command.

Uses Click's CliRunner plus mocked console / connector so no real DB or
filesystem writes are needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from click.testing import CliRunner
from nlqueries.cli.main import cli
from nlqueries.connectors.base import DatabaseConnector, QueryRecord, QueryResult, SchemaSpec

# ---------------------------------------------------------------------------
# Minimal stub connector
# ---------------------------------------------------------------------------

_FAKE_CFG: dict[str, Any] = {
    "db_type": "postgres",
    "url": "postgresql://user:pass@localhost/testdb",
}


class _StubConnector(DatabaseConnector):
    """Returns canned query records; connect/schema are no-ops."""

    def connect(self, credentials: dict[str, Any]) -> None:
        pass

    def test_connection(self) -> bool:
        return True

    def extract_schema(self) -> SchemaSpec:
        raise RuntimeError("stub — no schema")

    def extract_query_history(self, days: int = 30, limit: int = 500) -> list[QueryRecord]:
        return [
            QueryRecord(
                sql=f"SELECT id FROM users WHERE status = 'active_{i}'",
                execution_count=5,
                avg_duration_ms=None,
                last_executed=None,
            )
            for i in range(20)
        ]

    def execute_query(self, sql: str) -> QueryResult:
        return QueryResult(columns=[], rows=[], row_count=0, execution_time_ms=0.0, error=None)


def _invoke_process_history(
    tmp_path: Path,
    extra_args: list[str] | None = None,
) -> tuple[int, list[str]]:
    """Invoke 'nlqueries process-history' with mocked connector + console.

    Returns (exit_code, list_of_print_call_arg_strings).
    """
    runner = CliRunner()
    mock_console = MagicMock()
    args = [
        "process-history",
        "dvdrental",
        "--days",
        "30",
        "--min-executions",
        "1",
        "--no-annotate",
        "--no-embed",
        *(extra_args or []),
    ]

    with (
        patch("nlqueries.cli.main.console", mock_console),
        patch("nlqueries.cli.main.err_console", MagicMock()),
        patch("nlqueries.cli.main._require_connector", return_value=_FAKE_CFG),
        patch("nlqueries.cli.main._resolve_alias", side_effect=lambda x: x),
        patch("nlqueries.cli.main._load_password", return_value=None),
        patch("nlqueries.cli.main.CONNECTOR_REGISTRY", {"postgres": lambda: _StubConnector()}),
        patch("nlqueries.processing.pipeline.CAPSULES_DIR", tmp_path),
    ):
        result = runner.invoke(cli, args)

    print_calls = [str(call) for call in mock_console.print.call_args_list]
    return result.exit_code, print_calls


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestProcessHistoryStageOutput:
    def test_exits_cleanly(self, tmp_path: Path) -> None:
        exit_code, _ = _invoke_process_history(tmp_path)
        assert exit_code == 0

    def test_shows_stage_1_extract(self, tmp_path: Path) -> None:
        """Stage [1] extraction line is printed after query history is fetched."""
        _, calls = _invoke_process_history(tmp_path)
        assert any("[1]" in c for c in calls), f"[1] not found in: {calls}"

    def test_shows_stage_2_filter(self, tmp_path: Path) -> None:
        """Stage [2] filter line is printed after deduplication."""
        _, calls = _invoke_process_history(tmp_path)
        assert any("[2]" in c for c in calls), f"[2] not found in: {calls}"

    def test_shows_stage_3_capsules(self, tmp_path: Path) -> None:
        """Stage [3] capsule line is printed after clustering and parameterization."""
        _, calls = _invoke_process_history(tmp_path)
        assert any("[3]" in c for c in calls), f"[3] not found in: {calls}"

    def test_no_annotate_omits_stage_4(self, tmp_path: Path) -> None:
        """--no-annotate skips the annotation stage so [4] never appears."""
        _, calls = _invoke_process_history(tmp_path)
        assert not any("[4]" in c for c in calls), "Stage [4] should be absent without --annotate"

    def test_stage_1_reports_record_count(self, tmp_path: Path) -> None:
        """The [1] line includes the number of raw records extracted."""
        _, calls = _invoke_process_history(tmp_path)
        stage1 = next((c for c in calls if "[1]" in c and "Extracted" in c), None)
        assert stage1 is not None, "Expected '[1] Extracted N raw records' in output"
        # _StubConnector returns 20 records
        assert "20" in stage1


# ---------------------------------------------------------------------------
# cache list command
# ---------------------------------------------------------------------------


class TestCacheListCommand:
    def _make_entry_payload(self, question: str, agent_type: str = "sql") -> dict:
        from datetime import UTC, datetime

        record = MagicMock()
        record.payload = {
            "question": question,
            "resolved_question": question,
            "agent_type": agent_type,
            "answer": f"Answer to {question}",
            "sql": "SELECT COUNT(*) FROM film",
            "created_at": datetime.now(UTC).isoformat(),
            "hit_count": 2,
        }
        return record

    def test_cache_list_shows_questions(self) -> None:
        """cache list prints a table containing the cached question text."""
        runner = CliRunner()
        record = self._make_entry_payload("How many films are there?")
        mock_client = MagicMock()
        mock_client.scroll.return_value = ([record], None)
        mock_client.get_collection.return_value.points_count = 1

        with (
            patch("nlqueries.cli.main._resolve_alias", side_effect=lambda x: x),
            patch("nlqueries.cache.semantic_cache._get_client", return_value=mock_client),
        ):
            result = runner.invoke(cli, ["cache", "list", "dvdrental"])

        assert result.exit_code == 0, result.output
        assert "How many films are there?" in result.output

    def test_cache_list_empty_shows_message(self) -> None:
        """cache list prints a helpful message when the cache is empty."""
        runner = CliRunner()
        mock_client = MagicMock()
        mock_client.scroll.return_value = ([], None)

        with (
            patch("nlqueries.cli.main._resolve_alias", side_effect=lambda x: x),
            patch("nlqueries.cache.semantic_cache._get_client", return_value=mock_client),
        ):
            result = runner.invoke(cli, ["cache", "list", "dvdrental"])

        assert result.exit_code == 0, result.output
        assert "No cached entries" in result.output
