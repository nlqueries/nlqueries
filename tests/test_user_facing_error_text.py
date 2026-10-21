"""Driver and connectors-file text stays out of what a user is shown.

A driver's error carries the statement, its parameters and sometimes the
connection string on the lines after its first; PyYAML's parse error quotes the
line it stopped on. Either can hold a password or a host. The MCP ``query`` tool
shows the first line and a reference id, with the full text in the server log;
a connectors file that does not parse is reported by line number and problem.
"""

from __future__ import annotations

import logging
import re
import traceback
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner
from nlqueries.cli.main import cli

SECRET = "hunter2"
HOST = "10.20.30.40"
DRIVER_ERROR = (
    "(psycopg.OperationalError) could not run the statement\n"
    f"DETAIL: dsn host={HOST} user=app password={SECRET}"
)
REFERENCE = re.compile(r"\(reference ([0-9a-f]{8})\)")

# The fourth line is not valid YAML, and it is the line holding the password.
MALFORMED = f"good:\n  db_type: postgres\n  host: h\nbad: password: {SECRET}\n"


def _result_with_error(error: str) -> MagicMock:
    sql_result = MagicMock()
    sql_result.error = error
    result = MagicMock()
    result.answer = ""
    result.agent_type = "sql"
    result.sql = "SELECT 1"
    result.sql_result = sql_result
    result.citations = []
    result.latency_ms = 1
    result.provenance = None
    return result


def _referenced_in_log(out: str, caplog: pytest.LogCaptureFixture) -> None:
    """*out* names a reference, and the log entry with it holds the full text."""
    match = REFERENCE.search(out)
    assert match, out
    logged = [r.getMessage() for r in caplog.records if match.group(1) in r.getMessage()]
    assert logged and SECRET in logged[0] and HOST in logged[0]


# --- The MCP query tool --------------------------------------------------------


@pytest.mark.anyio
async def test_a_database_error_is_shown_without_what_the_driver_quoted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from nlqueries.mcp_server.server import query

    result = _result_with_error(DRIVER_ERROR)
    with (
        patch(
            "nlqueries.orchestrator.sync_runner.run_query",
            new_callable=AsyncMock,
            return_value=result,
        ),
        caplog.at_level(logging.WARNING, logger="nlqueries.mcp_server.server"),
    ):
        out = await query("How many orders?", "sales")

    assert "could not run the statement" in out
    assert "password=" not in out and SECRET not in out and HOST not in out
    _referenced_in_log(out, caplog)
    # Only the reply is cut down: the result keeps the raw text the correction
    # loop and the not-permitted check read.
    assert result.sql_result.error == DRIVER_ERROR


@pytest.mark.anyio
async def test_a_failed_query_is_shown_without_what_the_driver_quoted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from nlqueries.mcp_server.server import query

    with (
        patch(
            "nlqueries.orchestrator.sync_runner.run_query",
            new_callable=AsyncMock,
            side_effect=RuntimeError(DRIVER_ERROR),
        ),
        caplog.at_level(logging.WARNING, logger="nlqueries.mcp_server.server"),
    ):
        out = await query("How many orders?", "sales")

    assert out.startswith("Query failed: RuntimeError: (psycopg.OperationalError)")
    assert "password=" not in out and SECRET not in out and HOST not in out
    match = REFERENCE.search(out)
    assert match, out
    # `_log.exception`: the full error is in the record's traceback.
    (record,) = [r for r in caplog.records if match.group(1) in r.getMessage()]
    assert record.exc_info is not None
    assert SECRET in str(record.exc_info[1]) and HOST in str(record.exc_info[1])


# --- A connectors file that does not parse -------------------------------------


@pytest.fixture
def malformed_file(tmp_path: Path) -> Path:
    f = tmp_path / "connectors.yaml"
    f.write_text(MALFORMED, encoding="utf-8")
    return f


def test_the_mcp_listing_names_the_line_not_its_text(malformed_file: Path) -> None:
    from nlqueries.mcp_server.server import list_connectors

    with patch("nlqueries.mcp_server.server.config.CONNECTORS_FILE", malformed_file):
        out = list_connectors()

    assert out == "Failed to read connectors file: line 4: mapping values are not allowed here"


def test_the_loader_names_the_line_not_its_text(malformed_file: Path) -> None:
    from nlqueries.connectors.loader import ConnectorsFileError, _load_connectors

    with (
        patch("nlqueries.connectors.loader.config.CONNECTORS_FILE", malformed_file),
        pytest.raises(ConnectorsFileError) as refused,
    ):
        _load_connectors()

    assert str(refused.value) == (
        "Failed to read connectors file: line 4: mapping values are not allowed here"
    )
    # Not chained to PyYAML's error either, which would print the line in any
    # traceback logged for this.
    assert SECRET not in "".join(traceback.format_exception(refused.value))


def test_the_cli_names_the_line_not_its_text(malformed_file: Path) -> None:
    with patch("nlqueries.connectors.loader.config.CONNECTORS_FILE", malformed_file):
        result = CliRunner().invoke(cli, ["connectors"])

    assert result.exit_code == 1, result.output
    # A ClickException's exit, not the parse error escaping as a traceback.
    assert isinstance(result.exception, SystemExit), result.exception
    assert "Failed to read connectors file: line 4: mapping values are not allowed here" in (
        result.output
    )
    assert SECRET not in result.output


def test_health_reports_the_file_and_runs_the_other_checks(malformed_file: Path) -> None:
    """A file that does not parse is one failed check, not the end of the run."""
    import json

    from nlqueries.cli.main import _CheckResult

    others = {
        name: MagicMock(return_value=_CheckResult(service, "ok", "up"))
        for name, service in [
            ("_check_qdrant", "Qdrant"),
            ("_check_llm", "LLM"),
            ("_check_embedding", "Embedding"),
            ("_check_config", "Config"),
        ]
    }
    with (
        patch("nlqueries.connectors.loader.config.CONNECTORS_FILE", malformed_file),
        patch.multiple("nlqueries.cli.main", **others),
    ):
        result = CliRunner().invoke(cli, ["health", "--json"])

    assert result.exit_code == 1, result.output
    report = json.loads(result.output)
    by_service = {c["service"]: c for c in report["checks"]}
    assert set(by_service) == {"Qdrant", "Database", "LLM", "Embedding", "Config"}
    database = by_service["Database"]
    assert database["status"] == "fail"
    assert database["detail"].startswith(
        "Failed to read connectors file: line 4: mapping values are not allowed here"
    )
    assert report["healthy"] is False
    assert SECRET not in result.output


def test_the_mcp_listing_reads_the_file_through_the_loader(tmp_path: Path) -> None:
    """One reader: whatever the loader refuses, the listing reports the same way."""
    from nlqueries.connectors.loader import ConnectorsFileError
    from nlqueries.mcp_server.server import list_connectors

    # Valid on its own, so only the loader's refusal can produce the message.
    f = tmp_path / "connectors.yaml"
    f.write_text("good:\n  db_type: postgres\n", encoding="utf-8")
    with (
        patch("nlqueries.mcp_server.server.config.CONNECTORS_FILE", f),
        patch(
            "nlqueries.connectors.loader._read_connectors_root",
            side_effect=ConnectorsFileError("line 9: a problem the loader found"),
        ),
    ):
        out = list_connectors()

    assert out == "Failed to read connectors file: line 9: a problem the loader found"
