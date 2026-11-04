"""`nlqueries query` shows what still looked wrong about the rows.

The orchestrator corrects what it can (`_second_look`); the rest reaches the
caller as notes on the result, which the CLI prints after the rows and puts in
its JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from click.testing import CliRunner
from nlqueries.cli import main as cli_main
from nlqueries.cli.main import cli
from nlqueries.connectors.base import QueryResult
from nlqueries.orchestrator.sync_runner import AgentQueryResult

AGENT = "sales"
NOTE = (
    "No rows matched. If you expected some, a filter may be narrower than intended, "
    "for example a value spelled differently in the data."
)


def _result(notes: list[str]) -> AgentQueryResult:
    return AgentQueryResult(
        question="lost orders",
        resolved_question="lost orders",
        agent_type="sql",
        answer="",
        sql="SELECT id FROM orders WHERE status = 'lost'",
        sql_result=QueryResult(
            columns=["id"], rows=[], row_count=0, execution_time_ms=1.0, error=None
        ),
        sql_is_valid=True,
        execution_mode="execute",
        citations=[],
        merged_answer=None,
        latency_ms=1,
        session_id=None,
        result_notes=notes,
    )


def _query(tmp_path: Path, notes: list[str], *extra: str) -> Any:
    (tmp_path / f"{AGENT}.yaml").write_text("{}", encoding="utf-8")
    with (
        patch.object(cli_main, "_load_connectors", return_value={}),
        patch.object(cli_main, "KB_PATH", tmp_path),
        patch(
            "nlqueries.orchestrator.sync_runner.run_query_sync",
            return_value=_result(notes),
        ),
    ):
        return CliRunner().invoke(
            cli, ["query", AGENT, "lost orders", "--no-execute", "--no-session", *extra]
        )


def test_a_note_is_printed_after_the_rows(tmp_path: Path) -> None:
    result = _query(tmp_path, [NOTE])

    assert result.exit_code == 0, result.output
    flat = " ".join(result.output.split())
    assert "Note :" in flat
    assert "No rows matched." in flat
    assert flat.index("0 row(s)") < flat.index("No rows matched.")


def test_no_note_line_without_notes(tmp_path: Path) -> None:
    result = _query(tmp_path, [])

    assert result.exit_code == 0, result.output
    assert "Note" not in result.output


def test_the_json_carries_the_notes(tmp_path: Path) -> None:
    result = _query(tmp_path, [NOTE], "--json")

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["result_notes"] == [NOTE]
