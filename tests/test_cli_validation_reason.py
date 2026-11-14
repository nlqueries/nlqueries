"""`nlqueries query` says why a statement failed validation, not only that it did.

A statement the output budget cut off comes back with no SQL and a reason
naming ``LLM_MAX_OUTPUT_TOKENS``; without the reason the CLI printed nothing at
all for it, and for any other invalid statement only "failed validation".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from click.testing import CliRunner
from nlqueries.cli import main as cli_main
from nlqueries.cli.main import cli
from nlqueries.orchestrator.sync_runner import AgentQueryResult

AGENT = "sales"
CUT_OFF = (
    "The answer was cut off at the output budget (1024 tokens) before the statement "
    "was complete, and a retry with 2048 tokens was cut off too. Raise "
    "LLM_MAX_OUTPUT_TOKENS."
)


def _result(sql: str | None, error: str | None, code: str | None) -> AgentQueryResult:
    return AgentQueryResult(
        question="big orders",
        resolved_question="big orders",
        agent_type="sql",
        answer="",
        sql=sql,
        sql_result=None,
        sql_is_valid=error is None,
        execution_mode="generate_only",
        citations=[],
        merged_answer=None,
        latency_ms=1,
        session_id=None,
        validation_error=error,
        validation_code=code,
    )


def _query(tmp_path: Path, result: AgentQueryResult, *extra: str) -> Any:
    (tmp_path / f"{AGENT}.yaml").write_text("{}", encoding="utf-8")
    with (
        patch.object(cli_main, "_load_connectors", return_value={}),
        patch.object(cli_main, "KB_PATH", tmp_path),
        patch("nlqueries.orchestrator.sync_runner.run_query_sync", return_value=result),
    ):
        return CliRunner().invoke(
            cli, ["query", AGENT, "big orders", "--no-execute", "--no-session", *extra]
        )


def _json(output: str) -> Any:
    """The JSON document, after the warning line: the warning goes to stderr,
    which older Click's runner mixes into the output."""
    return json.loads(output[output.index("{") :])


def test_a_statement_cut_off_for_good_is_reported_with_its_reason(tmp_path: Path) -> None:
    out = _query(tmp_path, _result(None, CUT_OFF, "output_budget_exhausted"))

    assert out.exit_code == 0, out.output
    flat = " ".join(out.output.split())
    assert "Not executed:" in flat
    assert "Raise LLM_MAX_OUTPUT_TOKENS." in flat


def test_any_other_invalid_statement_is_reported_with_its_reason(tmp_path: Path) -> None:
    out = _query(
        tmp_path,
        _result("SELECT * FROM ghost", "Table 'ghost' not found", "invalid_sql"),
    )

    assert out.exit_code == 0, out.output
    assert "Table 'ghost' not found" in " ".join(out.output.split())


def test_the_json_carries_the_verdict_reason_and_code(tmp_path: Path) -> None:
    out = _query(tmp_path, _result(None, CUT_OFF, "output_budget_exhausted"), "--json")

    assert out.exit_code == 0, out.output
    body = _json(out.output)
    assert body["sql"] is None
    assert body["sql_is_valid"] is False
    assert body["validation_code"] == "output_budget_exhausted"
    assert body["validation_error"] == CUT_OFF


def test_a_valid_statement_reports_nothing(tmp_path: Path) -> None:
    out = _query(tmp_path, _result("SELECT id FROM orders", None, None), "--json")

    body = _json(out.output)
    assert (body["validation_error"], body["validation_code"]) == (None, None)
