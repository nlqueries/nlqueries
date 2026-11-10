"""`nlqueries eval` as a CI gate: what it runs, and what it exits with."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import yaml
from click.testing import CliRunner
from nlqueries.cli import main as cli_main
from nlqueries.cli.main import cli
from nlqueries.orchestrator.sync_runner import AgentQueryResult

_ID = "postgres:localhost:shop"


def _result(sql: str | None) -> AgentQueryResult:
    return AgentQueryResult(
        question="q",
        resolved_question="q",
        agent_type="sql",
        answer="",
        sql=sql,
        sql_result=None,
        sql_is_valid=True,
        execution_mode="generate_only",
        citations=[],
        merged_answer=None,
        latency_ms=1,
        session_id=None,
    )


def _invoke(tmp_path: Path, kb: dict[str, Any], args: list[str]) -> Any:
    from nlqueries.state_files import agent_storage_id

    (tmp_path / f"{agent_storage_id(_ID)}.yaml").write_text(yaml.safe_dump(kb), encoding="utf-8")
    with (
        patch.object(cli_main, "KB_PATH", tmp_path),
        patch.object(cli_main, "_resolve_alias", side_effect=lambda v: _ID if v == "shop" else v),
        patch.object(cli_main, "_resolve_dialect", return_value="postgres"),
        patch.object(cli_main, "_load_connectors", return_value={_ID: {"alias": "shop"}}),
        patch(
            "nlqueries.orchestrator.sync_runner.run_query_sync",
            side_effect=lambda *a, **k: _result("SELECT id FROM orders"),
        ),
    ):
        return CliRunner().invoke(cli, ["eval", *args])


_KB = {"schema": {"tables": [{"name": "orders"}]}}


def test_a_run_with_no_cases_fails(tmp_path: Path) -> None:
    """A gate that passes when it checked nothing passes a KB that lost its
    capsules, or a golden file that matched no case, without anyone seeing."""
    result = _invoke(tmp_path, _KB, ["shop"])

    assert result.exit_code != 0, result.output


def test_a_golden_case_named_by_alias_is_run(tmp_path: Path) -> None:
    golden = tmp_path / "golden.yaml"
    golden.write_text(
        yaml.safe_dump([{"q": "how many orders?", "agent": "shop", "expect_tables": ["orders"]}]),
        encoding="utf-8",
    )

    result = _invoke(tmp_path, _KB, ["shop", "--golden", str(golden), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output[result.output.index("{") :])
    assert payload["agent_id"] == _ID
    assert [c["question"] for c in payload["cases"]] == ["how many orders?"]


def test_a_golden_case_named_by_alias_is_run_when_eval_is_given_the_id(tmp_path: Path) -> None:
    """A CI job names the connector by its ID; the golden file was written with
    the alias. The connector's own alias has to match too, or every golden case
    is dropped and the capsules alone pass the run (from #397's review)."""
    golden = tmp_path / "golden.yaml"
    golden.write_text(
        yaml.safe_dump([{"q": "how many orders?", "agent": "shop"}]), encoding="utf-8"
    )

    result = _invoke(tmp_path, _KB, [_ID, "--golden", str(golden), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output[result.output.index("{") :])
    assert [c["question"] for c in payload["cases"]] == ["how many orders?"]
