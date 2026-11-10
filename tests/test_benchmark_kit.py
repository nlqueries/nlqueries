"""The benchmark kit in ``benchmarks/`` runs as its README-less header says.

Nothing covered it, and three things had stopped being true: the compose file
mounted a ``seed.sql`` that did not exist, ``run_query`` was called with Core's
defaults (so nothing executed under ``generate_only`` and every question used
the cache), and cache hits and validity were guessed rather than read.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml
from benchmarks import run_bench
from nlqueries.execution import ExecutionMode
from nlqueries.orchestrator.sync_runner import AgentQueryResult

_ROOT = Path(__file__).resolve().parents[1]


def _result(**fields: Any) -> AgentQueryResult:
    base: dict[str, Any] = {
        "question": "q",
        "resolved_question": "q",
        "agent_type": "sql",
        "answer": "There are 7 orders.",
        "sql": "SELECT count(*) FROM orders",
        "sql_result": None,
        "sql_is_valid": True,
        "execution_mode": "execute_read_only",
        "citations": [],
        "merged_answer": None,
        "latency_ms": 12,
        "session_id": None,
    }
    base.update(fields)
    return AgentQueryResult(**base)


class _Recorder:
    def __init__(self, result: AgentQueryResult) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, question: str, agent_id: str, **kwargs: Any) -> AgentQueryResult:
        self.calls.append({"question": question, "agent_id": agent_id, **kwargs})
        return self.result


def _run_one(recorder: _Recorder, options: run_bench.RunOptions) -> run_bench.BenchRow:
    with patch("nlqueries.orchestrator.sync_runner.run_query", recorder):
        return asyncio.run(run_bench._run_one("q", "shop", ["orders"], ["7"], False, options))


def test_the_run_names_its_dialect_execution_and_cache() -> None:
    """Core's defaults are Postgres, the cache on, and generate_only, under
    which nothing executes; the harness then measured answer prose."""
    recorder = _Recorder(_result())

    _run_one(recorder, run_bench.RunOptions(dialect="sqlite", execute=True, use_cache=False))

    call = recorder.calls[0]
    assert call["dialect"] == "sqlite"
    assert call["use_cache"] is False
    assert call["execution"].mode is ExecutionMode.EXECUTE_READ_ONLY


def test_without_execute_the_run_only_generates() -> None:
    recorder = _Recorder(_result())

    _run_one(recorder, run_bench.RunOptions())

    assert recorder.calls[0]["execution"].mode is ExecutionMode.GENERATE_ONLY
    assert recorder.calls[0]["use_cache"] is True


def test_cache_hits_and_validity_are_read_from_the_result() -> None:
    """`_is_from_cache` always returned False, and validity was a search of
    the answer text for the string "validation_error"."""
    row = _run_one(_Recorder(_result(from_cache=True, sql_is_valid=False)), run_bench.RunOptions())

    assert row.from_cache is True
    assert row.is_valid is False


def test_a_golden_file_runs_end_to_end(tmp_path: Path) -> None:
    golden = tmp_path / "golden.yaml"
    golden.write_text(
        yaml.safe_dump([{"q": "how many orders?", "expect_tables": ["orders"]}]),
        encoding="utf-8",
    )
    recorder = _Recorder(_result())

    with patch("nlqueries.orchestrator.sync_runner.run_query", recorder):
        report = asyncio.run(run_bench.run_golden(golden, "shop", run_bench.RunOptions()))

    assert report.total == 1
    assert report.passed == 1
    assert "how many orders?" in run_bench._render_markdown(report, "smoke")


def test_the_kit_s_golden_set_asks_only_about_tables_the_seed_creates() -> None:
    """The kit pointed at tests/golden/questions.yaml, whose questions expect
    HR and finance tables the seed does not have, so most cases failed on the
    fixture whatever the model wrote (from #401's review)."""
    import re

    seed = (_ROOT / "benchmarks" / "seed.sql").read_text("utf-8")
    created = {name.lower() for name in re.findall(r"CREATE TABLE (\w+)", seed)}
    golden = yaml.safe_load((_ROOT / "benchmarks" / "golden.yaml").read_text("utf-8"))

    assert golden, "the kit's golden set is empty"
    for case in golden:
        missing = {t.lower() for t in case.get("expect_tables", [])} - created
        assert not missing, (
            f"{case['q']!r} expects {sorted(missing)}, which seed.sql does not create"
        )
        assert "agent" not in case, "--agent decides the agent; the set must not pin one"
    run_bench_text = (_ROOT / "benchmarks" / "run_bench.py").read_text("utf-8")
    assert "--golden benchmarks/golden.yaml" in run_bench_text


def test_the_compose_file_mounts_a_seed_that_exists() -> None:
    compose = yaml.safe_load((_ROOT / "benchmarks" / "docker-compose.yaml").read_text("utf-8"))
    mounts = compose["services"]["postgres"]["volumes"]
    local = [m.split(":")[0] for m in mounts if m.startswith("./")]

    assert local, "the compose file mounts no local seed"
    for path in local:
        assert (_ROOT / "benchmarks" / path).is_file(), f"{path} is mounted but missing"


@pytest.mark.parametrize(
    ("argv", "dialect", "execute", "use_cache"),
    [
        (["--agent", "a", "--golden", "g.yaml"], "postgres", False, True),
        (
            [
                "--agent",
                "a",
                "--golden",
                "g.yaml",
                "--dialect",
                "duckdb",
                "--execute",
                "--no-cache",
            ],
            "duckdb",
            True,
            False,
        ),
    ],
)
def test_the_options_come_from_the_command_line(
    argv: list[str], dialect: str, execute: bool, use_cache: bool
) -> None:
    options = run_bench.options_from(run_bench._parse_args(argv))

    assert options == run_bench.RunOptions(dialect=dialect, execute=execute, use_cache=use_cache)
