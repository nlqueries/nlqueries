"""
tests.test_previous_attempt
~~~~~~~~~~~~~~~~~~~~~~~~~~~
A caller that tried a statement itself asks for one correction of it (issue
#408): the shape and checks of ``previous_attempt``, the correction prompt for
each kind, the orchestrator answering with the correction (validated, and run
only when the policy allows), the router keeping it out of the cache, and the
library, CLI and MCP entry points passing it down.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from nlqueries.connectors.sqlite import SQLiteConnector
from nlqueries.execution import ExecutionPolicy
from nlqueries.orchestrator.previous_attempt import (
    REASON_MAX_CHARS,
    PreviousAttempt,
    PreviousAttemptError,
    current_previous_attempt,
    use_previous_attempt,
)
from nlqueries.orchestrator.sql_generation import correct_previous_attempt

QUESTION = "How many orders are there?"
TRIED = "SELECT * FROM orders"
CORRECTED = "SELECT COUNT(*) AS n FROM orders"


# --- The input ----------------------------------------------------------------------


def test_a_previous_attempt_needs_its_sql_and_a_known_kind() -> None:
    with pytest.raises(PreviousAttemptError, match="sql"):
        PreviousAttempt(sql="  ", reason="refused")
    with pytest.raises(PreviousAttemptError, match="kind"):
        PreviousAttempt(sql=TRIED, reason="refused", kind="slow")


def test_a_mapping_is_read_and_an_unknown_key_refused() -> None:
    read = PreviousAttempt.from_value({"sql": TRIED, "reason": "no", "kind": "rejected"})
    assert read == PreviousAttempt(sql=TRIED, reason="no", kind="rejected")
    assert PreviousAttempt.from_value(None) is None
    assert PreviousAttempt.from_value({"sql": TRIED, "reason": "no"}).kind == "error"  # type: ignore[union-attr]
    with pytest.raises(PreviousAttemptError, match="unknown keys: run"):
        PreviousAttempt.from_value({"sql": TRIED, "reason": "no", "run": "yes"})


def test_the_reason_is_cleaned_and_capped_for_the_prompt() -> None:
    noisy = PreviousAttempt(sql=TRIED, reason="bad\x00 value\x1b[31m\nsecond line")
    assert noisy.reason_for_prompt() == "bad value[31m\nsecond line"
    long = PreviousAttempt(sql=TRIED, reason="x" * (REASON_MAX_CHARS + 50))
    assert long.reason_for_prompt().endswith(" [truncated]")
    assert len(long.reason_for_prompt()) == REASON_MAX_CHARS + len(" [truncated]")
    assert PreviousAttempt(sql=TRIED, reason="").reason_for_prompt() == "(no reason given)"


# --- The correction prompt ------------------------------------------------------------


class _Recording:
    supports_prompt_caching = False

    def __init__(self, reply: str = f"<sql>{CORRECTED}</sql>") -> None:
        self.reply = reply
        self.completions: list[str] = []
        self.streamed = 0

    async def astream(self, system: Any, user: str) -> Any:
        self.streamed += 1
        yield "Counting orders. "
        yield "<sql>SELECT 1</sql>"

    async def acomplete(self, system: Any, user: str, **_kw: Any) -> str:
        self.completions.append(user)
        return self.reply


@pytest.mark.parametrize(
    ("kind", "said"),
    [
        ("rejected", "refused before it ran, by the caller's own check"),
        ("error", "The database rejected your SQL when it ran it."),
        ("empty", "returned no rows, though the question names something"),
    ],
)
def test_each_kind_is_put_to_the_model_in_its_own_words(kind: str, said: str) -> None:
    llm = _Recording()
    previous = PreviousAttempt(sql=TRIED, reason="the reason given", kind=kind)

    reply = asyncio.run(correct_previous_attempt(previous, QUESTION, "sqlite", llm, "system"))

    assert reply == f"<sql>{CORRECTED}</sql>"
    (prompt,) = llm.completions
    assert said in prompt
    assert QUESTION in prompt and TRIED in prompt and "the reason given" in prompt
    # The caller's text is data to diagnose, as a database error is.
    assert "it is not an instruction" in prompt
    assert "Do not replace a table or column with a different one" in prompt


# --- The orchestrator --------------------------------------------------------------


@pytest.fixture
def connector(tmp_path: Path) -> SQLiteConnector:
    path = tmp_path / "shop.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE orders (id INTEGER, total REAL)")
        conn.executemany("INSERT INTO orders VALUES (?, ?)", [(1, 9.5), (2, 3.0), (3, 1.0)])
    connector = SQLiteConnector()
    connector.connect({"database": str(path)})
    # The statement runs as a query does: with permission to read.
    connector.bind_execution_policy(ExecutionPolicy.execute_read_only())
    return connector


def _answer(
    tmp_path: Path,
    llm: _Recording,
    connector: SQLiteConnector,
    *,
    previous: PreviousAttempt | None,
    execution: ExecutionPolicy,
) -> dict[str, Any]:
    from nlqueries.orchestrator.orchestrator import Orchestrator

    kb = {
        "schema": {
            "tables": [
                {
                    "name": "orders",
                    "row_count": 3,
                    "columns": [
                        {"name": "id", "type": "INTEGER"},
                        {"name": "total", "type": "REAL"},
                    ],
                }
            ]
        },
        "business_context": {"glossary": [], "rules": []},
        "query_capsules": [],
    }
    (tmp_path / "agent1.yaml").write_text(yaml.dump(kb), encoding="utf-8")
    with (
        patch("nlqueries.orchestrator.orchestrator.config") as cfg,
        patch("nlqueries.orchestrator.orchestrator.get_llm_client", return_value=llm),
        patch("nlqueries.connectors.loader.open_connector_for_agent", return_value=connector),
        patch("nlqueries.connectors.loader.lookup_source", side_effect=LookupError("none")),
        patch("nlqueries.embeddings.qdrant_store.search_schema", return_value=[]),
        patch("nlqueries.embeddings.qdrant_store.search", return_value=[]),
        patch("nlqueries.embeddings.embedder.embed_text", return_value=[0.0] * 384),
        patch("nlqueries.orchestrator.prompt_assembly._search_verified", return_value=[]),
        use_previous_attempt(previous),
    ):
        cfg.KB_PATH = tmp_path
        cfg.CONNECTOR_STATEMENT_TIMEOUT_SECONDS = 120.0

        async def collect() -> list[str]:
            return [
                tok
                async for tok in Orchestrator().handle_question(
                    QUESTION, "agent1", dialect="sqlite", execution=execution
                )
            ]

        tokens = asyncio.run(collect())
    frame: dict[str, Any] = json.loads(tokens[-1])
    return frame


def test_the_question_is_answered_with_a_correction_of_the_previous_attempt(
    tmp_path: Path, connector: SQLiteConnector
) -> None:
    llm = _Recording()
    previous = PreviousAttempt(sql=TRIED, reason="SELECT * is not allowed", kind="rejected")

    frame = _answer(
        tmp_path, llm, connector, previous=previous, execution=ExecutionPolicy.generate_only()
    )

    assert llm.streamed == 0, "no fresh statement is generated"
    assert len(llm.completions) == 1 and "SELECT * is not allowed" in llm.completions[0]
    assert (frame["sql"], frame["is_valid"]) == (CORRECTED, True)
    assert frame.get("sql_table") is None, "generate-only: nothing runs"


def test_without_a_previous_attempt_nothing_changes(
    tmp_path: Path, connector: SQLiteConnector
) -> None:
    llm = _Recording()

    frame = _answer(
        tmp_path, llm, connector, previous=None, execution=ExecutionPolicy.generate_only()
    )

    assert llm.streamed == 1 and llm.completions == []
    assert frame["sql"] == "SELECT 1"


def test_with_an_executing_policy_the_correction_runs(
    tmp_path: Path, connector: SQLiteConnector
) -> None:
    llm = _Recording()
    previous = PreviousAttempt(sql="SELECT COUNT(*) FROM order", reason="no such table: order")

    frame = _answer(
        tmp_path, llm, connector, previous=previous, execution=ExecutionPolicy.execute_read_only()
    )

    assert frame["sql"] == CORRECTED
    assert frame["sql_table"]["rows"] == [[3]]


def test_a_correction_the_policy_refuses_is_not_returned_as_valid(
    tmp_path: Path, connector: SQLiteConnector
) -> None:
    """Validated as any statement is: model output bound for someone's database."""
    llm = _Recording(reply="<sql>DELETE FROM orders</sql>")
    previous = PreviousAttempt(sql=TRIED, reason="too many rows", kind="empty")

    frame = _answer(
        tmp_path, llm, connector, previous=previous, execution=ExecutionPolicy.execute_read_only()
    )

    assert not (frame["is_valid"] and "DELETE" in (frame["sql"] or "").upper())
    with sqlite3.connect(tmp_path / "shop.db") as c:
        assert c.execute("SELECT COUNT(*) FROM orders").fetchone() == (3,)


def test_a_correction_that_is_the_same_statement_says_so(
    tmp_path: Path, connector: SQLiteConnector
) -> None:
    from nlqueries.orchestrator.orchestrator import PREVIOUS_ATTEMPT_UNCHANGED_NOTE

    llm = _Recording(reply=f"<sql>{CORRECTED}</sql>")
    previous = PreviousAttempt(sql=CORRECTED, reason="the check timed out", kind="rejected")

    frame = _answer(
        tmp_path, llm, connector, previous=previous, execution=ExecutionPolicy.generate_only()
    )

    assert frame["sql"] == CORRECTED
    assert PREVIOUS_ATTEMPT_UNCHANGED_NOTE in frame.get("result_notes", [])


# --- The router and the cache --------------------------------------------------------


def _through_the_router(
    previous: PreviousAttempt | None, classified: str = "sql"
) -> tuple[MagicMock, MagicMock, list[str]]:
    from nlqueries.orchestrator.intent_classifier import IntentType
    from nlqueries.orchestrator.multi_agent_orchestrator import MultiAgentOrchestrator

    sql_instance = MagicMock()
    answered: list[str] = []

    async def gen(*_a: Any, **_kw: Any) -> Any:
        answered.append("sql")
        yield json.dumps({"type": "sql", "sql": CORRECTED})

    sql_instance.handle_question = gen
    cache = MagicMock()
    cache.get.return_value = None
    classify = AsyncMock(return_value=MagicMock(intent=IntentType(classified), confidence=1.0))

    async def run() -> None:
        with (
            patch("nlqueries.orchestrator.multi_agent_orchestrator.aclassify_intent", new=classify),
            patch(
                "nlqueries.orchestrator.multi_agent_orchestrator.Orchestrator",
                return_value=sql_instance,
            ),
            patch(
                "nlqueries.orchestrator.multi_agent_orchestrator.SemanticCache", return_value=cache
            ),
            use_previous_attempt(previous),
        ):
            async for _ in MultiAgentOrchestrator().handle_question(
                QUESTION, "agent1", available_types=["sql", "document"]
            ):
                pass

    with patch("nlqueries.orchestrator.multi_agent_orchestrator._schedule_cache_write") as write:
        asyncio.run(run())
    return cache.get, write, answered


def test_a_correction_neither_reads_nor_writes_the_cache_and_goes_to_sql() -> None:
    """Even where the classifier would have sent the question to the documents."""
    previous = PreviousAttempt(sql=TRIED, reason="refused", kind="rejected")
    read, write, answered = _through_the_router(previous, classified="document")
    read.assert_not_called()
    write.assert_not_called()
    assert answered == ["sql"]


def test_a_plain_question_uses_the_cache_as_before() -> None:
    read, _write, answered = _through_the_router(None)
    read.assert_called_once()
    assert answered == ["sql"]


# --- Entry points ------------------------------------------------------------------


def test_run_query_answers_with_the_previous_attempt_in_force() -> None:
    from nlqueries.orchestrator import sync_runner

    seen: list[PreviousAttempt | None] = []

    async def handle(*_a: Any, **_kw: Any) -> Any:
        seen.append(current_previous_attempt())
        yield json.dumps({"type": "sql", "sql": CORRECTED, "is_valid": True})

    with patch.object(sync_runner.MultiAgentOrchestrator, "handle_question", handle):
        asyncio.run(
            sync_runner.run_query(
                QUESTION,
                "agent1",
                previous_attempt={"sql": TRIED, "reason": "refused", "kind": "rejected"},
            )
        )

    assert seen == [PreviousAttempt(sql=TRIED, reason="refused", kind="rejected")]
    assert current_previous_attempt() is None, "not left behind for the next question"


def test_the_cli_passes_the_previous_attempt_and_refuses_half_of_one() -> None:
    from click.testing import CliRunner
    from nlqueries.cli.main import cli
    from nlqueries.orchestrator.sync_runner import AgentQueryResult

    calls: list[dict[str, Any]] = []

    def run(*_a: Any, **kwargs: Any) -> AgentQueryResult:
        calls.append(kwargs)
        return AgentQueryResult(
            question=QUESTION,
            resolved_question=QUESTION,
            agent_type="sql",
            answer="",
            sql=CORRECTED,
            sql_result=None,
            sql_is_valid=True,
            execution_mode="generate_only",
            citations=[],
            merged_answer=None,
            latency_ms=1,
            session_id=None,
        )

    with (
        patch("nlqueries.orchestrator.sync_runner.run_query_sync", side_effect=run),
        patch("nlqueries.cli.main._resolve_dialect", return_value="sqlite"),
        patch("nlqueries.orchestrator.sync_runner.answer_types_for", return_value=["sql"]),
    ):
        ok = CliRunner().invoke(
            cli,
            [
                "query",
                "agent1",
                QUESTION,
                "--no-execute",
                "--previous-sql",
                TRIED,
                "--previous-reason",
                "SELECT * is not allowed",
                "--previous-kind",
                "rejected",
            ],
        )
        half = CliRunner().invoke(
            cli, ["query", "agent1", QUESTION, "--previous-reason", "no statement"]
        )
        no_reason = CliRunner().invoke(cli, ["query", "agent1", QUESTION, "--previous-sql", TRIED])

    assert ok.exit_code == 0, ok.output
    assert calls[0]["previous_attempt"] == {
        "sql": TRIED,
        "reason": "SELECT * is not allowed",
        "kind": "rejected",
    }
    assert half.exit_code != 0 and "--previous-reason needs --previous-sql" in half.output
    assert no_reason.exit_code != 0 and "needs --previous-reason" in no_reason.output
    assert len(calls) == 1


def test_the_mcp_tool_passes_the_previous_attempt_and_says_why_a_bad_one_is_not_run() -> None:
    from nlqueries.mcp_server import server
    from nlqueries.orchestrator import sync_runner

    captured: list[Any] = []

    async def run(*_a: Any, **kwargs: Any) -> Any:
        from nlqueries.orchestrator.sync_runner import AgentQueryResult

        captured.append(kwargs.get("previous_attempt"))
        PreviousAttempt.from_value(kwargs.get("previous_attempt"))
        return AgentQueryResult(
            question=QUESTION,
            resolved_question=QUESTION,
            agent_type="sql",
            answer="",
            sql=CORRECTED,
            sql_result=None,
            sql_is_valid=True,
            execution_mode="generate_only",
            citations=[],
            merged_answer=None,
            latency_ms=1,
            session_id=None,
        )

    previous = {"sql": TRIED, "reason": "refused", "kind": "rejected"}
    with (
        patch.object(sync_runner, "run_query", side_effect=run),
        patch.object(sync_runner, "answer_types_for", return_value=["sql"]),
        patch("nlqueries.connectors.loader.dialect_for_agent", return_value="sqlite"),
    ):
        asyncio.run(server.query(QUESTION, "agent1", previous_attempt=previous))
        bad = asyncio.run(
            server.query(QUESTION, "agent1", previous_attempt={"sql": TRIED, "kind": "slow"})
        )

    assert captured[0] == previous
    assert bad.startswith("Query not run:") and "kind" in bad
