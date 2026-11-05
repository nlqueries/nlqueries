"""Asking the user which stored value was meant (``NLQ_AMBIGUITY=ask``).

Grounding leaves a value that is part of several different stored values as a
choice (see ``test_literal_grounding.py``). What these pin is the rest of the
way: nothing runs while a choice waits, the caller is given the options and the
key to answer with, the answer with the choice runs with it, and neither is
kept in the semantic cache, where it would answer the next user without asking.
"""

from __future__ import annotations

import asyncio
import json
import re
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import yaml
from click.testing import CliRunner
from nlqueries.connectors.base import QueryResult
from nlqueries.execution import ExecutionPolicy
from nlqueries.orchestrator.match_notes import (
    Choice,
    add_choice,
    any_chosen,
    choices_for,
    chosen_value,
    use_chosen_values,
)
from nlqueries.orchestrator.sql_generation import SQLGenerationResult
from nlqueries.orchestrator.sync_runner import AgentQueryResult

_CHOICE = Choice(
    key="schools.soc:Schools",
    column="soc",
    asked="Schools",
    options=("Juvenile Court Schools", "Special Education Schools"),
)
_ASKED_SQL = "SELECT id FROM schools WHERE soc = 'Schools'"


def test_a_choice_counts_only_while_the_statement_still_holds_its_value() -> None:
    assert choices_for([_CHOICE], _ASKED_SQL, "sqlite") == [_CHOICE]
    settled = "SELECT id FROM schools WHERE soc = 'Juvenile Court Schools'"
    assert choices_for([_CHOICE], settled, "sqlite") == []
    assert choices_for([_CHOICE], "SELECT FROM", "sqlite") == []


def test_chosen_values_are_seen_inside_and_not_after() -> None:
    assert chosen_value(_CHOICE.key) is None and not any_chosen()
    with use_chosen_values({_CHOICE.key: "Special Education Schools"}):
        assert chosen_value(_CHOICE.key) == "Special Education Schools"
        assert any_chosen()
    assert chosen_value(_CHOICE.key) is None and not any_chosen()


def test_a_choice_is_given_as_its_key_column_value_and_options() -> None:
    assert _CHOICE.as_dict() == {
        "key": "schools.soc:Schools",
        "column": "soc",
        "asked": "Schools",
        "options": ["Juvenile Court Schools", "Special Education Schools"],
    }


# ---------------------------------------------------------------------------
# The orchestrator: nothing runs while a choice waits
# ---------------------------------------------------------------------------


class _LLM:
    supports_prompt_caching = False

    async def astream(self, system: Any, user: str) -> Any:
        yield f"<sql>{_ASKED_SQL}</sql>"

    async def acomplete(self, system: Any, user: str, **_kw: Any) -> str:
        return ""


def _ask(sql: str, *, leave_choice: bool) -> tuple[dict[str, Any], MagicMock]:
    result = SQLGenerationResult(
        sql=sql, is_valid=True, validation_error=None, dialect="sqlite", attempt_count=1
    )

    async def validate(*_a: Any, **_kw: Any) -> SQLGenerationResult:
        if leave_choice:
            add_choice(_CHOICE)
        return result

    connector = MagicMock()
    connector.execute_query.return_value = QueryResult(
        columns=["id"], rows=[[1]], row_count=1, execution_time_ms=1.0, error=None
    )
    kb = {"schema": {"tables": [{"name": "schools", "columns": [{"name": "soc", "type": "TEXT"}]}]}}
    with tempfile.TemporaryDirectory() as tmpdir:
        safe = re.sub(r"[^\w.-]", "_", "agent1")
        (Path(tmpdir) / f"{safe}.yaml").write_text(yaml.dump(kb), encoding="utf-8")
        with (
            patch("nlqueries.orchestrator.orchestrator.config") as cfg,
            patch("nlqueries.orchestrator.orchestrator.get_llm_client", return_value=_LLM()),
            patch(
                "nlqueries.orchestrator.orchestrator.validate_and_repair",
                new=AsyncMock(side_effect=validate),
            ),
            patch("nlqueries.connectors.loader.open_connector_for_agent", return_value=connector),
            patch("nlqueries.connectors.loader.lookup_source", return_value=None),
            patch("nlqueries.embeddings.qdrant_store.search_schema", return_value=[]),
            patch("nlqueries.embeddings.qdrant_store.search", return_value=[]),
            patch("nlqueries.embeddings.embedder.embed_text", return_value=[0.0] * 384),
            patch("nlqueries.orchestrator.prompt_assembly._search_verified", return_value=[]),
        ):
            cfg.KB_PATH = Path(tmpdir)
            cfg.CONNECTOR_STATEMENT_TIMEOUT_SECONDS = 120.0
            from nlqueries.orchestrator.orchestrator import Orchestrator

            async def collect() -> list[str]:
                return [
                    tok
                    async for tok in Orchestrator().handle_question(
                        "students in schools",
                        "agent1",
                        dialect="sqlite",
                        execution=ExecutionPolicy.execute_read_only(),
                    )
                ]

            tokens = asyncio.run(collect())
    return json.loads(tokens[-1]), connector


def test_while_a_choice_waits_nothing_runs_and_the_options_are_given() -> None:
    frame, connector = _ask(_ASKED_SQL, leave_choice=True)
    connector.execute_query.assert_not_called()
    assert frame["sql_table"] is None
    assert frame["choices"] == [_CHOICE.as_dict()]


def test_a_choice_the_statement_no_longer_holds_does_not_stop_it() -> None:
    frame, connector = _ask(
        "SELECT id FROM schools WHERE soc = 'Juvenile Court Schools'", leave_choice=True
    )
    connector.execute_query.assert_called_once()
    assert "choices" not in frame


# ---------------------------------------------------------------------------
# run_query: the choices come back, the chosen values go in
# ---------------------------------------------------------------------------


def test_run_query_returns_the_choices_and_answers_with_the_chosen_values() -> None:
    from nlqueries.orchestrator import sync_runner

    seen: list[str | None] = []

    class _Multi:
        async def handle_question(self, *_a: Any, **_kw: Any) -> Any:
            seen.append(chosen_value(_CHOICE.key))
            yield json.dumps(
                {
                    "type": "sql",
                    "agent_type": "sql",
                    "sql": _ASKED_SQL,
                    "sql_table": None,
                    "choices": [_CHOICE.as_dict()],
                }
            )

    with patch.object(sync_runner, "MultiAgentOrchestrator", _Multi):
        result = asyncio.run(
            sync_runner.run_query(
                "students in schools",
                "agent1",
                chosen_values={_CHOICE.key: "Special Education Schools"},
            )
        )
    assert seen == ["Special Education Schools"]
    assert result.choices == [_CHOICE.as_dict()]


# ---------------------------------------------------------------------------
# The semantic cache: neither a pending choice nor a chosen answer is kept
# ---------------------------------------------------------------------------


def _through_the_router(frame: dict[str, Any], chosen: dict[str, str] | None = None) -> MagicMock:
    from nlqueries.orchestrator.intent_classifier import IntentType
    from nlqueries.orchestrator.multi_agent_orchestrator import MultiAgentOrchestrator

    sql_instance = MagicMock()

    async def gen(*_a: Any, **_kw: Any) -> Any:
        yield json.dumps(frame)

    sql_instance.handle_question = gen
    classified = MagicMock(intent=IntentType.sql, confidence=1.0)

    async def run() -> None:
        with (
            patch(
                "nlqueries.orchestrator.multi_agent_orchestrator.aclassify_intent",
                return_value=classified,
            ),
            patch(
                "nlqueries.orchestrator.multi_agent_orchestrator.Orchestrator",
                return_value=sql_instance,
            ),
            use_chosen_values(chosen or {}),
        ):
            async for _ in MultiAgentOrchestrator().handle_question(
                "students in schools", "agent1", available_types=["sql"]
            ):
                pass

    with patch("nlqueries.orchestrator.multi_agent_orchestrator._schedule_cache_write") as write:
        asyncio.run(run())
    return write


def test_an_answer_waiting_on_a_choice_is_not_cached() -> None:
    frame = {"type": "sql", "sql": _ASKED_SQL, "choices": [_CHOICE.as_dict()]}
    _through_the_router(frame).assert_not_called()


def test_an_answer_worked_out_with_a_chosen_value_is_not_cached() -> None:
    frame = {"type": "sql", "sql": "SELECT 1"}
    _through_the_router(frame, {_CHOICE.key: "Special Education Schools"}).assert_not_called()


def test_an_answer_with_nothing_to_choose_is_cached_as_before() -> None:
    _through_the_router({"type": "sql", "sql": "SELECT 1"}).assert_called_once()


def _lookups(chosen: dict[str, str] | None) -> MagicMock:
    """The semantic cache's reads for one question through the router."""
    cache = MagicMock()
    cache.get.return_value = None
    with patch("nlqueries.orchestrator.multi_agent_orchestrator.SemanticCache", return_value=cache):
        _through_the_router({"type": "sql", "sql": "SELECT 1"}, chosen)
    return cache.get


def test_a_question_asked_with_a_chosen_value_is_not_answered_from_the_cache() -> None:
    _lookups({_CHOICE.key: "Special Education Schools"}).assert_not_called()
    _lookups(None).assert_called_once()


def test_the_hybrid_route_gives_the_choice_back_and_merges_nothing() -> None:
    from nlqueries.orchestrator.intent_classifier import IntentType
    from nlqueries.orchestrator.multi_agent_orchestrator import MultiAgentOrchestrator

    sql_frame = {
        "type": "sql",
        "agent_type": "sql",
        "sql": _ASKED_SQL,
        "sql_table": None,
        "choices": [_CHOICE.as_dict()],
    }
    ran = AsyncMock(return_value=([json.dumps(sql_frame)], [], None, "docs"))
    classified = MagicMock(intent=IntentType.hybrid, confidence=1.0)

    async def run() -> list[str]:
        with (
            patch(
                "nlqueries.orchestrator.multi_agent_orchestrator.aclassify_intent",
                return_value=classified,
            ),
            patch("nlqueries.orchestrator.multi_agent_orchestrator._run_hybrid", new=ran),
            patch("nlqueries.orchestrator.multi_agent_orchestrator._merge_hybrid") as merge,
        ):
            frames = [
                tok
                async for tok in MultiAgentOrchestrator().handle_question(
                    "students in schools",
                    "agent1",
                    available_types=["sql", "document", "hybrid"],
                    use_cache=False,
                )
            ]
        merge.assert_not_called()
        return frames

    final = json.loads(asyncio.run(run())[-1])
    assert final["type"] == "hybrid"
    assert final["choices"] == [_CHOICE.as_dict()]
    assert final["merged_answer"] is None and final["sql"] == _ASKED_SQL


# ---------------------------------------------------------------------------
# The CLI and the MCP tool
# ---------------------------------------------------------------------------


def _result(choices: list[dict[str, Any]]) -> AgentQueryResult:
    return AgentQueryResult(
        question="students in schools",
        resolved_question="students in schools",
        agent_type="sql",
        answer="",
        sql=_ASKED_SQL,
        sql_result=None,
        sql_is_valid=True,
        execution_mode="execute_read_only",
        citations=[],
        merged_answer=None,
        latency_ms=1,
        session_id=None,
        choices=choices,
    )


def _cli(args: list[str], results: list[AgentQueryResult], interactive: bool = False) -> Any:
    from nlqueries.cli.main import cli

    runner = CliRunner()
    with (
        patch("nlqueries.orchestrator.sync_runner.run_query_sync", side_effect=results) as run,
        patch("nlqueries.orchestrator.sync_runner.answer_types_for", return_value=("sql",)),
        patch("nlqueries.cli.main._interactive", return_value=interactive),
    ):
        out = runner.invoke(
            cli,
            [
                "query",
                "agent1",
                "students in schools",
                "--dialect",
                "sqlite",
                "--no-session",
                *args,
            ],
            input="2\n" if interactive else None,
        )
    return out, run


def test_the_cli_passes_chosen_values_to_the_query() -> None:
    out, run = _cli(["--choose", _CHOICE.key, "Special Education Schools"], [_result([])])
    assert out.exit_code == 0, out.output
    assert run.call_args.kwargs["chosen_values"] == {_CHOICE.key: "Special Education Schools"}


def test_the_cli_says_what_to_choose_when_no_one_is_at_the_terminal() -> None:
    out, run = _cli([], [_result([_CHOICE.as_dict()])])
    assert out.exit_code == 0, out.output
    assert run.call_count == 1
    assert "Choose" in out.output and "Juvenile Court Schools" in out.output
    # Rich wraps at the terminal's width: compared with the line breaks taken out.
    assert "--choose 'schools.soc:Schools' \"<value>\"" in " ".join(out.output.split())


def test_the_cli_asks_at_the_terminal_and_answers_with_the_choice() -> None:
    out, run = _cli([], [_result([_CHOICE.as_dict()]), _result([])], interactive=True)
    assert out.exit_code == 0, out.output
    assert run.call_count == 2
    assert run.call_args.kwargs["chosen_values"] == {_CHOICE.key: "Special Education Schools"}
    assert "Which did you mean?" in out.output


def test_the_cli_json_carries_the_choices() -> None:
    out, _run = _cli(["--json"], [_result([_CHOICE.as_dict()])])
    assert json.loads(out.output)["choices"] == [_CHOICE.as_dict()]


def test_the_mcp_tool_gives_the_key_and_passes_choices_back() -> None:
    from nlqueries.mcp_server import server

    with (
        patch(
            "nlqueries.orchestrator.sync_runner.run_query",
            new=AsyncMock(return_value=_result([_CHOICE.as_dict()])),
        ) as run,
        patch("nlqueries.connectors.loader.dialect_for_agent", return_value="sqlite"),
        patch("nlqueries.orchestrator.sync_runner.answer_types_for", return_value=("sql",)),
    ):
        reply = asyncio.run(
            server.query("students in schools", "agent1", choices={_CHOICE.key: "x"})
        )
    assert run.call_args.kwargs["chosen_values"] == {_CHOICE.key: "x"}
    assert "**A value needs choosing**" in reply
    assert "Key: 'schools.soc:Schools'" in reply and "'Special Education Schools'" in reply
