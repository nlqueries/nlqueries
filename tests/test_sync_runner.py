"""Tests for nlqueries.orchestrator.sync_runner."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncGenerator
from unittest.mock import MagicMock, patch

from nlqueries.orchestrator.followup_resolver import ResolvedQuestion
from nlqueries.orchestrator.sync_runner import AgentQueryResult, run_query, run_query_sync

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_TEXT_TOKENS = [
    "Token one. ",
    "Token two. ",
    "Token three. ",
    "Token four. ",
    "Token five. ",
]

_SQL_FINAL_CHUNK = json.dumps(
    {
        "type": "sql",
        "sql": "SELECT count(*) FROM orders",
        "is_valid": True,
        "validation_error": None,
        "dialect": "postgres",
        "agent_type": "sql",
    }
)

_DOC_FINAL_CHUNK = json.dumps(
    {
        "type": "citations",
        "citations": [
            {
                "source_name": "policy.pdf",
                "page_number": 3,
                "excerpt": "Full refund within 30 days.",
            }
        ],
        "agent_type": "document",
    }
)

_NO_FOLLOWUP = ResolvedQuestion(
    original="How many orders?",
    resolved="How many orders?",
    is_followup=False,
    reasoning="No follow-up references detected.",
)


def _async_gen_factory(tokens: list[str]):  # type: ignore[return]
    """Return a coroutine function whose call yields *tokens* as an async gen."""

    async def _gen(*_args: object, **_kwargs: object):  # type: ignore[return]
        for t in tokens:
            yield t

    return _gen


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunQuery:
    """Tests for the async run_query function."""

    def test_run_query_collects_all_tokens(self) -> None:
        """run_query joins all text tokens into answer; final chunk is excluded."""
        all_tokens = _TEXT_TOKENS + [_SQL_FINAL_CHUNK]

        mock_orch = MagicMock()
        mock_orch.handle_question = _async_gen_factory(all_tokens)

        with (
            patch(
                "nlqueries.orchestrator.sync_runner.MultiAgentOrchestrator",
                return_value=mock_orch,
            ),
            patch(
                "nlqueries.orchestrator.sync_runner.resolve_followup",
                return_value=_NO_FOLLOWUP,
            ),
        ):
            result = asyncio.run(run_query("How many orders?", "agent1"))

        expected_answer = "".join(_TEXT_TOKENS)
        assert result.answer == expected_answer
        assert result.agent_type == "sql"

    def test_run_query_extracts_sql_from_final_chunk(self) -> None:
        """SQL from the final chunk is extracted into result.sql."""
        all_tokens = ["Some reasoning. ", _SQL_FINAL_CHUNK]

        mock_orch = MagicMock()
        mock_orch.handle_question = _async_gen_factory(all_tokens)

        resolved = ResolvedQuestion(
            original="Order count?",
            resolved="Order count?",
            is_followup=False,
            reasoning="No follow-up references detected.",
        )

        with (
            patch(
                "nlqueries.orchestrator.sync_runner.MultiAgentOrchestrator",
                return_value=mock_orch,
            ),
            patch(
                "nlqueries.orchestrator.sync_runner.resolve_followup",
                return_value=resolved,
            ),
        ):
            result = asyncio.run(run_query("Order count?", "agent1"))

        assert result.sql == "SELECT count(*) FROM orders"
        assert result.agent_type == "sql"
        assert result.citations == []
        assert result.merged_answer is None

    def test_latency_ms_is_positive(self) -> None:
        """latency_ms must be a non-negative integer."""
        mock_orch = MagicMock()
        mock_orch.handle_question = _async_gen_factory(["Thinking. ", _SQL_FINAL_CHUNK])

        resolved = ResolvedQuestion(
            original="q?",
            resolved="q?",
            is_followup=False,
            reasoning="No follow-up references detected.",
        )

        with (
            patch(
                "nlqueries.orchestrator.sync_runner.MultiAgentOrchestrator",
                return_value=mock_orch,
            ),
            patch(
                "nlqueries.orchestrator.sync_runner.resolve_followup",
                return_value=resolved,
            ),
        ):
            result = asyncio.run(run_query("q?", "agent1"))

        assert isinstance(result.latency_ms, int)
        assert result.latency_ms >= 0

    def test_run_query_sync_works_without_event_loop(self) -> None:
        """run_query_sync works when called from a thread with no running event loop."""
        results: list[AgentQueryResult] = []
        errors: list[Exception] = []

        def _thread_target() -> None:
            local_orch = MagicMock()
            local_orch.handle_question = _async_gen_factory(["Answer. ", _SQL_FINAL_CHUNK])
            resolved = ResolvedQuestion(
                original="test?",
                resolved="test?",
                is_followup=False,
                reasoning="No follow-up references detected.",
            )
            try:
                with (
                    patch(
                        "nlqueries.orchestrator.sync_runner.MultiAgentOrchestrator",
                        return_value=local_orch,
                    ),
                    patch(
                        "nlqueries.orchestrator.sync_runner.resolve_followup",
                        return_value=resolved,
                    ),
                ):
                    results.append(run_query_sync("test?", "agent1"))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        t = threading.Thread(target=_thread_target)
        t.start()
        t.join()

        assert not errors, f"Thread failed: {errors[0]}"
        assert len(results) == 1
        assert isinstance(results[0], AgentQueryResult)
        assert results[0].agent_type == "sql"


class TestExtraDynamicContext:
    """The enterprise Nexus injection seam (Task: core gap)."""

    def test_run_query_threads_extra_dynamic_context(self) -> None:
        captured: dict[str, object] = {}

        def _capturing(*_args: object, **kwargs: object) -> AsyncGenerator[str, None]:
            captured.update(kwargs)

            async def _gen() -> AsyncGenerator[str, None]:
                yield _SQL_FINAL_CHUNK

            return _gen()

        mock_orch = MagicMock()
        mock_orch.handle_question = _capturing

        with (
            patch(
                "nlqueries.orchestrator.sync_runner.MultiAgentOrchestrator",
                return_value=mock_orch,
            ),
            patch(
                "nlqueries.orchestrator.sync_runner.resolve_followup",
                return_value=_NO_FOLLOWUP,
            ),
        ):
            result = asyncio.run(
                run_query("How many orders?", "agent1", extra_dynamic_context="NEXUS-SECTION")
            )

        assert captured.get("extra_dynamic_context") == "NEXUS-SECTION"
        assert result.nexus_warnings == []  # defaults empty; core never populates it

    def test_nexus_warnings_defaults_empty(self) -> None:
        mock_orch = MagicMock()
        mock_orch.handle_question = _async_gen_factory([_SQL_FINAL_CHUNK])
        with (
            patch(
                "nlqueries.orchestrator.sync_runner.MultiAgentOrchestrator",
                return_value=mock_orch,
            ),
            patch(
                "nlqueries.orchestrator.sync_runner.resolve_followup",
                return_value=_NO_FOLLOWUP,
            ),
        ):
            result = asyncio.run(run_query("q?", "agent1"))
        assert result.nexus_warnings == []


class TestTruncationSurvivesTheChunk:
    """The `sql_table` frame is JSON on the way out and a QueryResult on the way
    back in, and `truncated`/`truncation_reason` were dropped at both ends. Every
    caller therefore saw the dataclass defaults — `False` and `None` — including
    for results the orchestrator had itself shortened at `_MAX_RESULT_ROWS`.

    Testing the builder alone would not have caught that: a frame can carry both
    keys perfectly and still lose them here. This is the seam.
    """

    @staticmethod
    def _run(sql_table: dict[str, object]) -> AgentQueryResult:
        chunk = json.dumps(
            {
                "type": "sql",
                "sql": "SELECT n FROM t",
                "is_valid": True,
                "validation_error": None,
                "dialect": "postgres",
                "agent_type": "sql",
                "sql_table": sql_table,
            }
        )
        mock_orch = MagicMock()
        mock_orch.handle_question = _async_gen_factory(["Reasoning. ", chunk])
        resolved = ResolvedQuestion(original="q", resolved="q", is_followup=False, reasoning="")
        with (
            patch(
                "nlqueries.orchestrator.sync_runner.MultiAgentOrchestrator",
                return_value=mock_orch,
            ),
            patch("nlqueries.orchestrator.sync_runner.resolve_followup", return_value=resolved),
        ):
            return asyncio.run(run_query("q", "agent1"))

    _COMPLETE: dict[str, object] = {
        "columns": ["n"],
        "rows": [[1]],
        "row_count": 1,
        "execution_time_ms": 1.0,
        "error": None,
        "truncated": False,
        "truncation_reason": None,
    }

    def test_a_complete_result_arrives_complete(self) -> None:
        """Canary: without it, a parser hardcoding `truncated=True` would satisfy
        the test below."""
        result = self._run(dict(self._COMPLETE))
        assert result.sql_result is not None
        assert result.sql_result.truncated is False
        assert result.sql_result.truncation_reason is None

    def test_truncation_arrives_with_its_reason(self) -> None:
        result = self._run(
            {**self._COMPLETE, "truncated": True, "truncation_reason": "orchestrator_row_cap"}
        )
        assert result.sql_result is not None
        assert result.sql_result.truncated is True
        assert result.sql_result.truncation_reason == "orchestrator_row_cap"

    def test_an_older_frame_without_the_keys_is_read_as_complete(self) -> None:
        """A frame written before this change carries neither key. Reading them
        with `.get` keeps that working rather than raising — the values are
        wrong, but they were wrong before too, and a cached entry must not break
        a live query."""
        older = {k: v for k, v in self._COMPLETE.items() if not k.startswith("trunc")}
        result = self._run(older)
        assert result.sql_result is not None
        assert result.sql_result.truncated is False
