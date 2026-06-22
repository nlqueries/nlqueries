"""Tests for nlqueries.orchestrator.sync_runner."""

from __future__ import annotations

import asyncio
import json
import threading
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
