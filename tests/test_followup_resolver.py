"""Tests for nlqueries.orchestrator.followup_resolver."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nlqueries.orchestrator.conversation import ConversationTurn, create_session
from nlqueries.orchestrator.followup_resolver import (
    ResolvedQuestion,
    aresolve_followup,
    resolve_followup,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_history(pairs: list[tuple[str, str]]) -> list[ConversationTurn]:
    """Build a flat list of ConversationTurns from (role, content) pairs."""
    session = create_session("test-agent")
    for role, content in pairs:
        session.add_turn(role, content)
    return session.turns


def _mock_llm(response_dict: dict) -> MagicMock:  # type: ignore[type-arg]
    """Return a mock LLMClient whose complete() returns *response_dict* as JSON."""
    mock = MagicMock()
    mock.complete.return_value = json.dumps(response_dict)
    return mock


def _failing_llm() -> MagicMock:
    """Return a mock LLMClient whose complete() raises an exception."""
    mock = MagicMock()
    mock.complete.side_effect = Exception("LLM unavailable")
    return mock


# ---------------------------------------------------------------------------
# test_no_signal_skips_llm
# ---------------------------------------------------------------------------


def test_no_signal_skips_llm() -> None:
    """A question with no follow-up signals must not invoke the LLM at all."""
    history = _make_history(
        [
            ("user", "Show me orders by region"),
            ("assistant", "Here are the orders grouped by region."),
        ]
    )

    with patch("nlqueries.orchestrator.followup_resolver.get_llm_client") as mock_get_llm:
        result = resolve_followup("How many orders last month?", history)

    mock_get_llm.assert_not_called()
    assert isinstance(result, ResolvedQuestion)
    assert result.is_followup is False
    assert result.resolved == "How many orders last month?"
    assert result.original == "How many orders last month?"
    assert result.reasoning == "No follow-up references detected."


# ---------------------------------------------------------------------------
# test_pronoun_resolved_with_history
# ---------------------------------------------------------------------------


def test_pronoun_resolved_with_history() -> None:
    """A question with a follow-up signal and non-empty history should be resolved."""
    history = _make_history(
        [
            ("user", "Show orders by region"),
            ("assistant", "Here are the orders grouped by region."),
        ]
    )

    mock_llm = _mock_llm(
        {
            "resolved": "Filter orders by region to North America",
            "reasoning": "'those' refers to the orders by region from the previous answer.",
        }
    )

    with patch(
        "nlqueries.orchestrator.followup_resolver.get_llm_client",
        return_value=mock_llm,
    ):
        result = resolve_followup("filter those to North America", history)

    assert result.is_followup is True
    assert result.resolved == "Filter orders by region to North America"
    assert result.original == "filter those to North America"
    assert len(result.reasoning) > 0
    mock_llm.complete.assert_called_once()


# ---------------------------------------------------------------------------
# test_empty_history_returns_original_unchanged
# ---------------------------------------------------------------------------


def test_empty_history_returns_original_unchanged() -> None:
    """When history is empty, follow-up signals must not trigger an LLM call."""
    with patch("nlqueries.orchestrator.followup_resolver.get_llm_client") as mock_get_llm:
        result = resolve_followup("filter those to North America", [])

    mock_get_llm.assert_not_called()
    assert result.is_followup is False
    assert result.resolved == "filter those to North America"
    assert result.original == "filter those to North America"


# ---------------------------------------------------------------------------
# test_llm_failure_returns_original
# ---------------------------------------------------------------------------


def test_llm_failure_returns_original() -> None:
    """When the LLM raises an exception, return the original question unchanged."""
    history = _make_history(
        [
            ("user", "Show orders by region"),
            ("assistant", "Here are the orders grouped by region."),
        ]
    )

    with patch(
        "nlqueries.orchestrator.followup_resolver.get_llm_client",
        return_value=_failing_llm(),
    ):
        result = resolve_followup("filter those to North America", history)

    assert result.is_followup is False
    assert result.resolved == "filter those to North America"
    assert result.original == "filter those to North America"


def test_uses_fast_tier_client() -> None:
    """resolve_followup must request the fast-tier LLM client."""
    history = _make_history(
        [
            ("user", "Show orders by region"),
            ("assistant", "Here are the orders grouped by region."),
        ]
    )
    mock_llm = _mock_llm(
        {"resolved": "Filter orders by region to North America", "reasoning": "resolved."}
    )

    with patch(
        "nlqueries.orchestrator.followup_resolver.get_llm_client",
        return_value=mock_llm,
    ) as mock_get_client:
        resolve_followup("filter those to North America", history)

    mock_get_client.assert_called_once_with(tier="fast")


def test_complete_called_with_max_tokens_200() -> None:
    """resolve_followup must cap the completion at 200 tokens."""
    history = _make_history(
        [
            ("user", "Show orders by region"),
            ("assistant", "Here are the orders grouped by region."),
        ]
    )
    mock_llm = _mock_llm(
        {"resolved": "Filter orders by region to North America", "reasoning": "resolved."}
    )

    with patch(
        "nlqueries.orchestrator.followup_resolver.get_llm_client",
        return_value=mock_llm,
    ):
        resolve_followup("filter those to North America", history)

    _, call_kwargs = mock_llm.complete.call_args
    assert call_kwargs.get("max_tokens") == 200


# ---------------------------------------------------------------------------
# The async variant (W-3)
#
# resolve_followup is a full LLM round trip, typically one to three seconds.
# Called bare inside an async generator it froze every other request on the
# worker for that whole time, which is why an async twin exists. The twin is
# only worth having if it answers identically, so that is what is asserted.
# ---------------------------------------------------------------------------


def _async_mock_llm(response_dict: dict) -> MagicMock:  # type: ignore[type-arg]
    """A mock LLMClient whose acomplete() returns *response_dict* as JSON."""
    mock = MagicMock()
    mock.acomplete = AsyncMock(return_value=json.dumps(response_dict))
    return mock


_IDENTICAL_CASES = [
    # (question, history_pairs, llm_response)
    (
        "How many orders last month?",
        [("user", "Show me orders by region"), ("assistant", "Here they are.")],
        None,  # no follow-up signal — never reaches the LLM
    ),
    (
        "filter those to North America",
        [],
        None,  # signal, but no history to resolve against
    ),
    (
        "filter those to North America",
        [("user", "Show orders by region"), ("assistant", "Here they are.")],
        {"resolved": "Filter orders by region to North America", "reasoning": "'those' refers."},
    ),
    (
        "what about them",
        [("user", "Top customers"), ("assistant", "Acme, Globex.")],
        {"resolved": "what about them", "reasoning": "Nothing to change."},
    ),
]


@pytest.mark.parametrize(("question", "pairs", "response"), _IDENTICAL_CASES)
def test_sync_and_async_resolve_identically(question, pairs, response) -> None:
    """Two implementations of one behaviour is a drift risk; this is the check
    that keeps them honest."""
    history = _make_history(pairs) if pairs else []

    with patch(
        "nlqueries.orchestrator.followup_resolver.get_llm_client",
        return_value=_mock_llm(response) if response else MagicMock(),
    ):
        sync_result = resolve_followup(question, history)

    with patch(
        "nlqueries.orchestrator.followup_resolver.get_llm_client",
        return_value=_async_mock_llm(response) if response else MagicMock(),
    ):
        async_result = asyncio.run(aresolve_followup(question, history))

    assert async_result == sync_result


def test_async_resolution_failure_returns_the_original_question() -> None:
    """Fail open, exactly as the sync path does: an unresolvable follow-up is
    still answerable as the user asked it."""
    history = _make_history([("user", "Show orders"), ("assistant", "Here.")])
    mock = MagicMock()
    mock.acomplete = AsyncMock(side_effect=Exception("LLM unavailable"))

    with patch("nlqueries.orchestrator.followup_resolver.get_llm_client", return_value=mock):
        result = asyncio.run(aresolve_followup("filter those to Europe", history))

    assert result.is_followup is False
    assert result.resolved == "filter those to Europe"


def test_async_resolution_does_not_block_the_event_loop() -> None:
    """The reason the async variant exists at all."""
    import time

    history = _make_history([("user", "Show orders"), ("assistant", "Here.")])

    async def _slow_acomplete(*_a: object, **_kw: object) -> str:
        await asyncio.sleep(0.3)
        return json.dumps({"resolved": "resolved question", "reasoning": "ok"})

    mock = MagicMock()
    mock.acomplete = _slow_acomplete

    async def run() -> float:
        started = time.monotonic()
        await asyncio.gather(
            aresolve_followup("filter those to Europe", history),
            aresolve_followup("what about those", history),
        )
        return time.monotonic() - started

    with patch("nlqueries.orchestrator.followup_resolver.get_llm_client", return_value=mock):
        elapsed = asyncio.run(run())

    assert elapsed < 0.5, f"two concurrent 300ms resolutions took {elapsed:.3f}s"
