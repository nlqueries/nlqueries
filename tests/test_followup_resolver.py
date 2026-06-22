"""Tests for nlqueries.orchestrator.followup_resolver."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from nlqueries.orchestrator.conversation import ConversationTurn, create_session
from nlqueries.orchestrator.followup_resolver import ResolvedQuestion, resolve_followup

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
