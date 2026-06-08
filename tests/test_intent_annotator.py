"""Tests for nlqueries.processing.intent_annotator (Task 4.1.2)."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import patch

import pytest
from nlqueries.llm.client import LLMClient
from nlqueries.processing.intent_annotator import (
    _BATCH_DELAY,
    _BATCH_SIZE,
    _MAX_INTENT_LENGTH,
    _SYSTEM_PROMPT,
    annotate_capsule,
    annotate_capsules,
)
from nlqueries.processing.parameterizer import Placeholder, QueryCapsule

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_capsule(index: int = 0) -> QueryCapsule:
    return QueryCapsule(
        template_sql=f"SELECT id FROM orders WHERE status = '[status:VARCHAR]' AND id > {index}",
        placeholders=[Placeholder(name="status", type="VARCHAR")],
        tables=["orders"],
        columns=["id", "status"],
        frequency=index + 1,
        auto_description=f"Query on orders filtering by status (capsule {index})",
        intent="",
    )


class _FixedLLMClient(LLMClient):
    """Returns the same fixed string for every complete() call."""

    def __init__(self, response: str = "What are the orders for a given status?") -> None:
        self._response = response

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        return self._response

    def stream(self, system: str, user: str) -> Iterator[str]:
        yield self._response


class _RecordingLLMClient(LLMClient):
    """Records every complete() call for later assertion."""

    def __init__(self, response: str = "mocked intent") -> None:
        self._response = response
        self.calls: list[dict[str, str]] = []

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        self.calls.append({"system": system, "user": user})
        return self._response

    def stream(self, system: str, user: str) -> Iterator[str]:
        yield self._response


# ---------------------------------------------------------------------------
# annotate_capsule — single-capsule behaviour
# ---------------------------------------------------------------------------


def test_annotate_capsule_sets_intent():
    capsule = _make_capsule(0)
    assert capsule.intent == ""
    annotate_capsule(capsule, _FixedLLMClient("Show me active orders."))
    assert capsule.intent == "Show me active orders."


def test_annotate_capsule_returns_same_capsule():
    capsule = _make_capsule(0)
    returned = annotate_capsule(capsule, _FixedLLMClient())
    assert returned is capsule


def test_annotate_capsule_truncates_long_response():
    long_response = "x" * 300
    capsule = _make_capsule(0)
    annotate_capsule(capsule, _FixedLLMClient(long_response))
    assert len(capsule.intent) == _MAX_INTENT_LENGTH
    assert capsule.intent == "x" * _MAX_INTENT_LENGTH


def test_annotate_capsule_exact_200_chars_not_truncated():
    response = "a" * _MAX_INTENT_LENGTH
    capsule = _make_capsule(0)
    annotate_capsule(capsule, _FixedLLMClient(response))
    assert capsule.intent == response
    assert len(capsule.intent) == _MAX_INTENT_LENGTH


def test_annotate_capsule_passes_correct_system_prompt():
    recorder = _RecordingLLMClient()
    annotate_capsule(_make_capsule(0), recorder)
    assert recorder.calls[0]["system"] == _SYSTEM_PROMPT


def test_annotate_capsule_includes_template_sql_in_user_message():
    capsule = _make_capsule(0)
    recorder = _RecordingLLMClient()
    annotate_capsule(capsule, recorder)
    assert capsule.template_sql in recorder.calls[0]["user"]


def test_annotate_capsule_includes_tables_in_user_message():
    capsule = _make_capsule(0)
    recorder = _RecordingLLMClient()
    annotate_capsule(capsule, recorder)
    assert "orders" in recorder.calls[0]["user"]


def test_annotate_capsule_handles_empty_tables_list():
    capsule = _make_capsule(0)
    capsule.tables = []
    recorder = _RecordingLLMClient("ok")
    annotate_capsule(capsule, recorder)
    assert "unknown" in recorder.calls[0]["user"]
    assert capsule.intent == "ok"


# ---------------------------------------------------------------------------
# annotate_capsules — batch processing
# ---------------------------------------------------------------------------


def test_annotate_capsules_annotates_all_20_capsules():
    """Spec requirement: all 20 capsules must receive a non-empty intent."""
    capsules = [_make_capsule(i) for i in range(20)]
    llm = _FixedLLMClient("What is the order count by status?")

    with patch("nlqueries.processing.intent_annotator.time.sleep"):
        annotate_capsules(capsules, llm)

    assert all(c.intent != "" for c in capsules)


def test_annotate_capsules_all_intents_non_empty():
    capsules = [_make_capsule(i) for i in range(20)]
    with patch("nlqueries.processing.intent_annotator.time.sleep"):
        annotate_capsules(capsules, _FixedLLMClient("intent"))
    assert all(c.intent for c in capsules)


def test_annotate_capsules_returns_all_capsules():
    capsules = [_make_capsule(i) for i in range(20)]
    with patch("nlqueries.processing.intent_annotator.time.sleep"):
        result = annotate_capsules(capsules, _FixedLLMClient())
    assert result is capsules
    assert len(result) == 20


def test_annotate_capsules_sleeps_between_batches_not_after_last():
    """With 20 capsules and batch_size=10 there must be exactly 1 sleep."""
    capsules = [_make_capsule(i) for i in range(20)]
    with patch("nlqueries.processing.intent_annotator.time.sleep") as mock_sleep:
        annotate_capsules(capsules, _FixedLLMClient(), batch_size=10)
    assert mock_sleep.call_count == 1
    mock_sleep.assert_called_once_with(_BATCH_DELAY)


def test_annotate_capsules_sleep_count_matches_batch_count_minus_one():
    """k batches → k-1 sleeps."""
    capsules = [_make_capsule(i) for i in range(20)]
    with patch("nlqueries.processing.intent_annotator.time.sleep") as mock_sleep:
        annotate_capsules(capsules, _FixedLLMClient(), batch_size=5)
    # 20 capsules / 5 per batch = 4 batches → 3 sleeps
    assert mock_sleep.call_count == 3


def test_annotate_capsules_no_sleep_when_single_batch():
    capsules = [_make_capsule(i) for i in range(5)]
    with patch("nlqueries.processing.intent_annotator.time.sleep") as mock_sleep:
        annotate_capsules(capsules, _FixedLLMClient(), batch_size=10)
    assert mock_sleep.call_count == 0


def test_annotate_capsules_default_batch_size_is_10():
    assert _BATCH_SIZE == 10


def test_annotate_capsules_empty_list_returns_empty():
    result = annotate_capsules([], _FixedLLMClient())
    assert result == []


def test_annotate_capsules_single_capsule():
    capsule = _make_capsule(0)
    with patch("nlqueries.processing.intent_annotator.time.sleep"):
        result = annotate_capsules([capsule], _FixedLLMClient("single"))
    assert len(result) == 1
    assert result[0].intent == "single"


@pytest.mark.parametrize("n_capsules", [1, 10, 11, 20, 25])
def test_annotate_capsules_parametrized_counts(n_capsules: int):
    capsules = [_make_capsule(i) for i in range(n_capsules)]
    with patch("nlqueries.processing.intent_annotator.time.sleep"):
        result = annotate_capsules(capsules, _FixedLLMClient("ok"), batch_size=10)
    assert all(c.intent == "ok" for c in result)
    assert len(result) == n_capsules
