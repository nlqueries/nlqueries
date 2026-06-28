"""Tests for nlqueries.processing.intent_annotator (Task 4.1.2)."""

from __future__ import annotations

import threading
from collections.abc import Iterator

import pytest
from nlqueries.llm.client import LLMClient
from nlqueries.processing.intent_annotator import (
    _BATCH_SIZE,
    _MAX_CONCURRENT,
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
# annotate_capsules — concurrent batch processing
# ---------------------------------------------------------------------------


def test_annotate_capsules_annotates_all_20_capsules():
    """All 20 capsules must receive a non-empty intent."""
    capsules = [_make_capsule(i) for i in range(20)]
    annotate_capsules(capsules, _FixedLLMClient("What is the order count by status?"))
    assert all(c.intent != "" for c in capsules)


def test_annotate_capsules_all_intents_non_empty():
    capsules = [_make_capsule(i) for i in range(20)]
    annotate_capsules(capsules, _FixedLLMClient("intent"))
    assert all(c.intent for c in capsules)


def test_annotate_capsules_returns_same_list():
    capsules = [_make_capsule(i) for i in range(20)]
    result = annotate_capsules(capsules, _FixedLLMClient())
    assert result is capsules
    assert len(result) == 20


def test_annotate_capsules_default_batch_size_constant_is_10():
    assert _BATCH_SIZE == 10


def test_annotate_capsules_empty_list_returns_empty():
    result = annotate_capsules([], _FixedLLMClient())
    assert result == []


def test_annotate_capsules_single_capsule():
    capsule = _make_capsule(0)
    result = annotate_capsules([capsule], _FixedLLMClient("single"))
    assert len(result) == 1
    assert result[0].intent == "single"


@pytest.mark.parametrize("n_capsules", [1, 10, 11, 20, 25])
def test_annotate_capsules_parametrized_counts(n_capsules: int):
    capsules = [_make_capsule(i) for i in range(n_capsules)]
    result = annotate_capsules(capsules, _FixedLLMClient("ok"))
    assert all(c.intent == "ok" for c in result)
    assert len(result) == n_capsules


def test_annotate_capsules_runs_concurrently():
    """At least _MAX_CONCURRENT threads must be active simultaneously."""
    barrier = threading.Barrier(_MAX_CONCURRENT, timeout=5)

    class _BarrierClient(LLMClient):
        def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
            barrier.wait()
            return "intent"

        def stream(self, system: str, user: str) -> Iterator[str]:
            yield "intent"

    capsules = [_make_capsule(i) for i in range(_MAX_CONCURRENT)]
    annotate_capsules(capsules, _BarrierClient())
    assert all(c.intent == "intent" for c in capsules)


def test_annotate_capsules_error_propagates():
    """An exception raised by one capsule must bubble out of annotate_capsules."""

    class _RaisingClient(LLMClient):
        def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
            raise ValueError("LLM down")

        def stream(self, system: str, user: str) -> Iterator[str]:
            yield ""

    with pytest.raises(ValueError, match="LLM down"):
        annotate_capsules([_make_capsule(0)], _RaisingClient())
