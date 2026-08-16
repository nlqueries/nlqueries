"""Tests for nlqueries.orchestrator.intent_classifier."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nlqueries.orchestrator.intent_classifier import (
    IntentClassificationResult,
    IntentType,
    aclassify_intent,
    classify_intent,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_llm(response_dict: dict) -> MagicMock:  # type: ignore[type-arg]
    """Return a mock LLMClient whose complete() returns *response_dict* as JSON."""
    mock = MagicMock()
    mock.complete.return_value = json.dumps(response_dict)
    return mock


# ---------------------------------------------------------------------------
# IntentType enum
# ---------------------------------------------------------------------------


class TestIntentType:
    def test_values_are_strings(self) -> None:
        assert IntentType.sql == "sql"
        assert IntentType.document == "document"
        assert IntentType.hybrid == "hybrid"
        assert IntentType.unclear == "unclear"

    def test_all_four_members_defined(self) -> None:
        members = {m.value for m in IntentType}
        assert members == {"sql", "document", "hybrid", "unclear"}


# ---------------------------------------------------------------------------
# Required spec tests
# ---------------------------------------------------------------------------


class TestClassifyIntentSpec:
    """The three tests explicitly required by the Task 12.1 spec."""

    def test_sql_question_classified_as_sql(self) -> None:
        """mock returns sql intent; assert result is IntentType.sql."""
        mock_llm = _mock_llm(
            {"intent": "sql", "confidence": 0.95, "reasoning": "Database count query."}
        )
        with patch(
            "nlqueries.orchestrator.intent_classifier.get_llm_client",
            return_value=mock_llm,
        ):
            result = classify_intent(
                "How many orders did we get last month?",
                available_agent_types=["sql", "document"],
            )

        assert result.intent == IntentType.sql
        assert result.confidence == pytest.approx(0.95)
        assert len(result.reasoning) > 0

    def test_malformed_llm_response_falls_back_to_unclear(self) -> None:
        """Non-JSON LLM response falls back to IntentType.unclear with 0.0 confidence."""
        mock_llm = MagicMock()
        mock_llm.complete.return_value = "not valid json at all"

        with patch(
            "nlqueries.orchestrator.intent_classifier.get_llm_client",
            return_value=mock_llm,
        ):
            result = classify_intent(
                "What is the meaning of life?",
                available_agent_types=["sql", "document"],
            )

        assert result.intent == IntentType.unclear
        assert result.confidence == 0.0

    def test_only_sql_available_overrides_hybrid(self) -> None:
        """When available_agent_types=['sql'] and LLM says hybrid, coerce to sql."""
        mock_llm = _mock_llm(
            {
                "intent": "hybrid",
                "confidence": 0.82,
                "reasoning": "Needs both DB and documents.",
            }
        )
        with patch(
            "nlqueries.orchestrator.intent_classifier.get_llm_client",
            return_value=mock_llm,
        ):
            result = classify_intent(
                "Which customers signed the enterprise contract and haven't ordered yet?",
                available_agent_types=["sql"],
            )

        assert result.intent == IntentType.sql
        assert result.confidence == pytest.approx(0.82)


# ---------------------------------------------------------------------------
# Additional coverage tests
# ---------------------------------------------------------------------------


class TestClassifyIntentExtra:
    def test_document_question_classified_as_document(self) -> None:
        """Document question returns IntentType.document."""
        mock_llm = _mock_llm(
            {
                "intent": "document",
                "confidence": 0.93,
                "reasoning": "Refund policy is document content.",
            }
        )
        with patch(
            "nlqueries.orchestrator.intent_classifier.get_llm_client",
            return_value=mock_llm,
        ):
            result = classify_intent(
                "What does our refund policy say?",
                available_agent_types=["sql", "document"],
            )

        assert result.intent == IntentType.document
        assert result.confidence == pytest.approx(0.93)

    def test_result_is_intent_classification_result_instance(self) -> None:
        """classify_intent always returns an IntentClassificationResult."""
        mock_llm = _mock_llm({"intent": "sql", "confidence": 0.9, "reasoning": "Count query."})
        with patch(
            "nlqueries.orchestrator.intent_classifier.get_llm_client",
            return_value=mock_llm,
        ):
            result = classify_intent("How many users?", available_agent_types=["sql"])

        assert isinstance(result, IntentClassificationResult)
        assert isinstance(result.intent, IntentType)
        assert isinstance(result.confidence, float)
        assert isinstance(result.reasoning, str)

    def test_missing_intent_key_falls_back_to_unclear(self) -> None:
        """JSON missing 'intent' key falls back to unclear."""
        mock_llm = MagicMock()
        mock_llm.complete.return_value = json.dumps(
            {"confidence": 0.5, "reasoning": "Missing key."}
        )
        with patch(
            "nlqueries.orchestrator.intent_classifier.get_llm_client",
            return_value=mock_llm,
        ):
            result = classify_intent("ambiguous question", available_agent_types=["sql"])

        assert result.intent == IntentType.unclear
        assert result.confidence == 0.0

    def test_unknown_intent_value_falls_back_to_unclear(self) -> None:
        """JSON with an unrecognised intent value falls back to unclear."""
        mock_llm = MagicMock()
        mock_llm.complete.return_value = json.dumps(
            {"intent": "unknown_type", "confidence": 0.7, "reasoning": "Some reasoning."}
        )
        with patch(
            "nlqueries.orchestrator.intent_classifier.get_llm_client",
            return_value=mock_llm,
        ):
            result = classify_intent("some question", available_agent_types=["sql"])

        assert result.intent == IntentType.unclear

    def test_only_document_available_overrides_hybrid(self) -> None:
        """When only 'document' is available and LLM says hybrid, coerce to document."""
        mock_llm = _mock_llm(
            {"intent": "hybrid", "confidence": 0.78, "reasoning": "Needs DB and docs."}
        )
        with patch(
            "nlqueries.orchestrator.intent_classifier.get_llm_client",
            return_value=mock_llm,
        ):
            result = classify_intent(
                "Summarise contract terms for our top customers",
                available_agent_types=["document"],
            )

        assert result.intent == IntentType.document

    def test_hybrid_passes_through_when_in_available_types(self) -> None:
        """hybrid intent is preserved when 'hybrid' is in available_agent_types."""
        mock_llm = _mock_llm(
            {
                "intent": "hybrid",
                "confidence": 0.85,
                "reasoning": "Needs both structured data and document context.",
            }
        )
        with patch(
            "nlqueries.orchestrator.intent_classifier.get_llm_client",
            return_value=mock_llm,
        ):
            result = classify_intent(
                "Which customers from Q3 report haven't placed an order?",
                available_agent_types=["sql", "document", "hybrid"],
            )

        assert result.intent == IntentType.hybrid

    def test_unclear_passes_through_regardless_of_available_types(self) -> None:
        """unclear intent passes through even if 'unclear' is not in available_agent_types."""
        mock_llm = _mock_llm(
            {"intent": "unclear", "confidence": 0.3, "reasoning": "Cannot determine."}
        )
        with patch(
            "nlqueries.orchestrator.intent_classifier.get_llm_client",
            return_value=mock_llm,
        ):
            result = classify_intent("...", available_agent_types=["sql"])

        assert result.intent == IntentType.unclear

    def test_llm_complete_called_exactly_once(self) -> None:
        """classify_intent calls llm.complete() exactly once per invocation."""
        mock_llm = _mock_llm({"intent": "sql", "confidence": 0.9, "reasoning": "DB query."})
        with patch(
            "nlqueries.orchestrator.intent_classifier.get_llm_client",
            return_value=mock_llm,
        ):
            classify_intent("How many orders?", available_agent_types=["sql"])

        mock_llm.complete.assert_called_once()

    def test_empty_available_types_returns_intent_unchanged(self) -> None:
        """With empty available_agent_types, the LLM intent passes through unchanged."""
        mock_llm = _mock_llm({"intent": "sql", "confidence": 0.9, "reasoning": "DB query."})
        with patch(
            "nlqueries.orchestrator.intent_classifier.get_llm_client",
            return_value=mock_llm,
        ):
            result = classify_intent("Count orders", available_agent_types=[])

        assert result.intent == IntentType.sql

    def test_confidence_preserved_after_coercion(self) -> None:
        """Confidence from LLM response is preserved even after intent coercion."""
        mock_llm = _mock_llm(
            {"intent": "hybrid", "confidence": 0.77, "reasoning": "Both sources needed."}
        )
        with patch(
            "nlqueries.orchestrator.intent_classifier.get_llm_client",
            return_value=mock_llm,
        ):
            result = classify_intent("query", available_agent_types=["sql"])

        assert result.intent == IntentType.sql
        assert result.confidence == pytest.approx(0.77)

    def test_whitespace_around_json_is_tolerated(self) -> None:
        """Leading/trailing whitespace in LLM response is stripped before JSON parsing."""
        mock_llm = MagicMock()
        mock_llm.complete.return_value = (
            '  \n{"intent": "document", "confidence": 0.88, "reasoning": "Doc query."}\n  '
        )
        with patch(
            "nlqueries.orchestrator.intent_classifier.get_llm_client",
            return_value=mock_llm,
        ):
            result = classify_intent("What is the policy?", available_agent_types=["document"])

        assert result.intent == IntentType.document
        assert result.confidence == pytest.approx(0.88)

    def test_uses_fast_tier_client(self) -> None:
        """classify_intent must request the fast-tier LLM client."""
        mock_llm = _mock_llm({"intent": "sql", "confidence": 0.9, "reasoning": "DB query."})
        with patch(
            "nlqueries.orchestrator.intent_classifier.get_llm_client",
            return_value=mock_llm,
        ) as mock_get_client:
            classify_intent("How many orders?", available_agent_types=["sql"])

        mock_get_client.assert_called_once_with(tier="fast")

    def test_complete_called_with_max_tokens_200(self) -> None:
        """classify_intent must cap the completion at 200 tokens."""
        mock_llm = _mock_llm({"intent": "sql", "confidence": 0.9, "reasoning": "DB query."})
        with patch(
            "nlqueries.orchestrator.intent_classifier.get_llm_client",
            return_value=mock_llm,
        ):
            classify_intent("How many orders?", available_agent_types=["sql"])

        _, call_kwargs = mock_llm.complete.call_args
        assert call_kwargs.get("max_tokens") == 200


# ---------------------------------------------------------------------------
# The async variant (W-3)
#
# classify_intent is a full LLM round trip called bare inside an async
# generator, where it froze every other request on the worker. The async twin
# is only worth having if it classifies identically.
# ---------------------------------------------------------------------------


def _async_mock_llm(response: dict | str) -> MagicMock:  # type: ignore[type-arg]
    raw = response if isinstance(response, str) else json.dumps(response)
    mock = MagicMock()
    mock.acomplete = AsyncMock(return_value=raw)
    return mock


_IDENTICAL_CASES = [
    ({"intent": "sql", "confidence": 0.95, "reasoning": "Counting rows."}, ["sql", "document"]),
    ({"intent": "document", "confidence": 0.8, "reasoning": "Policy text."}, ["sql", "document"]),
    # hybrid coerced down to the one available type
    ({"intent": "hybrid", "confidence": 0.7, "reasoning": "Both."}, ["sql"]),
    # an intent that is not available at all
    ({"intent": "document", "confidence": 0.9, "reasoning": "Docs."}, ["sql"]),
    # malformed responses must fall back the same way
    ("not valid json at all", ["sql", "document"]),
    ({"intent": "nonsense", "confidence": 0.5, "reasoning": "?"}, ["sql", "document"]),
]


@pytest.mark.parametrize(("response", "available"), _IDENTICAL_CASES)
def test_sync_and_async_classify_identically(response, available) -> None:
    """Two implementations of one behaviour is a drift risk — and this one
    decides SQL versus document routing, so drift would be a wrong answer
    rather than a slow one."""
    raw = response if isinstance(response, str) else json.dumps(response)
    sync_llm = MagicMock()
    sync_llm.complete.return_value = raw

    with patch("nlqueries.orchestrator.intent_classifier.get_llm_client", return_value=sync_llm):
        sync_result = classify_intent("a question", available_agent_types=available)

    with patch(
        "nlqueries.orchestrator.intent_classifier.get_llm_client",
        return_value=_async_mock_llm(response),
    ):
        async_result = asyncio.run(aclassify_intent("a question", available_agent_types=available))

    assert async_result == sync_result


def test_async_classification_uses_the_fast_tier() -> None:
    """Classification is a short-output auxiliary call; paying for the large
    model on every turn would be a quiet cost regression."""
    with patch(
        "nlqueries.orchestrator.intent_classifier.get_llm_client",
        return_value=_async_mock_llm({"intent": "sql", "confidence": 1.0, "reasoning": "x"}),
    ) as mock_get:
        asyncio.run(aclassify_intent("a question", available_agent_types=["sql"]))

    assert mock_get.call_args.kwargs["tier"] == "fast"


def test_async_classification_does_not_block_the_event_loop() -> None:
    """The reason the async variant exists at all."""
    import time

    async def _slow_acomplete(*_a: object, **_kw: object) -> str:
        await asyncio.sleep(0.3)
        return json.dumps({"intent": "sql", "confidence": 1.0, "reasoning": "ok"})

    mock = MagicMock()
    mock.acomplete = _slow_acomplete

    async def run() -> float:
        started = time.monotonic()
        await asyncio.gather(
            aclassify_intent("question one", available_agent_types=["sql", "document"]),
            aclassify_intent("question two", available_agent_types=["sql", "document"]),
        )
        return time.monotonic() - started

    with patch("nlqueries.orchestrator.intent_classifier.get_llm_client", return_value=mock):
        elapsed = asyncio.run(run())

    assert elapsed < 0.5, f"two concurrent 300ms classifications took {elapsed:.3f}s"
