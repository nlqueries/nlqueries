"""
nlqueries.orchestrator.intent_classifier
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
LLM-based intent classifier — routes a user question to the appropriate
agent type (SQL, document, hybrid, or unclear).

Public API
----------
``classify_intent``
    Call with a question and a list of available agent types.
    Returns an ``IntentClassificationResult`` with intent, confidence,
    and reasoning.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum

from nlqueries.llm import get_llm_client


class IntentType(StrEnum):
    sql = "sql"  # question best answered by querying a database
    document = "document"  # question best answered from uploaded documents
    hybrid = "hybrid"  # question needs both structured data and documents
    unclear = "unclear"  # cannot determine; ask the user to clarify


@dataclass
class IntentClassificationResult:
    intent: IntentType
    confidence: float  # 0.0–1.0, estimated from LLM response
    reasoning: str  # one-sentence explanation from the LLM


_SYSTEM_PROMPT = """\
You are an intent classifier for a natural-language query engine.

Given a user question and a list of available agent types, classify the question
into exactly one intent type and respond with a single JSON object.

Intent types:
- sql: Best answered by querying a structured database (tables, rows, aggregations).
- document: Best answered from uploaded text documents (PDFs, Word files, wikis).
- hybrid: Requires both structured database data AND document context.
- unclear: Intent cannot be determined from the question alone.

Respond with ONLY a single JSON object on one line — no other text:
{"intent": "<type>", "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}

Examples:
Q: "How many orders did we get last month?"
A: {"intent": "sql", "confidence": 0.95, "reasoning": "Count from a DB table."}

Q: "What does our refund policy say?"
A: {"intent": "document", "confidence": 0.93, "reasoning": "Policy lives in a doc."}

Q: "Average deal size for customers who signed the enterprise contract?"
A: {"intent": "hybrid", "confidence": 0.82, "reasoning": "Deal size in DB; contract in docs."}

Q: "Show me total revenue by region"
A: {"intent": "sql", "confidence": 0.97, "reasoning": "Revenue aggregation is a DB query."}

Q: "Summarise the onboarding guide for new sales reps"
A: {"intent": "document", "confidence": 0.96, "reasoning": "Onboarding guide is a document."}

Q: "Which customers from the Q3 report have not placed an order yet?"
A: {"intent": "hybrid", "confidence": 0.85, "reasoning": "Q3 list in doc; orders in DB."}
"""


def _build_user_prompt(question: str, available_agent_types: list[str]) -> str:
    types_str = ", ".join(available_agent_types) if available_agent_types else "sql"
    return (
        f"Available agent types: {types_str}\n\n"
        f"Question: {question}\n\n"
        "Classify the intent. If the best intent type is not in the available list, "
        "choose the closest available type. Respond with only the JSON object."
    )


def coerce_intent(intent: IntentType, available_agent_types: list[str]) -> IntentType:
    """Coerce *intent* to an available type when it isn't one of *available_agent_types*.

    ``hybrid`` falls back to ``sql`` (preferred) or ``document``.
    All other unavailable types fall back to ``unclear``.
    ``unclear`` always passes through unchanged.

    Public helper (promoted from the former module-private ``_coerce_intent``) so
    an embedder that classifies intent out-of-band — e.g. the enterprise
    Conversation Context Engine's TurnPlanner, which supplies
    ``intent_override`` to :meth:`MultiAgentOrchestrator.handle_question` — can
    validate its choice through the exact same rules the built-in classifier uses.
    """
    if not available_agent_types or intent.value in available_agent_types:
        return intent
    if intent == IntentType.unclear:
        return intent
    if intent == IntentType.hybrid:
        if "sql" in available_agent_types:
            return IntentType.sql
        if "document" in available_agent_types:
            return IntentType.document
    return IntentType.unclear


# Backward-compatible private alias (the function was module-private before it
# was promoted to public API). Retained so any internal/legacy import keeps working.
_coerce_intent = coerce_intent


def classify_intent(
    question: str,
    available_agent_types: list[str],
) -> IntentClassificationResult:
    """Classify *question* into the most appropriate agent-type intent.

    Calls the LLM with a structured few-shot prompt and parses the response
    as JSON.  Falls back to :attr:`IntentType.unclear` with ``confidence=0.0``
    if the response is malformed or unparseable.

    When the LLM returns an intent not present in *available_agent_types*, the
    result is coerced: ``hybrid`` is downgraded to ``sql`` (if available) or
    ``document`` (if available); other unavailable types become ``unclear``.

    Args:
        question:              The natural-language question to classify.
        available_agent_types: Agent types enabled for this deployment,
                               e.g. ``["sql", "document"]``.

    Returns:
        :class:`IntentClassificationResult` with ``intent``, ``confidence``,
        and ``reasoning``.
    """
    llm = get_llm_client(tier="fast")
    user_prompt = _build_user_prompt(question, available_agent_types)
    raw = llm.complete(_SYSTEM_PROMPT, user_prompt, max_tokens=200)
    return _parse_classification(raw, available_agent_types)


async def aclassify_intent(
    question: str,
    available_agent_types: list[str],
) -> IntentClassificationResult:
    """Async counterpart to :func:`classify_intent`.

    Same prompt, same parsing, same coercion — it awaits the LLM instead of
    blocking on it. The synchronous version was being called bare inside an
    async generator, where a one-to-three-second round trip froze every other
    request sharing that worker.

    The sync version stays, unchanged, for the CLI and MCP paths.
    """
    llm = get_llm_client(tier="fast")
    user_prompt = _build_user_prompt(question, available_agent_types)
    raw = await llm.acomplete(_SYSTEM_PROMPT, user_prompt, max_tokens=200)
    return _parse_classification(raw, available_agent_types)


def _parse_classification(
    raw: str,
    available_agent_types: list[str],
) -> IntentClassificationResult:
    """Shared by both variants, so the two cannot drift into classifying the
    same response differently."""
    try:
        parsed = json.loads(raw.strip())
        intent = IntentType(parsed["intent"])
        confidence = float(parsed["confidence"])
        reasoning = str(parsed["reasoning"])
    except (json.JSONDecodeError, KeyError, ValueError):
        return IntentClassificationResult(
            intent=IntentType.unclear,
            confidence=0.0,
            reasoning="Failed to parse LLM response.",
        )

    return IntentClassificationResult(
        intent=coerce_intent(intent, available_agent_types),
        confidence=confidence,
        reasoning=reasoning,
    )
