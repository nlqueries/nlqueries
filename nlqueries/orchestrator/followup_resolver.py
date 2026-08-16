"""
nlqueries.orchestrator.followup_resolver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Follow-up question resolution — detects and rewrites pronoun references and
contextual references in multi-turn conversations before the question is sent
to the intent classifier and orchestrators.

Public API
----------
``ResolvedQuestion``
    Holds the original question, resolved form, and resolution metadata.
``resolve_followup``
    Heuristic + LLM-based follow-up resolution with no-history fast path.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from nlqueries.llm import get_llm_client
from nlqueries.orchestrator.conversation import ConversationTurn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Follow-up signal vocabulary
# ---------------------------------------------------------------------------

_FOLLOWUP_SIGNALS: tuple[str, ...] = (
    "that",
    "those",
    "these",
    "it",
    "the same",
    "previous",
    "last one",
    "instead",
    "also",
    "as well",
    "but only",
    "filter",
    "narrow",
    "exclude",
    "what about",
    "and also",
    "show me more",
    "break it down",
    "how about",
)

_SYSTEM_PROMPT = """\
You are a follow-up question resolver for a natural-language query engine.

Given a conversation history and a follow-up question, rewrite the question
as a fully self-contained question by substituting all pronoun references and
contextual references with the actual entities from the history.

Respond with ONLY a single JSON object on one line — no other text:
{"resolved": "<fully self-contained question>", "reasoning": "<one sentence>"}
"""

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _has_followup_signal(question: str) -> bool:
    q_lower = question.lower()
    return any(signal in q_lower for signal in _FOLLOWUP_SIGNALS)


def _format_history_for_prompt(history: list[ConversationTurn], max_turns: int = 3) -> str:
    recent = history[-max_turns:]
    lines: list[str] = []
    for turn in recent:
        prefix = "User" if turn.role == "user" else "Assistant"
        lines.append(f"{prefix}: {turn.content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class ResolvedQuestion:
    original: str
    resolved: str  # self-contained question with all references resolved
    is_followup: bool  # True if resolution changed the question
    reasoning: str  # one-sentence explanation of what was resolved


def resolve_followup(
    question: str,
    history: list[ConversationTurn],
) -> ResolvedQuestion:
    """Resolve pronoun and contextual references in a follow-up question.

    Step 1 — Heuristic check: if the question contains none of the defined
    follow-up signals (case-insensitive), return the question unchanged
    without calling the LLM, saving latency and cost.

    Step 2 — LLM resolution: if a signal is detected and history is not
    empty, call the LLM (non-streaming, temperature=0.1) with the last 3
    conversation turns and the current question.  The LLM rewrites the
    question as fully self-contained.

    If history is empty or the LLM call fails for any reason, the original
    question is returned unchanged (``is_followup=False``).

    Args:
        question: The current user question, possibly a follow-up.
        history:  Prior turns from the conversation session.

    Returns:
        :class:`ResolvedQuestion` with ``original``, ``resolved``,
        ``is_followup``, and ``reasoning`` fields.
    """
    shortcut = _shortcut(question, history)
    if shortcut is not None:
        return shortcut

    try:
        llm = get_llm_client(tier="fast")
        raw = llm.complete(_SYSTEM_PROMPT, _build_user_prompt(question, history), max_tokens=200)
    except Exception:
        logger.debug("Follow-up resolution failed; returning original question.", exc_info=True)
        return _unresolved(question)

    return _parse_response(question, raw)


async def aresolve_followup(
    question: str,
    history: list[ConversationTurn],
) -> ResolvedQuestion:
    """Async counterpart to :func:`resolve_followup`.

    Identical in every respect except that it awaits the LLM instead of blocking
    on it. That difference matters only to the caller's event loop, and it
    matters a great deal: this is a full LLM round trip, typically one to three
    seconds, and the synchronous version was being called bare inside an async
    generator — freezing every other request on that worker for its duration.

    The sync version stays, unchanged, for the CLI and MCP paths.
    """
    shortcut = _shortcut(question, history)
    if shortcut is not None:
        return shortcut

    try:
        llm = get_llm_client(tier="fast")
        raw = await llm.acomplete(
            _SYSTEM_PROMPT, _build_user_prompt(question, history), max_tokens=200
        )
    except Exception:
        logger.debug("Follow-up resolution failed; returning original question.", exc_info=True)
        return _unresolved(question)

    return _parse_response(question, raw)


# ---------------------------------------------------------------------------
# Shared between the two variants.
#
# Everything except the LLM call itself lives here, so the sync and async paths
# cannot drift into answering the same question differently — which is the real
# risk of keeping two of anything.
# ---------------------------------------------------------------------------


def _shortcut(question: str, history: list[ConversationTurn]) -> ResolvedQuestion | None:
    """The two cases that need no LLM at all, or None to go on and ask one."""
    if not _has_followup_signal(question):
        return ResolvedQuestion(
            original=question,
            resolved=question,
            is_followup=False,
            reasoning="No follow-up references detected.",
        )
    if not history:
        return ResolvedQuestion(
            original=question,
            resolved=question,
            is_followup=False,
            reasoning="No conversation history available to resolve references.",
        )
    return None


def _build_user_prompt(question: str, history: list[ConversationTurn]) -> str:
    return (
        f"Conversation history:\n{_format_history_for_prompt(history)}\n\n"
        f"Follow-up question: {question}\n\n"
        "Rewrite the follow-up question as fully self-contained. "
        "Respond with only the JSON object."
    )


def _unresolved(question: str) -> ResolvedQuestion:
    """Fail open: an unresolvable follow-up is still answerable as asked."""
    return ResolvedQuestion(
        original=question,
        resolved=question,
        is_followup=False,
        reasoning="Resolution failed; original question returned.",
    )


def _parse_response(question: str, raw: str) -> ResolvedQuestion:
    try:
        parsed = json.loads(raw.strip())
        resolved_text = str(parsed["resolved"])
        reasoning = str(parsed["reasoning"])
    except Exception:
        logger.debug("Follow-up resolution failed; returning original question.", exc_info=True)
        return _unresolved(question)

    return ResolvedQuestion(
        original=question,
        resolved=resolved_text,
        is_followup=resolved_text != question,
        reasoning=reasoning,
    )
