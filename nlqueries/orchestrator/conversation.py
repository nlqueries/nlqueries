"""
nlqueries.orchestrator.conversation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Conversation session management and context building for multi-turn chat.

Public API
----------
``ConversationTurn``
    A single turn (user or assistant) with optional SQL and agent type.
``ConversationSession``
    A session with multiple turns, context-window slicing, and Anthropic
    message-format serialisation.
``create_session``
    Factory for a new empty session with a generated UUID.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


@dataclass
class ConversationTurn:
    role: str  # "user" | "assistant"
    content: str  # the message text
    agent_type: str | None  # "sql" | "document" | "hybrid" | None
    sql: str | None  # generated SQL when role=="assistant" and agent_type=="sql"
    timestamp: datetime


@dataclass
class ConversationSession:
    session_id: str  # UUID
    agent_id: str
    turns: list[ConversationTurn]
    created_at: datetime
    updated_at: datetime

    def add_turn(self, role: str, content: str, **kwargs: Any) -> None:
        """Append a turn and update updated_at."""
        now = datetime.now(UTC)
        raw_agent_type = kwargs.get("agent_type")
        raw_sql = kwargs.get("sql")
        turn = ConversationTurn(
            role=role,
            content=content,
            agent_type=str(raw_agent_type) if raw_agent_type is not None else None,
            sql=str(raw_sql) if raw_sql is not None else None,
            timestamp=now,
        )
        self.turns.append(turn)
        self.updated_at = now

    def get_context_window(self, max_turns: int = 6) -> list[ConversationTurn]:
        """Return the last max_turns turns (for prompt assembly)."""
        return self.turns[-max_turns:]

    def to_prompt_messages(self, max_turns: int = 6) -> list[dict[str, str]]:
        """Convert the context window to Anthropic message format.

        SQL strings are intentionally omitted from assistant content — only
        the natural-language response is included so the LLM sees clean
        conversational context without raw SQL noise.

        Returns:
            ``[{"role": "user"|"assistant", "content": "..."}, ...]``
        """
        return [
            {"role": turn.role, "content": turn.content}
            for turn in self.get_context_window(max_turns)
        ]


def create_session(agent_id: str) -> ConversationSession:
    """Return a new empty ConversationSession with a generated UUID."""
    now = datetime.now(UTC)
    return ConversationSession(
        session_id=str(uuid.uuid4()),
        agent_id=agent_id,
        turns=[],
        created_at=now,
        updated_at=now,
    )
