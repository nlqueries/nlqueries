"""
nlqueries.feedback.models
~~~~~~~~~~~~~~~~~~~~~~~~~
Plain dataclass for query feedback captured by the OSS CLI and the
enterprise API.  Intentionally keeps no dependency on SQLAlchemy or any
web framework so that ``core/`` can remain self-contained.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class QueryFeedback:
    """A single piece of user feedback on a generated SQL query.

    Fields
    ------
    question       The natural-language question that was asked.
    generated_sql  The SQL statement produced by the orchestrator.
    corrected_sql  Optional correction supplied by the user (``None`` if none).
    rating         ``"up"`` (positive) or ``"down"`` (negative) thumbs signal.
    agent_id       The agent whose knowledge base answered the question.
    timestamp      UTC timestamp; defaults to *now* when not supplied.
    """

    question: str
    generated_sql: str
    rating: str  # "up" | "down"
    agent_id: str
    corrected_sql: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if self.rating not in ("up", "down"):
            raise ValueError(f"rating must be 'up' or 'down', got {self.rating!r}")
