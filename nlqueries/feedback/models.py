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

#: Submitted by an operator running the CLI on the host.
SOURCE_CLI = "cli"
#: Submitted over the MCP transport, which has no authentication (SEC-05), so
#: the record identifies nobody.
SOURCE_MCP = "mcp"
#: Submitted through the enterprise API, which authenticates its callers.
SOURCE_API = "api"
#: Written before this field existed, so its origin cannot be established.
SOURCE_UNKNOWN = "unknown"

#: Sources whose records name a party who can be held to them. Promotion turns
#: a record into an exemplar the model is shown for every later question on that
#: agent, so it is limited to these unless an operator asks otherwise (SEC-10).
ATTRIBUTED_SOURCES = frozenset({SOURCE_CLI, SOURCE_API})


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
    source         Where the record came from. See the constants above; a record
                   written before this field existed loads as ``unknown``.
    """

    question: str
    generated_sql: str
    rating: str  # "up" | "down"
    agent_id: str
    corrected_sql: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    source: str = SOURCE_UNKNOWN

    def __post_init__(self) -> None:
        if self.rating not in ("up", "down"):
            raise ValueError(f"rating must be 'up' or 'down', got {self.rating!r}")
