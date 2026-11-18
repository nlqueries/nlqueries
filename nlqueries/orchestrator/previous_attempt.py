"""
nlqueries.orchestrator.previous_attempt
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A statement the caller already tried, and why it did not do (issue #408).

A caller that runs the SQL itself (``ExecutionPolicy.generate_only()``, usually
behind its own validator) has no way to ask for what NLQueries does for its own
executions: one corrected statement from the same prompt, told what went wrong.
A :class:`PreviousAttempt` carries that: the statement, the reason, and its
kind. Given one, the question is answered with a correction of that statement,
framed as the correction after a database error is, rather than a fresh one.

Like the values a user chose (:mod:`match_notes`), it is held for the request
in a context variable that :func:`use_previous_attempt` opens around a question,
so it reaches the orchestrator without being threaded through every layer.
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

#: What went wrong with the previous statement:
#: ``rejected``: the caller's own check refused it before it ran;
#: ``error``: the database refused it when it ran;
#: ``empty``: it ran and returned no rows, though the question names something
#: that must exist.
KINDS = ("rejected", "error", "empty")

#: The longest reason passed to the model. It is the caller's text, and may
#: quote values from the database as a driver error does, so it is capped as a
#: database error is (``_DB_ERROR_MAX_CHARS`` in ``sql_generation``).
REASON_MAX_CHARS = 1000


class PreviousAttemptError(ValueError):
    """A previous attempt that cannot be used: no SQL, or an unknown kind."""


@dataclass(frozen=True)
class PreviousAttempt:
    """A statement the caller tried, why it did not do, and how."""

    sql: str
    reason: str
    kind: str = "error"

    def __post_init__(self) -> None:
        if not self.sql or not self.sql.strip():
            raise PreviousAttemptError("previous_attempt.sql must be the statement that was tried")
        if self.kind not in KINDS:
            raise PreviousAttemptError(
                f"previous_attempt.kind must be one of {', '.join(KINDS)}, not {self.kind!r}"
            )

    @classmethod
    def from_value(
        cls, value: PreviousAttempt | Mapping[str, Any] | None
    ) -> PreviousAttempt | None:
        """*value* as a :class:`PreviousAttempt`: one already, a mapping with
        ``sql``, ``reason`` and ``kind`` keys, or ``None`` for none."""
        if value is None or isinstance(value, PreviousAttempt):
            return value
        if not isinstance(value, Mapping):
            raise PreviousAttemptError(
                "previous_attempt must be a mapping with sql, reason and kind"
            )
        unknown = set(value) - {"sql", "reason", "kind"}
        if unknown:
            raise PreviousAttemptError(
                f"previous_attempt has unknown keys: {', '.join(sorted(map(str, unknown)))}"
            )
        return cls(
            sql=str(value.get("sql") or ""),
            reason=str(value.get("reason") or ""),
            kind=str(value.get("kind") or "error"),
        )

    def reason_for_prompt(self) -> str:
        """The reason as the model is shown it: control characters other than
        line breaks and tabs removed, and capped at :data:`REASON_MAX_CHARS`."""
        cleaned = "".join(ch for ch in self.reason if ch in "\n\t" or ch.isprintable())
        cleaned = cleaned.strip() or "(no reason given)"
        if len(cleaned) > REASON_MAX_CHARS:
            cleaned = cleaned[:REASON_MAX_CHARS] + " [truncated]"
        return cleaned


_current: contextvars.ContextVar[PreviousAttempt | None] = contextvars.ContextVar(
    "nlqueries_previous_attempt", default=None
)


@contextmanager
def use_previous_attempt(previous: PreviousAttempt | None) -> Iterator[None]:
    """Answer the enclosed question with a correction of *previous*, if given."""
    token = _current.set(previous)
    try:
        yield
    finally:
        _current.reset(token)


def current_previous_attempt() -> PreviousAttempt | None:
    """The previous attempt the current question corrects, if any."""
    return _current.get()
