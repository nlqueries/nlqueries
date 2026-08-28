"""
nlqueries.auth.admission
~~~~~~~~~~~~~~~~~~~~~~~~
How much a caller may ask for, as opposed to what they are allowed to ask
(SEC-14).

Authorisation answers whether a principal may run `query` on an agent. It says
nothing about how often. A caller with a perfectly valid grant can call it in a
loop: each one is an LLM call the deployment pays for, a database query the
customer's server runs, and a slot in a process that has no cap on how many it
will start at once.

Two limits, both per principal:

*Rate* is a fixed window rather than a sliding one or a token bucket. A fixed
window permits a burst of up to twice the limit across a boundary, which is
worth knowing and is not worth more machinery here: the purpose is to stop a
loop, not to shape traffic.

*Concurrency* is a count of calls in flight. It is the one that matters for a
slow tool -- `query` is allowed forty-five seconds, so a caller who ignores the
rate limit's refusals can still hold open as many slow calls as they can start.

Both are per process. The MCP server is one process; a deployment running
several behind a load balancer gets the limit multiplied by that number, which
the documentation says rather than leaving to be discovered.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

#: Calls per principal per minute. Zero disables the limit.
RATE_LIMIT_ENV = "NLQ_MCP_RATE_LIMIT_PER_MINUTE"

#: Calls per principal in flight at once. Zero disables the limit.
CONCURRENCY_ENV = "NLQ_MCP_MAX_CONCURRENT"

#: Chosen to be well clear of a person using an MCP client -- a question every
#: few seconds is nowhere near it -- while still bounding a loop.
DEFAULT_RATE_LIMIT = 60

#: `query` may take forty-five seconds, so this is what stops a caller holding
#: open more of them than the process can serve.
DEFAULT_CONCURRENCY = 8

_WINDOW_SECONDS = 60.0


class TooManyRequests(Exception):
    """Raised when a caller has asked for more than their share.

    Separate from NotAuthorized: the caller may well be entitled to this call,
    just not to this many of them, and an operator reading the audit trail
    should not have to tell the two apart by message.
    """


@dataclass
class _Window:
    started: float
    count: int = 0


@dataclass
class AdmissionControl:
    """Per-principal rate and concurrency limits for one process."""

    rate_limit: int = DEFAULT_RATE_LIMIT
    max_concurrent: int = DEFAULT_CONCURRENCY
    _windows: dict[str, _Window] = field(default_factory=dict, repr=False)
    _in_flight: dict[str, int] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def _now(self) -> float:
        return time.monotonic()

    def acquire(self, subject: str) -> None:
        """Admit one call by *subject*, or raise.

        Both limits are checked and the slot taken under one lock, so two calls
        arriving together cannot both see the last free slot.
        """
        with self._lock:
            if self.rate_limit > 0:
                window = self._windows.get(subject)
                now = self._now()
                if window is None or now - window.started >= _WINDOW_SECONDS:
                    # A window that has rolled over is the moment to drop the
                    # ones that rolled over and were never used again. Without
                    # it the dict only ever grows: one entry per distinct
                    # subject for the life of the process, which for a server
                    # in front of an identity provider is one per user who has
                    # ever called. Sweeping here rather than on every call
                    # keeps it to once per subject per window.
                    self._evict_expired(now)
                    window = _Window(started=now)
                    self._windows[subject] = window
                if window.count >= self.rate_limit:
                    raise TooManyRequests(
                        f"Rate limit of {self.rate_limit} calls per minute reached."
                    )
                window.count += 1

            if self.max_concurrent > 0:
                in_flight = self._in_flight.get(subject, 0)
                if in_flight >= self.max_concurrent:
                    # The rate-limit window has already been charged for this
                    # call. That is deliberate: a caller hammering a full
                    # concurrency pool is making requests, and not counting them
                    # would let them retry without limit.
                    raise TooManyRequests(
                        f"Limit of {self.max_concurrent} calls in flight reached."
                    )
                self._in_flight[subject] = in_flight + 1

    def _evict_expired(self, now: float) -> None:
        """Drop windows that have expired. The caller holds the lock."""
        expired = [
            name
            for name, window in self._windows.items()
            if now - window.started >= _WINDOW_SECONDS
        ]
        for name in expired:
            del self._windows[name]

    def tracked_subjects(self) -> int:
        """How many rate-limit windows are held. For the tests."""
        with self._lock:
            return len(self._windows)

    def release(self, subject: str) -> None:
        """Give back a slot. Safe to call for a caller that never took one."""
        if self.max_concurrent <= 0:
            return
        with self._lock:
            remaining = self._in_flight.get(subject, 0) - 1
            if remaining > 0:
                self._in_flight[subject] = remaining
            else:
                self._in_flight.pop(subject, None)

    def in_flight(self, subject: str) -> int:
        with self._lock:
            return self._in_flight.get(subject, 0)


def _int_from_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("%s is not a number (%r); using %d.", name, raw, default)
        return default
    if value < 0:
        logger.warning("%s is negative (%d); using %d.", name, value, default)
        return default
    return value


def from_config() -> AdmissionControl:
    """The limits this deployment is configured for."""
    return AdmissionControl(
        rate_limit=_int_from_env(RATE_LIMIT_ENV, DEFAULT_RATE_LIMIT),
        max_concurrent=_int_from_env(CONCURRENCY_ENV, DEFAULT_CONCURRENCY),
    )
