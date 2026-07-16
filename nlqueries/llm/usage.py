# nlqueries-core — OSS (BSL 1.1)
"""
nlqueries.llm.usage
~~~~~~~~~~~~~~~~~~~~
A generic, OSS-neutral seam for *observing* LLM token usage. When a host
application (e.g. the enterprise cost-tracking layer) binds a sink via
:func:`use_usage_sink`, every completion made by the LLM clients appends a
:class:`UsageRecord` to it — the provider's real token counts when reported,
or a flagged estimate otherwise. When no sink is bound this is a complete
no-op, so behaviour is byte-identical for callers that don't opt in.

Task-local, exactly like :func:`nlqueries.llm.use_llm_override`: the sink is a
``contextvars`` value copied per asyncio Task and into ``asyncio.to_thread``
workers, so one sink bound for the duration of a request captures every LLM
call the request makes — across threads and providers — without leaking across
concurrent requests.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from contextvars import ContextVar
from dataclasses import dataclass

# Rough chars-per-token used only for the estimate fallback (providers that
# don't report usage). A budget, not exact accounting — the record is flagged
# ``estimated`` so the host can surface it as approximate.
_CHARS_PER_TOKEN = 4


@dataclass
class UsageRecord:
    """Token usage for a single LLM completion.

    ``input_tokens`` is the non-cached prompt; ``cache_read_tokens`` /
    ``cache_write_tokens`` are the (separately-priced) prompt-cache halves, 0
    for providers without prompt caching. ``estimated`` is True when the
    provider didn't report usage and the counts are a heuristic estimate.
    """

    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    estimated: bool = False


_sink: ContextVar[list[UsageRecord] | None] = ContextVar("nlqueries_llm_usage_sink", default=None)


@contextlib.contextmanager
def use_usage_sink(sink: list[UsageRecord] | None) -> Iterator[None]:
    """Bind *sink* for the duration of the ``with`` block (task-local).

    Every LLM completion made inside the block appends one :class:`UsageRecord`
    to *sink*. Passing ``None`` is a no-op binding, so callers can wrap a block
    unconditionally.
    """
    token = _sink.set(sink)
    try:
        yield
    finally:
        _sink.reset(token)


def current_usage_sink() -> list[UsageRecord] | None:
    """Return the sink bound in the current context, if any."""
    return _sink.get()


def record_usage(record: UsageRecord) -> None:
    """Append *record* to the bound sink, or do nothing when none is bound."""
    sink = _sink.get()
    if sink is not None:
        sink.append(record)


def estimate_tokens(text: str) -> int:
    """Cheap heuristic token count (~4 chars/token) for the estimate fallback."""
    if not text:
        return 0
    return max(1, len(text) // _CHARS_PER_TOKEN)
