# nlqueries-core — OSS (BSL 1.1)
"""
nlqueries.orchestrator.provenance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Answer **provenance**: a structured, additive record of how an answer was
produced — the route chosen, capsules retrieved, KB parts injected into the
prompt, cache hit/miss, validator warnings, and timings (SYL-1.1).

Collected through a context-local seam, exactly like :mod:`nlqueries.llm.usage`:
a host binds a :class:`Provenance` collector via :func:`use_provenance`, and each
orchestrator site records what it contributed through the ``record_*`` helpers.
When nothing is bound every helper is a complete no-op, so behaviour is
byte-identical for callers that don't ask for an explanation. The collector is a
``contextvars`` value, so it is copied per asyncio Task and into
``asyncio.to_thread`` workers — one collector bound for a request captures every
site the request touches without leaking across concurrent requests.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

# Bump only on a breaking shape change. Consumers (API, UI, persistence) key on
# this to stay compatible; it ships from day one.
SCHEMA_VERSION = "1"


@dataclass
class CapsuleRef:
    """A query capsule that was retrieved for the prompt."""

    id: str
    intent: str = ""


@dataclass
class CacheInfo:
    """Semantic-cache outcome for the turn."""

    hit: bool
    similarity: float | None = None
    tier: str = ""  # "exact" | "answer" | "template" | ""


@dataclass
class Provenance:
    """How an answer was produced. Every field is optional/additive."""

    schema_version: str = SCHEMA_VERSION
    route: str | None = None  # "sql" | "document" | "hybrid"
    intent_confidence: float | None = None
    capsules_used: list[CapsuleRef] = field(default_factory=list)
    # Which KB parts were injected, as stable tags, e.g. "glossary:active_customer",
    # "rule:exclude_cancelled", "table_desc:orders".
    prompt_sections: list[str] = field(default_factory=list)
    cache: CacheInfo | None = None
    validator: list[str] = field(default_factory=list)  # validator / nexus warnings
    timings: dict[str, float] = field(default_factory=dict)  # phase -> milliseconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "route": self.route,
            "intent_confidence": self.intent_confidence,
            "capsules_used": [{"id": c.id, "intent": c.intent} for c in self.capsules_used],
            "prompt_sections": list(self.prompt_sections),
            "cache": (
                None
                if self.cache is None
                else {
                    "hit": self.cache.hit,
                    "similarity": self.cache.similarity,
                    "tier": self.cache.tier,
                }
            ),
            "validator": list(self.validator),
            "timings": dict(self.timings),
        }


_collector: ContextVar[Provenance | None] = ContextVar("nlqueries_provenance", default=None)


@contextlib.contextmanager
def use_provenance(collector: Provenance | None) -> Iterator[None]:
    """Bind *collector* for the duration of the ``with`` block (task-local).

    Every ``record_*`` call inside the block writes to *collector*. Passing
    ``None`` is a no-op binding, so callers can wrap a block unconditionally.
    """
    token = _collector.set(collector)
    try:
        yield
    finally:
        _collector.reset(token)


def current_provenance() -> Provenance | None:
    """Return the collector bound in the current context, if any."""
    return _collector.get()


# ---------------------------------------------------------------------------
# Record helpers — each a no-op when no collector is bound.
# ---------------------------------------------------------------------------


def record_route(route: str) -> None:
    p = _collector.get()
    if p is not None:
        p.route = route


def record_intent_confidence(confidence: float) -> None:
    p = _collector.get()
    if p is not None:
        p.intent_confidence = confidence


def record_capsule(capsule_id: str, intent: str = "") -> None:
    p = _collector.get()
    if p is not None and not any(c.id == capsule_id for c in p.capsules_used):
        p.capsules_used.append(CapsuleRef(id=str(capsule_id), intent=intent or ""))


def record_prompt_section(section: str) -> None:
    p = _collector.get()
    if p is not None and section and section not in p.prompt_sections:
        p.prompt_sections.append(section)


def record_cache(hit: bool, similarity: float | None = None, tier: str = "") -> None:
    p = _collector.get()
    if p is not None:
        p.cache = CacheInfo(hit=hit, similarity=similarity, tier=tier)


def record_validator_warning(warning: str) -> None:
    p = _collector.get()
    if p is not None and warning and warning not in p.validator:
        p.validator.append(warning)


def record_timing(key: str, ms: float) -> None:
    p = _collector.get()
    if p is not None:
        p.timings[key] = float(ms)
