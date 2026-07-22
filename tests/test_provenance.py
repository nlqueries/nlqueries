# nlqueries-core — OSS (BSL 1.1)
"""Tests for nlqueries.orchestrator.provenance (SYL-1.1).

Covers the context-local collector seam, the guarantee that recording crosses the
async-generator boundary (the mechanism the orchestrator relies on), and the
``run_query(explain=...)`` wiring for the sql / document / hybrid routes plus a
cache-recording turn — with flag-off behaviour unchanged.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from unittest.mock import MagicMock, patch

from nlqueries.orchestrator.followup_resolver import ResolvedQuestion
from nlqueries.orchestrator.provenance import (
    SCHEMA_VERSION,
    Provenance,
    current_provenance,
    record_cache,
    record_capsule,
    record_intent_confidence,
    record_prompt_section,
    record_route,
    record_timing,
    record_validator_warning,
    use_provenance,
)
from nlqueries.orchestrator.sync_runner import AgentQueryResult, run_query

# ---------------------------------------------------------------------------
# Seam
# ---------------------------------------------------------------------------


def test_no_op_when_unbound() -> None:
    # None of these raise or leak when no collector is bound.
    record_route("sql")
    record_cache(hit=True, similarity=0.9, tier="answer")
    record_capsule("c1", "count")
    assert current_provenance() is None


def test_records_into_bound_collector_and_unbinds() -> None:
    p = Provenance()
    with use_provenance(p):
        assert current_provenance() is p
        record_route("hybrid")
        record_intent_confidence(0.82)
        record_capsule("cap-1", "count_orders")
        record_prompt_section("glossary:active_customer")
        record_cache(hit=True, similarity=0.98, tier="answer")
        record_validator_warning("unsanctioned join")
        record_timing("total_ms", 1234.0)
    assert current_provenance() is None
    assert p.route == "hybrid"
    assert p.intent_confidence == 0.82
    assert p.capsules_used[0].id == "cap-1"
    assert p.prompt_sections == ["glossary:active_customer"]
    assert p.cache is not None and p.cache.hit and p.cache.tier == "answer"
    assert p.validator == ["unsanctioned join"]
    assert p.timings["total_ms"] == 1234.0


def test_dedup_capsules_sections_and_warnings() -> None:
    p = Provenance()
    with use_provenance(p):
        record_capsule("cap-1", "x")
        record_capsule("cap-1", "y")  # same id → ignored
        record_prompt_section("rule:r")
        record_prompt_section("rule:r")
        record_validator_warning("w")
        record_validator_warning("w")
    assert len(p.capsules_used) == 1
    assert p.prompt_sections == ["rule:r"]
    assert p.validator == ["w"]


def test_to_dict_shape_and_schema_version() -> None:
    p = Provenance()
    with use_provenance(p):
        record_route("sql")
        record_capsule("c1", "count")
        record_cache(hit=False)
    d = p.to_dict()
    assert d["schema_version"] == SCHEMA_VERSION == "1"
    assert d["route"] == "sql"
    assert d["capsules_used"] == [{"id": "c1", "intent": "count"}]
    assert d["cache"] == {"hit": False, "similarity": None, "tier": ""}
    assert json.dumps(d)  # fully JSON-serializable


def test_recording_crosses_async_generator_boundary() -> None:
    """The orchestrator records inside an async generator driven by a caller that
    binds the collector — the contextvar must be visible during iteration."""

    async def _gen() -> AsyncGenerator[str, None]:
        record_route("sql")  # runs during __anext__, in the caller's context
        yield "tok"
        record_cache(hit=False)
        yield "done"

    async def _drive() -> Provenance:
        p = Provenance()
        with use_provenance(p):
            async for _ in _gen():
                pass
        return p

    p = asyncio.run(_drive())
    assert p.route == "sql"
    assert p.cache is not None and p.cache.hit is False


# ---------------------------------------------------------------------------
# run_query(explain=...) wiring — per route
# ---------------------------------------------------------------------------

_NO_FOLLOWUP = ResolvedQuestion(original="q", resolved="q", is_followup=False, reasoning="")


def _final_chunk(agent_type: str) -> str:
    if agent_type == "hybrid":
        return json.dumps({"type": "hybrid", "agent_type": "hybrid", "merged_answer": "M"})
    if agent_type == "document":
        return json.dumps({"type": "citations", "agent_type": "document", "citations": []})
    return json.dumps({"type": "sql", "agent_type": "sql", "sql": "SELECT 1", "is_valid": True})


def _run_with_recording_gen(agent_type: str, *, explain: bool) -> AgentQueryResult:
    """Drive run_query with a mock orchestrator whose handle_question records the
    route (as the real one does) and yields a final chunk for *agent_type*."""

    async def _gen(*_a: object, **_k: object) -> AsyncGenerator[str, None]:
        # Mimic what the real orchestrator sites do while the collector is bound.
        record_route(agent_type)
        record_cache(hit=False)
        record_prompt_section("table_desc:orders")
        yield "answer "
        yield _final_chunk(agent_type)

    mock_orch = MagicMock()
    mock_orch.handle_question = _gen
    with (
        patch("nlqueries.orchestrator.sync_runner.MultiAgentOrchestrator", return_value=mock_orch),
        patch("nlqueries.orchestrator.sync_runner.resolve_followup", return_value=_NO_FOLLOWUP),
    ):
        return asyncio.run(run_query("q", "agent", explain=explain))


def test_run_query_populates_provenance_for_each_route() -> None:
    for route in ("sql", "document", "hybrid"):
        result = _run_with_recording_gen(route, explain=True)
        prov = result.provenance
        assert prov is not None, f"{route}: provenance missing"
        assert prov.route == route
        assert prov.cache is not None and prov.cache.hit is False
        assert "table_desc:orders" in prov.prompt_sections
        assert "total_ms" in prov.timings  # sync_runner fills the wall-clock total
        assert prov.to_dict()["schema_version"] == "1"


def test_cache_hit_recorded_on_result() -> None:
    async def _gen(*_a: object, **_k: object) -> AsyncGenerator[str, None]:
        record_route("sql")
        record_cache(hit=True, similarity=0.99, tier="answer")
        yield "answer "
        yield _final_chunk("sql")

    mock_orch = MagicMock()
    mock_orch.handle_question = _gen
    with (
        patch("nlqueries.orchestrator.sync_runner.MultiAgentOrchestrator", return_value=mock_orch),
        patch("nlqueries.orchestrator.sync_runner.resolve_followup", return_value=_NO_FOLLOWUP),
    ):
        result = asyncio.run(run_query("q", "agent", explain=True))
    assert result.provenance is not None
    assert result.provenance.cache is not None
    assert result.provenance.cache.hit is True
    assert result.provenance.cache.tier == "answer"
    assert result.provenance.cache.similarity == 0.99


def test_flag_off_leaves_provenance_none() -> None:
    """Without explain, no collector is bound and the result carries no provenance
    (behaviour byte-identical to before SYL-1.1)."""
    result = _run_with_recording_gen("sql", explain=False)
    assert result.provenance is None
