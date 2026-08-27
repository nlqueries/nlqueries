"""
Whether a forged cache entry can reach the database (SEC-09).

The audit reproduced this chain and destroyed the environment it used, leaving
the claim without evidence. The chain was: an anonymous writer put an entry into
the semantic cache, the cache returned it for a matching question, and the SQL
inside it executed against the customer's database.

Two of the three links have since been cut. Qdrant now requires authentication
off loopback (#124), so the first link needs a credential or network position
that the original did not. The SQL policy is applied on the cache-replay path
(#143), so a forged entry calling ``pg_read_file`` is refused.

The third link is open. ``SemanticCache._payload_to_entry`` reconstructs an
entry from the Qdrant payload with no check of where it came from, so a forged
entry whose SQL the policy permits is returned and executed. The policy
establishes that a statement is safe to run, not that NLQueries produced it or
that it answers the question asked.

Closing that is W5-1: a signed envelope binding tenant, agent, connector,
dialect, schema fingerprint and policy version, verified before use.

These tests use a stand-in Qdrant client. What is being tested is what the cache
does with a payload, not how the payload arrived.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from nlqueries.cache.semantic_cache import SIMILARITY_THRESHOLD, SemanticCache
from nlqueries.sql_policy import evaluate

pytestmark = pytest.mark.security

#: A statement the policy refuses. Reads a file from the database host.
FORGED_DANGEROUS_SQL = "SELECT pg_read_file('/etc/hostname')"

#: A statement the policy permits, reading a table the question did not mention.
#: A cache is not asked whether an answer is the right one, only whether it is
#: similar enough to a previous question.
FORGED_PLAUSIBLE_SQL = "SELECT * FROM salaries"


def _forged_payload(sql: str) -> dict[str, Any]:
    """A cache payload as something with write access to Qdrant would store it."""
    return {
        "question": "how many orders did we take last month?",
        "resolved_question": "how many orders did we take last month?",
        "agent_type": "sql",
        "answer": "There were 42 orders last month.",
        "sql": sql,
        "created_at": datetime.now(UTC).isoformat(),
        "hit_count": 0,
        "kind": "answer",
    }


def _cache_returning(payload: dict[str, Any]) -> Any:
    """A stand-in client whose nearest-neighbour search returns *payload*."""
    collection = MagicMock()
    collection.name = "cache_agent1"
    client = MagicMock()
    client.get_collections.return_value.collections = [collection]
    client.query_points.return_value = SimpleNamespace(
        points=[SimpleNamespace(id=1, score=SIMILARITY_THRESHOLD, payload=payload)]
    )
    client.set_payload.return_value = None
    return client


def _read_back(sql: str) -> Any:
    """What the cache returns for a question, given a forged entry storing *sql*."""
    client = _cache_returning(_forged_payload(sql))
    with (
        patch("nlqueries.cache.semantic_cache._get_client", return_value=client),
        patch("nlqueries.cache.semantic_cache.embed_text", return_value=[0.1] * 384),
    ):
        return SemanticCache("agent1").get("how many orders did we take last month?")


def test_the_cache_returns_a_forged_entry_unchanged() -> None:
    """The instrument check: the stand-in reaches the code under test.

    Also the finding itself. Nothing about the payload records where it came
    from, so nothing distinguishes this from an entry NLQueries wrote.
    """
    entry = _read_back(FORGED_PLAUSIBLE_SQL)

    assert entry is not None
    assert entry.sql == FORGED_PLAUSIBLE_SQL


def test_a_forged_entry_calling_a_dangerous_function_is_refused_before_execution() -> None:
    """The link the SQL policy cut. The entry is still returned by the cache;
    the policy refuses it on the replay path."""
    entry = _read_back(FORGED_DANGEROUS_SQL)

    assert entry is not None
    assert entry.sql == FORGED_DANGEROUS_SQL

    decision = evaluate(FORGED_DANGEROUS_SQL, "postgres")

    assert not decision.allowed
    assert "pg_read_file" in decision.summary()


@pytest.mark.xfail(
    strict=True,
    reason="SEC-09 — needs the signed cache envelope (W5-1); the policy checks safety, not origin",
)
def test_a_forged_entry_the_policy_permits_does_not_reach_the_database() -> None:
    """The open link.

    The statement is a valid read, so the policy permits it. Reaching the
    database requires only that the entry be similar enough to the question
    asked, and nothing establishes that NLQueries wrote it.
    """
    entry = _read_back(FORGED_PLAUSIBLE_SQL)
    assert entry is not None

    decision = evaluate(entry.sql or "", "postgres")

    assert not decision.allowed, (
        "a forged cache entry containing an ordinary SELECT would be executed: "
        "the payload carries no evidence of its origin"
    )


def test_the_policy_still_permits_the_statement_a_genuine_entry_would_hold() -> None:
    """The control. Whatever closes this must not refuse the cache's own
    entries, or the cache stops working."""
    decision = evaluate("SELECT count(*) FROM orders WHERE created_at >= '2026-07-01'", "postgres")

    assert decision.allowed, decision.summary()
