"""
Whether a forged cache entry can reach the database (SEC-09).

The audit reproduced this chain and destroyed the environment it used, leaving
the claim without evidence. The chain was: an anonymous writer put an entry into
the semantic cache, the cache returned it for a matching question, and the SQL
inside it executed against the customer's database.

All three links are now cut. Qdrant requires authentication off loopback (#124).
Cache entries are signed for the context they were produced in and verified
before use (#148), so an entry this deployment did not write does not verify.
The SQL policy is applied on the cache-replay path (#143), so a statement that
did verify but calls a dangerous function is still refused.

The signature is what closes the finding. The policy establishes that a
statement is safe to run; it does not establish that NLQueries produced it, and
``SELECT * FROM salaries`` returned for a question about order counts is safe by
the policy's measure and wrong by any other.

These tests use a stand-in Qdrant client. What is under test is what the cache
does with a payload, not how the payload arrived.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from nlqueries.cache.envelope import CacheBinding, sign
from nlqueries.cache.semantic_cache import SIMILARITY_THRESHOLD, SemanticCache
from nlqueries.sql_policy import evaluate

pytestmark = pytest.mark.security

QUESTION = "how many orders did we take last month?"

#: A statement the policy refuses. Reads a file from the database host.
FORGED_DANGEROUS_SQL = "SELECT pg_read_file('/etc/hostname')"

#: A statement the policy permits, reading a table the question did not mention.
FORGED_PLAUSIBLE_SQL = "SELECT * FROM salaries"

BINDING = CacheBinding(
    agent_id="agent1",
    connector_fingerprint="conn-fp",
    dialect="postgres",
    schema_fingerprint="schema-fp",
    policy_version="1",
)

#: The key this deployment holds. An attacker with write access to Qdrant does
#: not have it, which is what the signature depends on.
KEY = b"the-deployment-key"


def _payload(sql: str) -> dict[str, Any]:
    """A cache payload as something with write access to Qdrant would store it."""
    return {
        "question": QUESTION,
        "resolved_question": QUESTION,
        "agent_type": "sql",
        "answer": "There were 42 orders last month.",
        "sql": sql,
        "created_at": datetime.now(UTC).isoformat(),
        "hit_count": 0,
        "kind": "answer",
    }


def _read_back(payload: dict[str, Any], binding: CacheBinding | None = BINDING) -> Any:
    """What the cache returns for QUESTION, given *payload* stored against it."""
    collection = MagicMock()
    collection.name = "cache_agent1"
    client = MagicMock()
    client.get_collections.return_value.collections = [collection]
    client.query_points.return_value = SimpleNamespace(
        points=[SimpleNamespace(id=1, score=SIMILARITY_THRESHOLD, payload=payload)]
    )
    client.set_payload.return_value = None

    with (
        patch("nlqueries.cache.semantic_cache._get_client", return_value=client),
        patch("nlqueries.cache.semantic_cache.embed_text", return_value=[0.1] * 384),
        patch("nlqueries.cache.envelope.signing_key", return_value=KEY),
    ):
        return SemanticCache("agent1", binding=binding).get(QUESTION)


def test_an_entry_this_deployment_signed_is_returned() -> None:
    """The control. Whatever refuses forged entries must not refuse genuine
    ones, or the cache stops working and is turned off."""
    entry = _read_back(sign(_payload("SELECT count(*) FROM orders"), BINDING, KEY))

    assert entry is not None
    assert entry.sql == "SELECT count(*) FROM orders"


def test_an_unsigned_forged_entry_is_a_miss() -> None:
    """SEC-09. The statement is an ordinary read that the policy permits, so
    before signing existed it was returned and executed."""
    assert evaluate(FORGED_PLAUSIBLE_SQL, "postgres").allowed

    assert _read_back(_payload(FORGED_PLAUSIBLE_SQL)) is None


def test_an_entry_signed_with_another_key_is_a_miss() -> None:
    """An attacker who can write to Qdrant but does not hold the key cannot
    produce a tag that verifies."""
    forged = sign(_payload(FORGED_PLAUSIBLE_SQL), BINDING, b"an-attacker-key")

    assert _read_back(forged) is None


def test_an_entry_signed_for_another_agent_is_a_miss() -> None:
    """A genuine entry cannot be moved between agents to answer a question the
    reader is not entitled to ask."""
    other = CacheBinding(**{**BINDING.__dict__, "agent_id": "agent2"})
    elsewhere = sign(_payload("SELECT count(*) FROM orders"), other, KEY)

    assert _read_back(elsewhere) is None


def test_an_entry_signed_against_another_schema_is_a_miss() -> None:
    """SQL generated against one schema is not valid evidence about another."""
    other = CacheBinding(**{**BINDING.__dict__, "schema_fingerprint": "changed"})
    stale = sign(_payload("SELECT count(*) FROM orders"), other, KEY)

    assert _read_back(stale) is None


def test_a_dangerous_statement_is_refused_even_if_it_verifies() -> None:
    """Defence in depth. Signing establishes origin; the policy establishes that
    the statement is safe to run, and both apply on the replay path."""
    entry = _read_back(sign(_payload(FORGED_DANGEROUS_SQL), BINDING, KEY))

    assert entry is not None
    assert entry.sql == FORGED_DANGEROUS_SQL

    decision = evaluate(FORGED_DANGEROUS_SQL, "postgres")

    assert not decision.allowed
    assert "pg_read_file" in decision.summary()


def test_a_cache_without_a_binding_reads_nothing() -> None:
    """There is nothing to verify against, so every read is a miss rather than
    an unverified hit."""
    assert _read_back(sign(_payload("SELECT 1"), BINDING, KEY), binding=None) is None
