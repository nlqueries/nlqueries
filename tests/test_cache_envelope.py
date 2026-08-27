"""
Signing and verification of semantic-cache entries.

The properties asserted here are what SEC-09 depends on: an entry that was not
signed with this key, or was altered, or was produced for a different context,
must not verify.
"""

from __future__ import annotations

import stat
import sys

import pytest
from nlqueries.cache.envelope import (
    ENVELOPE_VERSION,
    SIGNATURE_KEY,
    SIGNED_FIELDS,
    VERSION_KEY,
    CacheBinding,
    sign,
    signing_key,
    verify,
)

KEY = b"a-test-key-not-used-anywhere-else"

BINDING = CacheBinding(
    agent_id="agent1",
    connector_fingerprint="conn-fp",
    dialect="postgres",
    schema_fingerprint="schema-fp",
    policy_version="1",
)


def _payload(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "question": "how many orders last month?",
        "resolved_question": "how many orders last month?",
        "agent_type": "sql",
        "answer": "42",
        "sql": "SELECT count(*) FROM orders",
        "created_at": "2026-08-27T00:00:00+00:00",
        "kind": "answer",
        "hit_count": 0,
    }
    base.update(overrides)
    return base


def test_a_signed_payload_verifies() -> None:
    assert verify(sign(_payload(), BINDING, KEY), BINDING, KEY)


def test_an_unsigned_payload_does_not_verify() -> None:
    """Entries written before this change carry no signature, so they become
    cache misses rather than trusted input."""
    assert not verify(_payload(), BINDING, KEY)


@pytest.mark.parametrize("field", SIGNED_FIELDS)
def test_altering_any_signed_field_breaks_the_signature(field: str) -> None:
    signed = sign(_payload(), BINDING, KEY)

    assert not verify({**signed, field: "tampered"}, BINDING, KEY)


def test_hit_count_is_not_signed() -> None:
    """It is updated in place on every hit. Signing it would invalidate an entry
    the moment it was used."""
    signed = sign(_payload(), BINDING, KEY)

    assert verify({**signed, "hit_count": 99}, BINDING, KEY)


@pytest.mark.parametrize(
    "changed",
    [
        {"agent_id": "agent2"},
        {"connector_fingerprint": "other"},
        {"dialect": "mysql"},
        {"schema_fingerprint": "other"},
        {"policy_version": "2"},
    ],
    ids=lambda c: next(iter(c)),
)
def test_an_entry_does_not_verify_under_a_different_binding(changed: dict[str, str]) -> None:
    """A cached answer cannot be replayed against a different agent, connector,
    dialect, schema or policy version."""
    signed = sign(_payload(), BINDING, KEY)
    other = CacheBinding(**{**BINDING.__dict__, **changed})

    assert not verify(signed, other, KEY)


def test_the_dialect_comparison_ignores_case() -> None:
    signed = sign(_payload(), BINDING, KEY)
    other = CacheBinding(**{**BINDING.__dict__, "dialect": "POSTGRES"})

    assert verify(signed, other, KEY)


def test_a_different_key_does_not_verify() -> None:
    """The key is held outside Qdrant, so write access to the vector store does
    not confer the ability to produce a tag."""
    signed = sign(_payload(), BINDING, KEY)

    assert not verify(signed, BINDING, b"a-different-key")


def test_a_different_envelope_version_does_not_verify() -> None:
    """A change to the signed representation expires the cache rather than
    producing tags that disagree."""
    signed = sign(_payload(), BINDING, KEY)

    assert not verify({**signed, VERSION_KEY: "0"}, BINDING, KEY)


def test_an_empty_or_missing_signature_does_not_verify() -> None:
    signed = sign(_payload(), BINDING, KEY)

    assert not verify({**signed, SIGNATURE_KEY: ""}, BINDING, KEY)
    assert not verify({**signed, SIGNATURE_KEY: None}, BINDING, KEY)
    assert not verify({k: v for k, v in signed.items() if k != SIGNATURE_KEY}, BINDING, KEY)


def test_signing_adds_the_version(monkeypatch) -> None:
    signed = sign(_payload(), BINDING, KEY)

    assert signed[VERSION_KEY] == ENVELOPE_VERSION


class TestTheKey:
    def test_the_environment_is_used_when_set(self, monkeypatch) -> None:
        monkeypatch.setenv("NLQ_CACHE_SIGNING_KEY", "configured-key")

        assert signing_key() == b"configured-key"

    def test_a_key_is_generated_and_reused(self, monkeypatch, tmp_path) -> None:
        monkeypatch.delenv("NLQ_CACHE_SIGNING_KEY", raising=False)
        monkeypatch.setattr("nlqueries.cache.envelope.Path.home", lambda: tmp_path)

        first = signing_key()
        second = signing_key()

        assert first == second
        assert len(first) == 64

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
    def test_the_generated_key_is_readable_only_by_its_owner(self, monkeypatch, tmp_path) -> None:
        monkeypatch.delenv("NLQ_CACHE_SIGNING_KEY", raising=False)
        monkeypatch.setattr("nlqueries.cache.envelope.Path.home", lambda: tmp_path)

        signing_key()
        mode = (tmp_path / ".nlqueries" / "cache_signing_key").stat().st_mode

        assert not mode & (stat.S_IRGRP | stat.S_IROTH | stat.S_IWGRP | stat.S_IWOTH)
