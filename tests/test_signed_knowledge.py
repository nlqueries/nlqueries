"""Knowledge read back from Qdrant is only trusted when this deployment wrote it.

Two kinds: verified question/SQL pairs, which reach the prompt as user-confirmed
examples, and semantic-cache entries, whose SQL can run without a model in
between. Pairs are signed now. Cache entries already were, but the vector a hit
is found by was not, so a signed entry could be moved onto another question.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from nlqueries import config
from nlqueries.cache import envelope
from nlqueries.cache import semantic_cache as sc
from nlqueries.cache.envelope import CacheBinding, sign, sign_pair, verify_pair
from nlqueries.feedback.promoter import _verified_collection
from nlqueries.orchestrator import prompt_assembly
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

KEY = b"k" * 64
OTHER_KEY = b"o" * 64
AGENT = "postgres:localhost:shop"

PAIR = {
    "question": "How many orders last month?",
    "sql": "SELECT COUNT(*) FROM orders",
    "agent_id": AGENT,
    "verified": True,
}

# ---------------------------------------------------------------------------
# Signing a pair
# ---------------------------------------------------------------------------


class TestPairSignatures:
    def test_a_signed_pair_verifies(self) -> None:
        assert verify_pair(sign_pair(PAIR, KEY), KEY)

    def test_an_unsigned_pair_does_not(self) -> None:
        """Pairs promoted before signing existed are in this state."""
        assert not verify_pair(dict(PAIR), KEY)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("question", "How many refunds last month?"),
            ("sql", "SELECT * FROM customers"),
            ("agent_id", "postgres:localhost:other"),
            ("verified", False),
        ],
    )
    def test_changing_a_signed_field_breaks_it(self, field: str, value: Any) -> None:
        signed = sign_pair(PAIR, KEY)
        assert not verify_pair({**signed, field: value}, KEY)

    def test_another_key_does_not_verify_it(self) -> None:
        assert not verify_pair(sign_pair(PAIR, KEY), OTHER_KEY)

    def test_another_version_does_not_verify(self) -> None:
        signed = sign_pair(PAIR, KEY)
        assert not verify_pair({**signed, envelope.PAIR_VERSION_KEY: "0"}, KEY)

    def test_a_cache_entrys_tag_is_not_a_pairs(self) -> None:
        """The two kinds of tag are made over differently labelled messages,
        so neither can stand in for the other."""
        binding = CacheBinding(AGENT, "c", "postgres", "s", "3")
        as_cache_entry = sign(dict(PAIR), binding, KEY)
        forged = {**as_cache_entry, envelope.PAIR_VERSION_KEY: envelope.PAIR_VERSION}
        assert not verify_pair(forged, KEY)

    def test_with_an_unusable_key_nothing_verifies_and_it_is_said_once(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        signed = sign_pair(PAIR, KEY)
        monkeypatch.setenv(envelope._KEY_ENV, "x")
        monkeypatch.setattr(envelope, "_KEY_ERROR_LOGGED", [])
        with caplog.at_level(logging.ERROR, logger=envelope.__name__):
            assert not verify_pair(signed)
            assert not verify_pair(signed)
        assert len([r for r in caplog.records if "disabled" in r.getMessage()]) == 1


def test_promote_feedback_writes_signed_pairs(monkeypatch: pytest.MonkeyPatch) -> None:
    from nlqueries.feedback.promoter import promote_feedback

    from tests.test_promoter import _make_feedback_record, _make_kb

    monkeypatch.setattr(envelope, "signing_key", lambda: KEY)
    written: list[dict[str, Any]] = []
    client = MagicMock()
    client.upsert.side_effect = lambda collection_name, points: written.extend(
        p.payload for p in points
    )
    with (
        patch("nlqueries.feedback.store.load_feedback", return_value=[_make_feedback_record()]),
        patch("nlqueries.feedback.promoter._load_kb", return_value=_make_kb(["orders"])),
        patch("nlqueries.embeddings.qdrant_store.ensure_collection", return_value=None),
        patch(
            "nlqueries.embeddings.embedder.embed_batch",
            side_effect=lambda qs: [[0.1] * 384 for _ in qs],
        ),
        patch("qdrant_client.QdrantClient", return_value=client),
    ):
        assert promote_feedback("agent1") == 1

    assert written and all(verify_pair(p, KEY) for p in written)


# ---------------------------------------------------------------------------
# Reading pairs
# ---------------------------------------------------------------------------


def _unit(index: int) -> list[float]:
    vector = [0.0] * config.EMBED_DIMENSIONS
    vector[index] = 1.0
    return vector


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> Iterator[QdrantClient]:
    monkeypatch.setattr(envelope, "signing_key", lambda: KEY)
    monkeypatch.setattr(prompt_assembly, "_UNSIGNED_PAIRS_WARNED", set())
    monkeypatch.setattr(prompt_assembly, "_REPOINTED_PAIRS_WARNED", set())
    monkeypatch.setattr(prompt_assembly, "_PAIR_RECHECK_FAILED_WARNED", set())
    # A pair's own question embeds to where the asked question is, as it does
    # for a pair stored honestly; "refunds" questions embed elsewhere.
    monkeypatch.setattr(
        "nlqueries.embeddings.embedder.embed_batch",
        lambda texts: [_unit(1) if "refunds" in t else _unit(0) for t in texts],
    )
    client = QdrantClient(":memory:")
    client.create_collection(
        _verified_collection(AGENT),
        vectors_config=VectorParams(size=config.EMBED_DIMENSIONS, distance=Distance.COSINE),
    )
    with patch("qdrant_client.QdrantClient", return_value=client):
        yield client
    client.close()


def _put(client: QdrantClient, point_id: int, payload: dict[str, Any], vector: list[float]) -> None:
    client.upsert(
        _verified_collection(AGENT),
        points=[PointStruct(id=point_id, vector=vector, payload=payload)],
    )


def _search(top_k: int = 3) -> list[dict[str, Any]]:
    return prompt_assembly._search_verified(
        f"agent_{AGENT}_schema", PAIR["question"], top_k=top_k, vector=_unit(0)
    )


class TestReadingPairs:
    def test_a_signed_pair_is_used(self, store: QdrantClient) -> None:
        _put(store, 1, sign_pair(PAIR, KEY), _unit(0))
        assert [p["sql"] for p in _search()] == [PAIR["sql"]]

    def test_a_pair_written_straight_into_qdrant_is_not(
        self, store: QdrantClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Anyone able to write to Qdrant could add a `verified: true` point,
        and it was shown to the model as a user-confirmed example."""
        planted = {**PAIR, "sql": "SELECT * FROM customers"}
        _put(store, 1, planted, _unit(0))
        _put(store, 2, sign_pair(PAIR, KEY), _unit(0))

        with caplog.at_level(logging.WARNING, logger=prompt_assembly.__name__):
            found = _search()
            _search()

        assert [p["sql"] for p in found] == [PAIR["sql"]]
        warned = [r for r in caplog.records if "not signed" in r.getMessage()]
        assert len(warned) == 1 and "promote-feedback" in warned[0].getMessage()

    def test_unsigned_pairs_do_not_crowd_out_signed_ones(self, store: QdrantClient) -> None:
        """The nearest k are read with room to spare, so a few unsigned pairs
        closer to the question than the real one do not push it out of the
        prompt. A bound, not a guarantee: more than k times the candidate factor
        would."""
        for i in range(3):
            _put(store, 10 + i, {**PAIR, "sql": f"SELECT {i}"}, _unit(0))
        near = [0.0] * config.EMBED_DIMENSIONS
        near[0], near[1] = 0.9, 0.1
        _put(store, 1, sign_pair(PAIR, KEY), near)

        assert [p["sql"] for p in _search(top_k=1)] == [PAIR["sql"]]

    def test_each_agent_with_unsigned_pairs_is_named(
        self, store: QdrantClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An MCP server serves several agents, and the upgrade step the warning
        asks for runs per agent: a once-per-process warning named only the
        first, and the others' examples went missing in silence."""
        other = "postgres:localhost:other"
        store.create_collection(
            _verified_collection(other),
            vectors_config=VectorParams(size=config.EMBED_DIMENSIONS, distance=Distance.COSINE),
        )
        _put(store, 1, PAIR, _unit(0))
        store.upsert(
            _verified_collection(other),
            points=[PointStruct(id=1, vector=_unit(0), payload={**PAIR, "agent_id": other})],
        )

        with caplog.at_level(logging.WARNING, logger=prompt_assembly.__name__):
            _search()
            prompt_assembly._search_verified(
                f"agent_{other}_schema", PAIR["question"], top_k=3, vector=_unit(0)
            )
            _search()

        warned = [r.getMessage() for r in caplog.records if "not signed" in r.getMessage()]
        assert len(warned) == 2
        assert any(repr(AGENT) in m for m in warned) and any(repr(other) in m for m in warned)

    def test_pairs_that_cannot_be_re_embedded_are_said_to_be_left_out(
        self, store: QdrantClient, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Without a word, an embedder failing here looked like an agent with no
        verified examples at all."""

        def _down(texts: list[str]) -> list[list[float]]:
            raise ConnectionError("embedder restarting")

        monkeypatch.setattr("nlqueries.embeddings.embedder.embed_batch", _down)
        _put(store, 1, sign_pair(PAIR, KEY), _unit(0))

        with caplog.at_level(logging.WARNING, logger=prompt_assembly.__name__):
            assert _search() == []
            _search()

        warned = [r.getMessage() for r in caplog.records if "re-embed" in r.getMessage()]
        assert len(warned) == 1 and "ConnectionError" in warned[0]
        assert not any("should not happen" in r.getMessage() for r in caplog.records)

    def test_another_agents_signed_pair_is_not_used_even_if_qdrant_returns_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The agent filter runs inside Qdrant, which is the party not trusted
        here: a server that ignores it can return a genuinely signed pair that
        belongs to another agent."""
        monkeypatch.setattr(envelope, "signing_key", lambda: KEY)
        monkeypatch.setattr(prompt_assembly, "_UNSIGNED_PAIRS_WARNED", set())
        theirs = sign_pair({**PAIR, "agent_id": "postgres:localhost:other"}, KEY)
        hit = MagicMock(score=0.99, payload=theirs)
        coll = MagicMock()
        coll.name = _verified_collection(AGENT)
        client = MagicMock()
        client.get_collections.return_value.collections = [coll]
        client.query_points.return_value = MagicMock(points=[hit])
        with patch("qdrant_client.QdrantClient", return_value=client):
            assert _search() == []

    def test_a_signed_pair_moved_onto_another_questions_vector_is_not_used(
        self, store: QdrantClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The signature does not cover the vector a pair is stored under, so a
        genuine pair copied onto another question's vector was shown as a
        confirmed example for that question. Its own question is checked now."""
        other = sign_pair({**PAIR, "question": "How many refunds last month?"}, KEY)
        _put(store, 1, other, _unit(0))
        with caplog.at_level(logging.WARNING, logger=prompt_assembly.__name__):
            assert _search() == []
            assert _search() == []
        warned = [r for r in caplog.records if "does not match" in r.getMessage()]
        assert len(warned) == 1

    def test_a_signed_pair_relabelled_for_another_agent_is_not_used(
        self, store: QdrantClient
    ) -> None:
        theirs = sign_pair({**PAIR, "agent_id": "postgres:localhost:other"}, KEY)
        _put(store, 1, {**theirs, "agent_id": AGENT}, _unit(0))
        assert _search() == []


# ---------------------------------------------------------------------------
# Cache entries moved onto another question's vector
# ---------------------------------------------------------------------------

BINDING = CacheBinding("agent1", "conn-fp", "postgres", "schema-fp", "3")


@pytest.fixture
def cache_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envelope, "signing_key", lambda: KEY)
    sc._known_collections.discard("cache_agent1")
    sc._missing_collections.pop("cache_agent1", None)


def _embedding(vectors: dict[str, list[float]], default: list[float]) -> Any:
    return lambda text: vectors.get(text, default)


def _hit(payload: dict[str, Any], score: float) -> MagicMock:
    hit = MagicMock()
    hit.score, hit.payload, hit.id = score, payload, 7
    return hit


def _client(tier1: list[Any], tier2: list[Any] | None = None) -> MagicMock:
    coll = MagicMock()
    coll.name = "cache_agent1"
    client = MagicMock()
    client.get_collections.return_value.collections = [coll]
    client.retrieve.return_value = []
    responses = [MagicMock(points=tier1)]
    if tier2 is not None:
        responses.append(MagicMock(points=tier2))
    client.query_points.side_effect = responses
    return client


def _answer(question: str) -> dict[str, Any]:
    return sign(
        {
            "question": question,
            "resolved_question": question,
            "agent_type": "sql",
            "answer": "There were 42.",
            "sql": "SELECT count(*) FROM orders",
            "created_at": datetime.now(UTC).isoformat(),
            "hit_count": 0,
            "kind": "answer",
        },
        BINDING,
        KEY,
    )


class TestRepointedCacheEntries:
    ASKED = "how many orders were there"
    OTHER = "how many refunds were there"

    def _get(self, stored_question: str) -> Any:
        vectors = {self.ASKED: _unit(0), self.OTHER: _unit(1)}
        client = _client([_hit(_answer(stored_question), 0.99)], tier2=[])
        with (
            patch.object(sc, "_get_client", return_value=client),
            patch.object(sc, "embed_text", side_effect=_embedding(vectors, _unit(0))),
        ):
            return sc.SemanticCache("agent1", binding=BINDING).get(self.ASKED, vector=_unit(0))

    def test_an_entry_found_by_its_own_question_is_served(self, cache_key: None) -> None:
        entry = self._get(self.ASKED)
        assert entry is not None and entry.question == self.ASKED

    def test_a_signed_entry_moved_onto_another_questions_vector_is_not(
        self, cache_key: None, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Its signature is genuine, but it answers another question: Qdrant
        found it by a vector the signature does not cover. Said once per point,
        as tier 0 says when a point holds the wrong question."""
        monkeypatch.setattr(sc, "_REPOINTED_POINTS_LOGGED", set())
        with caplog.at_level(logging.WARNING, logger=sc.__name__):
            assert self._get(self.OTHER) is None
            assert self._get(self.OTHER) is None
        warned = [r for r in caplog.records if "does not match" in r.getMessage()]
        assert len(warned) == 1 and "tier 1" in warned[0].getMessage()

    def test_an_embedder_failure_is_a_miss_not_a_repointing(
        self, cache_key: None, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """With the query's vector supplied, re-embedding the stored question is
        the only embed: an embedder that fails there was reported as a
        collection written to directly, and the point marked as reported."""
        monkeypatch.setattr(sc, "_REPOINTED_POINTS_LOGGED", set())
        monkeypatch.setattr(sc, "_RECHECK_EMBED_FAILED_LOGGED", set())
        client = _client([_hit(_answer(self.ASKED), 0.99)], tier2=[])
        with (
            patch.object(sc, "_get_client", return_value=client),
            patch.object(sc, "embed_text", side_effect=ConnectionError("embedder restarting")),
            caplog.at_level(logging.WARNING, logger=sc.__name__),
        ):
            assert (
                sc.SemanticCache("agent1", binding=BINDING).get(self.ASKED, vector=_unit(0)) is None
            )

        messages = [r.getMessage() for r in caplog.records]
        assert any("could not re-embed" in m and "tier 1" in m for m in messages)
        assert not any("does not match" in m for m in messages)
        assert set() == sc._REPOINTED_POINTS_LOGGED

    def test_a_template_moved_onto_another_questions_vector_is_not(self, cache_key: None) -> None:
        template = sign(
            {
                "question": "refunds after <DATE>",
                "resolved_question": "refunds after <DATE>",
                "agent_type": "sql",
                "answer": "Found results.",
                "sql": "SELECT * FROM orders WHERE d >= '[d:DATE]'",
                "created_at": datetime.now(UTC).isoformat(),
                "hit_count": 0,
                "kind": "template",
            },
            BINDING,
            KEY,
        )
        vectors = {"orders after <DATE>": _unit(0), "refunds after <DATE>": _unit(1)}

        def get(stored: dict[str, Any]) -> Any:
            client = _client([], tier2=[_hit(stored, 0.95)])
            with (
                patch.object(sc, "_get_client", return_value=client),
                patch.object(sc, "embed_text", side_effect=_embedding(vectors, _unit(0))),
            ):
                return sc.SemanticCache("agent1", binding=BINDING).get("orders after 2024-06-01")

        assert get(template) is None
        own = sign({**template, "question": "orders after <DATE>"}, BINDING, KEY)
        assert get(own) is not None

    def test_the_check_tolerates_float_noise_but_not_a_different_question(self) -> None:
        # Cosine 0.985: under the 0.99 threshold, within the tolerance.
        nearly = [0.0] * config.EMBED_DIMENSIONS
        nearly[0], nearly[1] = 0.985, (1 - 0.985**2) ** 0.5
        with patch.object(sc, "embed_text", return_value=nearly):
            assert sc._stored_question_matches("q", _unit(0), 0.99)
        with patch.object(sc, "embed_text", return_value=_unit(1)):
            assert sc._stored_question_matches("q", _unit(0), 0.99) is False
        assert sc._stored_question_matches(None, _unit(0), 0.99) is False
        # Not checkable is not a mismatch.
        with patch.object(sc, "embed_text", side_effect=ConnectionError("down")):
            assert sc._stored_question_matches("q", _unit(0), 0.99, "tier 1") is None
