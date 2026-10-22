"""Optional encryption of cached answers and document chunk text in Qdrant.

The cache and the document store run against an in-memory Qdrant, so what is
asserted about storage is read back from the store itself.
"""

from __future__ import annotations

import hashlib
import math
import os
import stat
import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from nlqueries import at_rest, config
from nlqueries.cache import semantic_cache
from nlqueries.cache.envelope import CacheBinding
from nlqueries.cache.semantic_cache import SemanticCache
from nlqueries.document_connectors.base import DocumentChunk
from nlqueries.embeddings import qdrant_store
from nlqueries.orchestrator.document_retrieval import retrieve_for_question
from qdrant_client import QdrantClient

ANSWER = "There were 42 orders from Acme Corp in June."
QUESTION = "How many orders did Acme place in June?"
CHUNK = "Refunds are accepted within 30 days of delivery."
BINDING = CacheBinding(
    agent_id="sales",
    connector_fingerprint="fp",
    dialect="postgres",
    schema_fingerprint="schema",
    policy_version="1",
)


def _vector(text: str) -> list[float]:
    vec = [0.0] * config.EMBED_DIMENSIONS
    for word in text.lower().split():
        digest = hashlib.sha256(word.strip(".,?!").encode()).digest()
        vec[int.from_bytes(digest[:4], "big") % config.EMBED_DIMENSIONS] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


@pytest.fixture(autouse=True)
def _clean_keys(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """No key from the environment, a state directory of the test's own."""
    monkeypatch.delenv(at_rest.KEY_ENV, raising=False)
    monkeypatch.delenv(at_rest.KEY_FILE_ENV, raising=False)
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(config, "CACHE_ENCRYPTION", False)
    at_rest._warned.clear()
    yield
    at_rest._cipher.cache_clear()


def _encrypt_with(monkeypatch: pytest.MonkeyPatch, key: str | None = "first key") -> None:
    monkeypatch.setattr(config, "CACHE_ENCRYPTION", True)
    if key is not None:
        monkeypatch.setenv(at_rest.KEY_ENV, key)


@pytest.fixture
def qdrant(monkeypatch: pytest.MonkeyPatch) -> Iterator[QdrantClient]:
    from nlqueries.embeddings import embedder

    client = QdrantClient(":memory:")
    monkeypatch.setattr(qdrant_store, "_client", client)
    monkeypatch.setattr(semantic_cache, "_cache_client", client)
    monkeypatch.setattr(semantic_cache, "embed_text", _vector)
    monkeypatch.setattr(embedder, "embed_batch", lambda texts: [_vector(t) for t in texts])
    monkeypatch.setattr(embedder, "embed_text", _vector)
    monkeypatch.setattr("nlqueries.cache.envelope.signing_key", lambda: b"signing key")
    collection = semantic_cache.cache_collection_name("sales")
    semantic_cache._known_collections.discard(collection)
    semantic_cache._missing_collections.pop(collection, None)
    yield client
    client.close()


def _payloads(client: QdrantClient, collection: str) -> list[dict[str, object]]:
    points, _ = client.scroll(collection_name=collection, limit=100, with_payload=True)
    return [dict(p.payload or {}) for p in points]


def _cache() -> SemanticCache:
    return SemanticCache("sales", binding=BINDING)


def _put(cache: SemanticCache) -> None:
    result = SimpleNamespace(
        resolved_question=QUESTION,
        agent_type="sql",
        answer=ANSWER,
        sql="SELECT count(*) FROM orders WHERE customer = 'Acme Corp'",
    )
    cache.put(QUESTION, result)


def _chunk(text: str, index: int = 0) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=hashlib.sha256(f"policy:{index}".encode()).hexdigest()[:16],
        source_id="policy",
        source_name="policy.md",
        page_number=None,
        chunk_index=index,
        text=text,
        metadata={},
    )


# --- the sealing itself -----------------------------------------------------------


def test_off_stores_text_as_it_is() -> None:
    assert at_rest.seal(ANSWER, at_rest.CACHE_ANSWER) == ANSWER
    assert at_rest.unseal(ANSWER, at_rest.CACHE_ANSWER) == ANSWER


def test_a_sealed_value_opens_with_its_key_and_shows_nothing(monkeypatch) -> None:
    _encrypt_with(monkeypatch)
    sealed = at_rest.seal(ANSWER, at_rest.CACHE_ANSWER)

    assert sealed.startswith(at_rest.MARKER)
    assert "Acme" not in sealed and "42" not in sealed
    assert at_rest.seal(ANSWER, at_rest.CACHE_ANSWER) != sealed  # a fresh nonce each time
    assert at_rest.unseal(sealed, at_rest.CACHE_ANSWER) == ANSWER


def test_another_key_does_not_open_it(monkeypatch, caplog) -> None:
    _encrypt_with(monkeypatch, "first key")
    sealed = at_rest.seal(ANSWER, at_rest.CACHE_ANSWER)
    monkeypatch.setenv(at_rest.KEY_ENV, "second key")

    assert at_rest.unseal(sealed, at_rest.CACHE_ANSWER) is None
    assert at_rest.unseal(sealed, at_rest.CACHE_ANSWER) is None
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1  # once per process, not once per read
    assert "another key" in warnings[0].getMessage()


def test_an_answer_cannot_be_moved_into_a_chunk(monkeypatch) -> None:
    _encrypt_with(monkeypatch)
    sealed = at_rest.seal(ANSWER, at_rest.CACHE_ANSWER)
    assert at_rest.unseal(sealed, at_rest.CHUNK_TEXT) is None


def test_a_damaged_value_does_not_open(monkeypatch) -> None:
    _encrypt_with(monkeypatch)
    sealed = at_rest.seal(ANSWER, at_rest.CACHE_ANSWER)
    flipped = sealed[:-4] + ("A" if sealed[-4] != "A" else "B") + sealed[-3:]
    assert at_rest.unseal(flipped, at_rest.CACHE_ANSWER) is None
    assert at_rest.unseal(at_rest.MARKER + "!!", at_rest.CACHE_ANSWER) is None


def test_with_no_key_anywhere_a_sealed_value_is_unreadable(monkeypatch) -> None:
    _encrypt_with(monkeypatch)
    sealed = at_rest.seal(ANSWER, at_rest.CACHE_ANSWER)
    monkeypatch.delenv(at_rest.KEY_ENV)
    assert at_rest.unseal(sealed, at_rest.CACHE_ANSWER) is None


def test_a_key_is_generated_once_and_kept_private(monkeypatch) -> None:
    _encrypt_with(monkeypatch, key=None)
    sealed = at_rest.seal(ANSWER, at_rest.CACHE_ANSWER)
    path = at_rest.key_path()
    first = path.read_bytes()

    assert len(first) == 64
    at_rest.seal(ANSWER, at_rest.CACHE_ANSWER)
    assert path.read_bytes() == first  # reused, not replaced
    assert at_rest.unseal(sealed, at_rest.CACHE_ANSWER) == ANSWER
    if sys.platform != "win32":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert not [p for p in path.parent.iterdir() if p.name.endswith(".tmp")]


def test_a_key_another_process_wrote_first_is_the_one_used(monkeypatch) -> None:
    """Two processes generating at once must not end up with two keys."""
    _encrypt_with(monkeypatch, key=None)
    path = at_rest.key_path()
    real_link = os.link

    def lose_the_race(src, dst):
        Path(dst).write_bytes(b"the other process's key")
        return real_link(src, dst)

    monkeypatch.setattr(at_rest.os, "link", lose_the_race)
    sealed = at_rest.seal(ANSWER, at_rest.CACHE_ANSWER)

    assert path.read_bytes() == b"the other process's key"
    assert at_rest.unseal(sealed, at_rest.CACHE_ANSWER) == ANSWER


def test_a_key_that_cannot_be_written_is_an_error_not_a_key_in_memory(monkeypatch) -> None:
    _encrypt_with(monkeypatch, key=None)

    def refuse(*args, **kwargs):
        raise PermissionError("read-only file system")

    monkeypatch.setattr(at_rest.tempfile, "mkstemp", refuse)
    with pytest.raises(RuntimeError, match=at_rest.KEY_ENV):
        at_rest.seal(ANSWER, at_rest.CACHE_ANSWER)


def test_a_key_file_is_read(monkeypatch, tmp_path) -> None:
    key_file = tmp_path / "key"
    key_file.write_text("mounted key\n")
    _encrypt_with(monkeypatch, key=None)
    monkeypatch.setenv(at_rest.KEY_FILE_ENV, str(key_file))
    sealed = at_rest.seal(ANSWER, at_rest.CACHE_ANSWER)

    monkeypatch.delenv(at_rest.KEY_FILE_ENV)
    monkeypatch.setenv(at_rest.KEY_ENV, "mounted key")
    assert at_rest.unseal(sealed, at_rest.CACHE_ANSWER) == ANSWER
    assert not at_rest.key_path().exists()


def test_a_configured_key_file_that_is_missing_refuses_to_seal(monkeypatch, tmp_path) -> None:
    _encrypt_with(monkeypatch, key=None)
    monkeypatch.setenv(at_rest.KEY_FILE_ENV, str(tmp_path / "absent"))
    with pytest.raises(RuntimeError, match="cannot be read"):
        at_rest.seal(ANSWER, at_rest.CACHE_ANSWER)
    assert not at_rest.key_path().exists()  # no quiet fallback to a generated key


# --- the cache --------------------------------------------------------------------


def test_a_cached_answer_round_trips_and_is_stored_sealed(monkeypatch, qdrant) -> None:
    _encrypt_with(monkeypatch)
    _put(_cache())

    entry = _cache().get(QUESTION)
    assert entry is not None and entry.answer == ANSWER
    (payload,) = _payloads(qdrant, semantic_cache.cache_collection_name("sales"))
    assert str(payload["answer"]).startswith(at_rest.MARKER)
    assert ANSWER not in str(payload)


def test_a_template_entry_is_sealed_too(monkeypatch, qdrant) -> None:
    _encrypt_with(monkeypatch)
    result = SimpleNamespace(
        resolved_question="orders with a total over 100",
        agent_type="sql",
        answer="Acme Corp has 7 orders over 100.",
        sql="SELECT * FROM orders WHERE total > 100",
    )
    _cache().put("orders with a total over 100", result)

    payloads = _payloads(qdrant, semantic_cache.cache_collection_name("sales"))
    assert sorted(str(p["kind"]) for p in payloads) == ["answer", "template"]
    for payload in payloads:
        assert str(payload["answer"]).startswith(at_rest.MARKER)
        assert "Acme" not in str(payload)


def test_a_cached_answer_under_another_key_is_a_miss(monkeypatch, qdrant) -> None:
    _encrypt_with(monkeypatch, "first key")
    _put(_cache())
    monkeypatch.setenv(at_rest.KEY_ENV, "second key")

    assert _cache().get(QUESTION) is None
    (listed,) = _cache().list_entries()
    assert listed.answer == "(encrypted; not readable with the current key)"


def test_with_encryption_off_the_cache_is_as_before(qdrant) -> None:
    _put(_cache())

    (payload,) = _payloads(qdrant, semantic_cache.cache_collection_name("sales"))
    assert payload["answer"] == ANSWER
    entry = _cache().get(QUESTION)
    assert entry is not None and entry.answer == ANSWER


def test_turning_it_on_keeps_plaintext_entries_readable(monkeypatch, qdrant) -> None:
    _put(_cache())
    _encrypt_with(monkeypatch)

    entry = _cache().get(QUESTION)
    assert entry is not None and entry.answer == ANSWER


def test_turning_it_off_keeps_sealed_entries_readable_while_the_key_is_there(
    monkeypatch, qdrant
) -> None:
    _encrypt_with(monkeypatch)
    _put(_cache())
    monkeypatch.setattr(config, "CACHE_ENCRYPTION", False)

    entry = _cache().get(QUESTION)
    assert entry is not None and entry.answer == ANSWER


# --- document chunks --------------------------------------------------------------


def test_chunk_text_is_stored_sealed_and_searched_in_the_clear(monkeypatch, qdrant) -> None:
    _encrypt_with(monkeypatch)
    qdrant_store.upsert_chunks("doc_policy_chunks", [_chunk(CHUNK)])

    (payload,) = _payloads(qdrant, "doc_policy_chunks")
    assert str(payload["text"]).startswith(at_rest.MARKER)
    assert "Refunds" not in str(payload)

    (found,) = qdrant_store.search_chunks("doc_policy_chunks", "refunds within 30 days")
    assert found.text == CHUNK
    (retrieved,) = retrieve_for_question("refunds within 30 days", "doc_policy_chunks").chunks
    assert retrieved.text == CHUNK


def test_a_chunk_under_another_key_is_skipped_and_plaintext_ones_still_read(
    monkeypatch, qdrant
) -> None:
    qdrant_store.upsert_chunks("doc_policy_chunks", [_chunk(CHUNK, 0)])  # before turning it on
    _encrypt_with(monkeypatch, "first key")
    qdrant_store.upsert_chunks("doc_policy_chunks", [_chunk("Refunds need a receipt.", 1)])
    monkeypatch.setenv(at_rest.KEY_ENV, "second key")

    found = qdrant_store.search_chunks("doc_policy_chunks", "refunds", top_k=5)
    assert [c.text for c in found] == [CHUNK]
    retrieved = retrieve_for_question("refunds", "doc_policy_chunks", top_k=5)
    assert [c.text for c in retrieved.chunks] == [CHUNK]
    assert len(retrieved.citations) == 1


def test_with_encryption_off_chunks_are_stored_as_before(qdrant) -> None:
    qdrant_store.upsert_chunks("doc_policy_chunks", [_chunk(CHUNK)])
    (payload,) = _payloads(qdrant, "doc_policy_chunks")
    assert payload["text"] == CHUNK
