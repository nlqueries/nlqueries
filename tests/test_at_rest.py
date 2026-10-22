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
import time
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
    monkeypatch.setattr(at_rest, "_warned_short", False)
    monkeypatch.setattr(at_rest, "_gave_up_waiting", False)
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
    warnings = [r for r in caplog.records if "could not be read" in r.getMessage()]
    assert len(warnings) == 1  # once per process, not once per read
    assert "another key" in warnings[0].getMessage()


def test_unreadable_chunks_warn_once_for_the_kind_not_per_document(monkeypatch, caplog) -> None:
    """A chunk's associated data carries its document; the warning must not."""
    _encrypt_with(monkeypatch, "first key")
    sealed = {
        source: at_rest.seal(CHUNK, at_rest.chunk_text_purpose(source))
        for source in ("policy", "salaries")
    }
    monkeypatch.setenv(at_rest.KEY_ENV, "second key")

    for source, value in sealed.items():
        assert at_rest.unseal(value, at_rest.chunk_text_purpose(source)) is None

    warnings = [r.getMessage() for r in caplog.records if "could not be read" in r.getMessage()]
    assert len(warnings) == 1
    assert "nlqueries document chunk text could not be read" in warnings[0]
    assert "\x00" not in warnings[0] and "policy" not in warnings[0]


def test_each_kind_warns_once_of_its_own(monkeypatch, caplog) -> None:
    """Once per kind, not once in all: an answer and a chunk are two warnings."""
    _encrypt_with(monkeypatch, "first key")
    answer = at_rest.seal(ANSWER, at_rest.CACHE_ANSWER)
    chunk = at_rest.seal(CHUNK, at_rest.chunk_text_purpose("policy"))
    monkeypatch.setenv(at_rest.KEY_ENV, "second key")

    for _ in range(2):
        assert at_rest.unseal(answer, at_rest.CACHE_ANSWER) is None
        assert at_rest.unseal(chunk, at_rest.chunk_text_purpose("policy")) is None

    warnings = [r.getMessage() for r in caplog.records if "could not be read" in r.getMessage()]
    assert len(warnings) == 2
    assert "nlqueries cache answer could not be read" in warnings[0]
    assert "nlqueries document chunk text could not be read" in warnings[1]


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

    theirs = b"ab" * 32

    def lose_the_race(src, dst):
        Path(dst).write_bytes(theirs)
        return real_link(src, dst)

    monkeypatch.setattr(at_rest.os, "link", lose_the_race)
    sealed = at_rest.seal(ANSWER, at_rest.CACHE_ANSWER)

    assert path.read_bytes() == theirs
    assert at_rest.unseal(sealed, at_rest.CACHE_ANSWER) == ANSWER


def _no_hard_links(monkeypatch) -> None:
    def refuse(src, dst):
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(at_rest.os, "link", refuse)


def test_without_hard_links_the_key_is_created_in_place_once(monkeypatch) -> None:
    """The only path taken on a filesystem that cannot link."""
    _encrypt_with(monkeypatch, key=None)
    _no_hard_links(monkeypatch)
    sealed = at_rest.seal(ANSWER, at_rest.CACHE_ANSWER)
    path = at_rest.key_path()
    first = path.read_bytes()

    assert len(first) == 64
    at_rest.seal(ANSWER, at_rest.CACHE_ANSWER)
    assert path.read_bytes() == first
    assert at_rest.unseal(sealed, at_rest.CACHE_ANSWER) == ANSWER
    if sys.platform != "win32":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert not [p for p in path.parent.iterdir() if p.name.endswith(".tmp")]


def test_a_key_still_being_written_is_not_used(monkeypatch) -> None:
    """Seen half-written, a key would seal text no other process can open."""
    _encrypt_with(monkeypatch, key=None)
    _no_hard_links(monkeypatch)
    path = at_rest.key_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"ab" * 10)  # the winner is 20 bytes into its 64
    monkeypatch.setattr(at_rest, "_KEY_WAIT_SECONDS", 0.2)

    with pytest.raises(RuntimeError, match="no complete key"):
        at_rest.seal(ANSWER, at_rest.CACHE_ANSWER)
    assert at_rest.unseal(at_rest.MARKER + "AAAA", at_rest.CACHE_ANSWER) is None

    # Left half written for good: the next write does not wait again.
    monkeypatch.setattr(at_rest, "_KEY_WAIT_SECONDS", 30.0)
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="no complete key"):
        at_rest.seal(ANSWER, at_rest.CACHE_ANSWER)
    assert time.monotonic() - started < 5


def test_a_stored_file_that_is_not_a_key_is_not_waited_on(monkeypatch) -> None:
    """An operator's own key put at that path, say: it will never become one."""
    _encrypt_with(monkeypatch, key=None)
    _no_hard_links(monkeypatch)
    path = at_rest.key_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"correct horse battery staple")
    monkeypatch.setattr(at_rest, "_KEY_WAIT_SECONDS", 30.0)

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="no complete key"):
        at_rest.seal(ANSWER, at_rest.CACHE_ANSWER)
    assert time.monotonic() - started < 5


def test_without_hard_links_the_key_is_created_at_0600_not_chmodded(monkeypatch) -> None:
    """The mode is given when the file is created, before the key is in it."""
    _encrypt_with(monkeypatch, key=None)
    _no_hard_links(monkeypatch)
    opened: list[tuple[int, int]] = []
    real_open = at_rest.os.open

    def record(path, flags, mode=0o777, *args, **kwargs):
        if Path(path) == at_rest.key_path():  # not mkstemp's own temporary file
            opened.append((flags, mode))
        return real_open(path, flags, mode, *args, **kwargs)

    monkeypatch.setattr(at_rest.os, "open", record)
    at_rest.seal(ANSWER, at_rest.CACHE_ANSWER)

    ((flags, mode),) = opened
    assert mode == 0o600
    assert flags & os.O_EXCL and flags & os.O_CREAT


def test_without_hard_links_the_key_is_on_disk_before_it_is_used(monkeypatch) -> None:
    """Synced as the linked key is: an empty file after a power cut never completes."""
    _encrypt_with(monkeypatch, key=None)
    _no_hard_links(monkeypatch)
    synced: list[bytes] = []
    real_fsync = at_rest.os.fsync

    def record(fd):
        real_fsync(fd)
        synced.append(at_rest.key_path().read_bytes() if at_rest.key_path().exists() else b"")

    monkeypatch.setattr(at_rest.os, "fsync", record)
    at_rest.seal(ANSWER, at_rest.CACHE_ANSWER)

    # The temporary file's sync happens before the key path exists; the
    # fallback's happens with the key written to it.
    assert len(synced[-1]) == 64


def test_the_loser_waits_for_the_winners_key_to_be_complete(monkeypatch) -> None:
    _encrypt_with(monkeypatch, key=None)
    _no_hard_links(monkeypatch)
    path = at_rest.key_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"cd" * 10)
    theirs = b"cd" * 32

    def finish_writing(seconds):
        path.write_bytes(theirs)

    monkeypatch.setattr(at_rest.time, "sleep", finish_writing)
    sealed = at_rest.seal(ANSWER, at_rest.CACHE_ANSWER)

    monkeypatch.setenv(at_rest.KEY_ENV, theirs.decode())
    assert at_rest.unseal(sealed, at_rest.CACHE_ANSWER) == ANSWER


def test_a_key_that_cannot_be_written_is_an_error_not_a_key_in_memory(monkeypatch) -> None:
    _encrypt_with(monkeypatch, key=None)

    def refuse(*args, **kwargs):
        raise PermissionError("read-only file system")

    monkeypatch.setattr(at_rest.tempfile, "mkstemp", refuse)
    with pytest.raises(RuntimeError, match=at_rest.KEY_ENV):
        at_rest.seal(ANSWER, at_rest.CACHE_ANSWER)


def test_a_stored_key_this_process_cannot_read_is_no_key(monkeypatch) -> None:
    """Written by another user at 0600, say: reading treats it as missing and
    writing fails naming the settings, rather than either raising PermissionError."""
    _encrypt_with(monkeypatch, "first key")
    sealed = at_rest.seal(ANSWER, at_rest.CACHE_ANSWER)
    monkeypatch.delenv(at_rest.KEY_ENV)
    path = at_rest.key_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"0" * 64)
    real_read = Path.read_bytes

    def refuse(self):
        if self == path:
            raise PermissionError(13, "Permission denied", str(self))
        return real_read(self)

    monkeypatch.setattr(Path, "read_bytes", refuse)
    monkeypatch.setattr(at_rest, "_KEY_WAIT_SECONDS", 30.0)

    assert at_rest.unseal(sealed, at_rest.CACHE_ANSWER) is None
    started = time.monotonic()
    with pytest.raises(RuntimeError, match=at_rest.KEY_ENV):
        at_rest.seal(ANSWER, at_rest.CACHE_ANSWER)
    assert time.monotonic() - started < 5  # nothing to wait for: it will not change


def test_a_short_configured_key_is_warned_about_once(monkeypatch, caplog) -> None:
    _encrypt_with(monkeypatch, "hunter2")
    at_rest.seal(ANSWER, at_rest.CACHE_ANSWER)
    at_rest.seal(ANSWER, at_rest.CACHE_ANSWER)
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "7 characters" in warnings[0] and "openssl rand -hex 32" in warnings[0]


def test_a_long_configured_key_is_not_warned_about(monkeypatch, caplog) -> None:
    _encrypt_with(monkeypatch, "a" * 64)
    at_rest.seal(ANSWER, at_rest.CACHE_ANSWER)
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


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


def test_a_chunk_moved_to_another_document_does_not_open(monkeypatch, qdrant) -> None:
    """Chunk payloads are not signed; the sealed text is bound to its document."""
    from qdrant_client.models import PointStruct

    _encrypt_with(monkeypatch)
    qdrant_store.upsert_chunks("doc_policy_chunks", [_chunk(CHUNK)])
    points, _ = qdrant.scroll(
        collection_name="doc_policy_chunks", limit=10, with_payload=True, with_vectors=True
    )
    (point,) = points
    moved = {**(point.payload or {}), "source_id": "salaries"}
    qdrant_store.upsert_chunks("doc_salaries_chunks", [_chunk("placeholder")])
    qdrant.upsert(
        collection_name="doc_salaries_chunks",
        points=[PointStruct(id=point.id, vector=point.vector, payload=moved)],
    )

    assert qdrant_store.chunk_from_payload(point.payload or {}) is not None
    assert qdrant_store.chunk_from_payload(moved) is None


def test_the_first_key_leaves_the_state_directory_as_it_was(monkeypatch) -> None:
    """It also holds the knowledge base and the connector registry."""
    _encrypt_with(monkeypatch, key=None)
    state = at_rest.key_path().parent
    state.mkdir(parents=True)
    calls: list[object] = []
    real_chmod = Path.chmod

    def record(self, mode, *args, **kwargs):
        calls.append(self)
        return real_chmod(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "chmod", record)
    at_rest.seal(ANSWER, at_rest.CACHE_ANSWER)

    assert state not in calls
    assert at_rest.key_path().exists()


def test_with_encryption_off_chunks_are_stored_as_before(qdrant) -> None:
    qdrant_store.upsert_chunks("doc_policy_chunks", [_chunk(CHUNK)])
    (payload,) = _payloads(qdrant, "doc_policy_chunks")
    assert payload["text"] == CHUNK
