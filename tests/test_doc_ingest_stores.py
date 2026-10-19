"""`doc-ingest` and the two `doc-sync` commands store what they chunk.

They used to chunk a source, print a count and keep nothing, so `doc-ask`
searched a collection no core command ever filled. These tests run the CLI
against an in-memory Qdrant with a deterministic embedder and search the
result the way `doc-ask` does.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner
from nlqueries import config
from nlqueries.cli.main import cli
from nlqueries.document_connectors.base import DocumentChunk
from nlqueries.embeddings import qdrant_store
from qdrant_client import QdrantClient


def _fake_vector(text: str) -> list[float]:
    """Bag of words hashed into the embedding width, L2-normalised."""
    vec = [0.0] * config.EMBED_DIMENSIONS
    for word in text.lower().split():
        digest = hashlib.sha256(word.strip(".,?!").encode()).digest()
        vec[int.from_bytes(digest[:4], "big") % config.EMBED_DIMENSIONS] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


@pytest.fixture
def qdrant(monkeypatch: pytest.MonkeyPatch) -> Iterator[QdrantClient]:
    from nlqueries.embeddings import embedder

    client = QdrantClient(":memory:")
    monkeypatch.setattr(qdrant_store, "_client", client)
    monkeypatch.setattr(embedder, "embed_batch", lambda texts: [_fake_vector(t) for t in texts])
    monkeypatch.setattr(embedder, "embed_text", _fake_vector)
    yield client
    client.close()


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _stored_texts(client: QdrantClient, collection: str) -> list[str]:
    points, _ = client.scroll(collection_name=collection, limit=1000, with_payload=True)
    return sorted(str((p.payload or {}).get("text", "")) for p in points)


def _chunk(source_id: str, index: int, text: str) -> DocumentChunk:
    chunk_id = hashlib.sha256(f"{source_id}:None:{index}".encode()).hexdigest()[:16]
    return DocumentChunk(
        chunk_id=chunk_id,
        source_id=source_id,
        source_name="test",
        page_number=None,
        chunk_index=index,
        text=text,
        metadata={},
    )


def test_an_ingested_document_can_be_found_by_doc_ask(tmp_path: Path, qdrant: QdrantClient) -> None:
    path = _write(
        tmp_path,
        "policy.md",
        "# Refunds\nRefunds are accepted within thirty days.\n\n"
        "# Shipping\nOrders ship from the Leeds warehouse.\n",
    )

    result = CliRunner().invoke(cli, ["doc-ingest", "policy", str(path)])

    assert result.exit_code == 0, result.output
    assert "doc_policy_chunks" in result.output
    hits = qdrant_store.search_chunks("doc_policy_chunks", "refunds thirty days", top_k=1)
    assert len(hits) == 1
    assert "thirty days" in hits[0].text
    assert hits[0].source_id == "policy"


def test_ingesting_again_replaces_the_earlier_version(tmp_path: Path, qdrant: QdrantClient) -> None:
    long_doc = "".join(f"# Part {i}\nSection {i} of the old handbook.\n\n" for i in range(4))
    path = _write(tmp_path, "handbook.md", long_doc)
    assert CliRunner().invoke(cli, ["doc-ingest", "hb", str(path)]).exit_code == 0
    assert len(_stored_texts(qdrant, "doc_hb_chunks")) == 4

    path.write_text("# Part 0\nThe new handbook has one section.\n", encoding="utf-8")
    result = CliRunner().invoke(cli, ["doc-ingest", "hb", str(path)])

    assert result.exit_code == 0, result.output
    texts = _stored_texts(qdrant, "doc_hb_chunks")
    assert len(texts) == 1
    assert "new handbook" in texts[0]


def test_an_empty_read_stores_nothing_and_keeps_the_stored_copy(
    tmp_path: Path, qdrant: QdrantClient
) -> None:
    """A wrong space key, a page whose text is in blocks the connector skips,
    or a scanned PDF all read as empty; storing that would delete a good copy."""
    path = _write(tmp_path, "notes.txt", "Some notes worth keeping.\n")
    assert CliRunner().invoke(cli, ["doc-ingest", "notes", str(path)]).exit_code == 0

    path.write_text("", encoding="utf-8")
    result = CliRunner().invoke(cli, ["doc-ingest", "notes", str(path)])

    assert result.exit_code == 1
    assert "produced no text" in result.output
    assert "--allow-empty" in result.output
    assert "Ingestion complete" not in result.output
    assert _stored_texts(qdrant, "doc_notes_chunks") == ["Some notes worth keeping."]


def test_allow_empty_removes_the_stored_copy(tmp_path: Path, qdrant: QdrantClient) -> None:
    path = _write(tmp_path, "notes.txt", "Some notes worth keeping.\n")
    assert CliRunner().invoke(cli, ["doc-ingest", "notes", str(path)]).exit_code == 0

    path.write_text("", encoding="utf-8")
    result = CliRunner().invoke(cli, ["doc-ingest", "notes", str(path), "--allow-empty"])

    assert result.exit_code == 0, result.output
    assert _stored_texts(qdrant, "doc_notes_chunks") == []


def test_replacing_one_source_keeps_another_in_the_same_collection(
    qdrant: QdrantClient,
) -> None:
    qdrant_store.replace_chunks(
        "shared", "a", [_chunk("a", 0, "alpha one"), _chunk("a", 1, "alpha two")]
    )
    qdrant_store.replace_chunks("shared", "b", [_chunk("b", 0, "beta one")])

    qdrant_store.replace_chunks("shared", "a", [_chunk("a", 0, "alpha new")])

    assert _stored_texts(qdrant, "shared") == ["alpha new", "beta one"]


def test_a_failed_store_keeps_the_previous_version(
    qdrant: QdrantClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nlqueries.embeddings import embedder

    qdrant_store.replace_chunks("doc_x_chunks", "x", [_chunk("x", 0, "kept text")])

    def _down(texts: list[str]) -> list[list[float]]:
        raise ConnectionError("embedding daemon unavailable")

    monkeypatch.setattr(embedder, "embed_batch", _down)
    with pytest.raises(ConnectionError):
        qdrant_store.replace_chunks("doc_x_chunks", "x", [_chunk("x", 0, "replacement")])

    assert _stored_texts(qdrant, "doc_x_chunks") == ["kept text"]


def test_a_failed_delete_after_the_store_says_both_versions_remain(
    tmp_path: Path, qdrant: QdrantClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one window where a failure leaves a blend: the new chunks are in,
    the old tail is not out. The error has to say so, not "nothing changed"."""
    long_doc = "".join(f"# Part {i}\nSection {i} of the old handbook.\n\n" for i in range(3))
    path = _write(tmp_path, "handbook.md", long_doc)
    assert CliRunner().invoke(cli, ["doc-ingest", "hb", str(path)]).exit_code == 0

    def _delete_fails(*args: Any, **kwargs: Any) -> None:
        raise TimeoutError("timed out deleting points")

    monkeypatch.setattr(qdrant, "delete", _delete_fails)
    path.write_text("# Part 0\nThe new handbook has one section.\n", encoding="utf-8")
    result = CliRunner().invoke(cli, ["doc-ingest", "hb", str(path)])

    assert result.exit_code == 1
    assert "Stored, but not cleanly" in result.output
    assert "both versions" in result.output
    assert "timed out deleting points" in result.output
    assert "Ingestion complete" not in result.output
    texts = _stored_texts(qdrant, "doc_hb_chunks")
    assert len(texts) == 3 and any("new handbook" in t for t in texts)


def test_doc_ingest_fails_when_nothing_could_be_stored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reporting "Ingestion complete" with nothing stored is the bug this fixes."""

    def _refuse(*args: Any, **kwargs: Any) -> None:
        raise ConnectionError("Connection refused: localhost:6333")

    monkeypatch.setattr(qdrant_store, "replace_chunks", _refuse)
    path = _write(tmp_path, "a.md", "# A\nBody.\n")

    result = CliRunner().invoke(cli, ["doc-ingest", "a", str(path)])

    assert result.exit_code == 1
    assert "Could not store the chunks" in result.output
    assert "localhost:6333" in result.output
    assert "was not deleted" in result.output
    assert "Ingestion complete" not in result.output


def test_notion_since_with_an_unchanged_page_leaves_the_stored_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nlqueries.document_connectors import notion

    stored: list[Any] = []
    monkeypatch.setenv("NOTION_API_TOKEN", "secret")
    monkeypatch.setattr(notion.NotionConnector, "ingest", lambda self, *a, **k: [])
    monkeypatch.setattr(qdrant_store, "replace_chunks", lambda *a: stored.append(a))

    result = CliRunner().invoke(
        cli, ["doc-sync-notion", "wiki", "page1", "--since", "2026-01-01T00:00:00+00:00"]
    )

    assert result.exit_code == 0, result.output
    assert stored == []


def test_notion_full_sync_reading_nothing_stores_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    from nlqueries.document_connectors import notion

    stored: list[Any] = []
    monkeypatch.setenv("NOTION_API_TOKEN", "secret")
    monkeypatch.setattr(notion.NotionConnector, "ingest", lambda self, *a, **k: [])
    monkeypatch.setattr(qdrant_store, "replace_chunks", lambda *a: stored.append(a))

    result = CliRunner().invoke(cli, ["doc-sync-notion", "wiki", "page1"])

    assert result.exit_code == 1
    assert "produced no text" in result.output
    assert stored == []


def test_notion_sync_stores_the_page(monkeypatch: pytest.MonkeyPatch) -> None:
    from nlqueries.document_connectors import notion

    stored: list[Any] = []
    chunks = [_chunk("wiki", 0, "page text")]
    monkeypatch.setenv("NOTION_API_TOKEN", "secret")
    monkeypatch.setattr(notion.NotionConnector, "ingest", lambda self, *a, **k: chunks)
    monkeypatch.setattr(qdrant_store, "replace_chunks", lambda *a: stored.append(a))

    result = CliRunner().invoke(
        cli, ["doc-sync-notion", "wiki", "page1", "--since", "2026-01-01T00:00:00+00:00"]
    )

    assert result.exit_code == 0, result.output
    assert stored == [("doc_wiki_chunks", "wiki", chunks)]


_CONFLUENCE_ARGS = [
    "doc-sync-confluence",
    "eng",
    "ENG",
    "--base-url",
    "https://acme.atlassian.net",
    "--username",
    "alice@acme.com",
]
_SINCE = ["--since", "2026-01-01T00:00:00+00:00"]


def _stub_confluence(
    monkeypatch: pytest.MonkeyPatch, *, changed: bool, whole: list[DocumentChunk]
) -> tuple[MagicMock, MagicMock, list[Any]]:
    from nlqueries.document_connectors import confluence

    probe = MagicMock(return_value=changed)
    ingest = MagicMock(return_value=whole)
    stored: list[Any] = []
    monkeypatch.setenv("CONFLUENCE_API_TOKEN", "secret")
    monkeypatch.setattr(
        confluence.ConfluenceConnector, "has_changes_since", lambda self, *a: probe(*a)
    )
    monkeypatch.setattr(
        confluence.ConfluenceConnector, "ingest", lambda self, *a, **k: ingest(*a, **k)
    )
    monkeypatch.setattr(qdrant_store, "replace_chunks", lambda *a: stored.append(a))
    return probe, ingest, stored


def test_confluence_since_with_changes_reads_the_whole_space_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A --since read numbers only the changed pages' chunks from zero, so
    storing it would overwrite other pages' chunks. --since only decides
    whether to read, and the one read is of the whole space."""
    whole = [_chunk("eng", 0, "page one"), _chunk("eng", 1, "changed page")]
    probe, ingest, stored = _stub_confluence(monkeypatch, changed=True, whole=whole)

    result = CliRunner().invoke(cli, _CONFLUENCE_ARGS + _SINCE)

    assert result.exit_code == 0, result.output
    probe.assert_called_once()
    ingest.assert_called_once()
    assert ingest.call_args.kwargs.get("since") is None
    assert stored == [("doc_eng_chunks", "eng", whole)]


def test_confluence_since_with_no_changes_reads_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    probe, ingest, stored = _stub_confluence(monkeypatch, changed=False, whole=[])

    result = CliRunner().invoke(cli, _CONFLUENCE_ARGS + _SINCE)

    assert result.exit_code == 0, result.output
    probe.assert_called_once()
    ingest.assert_not_called()
    assert stored == []
    assert "stored copy is unchanged" in result.output


def test_confluence_full_sync_reading_nothing_stores_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mistyped space key or a lost permission returns no pages."""
    probe, ingest, stored = _stub_confluence(monkeypatch, changed=True, whole=[])

    result = CliRunner().invoke(cli, _CONFLUENCE_ARGS)

    assert result.exit_code == 1
    assert "produced no text" in result.output
    probe.assert_not_called()
    assert stored == []


def test_confluence_sync_stores_nothing_when_the_space_could_not_be_read_whole(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nlqueries.document_connectors.confluence import IncompleteSpaceRead

    probe, ingest, stored = _stub_confluence(monkeypatch, changed=True, whole=[])
    ingest.side_effect = IncompleteSpaceRead("the search stopped advancing after 100 page(s)")

    result = CliRunner().invoke(cli, _CONFLUENCE_ARGS)

    assert result.exit_code == 1
    assert "stopped advancing" in result.output
    assert stored == []
