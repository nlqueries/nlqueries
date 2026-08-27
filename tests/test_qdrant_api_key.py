"""Every Qdrant client must send ``QDRANT_API_KEY`` when one is configured.

A Qdrant instance running with ``service.api_key`` rejects unauthenticated
requests with 401. Every path here degrades gracefully on failure — the semantic
cache skips silently and retrieval falls back to full-YAML injection, so a
client that omits the key does not fail visibly. It stops using the vector store
instead, which degrades answer quality and increases cost without an error.
"""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "nlqueries"


@pytest.fixture
def qdrant_class() -> MagicMock:
    """Patch the lazily-imported ``QdrantClient`` class and hand back the mock."""
    fake = MagicMock()
    with patch("qdrant_client.QdrantClient", fake):
        yield fake


def test_semantic_cache_client_sends_the_api_key(qdrant_class: MagicMock) -> None:
    from nlqueries.cache import semantic_cache

    with (
        patch.object(semantic_cache, "_cache_client", None),
        patch("nlqueries.config.QDRANT_API_KEY", "s3cret"),
    ):
        semantic_cache._get_client()

    assert qdrant_class.call_args.kwargs["api_key"] == "s3cret"


def test_embeddings_store_client_sends_the_api_key(qdrant_class: MagicMock) -> None:
    from nlqueries.embeddings import qdrant_store

    with (
        patch.object(qdrant_store, "_client", None),
        patch("nlqueries.config.QDRANT_API_KEY", "s3cret"),
    ):
        qdrant_store._get_client()

    assert qdrant_class.call_args.kwargs["api_key"] == "s3cret"


def test_no_key_configured_passes_none_not_empty_string(qdrant_class: MagicMock) -> None:
    """An unauthenticated instance is the common local setup.

    ``api_key=""`` is not the same as ``api_key=None``: the empty string makes
    the client send an empty ``api-key`` header, which an unauthenticated Qdrant
    ignores but a proxy in front of it may not. Pass None.
    """
    from nlqueries.embeddings import qdrant_store

    with (
        patch.object(qdrant_store, "_client", None),
        patch("nlqueries.config.QDRANT_API_KEY", ""),
    ):
        qdrant_store._get_client()

    assert qdrant_class.call_args.kwargs["api_key"] is None


def test_every_qdrant_client_in_the_package_passes_an_api_key() -> None:
    """Guard for the two call sites built inline, and for future ones.

    ``feedback/promoter.py`` and ``orchestrator/prompt_assembly.py`` construct a
    client inside a try/except rather than through a cached getter, so they can't
    be covered by the fixtures above without contorting the tests.
    """
    offenders: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            is_qdrant_call = (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "QdrantClient"
            )
            if is_qdrant_call and not any(kw.arg == "api_key" for kw in node.keywords):
                offenders.append(f"{path.relative_to(PACKAGE_ROOT.parent)}:{node.lineno}")

    assert not offenders, (
        f"These QdrantClient calls don't pass api_key: {offenders}. Against an "
        "authenticated Qdrant they get 401, and the caller degrades silently. Use "
        "api_key=config.QDRANT_API_KEY or None."
    )
