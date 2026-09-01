"""The embedding model can be swapped, and a wrong-width one cannot get in.

``NLQ_EMBED_MODEL`` exists so an operator can replace the weights without
waiting for a release — the case that matters is a published vulnerability in a
model artefact, where our release cadence should not be in the way.

The failure that invites is a model of a different width. The caches and the
Qdrant collection are created at ``EMBED_DIMENSIONS``; a mismatched model does
not raise on its own, it writes vectors that do not mean what the collection
thinks they mean. The symptom is bad neighbours, later, with nothing in a log
connecting them to a swapped model — so it is refused at load, where the name
of the offending model is still in hand.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest
from nlqueries import config
from nlqueries.embeddings.embedder import check_model_width


class _Model:
    def __init__(self, width: int | None) -> None:
        self._width = width

    def get_sentence_embedding_dimension(self) -> int | None:
        return self._width


def test_a_model_of_the_wrong_width_is_refused() -> None:
    with pytest.raises(RuntimeError) as raised:
        check_model_width(_Model(768), "some-other-model")  # type: ignore[arg-type]
    message = str(raised.value)
    # The name has to be in the message: at this point it is the only thing
    # that tells an operator which knob they turned.
    assert "some-other-model" in message
    assert "768" in message
    assert str(config.EMBED_DIMENSIONS) in message


def test_a_model_of_the_right_width_passes() -> None:
    check_model_width(_Model(config.EMBED_DIMENSIONS), "all-MiniLM-L6-v2")  # type: ignore[arg-type]


def test_a_model_that_will_not_say_its_width_is_not_refused() -> None:
    """Some implementations return None. Refusing on "unknown" would block a
    legitimate swap over a missing attribute, which is worse than the risk."""
    check_model_width(_Model(None), "quiet-model")  # type: ignore[arg-type]


class _RenamedModel:
    """A model exposing only the post-rename accessor."""

    def __init__(self, width: int | None) -> None:
        self._width = width

    def get_embedding_dimension(self) -> int | None:
        return self._width


def test_the_width_is_read_through_either_accessor() -> None:
    """sentence-transformers renamed ``get_sentence_embedding_dimension`` to
    ``get_embedding_dimension``; the old name still works and now warns.

    ``_Model`` above has only the old name, so every other test here exercises
    that path. This one has only the new name. Without it the check would rest
    on whichever version happens to be installed, and the guard would go quiet
    -- returning None, refusing nothing -- on the day the old name is removed.
    """
    with pytest.raises(RuntimeError) as raised:
        check_model_width(_RenamedModel(768), "some-other-model")  # type: ignore[arg-type]
    assert "768" in str(raised.value)

    check_model_width(_RenamedModel(config.EMBED_DIMENSIONS), "all-MiniLM-L6-v2")  # type: ignore[arg-type]


def test_the_daemon_and_the_embedder_cannot_disagree() -> None:
    """The property behind single-sourcing the name.

    Vectors written through the daemon are read back through the in-process
    embedder. Two hardcoded copies of the model name would not raise if they
    drifted apart; they would return wrong neighbours.
    """
    from nlqueries.embeddings import embed_server, embedder

    assert embedder._MODEL_NAME == embed_server._MODEL_NAME == config.EMBED_MODEL


def test_the_default_is_unchanged_and_the_override_is_honoured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default must not move: an existing deployment's collections were built
    with it, so a changed default would silently invalidate them."""
    monkeypatch.delenv("NLQ_EMBED_MODEL", raising=False)
    reloaded: Any = importlib.reload(config)
    assert reloaded.EMBED_MODEL == "all-MiniLM-L6-v2"

    monkeypatch.setenv("NLQ_EMBED_MODEL", "/models/pinned-copy")
    reloaded = importlib.reload(config)
    assert reloaded.EMBED_MODEL == "/models/pinned-copy", (
        "a filesystem path is the point: it is how a baked image's weights get replaced"
    )

    monkeypatch.delenv("NLQ_EMBED_MODEL", raising=False)
    importlib.reload(config)


def test_every_store_is_built_for_the_width_the_model_produces() -> None:
    """The contract has to be one number, or the guard above is decorative.

    `check_model_width` refuses a model that does not match EMBED_DIMENSIONS.
    That means nothing if the collections are built from their own copies of the
    number: raising the constant to admit a wider model would leave the caches
    and collections still created at the old width, and the guard would wave
    through vectors the stores cannot hold.
    """
    from nlqueries.cache.semantic_cache import CACHE_VECTOR_SIZE
    from nlqueries.embeddings.qdrant_store import DOCUMENT_VECTOR_SIZE
    from nlqueries.feedback.promoter import VERIFIED_VECTOR_SIZE

    assert CACHE_VECTOR_SIZE == config.EMBED_DIMENSIONS
    assert DOCUMENT_VECTOR_SIZE == config.EMBED_DIMENSIONS
    assert VERIFIED_VECTOR_SIZE == config.EMBED_DIMENSIONS


def test_no_module_states_the_vector_width_for_itself() -> None:
    """Catches the next one, which the equality checks above cannot.

    They name three constants that exist today; a fourth store added with its
    own literal would pass them while being exactly the drift this closes. So
    the source is searched instead -- the width is declared in config or not at
    all.
    """
    import re
    from pathlib import Path

    package = Path(config.__file__).parent
    pattern = re.compile(r"=\s*" + str(config.EMBED_DIMENSIONS) + r"\b")
    stray = [
        f"{path.relative_to(package)}:{n}: {line.strip()}"
        for path in package.rglob("*.py")
        if path.name != "config.py"
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if pattern.search(line)
    ]
    assert not stray, (
        "the embedding width is stated outside config.EMBED_DIMENSIONS:\n  " + "\n  ".join(stray)
    )
