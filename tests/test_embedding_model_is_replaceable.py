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
import sys
from typing import Any
from unittest.mock import MagicMock, patch

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


# ---------------------------------------------------------------------------
# The cross-process half of the property (#175 review)
# ---------------------------------------------------------------------------


def _reset_daemon_check(embedder: Any) -> None:
    embedder._daemon_model_checked = False
    embedder._daemon_model_mismatch = None


def test_the_daemon_reports_its_model_over_healthz() -> None:
    """Asserted on the HTTP response, not on the handler attribute.

    The first version of this test checked that `build_server` stored the name,
    which is not the property that matters: deleting the line that puts it in
    the `/healthz` body left the attribute set and the test green, so the guard
    passed over the exact thing it was written for. `/healthz` is the only
    channel by which another process can learn this, so the response is what
    has to carry it.
    """
    import json as _json
    import socket
    import threading
    import time
    import urllib.request

    from nlqueries.embeddings.embed_server import build_server

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    server = build_server(
        port,
        encoder=lambda texts: [[0.0] * config.EMBED_DIMENSIONS for _ in texts],
        model_name="some-pinned-model",
        backend="torch",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.05)
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=5) as response:
            body = _json.loads(response.read())
    finally:
        server.shutdown()
        server.server_close()

    assert body["model"] == "some-pinned-model"


def test_a_daemon_running_another_model_is_not_trusted() -> None:
    """The failure a per-process constant cannot reach.

    Single-sourcing the name makes the embedder and the daemon agree inside one
    process. The daemon is a separate one, and an operator who changes
    NLQ_EMBED_MODEL and restarts only the API gets vectors from one model read
    back against a collection built by another -- with nothing raised, and
    invisible to `check_model_width`, because both models may be 384 wide.
    """
    from nlqueries.embeddings import embedder

    _reset_daemon_check(embedder)
    with patch.object(embedder, "_daemon_get", return_value={"model": "other-model"}) as probe:
        for _ in range(3):
            with pytest.raises(embedder.EmbeddingServiceUnavailable) as raised:
                embedder._require_daemon_model_agreement()
            assert raised.value.reason == embedder.MISMATCHED
        assert "other-model" in str(raised.value)
        # Re-asked every time rather than remembered, which is what makes the
        # recovery below possible. This is the degraded path already, so the
        # extra request costs nothing that matters.
        assert probe.call_count == 3
    _reset_daemon_check(embedder)


def test_a_daemon_restarted_with_the_right_model_is_trusted_again() -> None:
    """The remedy this reason names is restarting the daemon.

    Remembering the mismatch meant an operator who did exactly that found
    nothing had changed: with EMBED_SERVER_REQUIRED set every embed kept
    raising, still telling them to restart the daemon they had just restarted,
    until the API process was restarted too. The absent case never had that
    problem, but only because its exception leaves the checked flag false --
    accidentally right, where this is now deliberate.
    """
    from nlqueries.embeddings import embedder

    reported = {"model": "other-model"}
    _reset_daemon_check(embedder)
    with patch.object(embedder, "_daemon_get", side_effect=lambda _path: dict(reported)):
        with pytest.raises(embedder.EmbeddingServiceUnavailable):
            embedder._require_daemon_model_agreement()

        reported["model"] = embedder._MODEL_NAME  # the operator restarts the daemon
        embedder._require_daemon_model_agreement()  # must not raise

        # And having agreed, it goes back to being remembered.
        embedder._require_daemon_model_agreement()
    _reset_daemon_check(embedder)


def test_a_reply_that_is_not_an_object_falls_back_instead_of_raising() -> None:
    """Something other than the daemon bound to this port.

    `_daemon_send` exists to turn every failure into the one exception the
    callers catch. Returning the body unexamined for a keyless read left
    `.get(...)` outside that protection, so valid JSON that is not an object
    raised AttributeError straight past `except EmbeddingServiceUnavailable` --
    where the same process already translated a malformed POST reply and
    carried on.
    """
    from nlqueries.embeddings import embedder

    class _Reply:
        def read(self) -> bytes:
            return b"[1, 2, 3]"

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *_exc: Any) -> bool:
            return False

    _reset_daemon_check(embedder)
    with (
        patch.object(embedder.urllib.request, "urlopen", return_value=_Reply()),
        pytest.raises(embedder.EmbeddingServiceUnavailable),
    ):
        embedder._require_daemon_model_agreement()
    _reset_daemon_check(embedder)


def test_a_matching_daemon_is_asked_once_and_then_trusted() -> None:
    """The cost of the check has to be one request, not one per embed."""
    from nlqueries.embeddings import embedder

    _reset_daemon_check(embedder)
    with patch.object(
        embedder, "_daemon_get", return_value={"model": embedder._MODEL_NAME}
    ) as probe:
        for _ in range(5):
            embedder._require_daemon_model_agreement()  # must not raise
        assert probe.call_count == 1
    _reset_daemon_check(embedder)


def test_a_daemon_that_does_not_say_is_treated_as_agreeing() -> None:
    """An older daemon reports no model at all. Refusing every one of those
    would break a rolling upgrade to gain nothing this process can verify."""
    from nlqueries.embeddings import embedder

    _reset_daemon_check(embedder)
    with patch.object(embedder, "_daemon_get", return_value={"status": "ok"}):
        embedder._require_daemon_model_agreement()  # must not raise
    _reset_daemon_check(embedder)


def test_a_refused_model_is_not_loaded_again_on_every_call() -> None:
    """`_model` is assigned only after the width check passes, so a mis-sized
    model meant a fresh SentenceTransformer -- seconds, and about a gigabyte --
    constructed per call in order to reject it again. Callers such as
    `promote_feedback` and the semantic cache catch and carry on, so a
    misconfiguration degraded into repeated loads on the request path instead of
    one visible failure.
    """
    from nlqueries.embeddings import embedder

    class _WrongWidth:
        def get_embedding_dimension(self) -> int:
            return config.EMBED_DIMENSIONS + 128

    loads = 0

    def _construct(_name: str) -> _WrongWidth:
        nonlocal loads
        loads += 1
        return _WrongWidth()

    fake_st = MagicMock()
    fake_st.SentenceTransformer = _construct

    saved_model, saved_refusal = embedder._model, embedder._model_refusal
    embedder._model, embedder._model_refusal = None, None
    try:
        with patch.dict(sys.modules, {"sentence_transformers": fake_st}):
            for _ in range(4):
                with pytest.raises(RuntimeError):
                    embedder._get_model()
        assert loads == 1, f"the model was constructed {loads} times to be refused each time"
    finally:
        embedder._model, embedder._model_refusal = saved_model, saved_refusal


def test_the_remembered_refusal_is_a_fresh_exception_each_time() -> None:
    """Re-raising one object appends a frame to its ``__traceback__`` per raise.

    The refusal is a module global and callers such as ``promote_feedback``
    catch and carry on, so that chain would grow -- and keep every frame it
    references alive -- for the life of a misconfigured process. A slow leak, on
    the one request path this caching was added to protect.
    """
    import traceback

    from nlqueries.embeddings import embedder

    class _WrongWidth:
        def get_embedding_dimension(self) -> int:
            return config.EMBED_DIMENSIONS + 128

    fake_st = MagicMock()
    fake_st.SentenceTransformer = lambda _name: _WrongWidth()

    saved_model, saved_refusal = embedder._model, embedder._model_refusal
    embedder._model, embedder._model_refusal = None, None
    raised: list[BaseException] = []
    try:
        with patch.dict(sys.modules, {"sentence_transformers": fake_st}):
            for _ in range(4):
                try:
                    embedder._get_model()
                except RuntimeError as exc:
                    raised.append(exc)
    finally:
        embedder._model, embedder._model_refusal = saved_model, saved_refusal

    assert len(raised) == 4
    assert len({id(exc) for exc in raised}) == 4, "the same exception object was re-raised"
    depths = [len(traceback.extract_tb(exc.__traceback__)) for exc in raised]
    # The first raise travels through `check_model_width` and is naturally the
    # deeper one. The property is that the repeats do not *accumulate*: with a
    # single object re-raised, every raise appended a frame and these climbed.
    repeats = depths[1:]
    assert len(set(repeats)) == 1, f"traceback depth grew across raises: {depths}"
    assert max(repeats) <= depths[0], f"a repeat was deeper than the original: {depths}"
