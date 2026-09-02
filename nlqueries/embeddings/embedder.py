"""
nlqueries.embeddings.embedder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Sentence-transformer embedding module (OSS, local, free).

Uses ``all-MiniLM-L6-v2`` (384 dimensions).  When the embedding daemon is
running (``nlqueries embed-server start``), requests are forwarded to it over
localhost so the model is never re-loaded per CLI invocation (~10 ms vs ~9 s).
Falls back to loading the model in-process when the daemon is not running.

That fallback is right for the CLI and wrong for a server. In a uvicorn worker
it loads roughly a gigabyte of torch in the middle of a request, taking seconds,
and then keeps it — so a daemon that is merely *busy* leaves every worker
permanently fatter. Set ``EMBED_SERVER_REQUIRED`` on a server and the same
situation raises :class:`EmbeddingServiceUnavailable` instead, naming which of
the two problems it is: nothing listening, or something listening that could not
answer in time.
"""

from __future__ import annotations

import functools
import json as _json
import logging
import os as _os
import socket
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any, cast

from nlqueries import config

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Daemon connection
# ---------------------------------------------------------------------------

_DAEMON_PORT = int(_os.environ.get("EMBED_SERVER_PORT", "8765"))
_DAEMON_BASE = f"http://127.0.0.1:{_DAEMON_PORT}"

#: Whether the daemon's model has been compared with this process's, and the
#: mismatch if there was one. See `_require_daemon_model_agreement`.
_daemon_model_checked = False
_daemon_model_mismatch: str | None = None

#: The width refusal, once raised, so it costs one model load rather than one
#: per call. See `_get_model`.
_model_refusal: Exception | None = None

# Why the daemon did not answer. The two have different remedies — start it
# versus give it more room — so the difference is worth carrying rather than
# flattening into "unavailable".
ABSENT = "absent"  # nothing listening: no daemon on this host
FAILING = "failing"  # something listening, but it timed out or errored
#: Answering, and answering with the wrong model. Distinct from both above
#: because the remedy differs: restart the daemon so it picks up
#: NLQ_EMBED_MODEL.
MISMATCHED = "running a different model"


class EmbeddingServiceUnavailable(RuntimeError):
    """The embedding daemon was required for this call and could not serve it.

    Raised only when ``EMBED_SERVER_REQUIRED`` is set. Without it the caller
    falls back to embedding in-process, which is right for the CLI and wrong for
    a server — see the config docstring.
    """

    def __init__(self, reason: str, detail: str) -> None:
        self.reason = reason
        super().__init__(
            f"Embedding daemon {reason}: {detail}. "
            f"EMBED_SERVER_REQUIRED is set, so this call will not load a model "
            f"in-process instead."
        )


def _daemon_post(path: str, payload: dict[str, Any], key: str) -> Any:
    """POST to the daemon, translating transport failures into a reason.

    Raises:
        EmbeddingServiceUnavailable: always, on failure. Callers decide whether
            to propagate it or fall back, based on EMBED_SERVER_REQUIRED.
    """
    return _daemon_send(
        urllib.request.Request(
            f"{_DAEMON_BASE}{path}",
            data=_json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        ),
        key,
    )


def _daemon_get(path: str) -> Any:
    """GET from the daemon, with the same failure translation as a POST."""
    return _daemon_send(urllib.request.Request(f"{_DAEMON_BASE}{path}"))


def _daemon_send(request: urllib.request.Request, key: str | None = None) -> Any:
    """Send *request* and return its JSON body, or *key* within it.

    Factored out of ``_daemon_post`` so a GET is translated identically: the
    chain below is what tells a daemon that is absent from one that is failing,
    and a second copy of it would drift from this one.
    """
    try:
        with urllib.request.urlopen(
            request, timeout=config.EMBED_CLIENT_TIMEOUT_SECONDS
        ) as response:
            body = _json.loads(response.read())
            # Indexed inside the try deliberately: a reply missing the key is a
            # malformed reply, and the clauses below turn it into a failure
            # rather than letting a KeyError escape the client.
            return body if key is None else body[key]
    except urllib.error.HTTPError as exc:
        # It answered, so it exists — it is just not healthy.
        raise EmbeddingServiceUnavailable(FAILING, f"HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        # A refused connection means no daemon. A timeout means one that is too
        # busy to answer — which is precisely when falling back to loading a
        # model in this process does the most damage.
        if isinstance(exc.reason, TimeoutError | socket.timeout):
            raise EmbeddingServiceUnavailable(
                FAILING, f"no response in {config.EMBED_CLIENT_TIMEOUT_SECONDS}s"
            ) from exc
        raise EmbeddingServiceUnavailable(ABSENT, str(exc.reason)) from exc
    except TimeoutError as exc:  # raised directly by the socket on some platforms
        raise EmbeddingServiceUnavailable(
            FAILING, f"no response in {config.EMBED_CLIENT_TIMEOUT_SECONDS}s"
        ) from exc
    except Exception as exc:  # noqa: BLE001 — a malformed reply is still a failure
        raise EmbeddingServiceUnavailable(FAILING, str(exc)) from exc


def _fall_back_or_raise(exc: EmbeddingServiceUnavailable) -> None:
    """Log the degradation, and re-raise it when the daemon is mandatory.

    The fallback used to be silent, which is how a deployment could spend weeks
    loading torch models into its request path without anyone knowing. It is a
    warning now whether or not it is fatal.
    """
    if config.EMBED_SERVER_REQUIRED:
        raise exc
    logger.warning(
        "Embedding daemon %s (%s); embedding in-process instead. "
        "This loads a full model into THIS process — set EMBED_SERVER_REQUIRED "
        "on a server to make it a failure instead.",
        exc.reason,
        exc,
    )


def _require_daemon_model_agreement() -> None:
    """Check once that the daemon loaded the model this process expects.

    ``EMBED_MODEL`` is single-sourced, which makes the embedder and the daemon
    agree *within* one process. The daemon is a separate process and says
    nothing about which model produced the vectors it returns, so an operator
    who changes ``NLQ_EMBED_MODEL`` and restarts the API while the daemon keeps
    running -- or who starts the daemon from an environment where the variable
    is unset -- gets vectors from one model read back against a collection built
    by another. Nothing raises, and ``check_model_width`` cannot see it, because
    both models may well be 384 wide. That is the failure this module was
    changed to prevent, on the one axis a per-process constant cannot reach.

    Raised as the daemon being unavailable rather than as a fatal error, so the
    existing policy decides: ``EMBED_SERVER_REQUIRED`` makes it a failure, and
    otherwise this process embeds in-process with the model it was actually
    configured with -- slower, and correct.

    Checked once and remembered in both directions: a daemon does not change
    model without restarting, and a ``/healthz`` per embed would double the
    request count to learn a fact that cannot have changed. A daemon that is
    absent leaves this unchecked, so it is checked again when one appears.
    """
    global _daemon_model_checked, _daemon_model_mismatch  # noqa: PLW0603

    if not _daemon_model_checked:
        reported = _daemon_get("/healthz").get("model")
        _daemon_model_checked = True
        # An older daemon reports no model at all. Treated as agreement: it
        # cannot be compared, and refusing every such daemon would break a
        # rolling upgrade to gain nothing this process is able to verify.
        if reported and reported != _MODEL_NAME:
            _daemon_model_mismatch = (
                f"loaded {reported!r} while this process expects {_MODEL_NAME!r}"
            )

    if _daemon_model_mismatch:
        raise EmbeddingServiceUnavailable(MISMATCHED, _daemon_model_mismatch)


def _try_daemon_single(text: str) -> list[float] | None:
    """Return the embedding vector from the daemon, or None to fall back."""
    try:
        _require_daemon_model_agreement()
        return cast(list[float], _daemon_post("/embed", {"text": text}, "vector"))
    except EmbeddingServiceUnavailable as exc:
        _fall_back_or_raise(exc)
        return None


def _try_daemon_batch(texts: list[str]) -> list[list[float]] | None:
    """Return batch vectors from the daemon, or None to fall back."""
    try:
        _require_daemon_model_agreement()
        return cast(list[list[float]], _daemon_post("/embed-batch", {"texts": texts}, "vectors"))
    except EmbeddingServiceUnavailable as exc:
        _fall_back_or_raise(exc)
        return None


# ---------------------------------------------------------------------------
# Local model fallback (lazy singleton)
# ---------------------------------------------------------------------------

#: Kept as a module attribute because callers and tests refer to it, but the
#: value now comes from one place -- see :data:`nlqueries.config.EMBED_MODEL`.
_MODEL_NAME = config.EMBED_MODEL
_model: SentenceTransformer | None = None


def check_model_width(model: SentenceTransformer, name: str) -> None:
    """Refuse a model whose vectors would not fit the stores.

    ``NLQ_EMBED_MODEL`` exists so an operator can replace the weights without
    waiting for a release. The failure that invites is a model of a different
    width: the caches and the Qdrant collection are created at
    :data:`~nlqueries.config.EMBED_DIMENSIONS`, and a mismatch does not raise on
    its own. It writes vectors that simply do not mean what the collection
    thinks they mean, and the symptom is bad neighbours, months later, with
    nothing in a log to connect them to a swapped model.

    Checked at load, where the name of the offending model is still in hand.
    """
    # `get_sentence_embedding_dimension` was renamed to `get_embedding_dimension`
    # and now emits a FutureWarning. Both are read rather than one chosen: the
    # declared floor is `sentence-transformers>=3.0`, which has only the old
    # name, while the new one is what survives its removal.
    measure = getattr(model, "get_embedding_dimension", None)
    if measure is None:
        measure = model.get_sentence_embedding_dimension
    width = measure()
    if width is not None and width != config.EMBED_DIMENSIONS:
        raise RuntimeError(
            f"embedding model {name!r} produces {width}-dimension vectors, but this "
            f"deployment's caches and collections are built for {config.EMBED_DIMENSIONS}. "
            "Point NLQ_EMBED_MODEL at a model of the same width, or rebuild the "
            "vector stores for the new one."
        )


def _get_model() -> SentenceTransformer:
    """Return the shared ``SentenceTransformer`` instance, loading it on first call."""
    global _model, _model_refusal  # noqa: PLW0603
    if _model_refusal is not None:
        # Remembered, not re-derived. The refusal is permanent for this process
        # and the load in front of it costs seconds and about a gigabyte --
        # `_model` is assigned only once the check passes, so without this every
        # call built a fresh SentenceTransformer in order to reject it again.
        # Callers such as `promote_feedback` and the semantic cache catch and
        # carry on, which turned a misconfiguration into a full model load per
        # request rather than one visible failure.
        raise _model_refusal
    if _model is None:
        from sentence_transformers import SentenceTransformer  # deferred — heavy import

        loaded = SentenceTransformer(_MODEL_NAME)
        try:
            check_model_width(loaded, _MODEL_NAME)
        except RuntimeError as exc:
            _model_refusal = exc
            raise
        _model = loaded
    return _model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=2048)
def embed_text(text: str) -> list[float]:
    """Embed a single string and return a list of 384 floats (L2-normalised).

    Tries the embedding daemon first; falls back to loading the local model.
    Results are memoised (LRU, up to 2 048 unique strings) so repeated calls
    within a process — e.g. the same question hitting the semantic cache and
    the Qdrant search in the same request — cost zero additional compute.

    The returned list must not be mutated by callers.
    """
    result = _try_daemon_single(text)
    if result is not None:
        return result
    model = _get_model()
    # ndarray.tolist() is typed as Any, so the annotation is the only thing
    # asserting the shape here — same reason the daemon responses above are
    # cast. The model is fixed at 384 dimensions (see _MODEL_NAME).
    return cast(list[float], model.encode(text, normalize_embeddings=True).tolist())


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings and return one 384-float vector per input (L2-normalised).

    Tries the embedding daemon first; falls back to loading the local model.
    """
    result = _try_daemon_batch(texts)
    if result is not None:
        return result
    model = _get_model()
    vectors = model.encode(texts, batch_size=64, normalize_embeddings=True)
    return [v.tolist() for v in vectors]
