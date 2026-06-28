"""
nlqueries.embeddings.embedder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Sentence-transformer embedding module (OSS, local, free).

Uses ``all-MiniLM-L6-v2`` (384 dimensions).  When the embedding daemon is
running (``nlqueries embed-server start``), requests are forwarded to it over
localhost so the model is never re-loaded per CLI invocation (~10 ms vs ~9 s).
Falls back to loading the model in-process when the daemon is not running.
"""

from __future__ import annotations

import json as _json
import os as _os
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Daemon connection
# ---------------------------------------------------------------------------

_DAEMON_PORT = int(_os.environ.get("EMBED_SERVER_PORT", "8765"))
_DAEMON_BASE = f"http://127.0.0.1:{_DAEMON_PORT}"


def _try_daemon_single(text: str) -> list[float] | None:
    """Return the embedding vector from the daemon, or None if unreachable."""
    try:
        body = _json.dumps({"text": text}).encode()
        req = urllib.request.Request(
            f"{_DAEMON_BASE}/embed",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            return cast(list[float], _json.loads(resp.read())["vector"])
    except Exception:  # noqa: BLE001
        return None


def _try_daemon_batch(texts: list[str]) -> list[list[float]] | None:
    """Return batch vectors from the daemon, or None if unreachable."""
    try:
        body = _json.dumps({"texts": texts}).encode()
        req = urllib.request.Request(
            f"{_DAEMON_BASE}/embed-batch",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            return cast(list[list[float]], _json.loads(resp.read())["vectors"])
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Local model fallback (lazy singleton)
# ---------------------------------------------------------------------------

_MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Return the shared ``SentenceTransformer`` instance, loading it on first call."""
    global _model  # noqa: PLW0603
    if _model is None:
        from sentence_transformers import SentenceTransformer  # deferred — heavy import

        _model = SentenceTransformer(_MODEL_NAME)
    return _model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def embed_text(text: str) -> list[float]:
    """Embed a single string and return a list of 384 floats (L2-normalised).

    Tries the embedding daemon first; falls back to loading the local model.
    """
    result = _try_daemon_single(text)
    if result is not None:
        return result
    model = _get_model()
    return model.encode(text, normalize_embeddings=True).tolist()


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
