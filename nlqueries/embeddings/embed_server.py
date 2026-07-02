"""
nlqueries.embeddings.embed_server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Lightweight embedding daemon.  Loads all-MiniLM-L6-v2 once and serves
POST /embed and POST /embed-batch over localhost:8765.

Two backends are supported via the ``EMBED_BACKEND`` environment variable:

``torch`` (default)
    Loads ``all-MiniLM-L6-v2`` via ``sentence-transformers``/PyTorch.
    No extra dependencies beyond the base install.

``onnx``
    Loads the model via ``optimum[onnxruntime]`` and performs mean-pooling
    + L2 normalisation with NumPy — no PyTorch required at serve time.
    Cold-start is ~3–5× faster than the torch backend.
    Requires: ``pip install "optimum[onnxruntime]" transformers``

Run via:  python -m nlqueries.embeddings.embed_server [--port N]
Managed:  nlqueries embed-server start / stop / status
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import signal
import subprocess
import sys
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, cast

_MODEL_NAME = "all-MiniLM-L6-v2"
_DEFAULT_PORT = 8765
_PID_FILE = pathlib.Path.home() / ".nlqueries" / "embed-server.pid"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Process-existence helper
# ---------------------------------------------------------------------------


def is_pid_alive(pid: int) -> bool:
    """Return True if a process with *pid* is currently running.

    Uses tasklist on Windows because os.kill(pid, 0) maps to CTRL_C_EVENT
    (value 0) there, which sends Ctrl+C to the whole console group instead of
    checking process existence.
    """
    if sys.platform == "win32":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return str(pid) in result.stdout
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Mean-pool + L2-normalise (used by the ONNX backend)
# ---------------------------------------------------------------------------


def _mean_pool_normalize(
    token_embeddings: Any,  # (batch, seq, hidden) — numpy array or array-like
    attention_mask: Any,  # (batch, seq) — numpy array or array-like
) -> list[list[float]]:
    """Mean-pool token embeddings and L2-normalise each row.

    Replicates sentence-transformers' pooling behaviour so the ONNX backend
    produces vectors that are numerically equivalent to the torch backend.

    Args:
        token_embeddings: ``last_hidden_state`` output of the ONNX model,
                          shape ``(batch, seq_len, hidden_dim)``.
        attention_mask:   Tokenizer attention mask, shape ``(batch, seq_len)``.

    Returns:
        ``list[list[float]]`` of length *batch*, each inner list having
        ``hidden_dim`` elements.
    """
    import numpy as np  # noqa: PLC0415

    token_embeddings = np.array(token_embeddings, dtype=np.float32)
    mask = np.array(attention_mask, dtype=np.float32)[:, :, np.newaxis]  # (B, S, 1)

    sum_emb = np.sum(token_embeddings * mask, axis=1)  # (B, H)
    sum_mask = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)  # (B, 1)
    embeddings = sum_emb / sum_mask  # (B, H)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.where(norms > 0, norms, 1.0)
    return cast(list[list[float]], normalized.tolist())


# ---------------------------------------------------------------------------
# Backend loaders — return an encoder callable (texts) -> list[list[float]]
# ---------------------------------------------------------------------------


def _load_torch_encoder() -> Callable[[list[str]], list[list[float]]]:
    """Load sentence-transformers and return a batch-encode callable."""
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    model = SentenceTransformer(_MODEL_NAME)

    def _encode(texts: list[str]) -> list[list[float]]:
        return cast(list[list[float]], model.encode(texts, normalize_embeddings=True).tolist())

    return _encode


def _load_onnx_encoder() -> Callable[[list[str]], list[list[float]]]:
    """Load the model via ONNX Runtime and return a batch-encode callable.

    Requires ``optimum[onnxruntime]`` and ``transformers``.  The model is
    exported to ONNX on first load (cached in the HuggingFace hub cache).
    """
    from optimum.onnxruntime import ORTModelForFeatureExtraction  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415

    ort_model = ORTModelForFeatureExtraction.from_pretrained(_MODEL_NAME, export=True)
    tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)

    def _encode(texts: list[str]) -> list[list[float]]:
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="np")
        outputs = ort_model(**inputs)
        return _mean_pool_normalize(outputs.last_hidden_state, inputs["attention_mask"])

    return _encode


def _load_encoder(backend: str) -> Callable[[list[str]], list[list[float]]]:
    """Return the appropriate encoder callable for *backend*.

    Args:
        backend: ``"onnx"`` or any other value (treated as ``"torch"``).

    Returns:
        A callable ``(texts: list[str]) -> list[list[float]]`` that embeds a
        batch of strings and returns L2-normalised vectors.
    """
    if backend == "onnx":
        logger.info("Loading ONNX Runtime embedding backend (%s)", _MODEL_NAME)
        return _load_onnx_encoder()
    logger.info("Loading torch/sentence-transformers backend (%s)", _MODEL_NAME)
    return _load_torch_encoder()


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------


class _EmbedHandler(BaseHTTPRequestHandler):
    # Set by serve() before the server starts; accepts a list[str] of texts.
    encoder: Callable[[list[str]], list[list[float]]] | None = None
    # Legacy attribute kept for backward compatibility with existing tests
    # that set ``_EmbedHandler.model`` directly.
    model: Any = None

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))

        if self.path == "/embed":
            texts = [body["text"]]
            vectors = self._encode(texts)
            result: dict[str, Any] = {"vector": vectors[0]}
        elif self.path == "/embed-batch":
            vectors = self._encode(body["texts"])
            result = {"vectors": vectors}
        else:
            self.send_response(404)
            self.end_headers()
            return

        payload = json.dumps(result).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _encode(self, texts: list[str]) -> list[list[float]]:
        """Dispatch to encoder callable or legacy model attribute."""
        cls = self.__class__
        if cls.encoder is not None:
            return cls.encoder(texts)
        # Legacy path: existing tests set _EmbedHandler.model directly.
        assert cls.model is not None, "neither encoder nor model is set"
        result = cls.model.encode(
            texts[0] if len(texts) == 1 else texts,
            normalize_embeddings=True,
        )
        vectors = result.tolist()
        if isinstance(vectors[0], float):
            return [vectors]  # single text → wrap in batch list
        return cast(list[list[float]], vectors)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass  # silence per-request access log


# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------


def serve(port: int = _DEFAULT_PORT, backend: str | None = None) -> None:
    """Load the embedding model/backend and start the HTTP server.

    Blocks until ``SIGTERM`` or ``SIGINT`` is received.

    Args:
        port:    TCP port to listen on (default 8765).
        backend: ``"torch"`` or ``"onnx"``.  When ``None``, reads
                 ``EMBED_BACKEND`` from the environment (default ``"torch"``).
    """
    if backend is None:
        backend = os.getenv("EMBED_BACKEND", "torch").lower()

    _PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    _PID_FILE.write_text(str(os.getpid()))
    try:
        encoder = _load_encoder(backend)
        logger.info("Embedding backend ready. Listening on localhost:%d", port)

        _EmbedHandler.encoder = encoder
        server = HTTPServer(("127.0.0.1", port), _EmbedHandler)

        def _shutdown(signum: int, frame: object) -> None:  # noqa: ARG001
            server.shutdown()

        signal.signal(signal.SIGTERM, _shutdown)
        server.serve_forever()
    finally:
        _PID_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=_DEFAULT_PORT)
    p.add_argument("--backend", default=None, choices=["torch", "onnx"])
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    serve(port=args.port, backend=args.backend)
