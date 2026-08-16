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

Concurrency
-----------
The daemon is threaded, with a bounded number of encodes running at once
(``EMBED_SERVER_MAX_CONCURRENCY``). It serves every process on the host and sits
on the hot path of every chat turn — two to three embeds per turn — so serving
one request at a time made it the ceiling for the whole machine.

The bound matters as much as the threading: every concurrent encode holds its
own activations, so an unbounded daemon turns a burst into an out-of-memory kill
rather than a queue. Set the limit to 1 to reproduce the old behaviour exactly.

``GET /healthz`` reports the backend and how many encodes are in flight, and is
answered outside the bound — a health check that queues behind the work it
reports on stops being useful exactly when it matters.

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
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType
from typing import Any, cast

from nlqueries import config

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
# Concurrency gate
# ---------------------------------------------------------------------------


class _Gate:
    """Bounds concurrent encodes and keeps count of them.

    Threading the server without this would trade one bottleneck for a worse
    failure: requests would no longer queue, they would all run, and each holds
    its own activations, so a burst becomes an out-of-memory kill instead of a
    slowdown. Queuing at a fixed depth is the behaviour to want — slower under
    overload, never dead.

    The count is kept here rather than derived from the semaphore so ``/healthz``
    can report it without touching a private attribute, and so the peak is
    available to tests that need to prove the bound actually holds.
    """

    def __init__(self, limit: int) -> None:
        self.limit = max(1, limit)
        self._semaphore = threading.Semaphore(self.limit)
        self._lock = threading.Lock()
        self.inflight = 0
        self.peak = 0

    def __enter__(self) -> _Gate:
        self._semaphore.acquire()
        with self._lock:
            self.inflight += 1
            self.peak = max(self.peak, self.inflight)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        with self._lock:
            self.inflight -= 1
        self._semaphore.release()


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------


class _EmbedHandler(BaseHTTPRequestHandler):
    # Set by serve() before the server starts; accepts a list[str] of texts.
    encoder: Callable[[list[str]], list[list[float]]] | None = None
    # Legacy attribute kept for backward compatibility with existing tests
    # that set ``_EmbedHandler.model`` directly.
    model: Any = None
    # Set by build_server(). Unset means unbounded, which only happens if a
    # caller constructs the handler itself — the legacy path.
    gate: _Gate | None = None
    backend: str = "unknown"

    def do_GET(self) -> None:  # noqa: N802
        """Report liveness and how busy the daemon is.

        The API's own health check previously had no visibility into this
        process at all: the daemon could be saturated, and every chat turn
        quietly paying a timeout, while everything upstream reported healthy.

        Deliberately outside the gate — a health check that queues behind the
        work it is reporting on stops being a health check the moment it matters.
        """
        if self.path != "/healthz":
            self.send_response(404)
            self.end_headers()
            return
        gate = self.__class__.gate
        self._respond(
            {
                "status": "ok",
                "backend": self.__class__.backend,
                "inflight": gate.inflight if gate else 0,
                "max_concurrency": gate.limit if gate else None,
            }
        )

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

        self._respond(result)

    def _respond(self, result: dict[str, Any]) -> None:
        payload = json.dumps(result).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _encode(self, texts: list[str]) -> list[list[float]]:
        """Dispatch to encoder callable or legacy model attribute, under the gate."""
        cls = self.__class__
        gate = cls.gate
        if gate is None:
            return self._encode_unguarded(texts)
        with gate:
            return self._encode_unguarded(texts)

    def _encode_unguarded(self, texts: list[str]) -> list[list[float]]:
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
# Server construction
# ---------------------------------------------------------------------------


class _ThreadingEmbedServer(ThreadingHTTPServer):
    """Threaded so one slow encode does not hold up the whole host.

    ``daemon_threads`` so shutdown is not held hostage by an in-flight encode:
    the daemon is restarted routinely and a model forward pass can take seconds.
    """

    daemon_threads = True


def build_server(
    port: int,
    *,
    encoder: Callable[[list[str]], list[list[float]]] | None = None,
    model: Any = None,
    max_concurrency: int | None = None,
    backend: str = "unknown",
) -> _ThreadingEmbedServer:
    """Build a configured embedding server without starting it.

    Each server gets its own handler subclass rather than mutating the shared
    ``_EmbedHandler`` class attributes, so two servers in one process — which is
    exactly what a concurrency test does — cannot overwrite each other's encoder.
    """
    limit = config.EMBED_SERVER_MAX_CONCURRENCY if max_concurrency is None else max_concurrency
    handler = type(
        "_BoundEmbedHandler",
        (_EmbedHandler,),
        {
            "encoder": staticmethod(encoder) if encoder is not None else None,
            "model": model,
            "gate": _Gate(limit),
            "backend": backend,
        },
    )
    return _ThreadingEmbedServer(("127.0.0.1", port), handler)


def _apply_torch_thread_limit() -> None:
    """Pin torch to one intra-op thread inside the daemon.

    With N requests in flight and torch defaulting to one thread per core, the
    process asks for N×cores threads to do work that has cores' worth of
    capacity. They contend rather than compute. One thread per request, N
    requests, is the arrangement that actually uses the machine.

    Set EMBED_SERVER_TORCH_THREADS=0 to leave torch's default alone.
    """
    threads = config.EMBED_SERVER_TORCH_THREADS
    if threads <= 0:
        return
    try:
        import torch  # noqa: PLC0415 — optional; the onnx backend has no torch

        torch.set_num_threads(threads)
        logger.info("torch intra-op threads set to %d", threads)
    except Exception as exc:  # noqa: BLE001 — a thread hint is never worth failing over
        logger.debug("could not set torch thread count: %s", exc)


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
        if backend != "onnx":
            _apply_torch_thread_limit()
        encoder = _load_encoder(backend)

        server = build_server(port, encoder=encoder, backend=backend)
        logger.info(
            "Embedding backend ready. Listening on localhost:%d, up to %d concurrent encodes",
            port,
            config.EMBED_SERVER_MAX_CONCURRENCY,
        )

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
