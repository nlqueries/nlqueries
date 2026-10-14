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

Encodes run on a fixed pool of that many threads, not on the thread that handled
the request, which is a new one for every request. On Windows the torch
backend's BLAS (MKL) keeps buffers per thread and never frees them when the
thread exits, so encoding on request threads leaked about 8 MB per call for
texts of varying length: a daemon grew to 38 GB over four days. Linux releases
them, but a fixed pool is right there too.

``GET /healthz`` reports the backend and how many encodes are in flight, and is
answered outside the bound — a health check that queues behind the work it
reports on stops being useful exactly when it matters.

Run via:  python -m nlqueries.embeddings.embed_server [--port N]
Managed:  nlqueries embed-server start / stop / status
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import queue
import signal
import subprocess
import sys
import threading
from collections.abc import Callable
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType
from typing import Any, cast

from nlqueries import config

#: One declaration, in config. The daemon and the in-process embedder must
#: agree: vectors written through one are read back through the other, and a
#: divergence returns wrong neighbours rather than raising anything.
_MODEL_NAME = config.EMBED_MODEL
_DEFAULT_PORT = 8765
#: The longest a request waits for its encode. Far beyond any encode and any
#: client's timeout; it exists so that a pool that has stopped taking work fails
#: the request, rather than holding it and the gate's slot forever (from #257's
#: review).
_ENCODE_WAIT_SECONDS = 300.0
#: Under `config.STATE_DIR` rather than a hardcoded home directory. A read-only
#: root filesystem left this with nowhere to go, so the server died at startup
#: with `[Errno 30]` and every natural-language query with it -- and nothing
#: could be configured to move it.
_PID_FILE = config.STATE_DIR / "embed-server.pid"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Process-existence helper
# ---------------------------------------------------------------------------


def is_pid_alive(pid: int) -> bool:
    """Return True if a process with *pid* is currently running.

    Uses tasklist on Windows because os.kill(pid, 0) maps to CTRL_C_EVENT
    (value 0) there, which sends Ctrl+C to the whole console group instead of
    checking process existence.

    Never true for 0 or a negative *pid*: tasklist lists PID 0 (the System Idle
    Process), and on POSIX ``os.kill(0, 0)`` signals the caller's own process
    group, so either would read a junk PID file as a running daemon.
    """
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except subprocess.TimeoutExpired:
            # Unknown, so assumed running: guessing "gone" would let a caller
            # remove the PID file of a live daemon. A slow tasklist is
            # transient, and `embed-server stop` asks again on its next poll.
            # Not an OSError, so uncaught it surfaced as a traceback.
            return True
        return str(pid) in result.stdout
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def read_pid_file() -> tuple[int | None, str | None]:
    """The running daemon's PID, read from the PID file and checked.

    Returns ``(pid, None)`` when the file names a running process, and
    ``(None, None)`` when there is no file. A stale file -- its process gone,
    or its content not a PID at all -- is removed, and ``(None, reason)`` says
    why, for the caller to report.

    ``start``, ``stop`` and ``status`` all read the file through this. Only
    ``status`` used to check the PID: ``start`` took the file's existence to mean
    a daemon was running, and ``stop`` signalled whatever PID it held, so a file
    left behind by a reboot or a killed terminal blocked the one and crashed the
    other on Windows.
    """
    try:
        raw = _PID_FILE.read_text().strip()
    except FileNotFoundError:
        return None, None
    try:
        pid = int(raw)
    except ValueError:
        pid = 0
    if pid > 0 and is_pid_alive(pid):
        return pid, None
    _PID_FILE.unlink(missing_ok=True)
    if pid > 0:
        return None, f"process {pid} not found"
    return None, f"it holds {raw!r}, not a PID" if raw else "it is empty"


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
    # The daemon is the path that populates the caches, so a mis-sized model
    # must not get past this point -- see embedder.check_model_width.
    from nlqueries.embeddings.embedder import check_model_width  # noqa: PLC0415

    check_model_width(model, _MODEL_NAME)

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

    # The torch path asks the model its width; an ORT model does not expose one,
    # so this measures a vector instead. Same purpose, once, at load: a mis-sized
    # model must not reach the caches, and this is the last point where the name
    # that caused it is still to hand.
    probe = _encode(["width probe"])
    if probe and len(probe[0]) != config.EMBED_DIMENSIONS:
        raise RuntimeError(
            f"embedding model {_MODEL_NAME!r} produces {len(probe[0])}-dimension vectors "
            f"through the ONNX backend, but this deployment's caches and collections "
            f"are built for {config.EMBED_DIMENSIONS}. Point NLQ_EMBED_MODEL at a model "
            "of the same width, or rebuild the vector stores for the new one."
        )

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
# Encode pool
# ---------------------------------------------------------------------------


class _EncodePool:
    """A fixed set of daemon threads that run encodes.

    Fixed, because a new thread per request leaked MKL's per-thread buffers on
    Windows (see the module docstring). Daemon, for the reason the server sets
    ``daemon_threads``: the daemon is restarted routinely, and a forward pass
    can take seconds. ``concurrent.futures.ThreadPoolExecutor`` would do the
    first and not the second: it joins its threads at interpreter exit, so a
    stopped daemon waited for the encode in flight, and a large batch could
    outlast ``embed-server stop``'s wait (from #257's review).

    Threads start as work arrives, up to *size*, so a server that is built and
    never used costs none.
    """

    def __init__(self, size: int) -> None:
        self.size = max(1, size)
        self._jobs: queue.SimpleQueue[tuple[Future[Any], Callable[[], Any]] | None] = (
            queue.SimpleQueue()
        )
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._closed = False
        # Whether running short of `size` has been logged since the last
        # thread that did start: once per episode, not once per request.
        self._shortfall_logged = False

    @property
    def running(self) -> int:
        """Encode threads started, which is fewer than `size` until enough
        requests have arrived, or while the host cannot start more."""
        return len(self._threads)

    def submit(self, fn: Callable[[], Any]) -> Future[Any]:
        future: Future[Any] = Future()
        with self._lock:
            if self._closed:
                raise RuntimeError("the encode pool has been shut down")
            if len(self._threads) < self.size:
                thread = threading.Thread(
                    target=self._work, name=f"embed-encode-{len(self._threads)}", daemon=True
                )
                # Counted once it runs. `start()` raises when the host is out of
                # threads, and a thread counted first kept its slot for good: at
                # a limit of 1, every later job waited for a worker that never
                # ran (from #257's review). With no worker at all this request
                # fails; with one, the job queues and the pool serves at the
                # capacity it has (from #258's review).
                try:
                    thread.start()
                except RuntimeError as exc:
                    if not self._threads:
                        raise
                    # Said, because nothing else would say it: the daemon keeps
                    # answering, only slower, and /healthz's max_concurrency
                    # still reads as configured (from #259's review).
                    if not self._shortfall_logged:
                        self._shortfall_logged = True
                        logger.warning(
                            "Could not start an encode thread (%s); running on %d of %d "
                            "until one starts. Requests queue for the threads there are.",
                            exc,
                            len(self._threads),
                            self.size,
                        )
                else:
                    self._threads.append(thread)
                    self._shortfall_logged = False
            self._jobs.put((future, fn))
        return future

    def shutdown(self) -> None:
        """Let idle threads finish. One busy with an encode finishes it, unless
        the process exits first -- it is a daemon thread."""
        with self._lock:
            self._closed = True
            for _ in self._threads:
                self._jobs.put(None)

    def _work(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            future, fn = job
            if not future.set_running_or_notify_cancel():
                continue
            try:
                result = fn()
            except BaseException as exc:  # noqa: BLE001 -- handed to the waiting request
                future.set_exception(exc)
            else:
                future.set_result(result)


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
    #: Where encodes run: a fixed set of threads, as many as the gate admits.
    #: Not the request's own thread, which is new for every request -- see the
    #: module docstring for what that cost on Windows. Unset only on the
    #: legacy path, like the gate.
    pool: _EncodePool | None = None
    backend: str = "unknown"
    #: The model this daemon loaded, reported by ``/healthz``.
    #:
    #: Distinct from ``model`` above, which is a model *object* on the
    #: legacy path. Single-sourcing ``EMBED_MODEL`` makes the name agree
    #: within a process; this is how a *different* process can find out,
    #: and the client compares it once before trusting any vector.
    model_name: str | None = None

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
        pool = self.__class__.pool
        self._respond(
            {
                "status": "ok",
                "backend": self.__class__.backend,
                # So a client can refuse vectors from a model other than the
                # one its collections were built with. A per-process constant
                # cannot reach across the socket; this can.
                "model": self.__class__.model_name,
                "inflight": gate.inflight if gate else 0,
                "max_concurrency": gate.limit if gate else None,
                # Started so far: below max_concurrency until enough requests
                # have arrived, or while the host cannot start threads.
                "encode_threads": pool.running if pool else None,
            }
        )

    #: Requests larger than this are refused without being read.
    MAX_BODY_BYTES = 10 * 1024 * 1024

    #: Applied to the connection socket. A client that declares a body and does
    #: not send it would otherwise hold a worker thread until it disconnects,
    #: and this server is threaded, so enough such connections stop it serving.
    timeout = 30

    def _fail(self, status: int, reason: str) -> None:
        """Answer with *status* and a short JSON reason.

        Answering matters: an unhandled exception closes the connection without
        a response, which a caller sees as the server having gone away.
        """
        body = json.dumps({"error": reason}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    #: Whether the request body has been taken off the socket.
    _body_consumed = False

    def _drain(self) -> None:
        """Read and discard the declared body, up to the size limit.

        Replying without consuming the body leaves unread bytes in the receive
        buffer, and closing the connection then sends a reset instead of the
        response. The client sees a dropped connection rather than the status.
        """
        original = self.connection.gettimeout()
        try:
            # Briefly, and by what arrives rather than by what was declared:
            # the declared length is either unknown here or known to be a lie,
            # and reading it would block until the connection times out.
            self.connection.settimeout(0.25)
            self.rfile.read(65536)
        except Exception:  # noqa: BLE001 - draining is best effort
            pass
        finally:
            with contextlib.suppress(Exception):
                self.connection.settimeout(original)

    def _read_body(self) -> Any:
        """Parse the request body, or raise ValueError with a reason."""
        raw_length = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_length)
        except (TypeError, ValueError):
            raise ValueError("Content-Length is not a number") from None
        if length < 0:
            raise ValueError("Content-Length is negative")
        if length > self.MAX_BODY_BYTES:
            raise ValueError(f"body exceeds {self.MAX_BODY_BYTES} bytes")

        raw = self.rfile.read(length)
        self._body_consumed = True
        if len(raw) != length:
            raise ValueError("body is shorter than Content-Length")
        try:
            parsed = json.loads(raw or b"{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise ValueError("body is not valid JSON") from None
        if not isinstance(parsed, dict):
            raise ValueError("body is not a JSON object")
        return parsed

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in ("/embed", "/embed-batch"):
            self._drain()
            self._fail(404, "no such endpoint")
            return

        try:
            body = self._read_body()
        except ValueError as exc:
            if not self._body_consumed:
                # Replying with bytes still unread closes the connection with a
                # reset on some platforms, and the client sees a dropped
                # connection rather than the status.
                self._drain()
            self._fail(400, str(exc))
            return
        except (TimeoutError, OSError):
            # The client stopped sending. Nothing to answer to.
            return

        try:
            if self.path == "/embed":
                text = body["text"]
                if not isinstance(text, str):
                    raise TypeError("'text' must be a string")
                result: dict[str, Any] = {"vector": self._encode([text])[0]}
            else:
                texts = body["texts"]
                if not isinstance(texts, list) or not all(isinstance(x, str) for x in texts):
                    raise TypeError("'texts' must be a list of strings")
                result = {"vectors": self._encode(texts)}
        except KeyError as exc:
            self._fail(400, f"missing field: {exc.args[0]}")
            return
        except TypeError as exc:
            self._fail(400, str(exc))
            return
        except Exception:
            logger.exception("Embedding request failed")
            self._fail(500, "embedding failed")
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
            if cls.pool is None:
                return self._encode_unguarded(texts)
            # The gate admits no more encodes than the pool's size, so with every
            # thread started this never queues inside the pool. Below its size
            # -- the host could not start a thread -- jobs do queue behind busy
            # threads, and the gate still bounds that queue to its limit less
            # the threads running.
            future = cls.pool.submit(lambda: self._encode_unguarded(texts))
            try:
                return cast(list[list[float]], future.result(timeout=_ENCODE_WAIT_SECONDS))
            except TimeoutError:
                # Its job may still be queued behind a busy worker. Cancelled,
                # the worker skips it instead of encoding for a request that
                # has gone (from #258's review).
                future.cancel()
                raise TimeoutError(
                    f"no encode thread finished this request within {_ENCODE_WAIT_SECONDS:g}s"
                ) from None

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
    The request threads only wait now; the encode itself runs on
    ``_EncodePool``'s threads, which are daemon threads for the same reason.
    """

    daemon_threads = True

    def handle_error(self, request: Any, client_address: Any) -> None:
        """Log an exception that escaped a handler, through the logger.

        socketserver prints it to stderr itself, past the logging
        configuration, so in JSON mode a raw multi-line traceback landed among
        the JSON lines.

        A client that left before its reply was written is routine, not a
        failure: the embedder gives up after ``EMBED_CLIENT_TIMEOUT_SECONDS``
        and embeds in-process, and the write then fails. It is one INFO line
        without a traceback, as the read side treats the same thing as a
        non-event; anything else is an ERROR with its traceback.
        """
        exc = sys.exc_info()[1]
        if isinstance(exc, ConnectionError | TimeoutError):
            logger.info(
                "A client at %s left before its reply was written (%s)",
                client_address,
                type(exc).__name__,
            )
            return
        logger.exception("A request from %s failed", client_address)

    def server_close(self) -> None:
        super().server_close()
        pool = getattr(self.RequestHandlerClass, "pool", None)
        if pool is not None:
            pool.shutdown()


def build_server(
    port: int,
    *,
    encoder: Callable[[list[str]], list[list[float]]] | None = None,
    model: Any = None,
    model_name: str | None = None,
    max_concurrency: int | None = None,
    backend: str = "unknown",
) -> _ThreadingEmbedServer:
    """Build a configured embedding server without starting it.

    Each server gets its own handler subclass rather than mutating the shared
    ``_EmbedHandler`` class attributes, so two servers in one process — which is
    exactly what a concurrency test does — cannot overwrite each other's encoder.
    """
    limit = config.EMBED_SERVER_MAX_CONCURRENCY if max_concurrency is None else max_concurrency
    gate = _Gate(limit)
    handler = type(
        "_BoundEmbedHandler",
        (_EmbedHandler,),
        {
            "encoder": staticmethod(encoder) if encoder is not None else None,
            "model": model,
            "model_name": model_name,
            "gate": gate,
            "pool": _EncodePool(gate.limit),
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

        server = build_server(port, encoder=encoder, model_name=_MODEL_NAME, backend=backend)
        logger.info(
            "Embedding backend ready with model %s. Listening on localhost:%d, "
            "up to %d concurrent encodes",
            _MODEL_NAME,
            port,
            config.EMBED_SERVER_MAX_CONCURRENCY,
        )

        def _shutdown(signum: int, frame: object) -> None:  # noqa: ARG001
            # On another thread: the handler runs on this one, inside
            # `serve_forever`, and `shutdown()` waits for that loop to finish --
            # called here it waited forever, so SIGTERM never stopped the daemon.
            threading.Thread(target=server.shutdown, daemon=True).start()

        signal.signal(signal.SIGTERM, _shutdown)
        try:
            server.serve_forever()
        finally:
            # Closes the socket and stops the encode pool's idle threads (from
            # #257's review). One busy with an encode is a daemon thread, so
            # the process does not wait for it.
            server.server_close()
    finally:
        _PID_FILE.unlink(missing_ok=True)


#: The format of the daemon's log lines when ``--log-format`` is not given:
#: ``text`` (the default) or ``json``.
LOG_FORMAT_ENV = "NLQ_LOG_FORMAT"


class JsonLogFormatter(logging.Formatter):
    """One JSON object per log line.

    The keys are those of the nlqueries-enterprise API's own lines --
    ``asctime``, ``name``, ``levelname``, ``message``, and ``exc_info`` for a
    traceback -- so a log reader that understands those reads these too. The
    enterprise API container runs this daemon beside the API, and its
    plain-text lines, "Embedding request failed" tracebacks included, were
    invisible to the support tooling that reads the API's JSON lines.
    """

    def __init__(self) -> None:
        super().__init__(datefmt="%Y-%m-%dT%H:%M:%S%z")

    def format(self, record: logging.LogRecord) -> str:
        out: dict[str, Any] = {
            "asctime": self.formatTime(record, self.datefmt),
            "name": record.name,
            "levelname": record.levelname,
            "message": record.getMessage(),
        }
        if record.exc_info:
            out["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            out["stack_info"] = self.formatStack(record.stack_info)
        return json.dumps(out, default=str)


def configure_logging(log_format: str | None = None) -> None:
    """Send the daemon's logs to stderr, as plain text or as JSON lines.

    *log_format* is ``text`` or ``json``; when it is not given,
    ``NLQ_LOG_FORMAT`` decides, and text is the default. Plain text is what
    ``logging.basicConfig`` writes. Any other value is text, with a warning
    that says so.
    """
    chosen = (log_format or os.environ.get(LOG_FORMAT_ENV) or "text").strip().lower()
    handler = logging.StreamHandler()
    handler.setFormatter(
        JsonLogFormatter() if chosen == "json" else logging.Formatter(logging.BASIC_FORMAT)
    )
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
    if chosen not in ("text", "json"):
        logger.warning("Unknown log format %r; writing plain text.", chosen)


def main(argv: list[str] | None = None) -> None:
    """``python -m nlqueries.embeddings.embed_server``."""
    import argparse  # noqa: PLC0415

    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=_DEFAULT_PORT)
    p.add_argument("--backend", default=None, choices=["torch", "onnx"])
    p.add_argument(
        "--log-format",
        default=None,
        choices=["text", "json"],
        help=f"Log line format (default: ${LOG_FORMAT_ENV}, else text).",
    )
    args = p.parse_args(argv)
    configure_logging(args.log_format)
    serve(port=args.port, backend=args.backend)


if __name__ == "__main__":
    main()
