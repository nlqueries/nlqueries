"""Tests for the persistent embedding daemon and daemon-first embedder (#32)."""

from __future__ import annotations

import socket
import threading
from unittest.mock import MagicMock, patch


def _free_port() -> int:
    """Return an available TCP port on localhost."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_server_thread(port: int) -> None:
    """Load a mock model and start the embed server in a daemon thread."""
    import time

    from nlqueries.embeddings.embed_server import build_server

    mock_model = MagicMock()
    # encode() returns a list-like; tolist() gives the final list
    mock_model.encode.side_effect = lambda text_or_texts, **_kw: (
        MagicMock(tolist=lambda: [0.0] * 384)
        if isinstance(text_or_texts, str)
        else MagicMock(tolist=lambda: [[0.0] * 384 for _ in text_or_texts])
    )

    # The legacy `model` attribute rather than an encoder callable: that path is
    # what deployed callers still use, so it keeps being exercised here.
    server = build_server(port, model=mock_model)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    # give the server a moment to start
    time.sleep(0.05)
    return server


# ---------------------------------------------------------------------------
# Server integration tests
# ---------------------------------------------------------------------------


def test_embed_endpoint_returns_384_dim_vector():
    """POST /embed must return a 384-element float list."""
    import json
    import urllib.request

    port = _free_port()
    _start_server_thread(port)

    body = json.dumps({"text": "hello"}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/embed",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        result = json.loads(resp.read())

    assert "vector" in result
    assert isinstance(result["vector"], list)
    assert len(result["vector"]) == 384


def test_embed_batch_endpoint_returns_correct_count():
    """POST /embed-batch must return one 384-dim vector per input text."""
    import json
    import urllib.request

    port = _free_port()
    _start_server_thread(port)

    texts = ["a", "b", "c"]
    body = json.dumps({"texts": texts}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/embed-batch",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        result = json.loads(resp.read())

    assert "vectors" in result
    assert len(result["vectors"]) == 3
    assert all(len(v) == 384 for v in result["vectors"])


# ---------------------------------------------------------------------------
# Concurrency
#
# The daemon serves every process on the host and is called two to three times
# per chat turn. Single-threaded, it was the ceiling for the whole machine: one
# embed at a time, everything else queued behind it.
# ---------------------------------------------------------------------------


def _slow_encoder(delay: float):
    """An encoder that takes *delay* seconds, so concurrency is measurable."""
    import time

    def _encode(texts: list[str]) -> list[list[float]]:
        time.sleep(delay)
        return [[0.0] * 384 for _ in texts]

    return _encode


def _post_embed(port: int, text: str = "hello", timeout: float = 30.0) -> dict:
    import json
    import urllib.request

    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/embed",
        data=json.dumps({"text": text}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _serve_in_background(server) -> None:
    threading.Thread(target=server.serve_forever, daemon=True).start()


def _fire_concurrently(port: int, count: int) -> float:
    """Send *count* embeds at once; return the wall time for all of them."""
    import time
    from concurrent.futures import ThreadPoolExecutor

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=count) as pool:
        list(pool.map(lambda i: _post_embed(port, f"text-{i}"), range(count)))
    return time.monotonic() - started


def test_concurrent_embeds_are_served_in_parallel():
    """Two 200 ms embeds must take about 200 ms, not 400.

    This is the whole point of the change: with the single-threaded server every
    concurrent chat turn on the host queued behind every other one.
    """
    from nlqueries.embeddings.embed_server import build_server

    port = _free_port()
    server = build_server(port, encoder=_slow_encoder(0.2), max_concurrency=4)
    _serve_in_background(server)

    elapsed = _fire_concurrently(port, 2)
    server.shutdown()

    assert elapsed < 0.35, f"two 200ms embeds took {elapsed:.3f}s — they serialised"


def test_concurrency_of_one_serialises_exactly_as_before():
    """The escape hatch has to actually be one at a time, or it is not an escape
    hatch — it is a different bug."""
    from nlqueries.embeddings.embed_server import build_server

    port = _free_port()
    server = build_server(port, encoder=_slow_encoder(0.15), max_concurrency=1)
    _serve_in_background(server)

    elapsed = _fire_concurrently(port, 3)
    gate = server.RequestHandlerClass.gate
    server.shutdown()

    assert elapsed >= 0.45, f"three serialised 150ms embeds took only {elapsed:.3f}s"
    assert gate.peak == 1


def test_the_gate_bounds_work_in_flight():
    """Threading without a bound trades a slowdown for an out-of-memory kill:
    every concurrent encode holds its own activations."""
    from nlqueries.embeddings.embed_server import build_server

    port = _free_port()
    server = build_server(port, encoder=_slow_encoder(0.1), max_concurrency=2)
    _serve_in_background(server)

    _fire_concurrently(port, 8)
    gate = server.RequestHandlerClass.gate
    server.shutdown()

    assert gate.peak <= 2, f"{gate.peak} encodes ran at once against a limit of 2"
    assert gate.inflight == 0, "the gate leaked a slot"


def test_healthz_answers_while_an_embed_is_in_flight():
    """A health check that queues behind the work it reports on stops being a
    health check exactly when it matters."""
    import json
    import time
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    from nlqueries.embeddings.embed_server import build_server

    port = _free_port()
    server = build_server(port, encoder=_slow_encoder(0.5), max_concurrency=1, backend="torch")
    _serve_in_background(server)

    with ThreadPoolExecutor(max_workers=2) as pool:
        embed = pool.submit(_post_embed, port)
        time.sleep(0.1)  # let the encode start and take the only slot

        started = time.monotonic()
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=5) as resp:
            health = json.loads(resp.read())
        health_elapsed = time.monotonic() - started
        embed.result()

    server.shutdown()

    assert health["status"] == "ok"
    assert health["backend"] == "torch"
    assert health["inflight"] == 1, "healthz should see the embed that is running"
    assert health_elapsed < 0.3, "healthz waited for the encode to finish"


def test_an_unknown_path_is_still_a_404():
    import urllib.error
    import urllib.request

    from nlqueries.embeddings.embed_server import build_server

    port = _free_port()
    server = build_server(port, encoder=_slow_encoder(0.0))
    _serve_in_background(server)

    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/nope", timeout=5)
        raised = None
    except urllib.error.HTTPError as exc:
        raised = exc.code
    finally:
        server.shutdown()

    assert raised == 404


def test_each_server_gets_its_own_encoder():
    """Two servers in one process must not overwrite each other's encoder, which
    handler-class attributes would have done."""
    from nlqueries.embeddings.embed_server import build_server

    first = build_server(_free_port(), encoder=lambda texts: [[1.0] for _ in texts])
    second = build_server(_free_port(), encoder=lambda texts: [[2.0] for _ in texts])

    assert first.RequestHandlerClass.encoder(["x"]) == [[1.0]]
    assert second.RequestHandlerClass.encoder(["x"]) == [[2.0]]


# ---------------------------------------------------------------------------
# Embedder unit tests (daemon path vs. fallback path)
# ---------------------------------------------------------------------------


def test_embed_text_uses_daemon_when_running():
    """embed_text must return the daemon vector without loading the local model."""
    sentinel = [0.42] * 384

    with (
        patch("nlqueries.embeddings.embedder._try_daemon_single", return_value=sentinel),
        patch("nlqueries.embeddings.embedder._get_model") as mock_get_model,
    ):
        from nlqueries.embeddings.embedder import embed_text

        result = embed_text("anything")

    assert result == sentinel
    mock_get_model.assert_not_called()


def test_embed_text_falls_back_to_local_model_when_daemon_down():
    """embed_text must load the local model when the daemon is unreachable."""
    import importlib

    sentinel = [0.99] * 384
    mock_model = MagicMock()
    mock_model.encode.return_value = MagicMock(tolist=lambda: sentinel)

    with (
        patch("nlqueries.embeddings.embedder._try_daemon_single", return_value=None),
        patch("nlqueries.embeddings.embedder._get_model", return_value=mock_model),
    ):
        from nlqueries.embeddings import embedder

        # reload to avoid cached module state from other tests
        importlib.reload(embedder)
        with (
            patch.object(embedder, "_try_daemon_single", return_value=None),
            patch.object(embedder, "_get_model", return_value=mock_model),
        ):
            result = embedder.embed_text("anything")

    assert result == sentinel
    mock_model.encode.assert_called_once()
