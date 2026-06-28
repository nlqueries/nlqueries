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
    from nlqueries.embeddings.embed_server import _EmbedHandler, HTTPServer

    mock_model = MagicMock()
    # encode() returns a list-like; tolist() gives the final list
    mock_model.encode.side_effect = lambda text_or_texts, **_kw: (
        MagicMock(tolist=lambda: [0.0] * 384)
        if isinstance(text_or_texts, str)
        else MagicMock(tolist=lambda: [[0.0] * 384 for _ in text_or_texts])
    )

    _EmbedHandler.model = mock_model
    server = HTTPServer(("127.0.0.1", port), _EmbedHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    # give the server a moment to start
    import time

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
# Embedder unit tests (daemon path vs. fallback path)
# ---------------------------------------------------------------------------


def test_embed_text_uses_daemon_when_running():
    """embed_text must return the daemon vector without loading the local model."""
    sentinel = [0.42] * 384

    with patch("nlqueries.embeddings.embedder._try_daemon_single", return_value=sentinel):
        with patch("nlqueries.embeddings.embedder._get_model") as mock_get_model:
            from nlqueries.embeddings.embedder import embed_text

            result = embed_text("anything")

    assert result == sentinel
    mock_get_model.assert_not_called()


def test_embed_text_falls_back_to_local_model_when_daemon_down():
    """embed_text must load the local model when the daemon is unreachable."""
    sentinel = [0.99] * 384
    mock_model = MagicMock()
    mock_model.encode.return_value = MagicMock(tolist=lambda: sentinel)

    with patch("nlqueries.embeddings.embedder._try_daemon_single", return_value=None):
        with patch("nlqueries.embeddings.embedder._get_model", return_value=mock_model):
            from nlqueries.embeddings import embedder

            # reload to avoid cached module state from other tests
            import importlib

            importlib.reload(embedder)
            with patch.object(embedder, "_try_daemon_single", return_value=None):
                with patch.object(embedder, "_get_model", return_value=mock_model):
                    result = embedder.embed_text("anything")

    assert result == sentinel
    mock_model.encode.assert_called_once()
