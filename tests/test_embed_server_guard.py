"""
The embedding server's handling of malformed requests (SEC-17).

It binds to loopback, so the caller is a local process rather than the network,
and the module notes that it serves every process on the host.

Measured before this was guarded: malformed JSON, a missing field, an empty
body and a non-numeric Content-Length each raised out of the handler, which
closed the connection with no response. A Content-Length larger than the body
sent blocked the read until the client gave up, holding a worker thread; the
server is threaded, so enough such connections stop it serving.
"""

from __future__ import annotations

import http.client
import json
import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest
from nlqueries.embeddings import embed_server as es


@pytest.fixture()
def server() -> Iterator[int]:
    """A running server with the model stubbed out. Yields its port."""
    original_encode = es._EmbedHandler._encode
    original_log = es._EmbedHandler.log_message
    es._EmbedHandler._encode = lambda self, texts: [[0.1] * 4 for _ in texts]  # type: ignore[method-assign]
    es._EmbedHandler.log_message = lambda *a, **k: None  # type: ignore[method-assign]

    srv = es._ThreadingEmbedServer(("127.0.0.1", 0), es._EmbedHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield int(srv.server_address[1])
    finally:
        srv.shutdown()
        srv.server_close()
        es._EmbedHandler._encode = original_encode  # type: ignore[method-assign]
        es._EmbedHandler.log_message = original_log  # type: ignore[method-assign]


def _post(port: int, path: str, body: Any = None, headers: dict[str, str] | None = None) -> Any:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    if headers is not None and body is None:
        conn.putrequest("POST", path)
        for key, value in headers.items():
            conn.putheader(key, value)
        conn.endheaders()
        conn.send(b"{}")
    else:
        conn.request("POST", path, body, {"Content-Type": "application/json"})
    return conn.getresponse()


def test_a_valid_request_is_answered(server: int) -> None:
    """The control. Whatever refuses malformed input must not refuse this."""
    response = _post(server, "/embed", json.dumps({"text": "hello"}))

    assert response.status == 200
    assert json.loads(response.read())["vector"] == [0.1] * 4


def test_a_valid_batch_is_answered(server: int) -> None:
    response = _post(server, "/embed-batch", json.dumps({"texts": ["a", "b"]}))

    assert response.status == 200
    assert len(json.loads(response.read())["vectors"]) == 2


@pytest.mark.parametrize(
    ("label", "body", "expected"),
    [
        ("malformed JSON", "{not json", "not valid JSON"),
        ("missing field", json.dumps({"nope": 1}), "missing field"),
        ("empty body", "", "missing field"),
        ("text is not a string", json.dumps({"text": 123}), "must be a string"),
        ("JSON is not an object", json.dumps([1, 2]), "not a JSON object"),
    ],
    ids=lambda v: v if isinstance(v, str) and " " in v else "",
)
def test_malformed_bodies_are_answered_with_400(
    server: int, label: str, body: str, expected: str
) -> None:
    """Each of these previously raised out of the handler, closing the
    connection without a response."""
    response = _post(server, "/embed", body)

    assert response.status == 400
    assert expected in json.loads(response.read())["error"]


def test_a_non_numeric_content_length_is_answered_with_400(server: int) -> None:
    response = _post(server, "/embed", None, {"Content-Length": "abc"})

    assert response.status == 400
    assert "not a number" in json.loads(response.read())["error"]


def test_an_oversized_body_is_refused_without_being_read(server: int) -> None:
    """The declared length is checked before the read, so the bytes are never
    allocated."""
    response = _post(server, "/embed", None, {"Content-Length": str(20 * 1024 * 1024)})

    assert response.status == 400
    assert "exceeds" in json.loads(response.read())["error"]


def test_an_unknown_path_is_answered_with_404(server: int) -> None:
    response = _post(server, "/nope", json.dumps({"text": "x"}))

    assert response.status == 404


def test_a_declared_but_unsent_body_releases_the_thread(server: int) -> None:
    """The connection is closed rather than held until the client gives up.

    Without the handler timeout this blocked in `rfile.read`, and the server is
    threaded, so repeating it exhausts the pool.
    """
    original_timeout = es._EmbedHandler.timeout
    es._EmbedHandler.timeout = 2  # type: ignore[assignment]
    try:
        started = time.monotonic()
        with pytest.raises(Exception):  # noqa: B017 - the connection closes; how varies by platform
            _post(server, "/embed", None, {"Content-Length": "999999"})
        elapsed = time.monotonic() - started

        assert elapsed < 8, f"the read blocked for {elapsed:.1f}s"
    finally:
        es._EmbedHandler.timeout = original_timeout  # type: ignore[assignment]

    # The server still serves other clients.
    assert _post(server, "/embed", json.dumps({"text": "hi"})).status == 200
