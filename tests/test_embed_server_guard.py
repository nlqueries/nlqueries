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
        for key, value in {"Content-Type": "application/json", **headers}.items():
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


# --- Who may POST, and how much -------------------------------------------------


def _post_as(port: int, host: str | None, content_type: str | None, body: Any) -> Any:
    """POST *body* with exactly the given ``Host`` and ``Content-Type``; None
    leaves the header out."""
    raw = json.dumps(body).encode()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.putrequest("POST", "/embed-batch", skip_host=True)
    if host is not None:
        conn.putheader("Host", host)
    if content_type is not None:
        conn.putheader("Content-Type", content_type)
    conn.putheader("Content-Length", str(len(raw)))
    conn.endheaders()
    conn.send(raw)
    return conn.getresponse()


@pytest.mark.parametrize(
    "host",
    [
        "127.0.0.1",
        "127.0.0.1:8765",
        "localhost",
        "localhost:1",
        "LOCALHOST:8765",
        "[::1]",
        "[::1]:8765",
    ],
)
def test_a_loopback_host_with_any_port_is_answered(server: int, host: str) -> None:
    response = _post_as(server, host, "application/json", {"texts": ["a"]})

    assert response.status == 200


@pytest.mark.parametrize(
    "host",
    [
        None,
        "",
        "example.com",
        "example.com:8765",
        "127.0.0.1.example.com",
        "localhost.example.com:8765",
        "127.0.0.2:8765",
        "0.0.0.0:8765",
        "[::1",
        "[::2]:8765",
        "localhost:port",
        "localhost:8765:1",
    ],
)
def test_any_other_host_is_refused_with_421(server: int, host: str | None) -> None:
    response = _post_as(server, host, "application/json", {"texts": ["a"]})

    assert response.status == 421
    assert "Host" in json.loads(response.read())["error"]


@pytest.mark.parametrize("content_type", ["application/json", "application/JSON; charset=utf-8"])
def test_a_json_content_type_is_answered(server: int, content_type: str) -> None:
    response = _post_as(server, f"127.0.0.1:{server}", content_type, {"texts": ["a"]})

    assert response.status == 200


@pytest.mark.parametrize(
    "content_type",
    [
        None,
        "text/plain",
        "application/x-www-form-urlencoded",
        "multipart/form-data",
        "application/jsonx",
    ],
)
def test_any_other_content_type_is_refused_with_415(server: int, content_type: str | None) -> None:
    response = _post_as(server, f"127.0.0.1:{server}", content_type, {"texts": ["a"]})

    assert response.status == 415
    assert "Content-Type" in json.loads(response.read())["error"]


def test_a_batch_at_the_text_limit_is_answered(server: int) -> None:
    response = _post(server, "/embed-batch", json.dumps({"texts": ["a"] * es.MAX_BATCH_TEXTS}))

    assert response.status == 200
    assert len(json.loads(response.read())["vectors"]) == es.MAX_BATCH_TEXTS


def test_a_513_text_batch_is_refused_with_413(server: int) -> None:
    assert es.MAX_BATCH_TEXTS == 512
    response = _post(server, "/embed-batch", json.dumps({"texts": ["a"] * 513}))

    assert response.status == 413
    assert "512 texts" in json.loads(response.read())["error"]


def test_a_batch_over_the_character_limit_is_refused_with_413(server: int) -> None:
    texts = ["x" * 1_000_000, "x" * 1_000_000, "x"]
    response = _post(server, "/embed-batch", json.dumps({"texts": texts}))

    assert response.status == 413
    assert "characters" in json.loads(response.read())["error"]


def test_a_batch_at_the_character_limit_is_answered(server: int) -> None:
    texts = ["x" * 1_000_000, "x" * 1_000_000]
    response = _post(server, "/embed-batch", json.dumps({"texts": texts}))

    assert response.status == 200


def test_a_single_text_over_the_character_limit_is_refused_with_413(server: int) -> None:
    response = _post(server, "/embed", json.dumps({"text": "x" * (es.MAX_BATCH_CHARS + 1)}))

    assert response.status == 413


def test_the_client_sends_what_the_daemon_accepts(
    server: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real client against the real handler: its Host and Content-Type pass."""
    from nlqueries.embeddings import embedder

    monkeypatch.setattr(embedder, "_DAEMON_BASE", f"http://127.0.0.1:{server}")

    assert embedder._daemon_post("/embed-batch", {"texts": ["a", "b"]}, "vectors") == [
        [0.1] * 4,
        [0.1] * 4,
    ]


def test_a_whole_collection_reaches_a_real_daemon_through_embed_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What `qdrant_store` does with a collection of 600: every text is embedded
    by the daemon, in requests it accepts, and nothing loads a model in-process."""
    from nlqueries.embeddings import embedder

    batches: list[int] = []

    def encode(texts: list[str]) -> list[list[float]]:
        batches.append(len(texts))
        return [[float(len(t))] for t in texts]

    srv = es.build_server(0, encoder=encode, model_name=embedder._MODEL_NAME)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        monkeypatch.setattr(embedder, "_DAEMON_BASE", f"http://127.0.0.1:{srv.server_address[1]}")
        monkeypatch.setattr(embedder, "_daemon_model_checked", False)
        monkeypatch.setattr(embedder, "_get_model", lambda: pytest.fail("loaded a model"))
        texts = ["x" * (i % 7) for i in range(600)]

        vectors = embedder.embed_batch(texts)
    finally:
        srv.shutdown()
        srv.server_close()

    assert sorted(batches) == [88, 512]
    assert vectors == [[float(len(t))] for t in texts]
