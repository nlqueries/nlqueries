"""
nlqueries.embeddings.embed_server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Lightweight embedding daemon.  Loads all-MiniLM-L6-v2 once and serves
POST /embed and POST /embed-batch over localhost:8765.

Run via:  python -m nlqueries.embeddings.embed_server [--port N]
Managed:  nlqueries embed-server start / stop / status
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import signal
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"
_DEFAULT_PORT = 8765
_PID_FILE = pathlib.Path.home() / ".nlqueries" / "embed-server.pid"

logger = logging.getLogger(__name__)


def _load_model() -> SentenceTransformer:
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    return SentenceTransformer(_MODEL_NAME)  # type: ignore[no-any-return]


class _EmbedHandler(BaseHTTPRequestHandler):
    model: SentenceTransformer | None = None  # set by serve() before the server starts

    def do_POST(self) -> None:  # noqa: N802
        assert self.model is not None
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))

        if self.path == "/embed":
            vector = self.model.encode(body["text"], normalize_embeddings=True).tolist()
            result = {"vector": vector}
        elif self.path == "/embed-batch":
            vectors = self.model.encode(body["texts"], normalize_embeddings=True).tolist()
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

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass  # silence per-request access log


def serve(port: int = _DEFAULT_PORT) -> None:
    """Load the model and start the HTTP server. Blocks until SIGTERM/SIGINT."""
    logger.info("Loading %s ...", _MODEL_NAME)
    model = _load_model()
    logger.info("Model loaded. Listening on localhost:%d", port)

    _EmbedHandler.model = model
    server = HTTPServer(("127.0.0.1", port), _EmbedHandler)

    def _shutdown(signum: int, frame: object) -> None:  # noqa: ARG001
        server.shutdown()

    signal.signal(signal.SIGTERM, _shutdown)

    _PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    _PID_FILE.write_text(str(os.getpid()))
    try:
        server.serve_forever()
    finally:
        _PID_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=_DEFAULT_PORT)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    serve(port=args.port)
