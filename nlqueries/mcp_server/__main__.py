"""
Makes ``python -m nlqueries.mcp_server`` start the server.

This module did not exist, and the Dockerfile's ``CMD`` was exactly that
command — so the published image could not start at all. The healthcheck then
curled ``/health``, which is an MCP *tool* rather than an HTTP route and would
have returned 404 even if the process had been running. Neither defect could
survive anyone running the image once; both survived because nobody did.

Configuration comes from the environment rather than argv because that is what a
container image can be handed. Defaults match ``main()``: stdio, loopback, 8000
— so running the module directly behaves like the CLI's ``mcp-server start``,
and only a deployment that sets these gets a network listener.
"""

from __future__ import annotations

import os

from nlqueries.mcp_server.server import main

_VALID_TRANSPORTS = ("stdio", "sse", "streamable-http")


def _transport() -> str:
    value = os.getenv("MCP_TRANSPORT", "stdio").strip().lower()
    if value not in _VALID_TRANSPORTS:
        raise SystemExit(f"MCP_TRANSPORT={value!r} is not one of {', '.join(_VALID_TRANSPORTS)}.")
    return value


def _port() -> int:
    raw = os.getenv("MCP_PORT", "8000").strip()
    try:
        return int(raw)
    except ValueError as exc:
        raise SystemExit(f"MCP_PORT={raw!r} is not a number.") from exc


if __name__ == "__main__":
    main(
        transport=_transport(),  # type: ignore[arg-type]
        host=os.getenv("MCP_HOST", "127.0.0.1").strip(),
        port=_port(),
    )
