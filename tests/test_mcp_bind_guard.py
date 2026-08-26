"""
The MCP listener stays on loopback while it has no authentication (SEC-06).

Every tool on this server is reachable without credentials — including `query`,
which runs SQL against a configured database, and `invalidate_cache`. The CLI
used to default `--host` to `0.0.0.0`, and `docs/cli-reference.md` showed that
binding as ordinary usage, so the documented quickstart published a
database-query API to whatever could route to the host.
"""

from __future__ import annotations

import pytest
from nlqueries.mcp_server.server import _INSECURE_BIND_ENV, _refuse_unauthenticated_exposure


class TestBindGuard:
    @pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "192.168.1.10"])
    def test_a_specific_interface_is_allowed(self, host: str) -> None:
        """The guard is about wildcards, not about staying on this machine: an
        operator naming one interface has made a choice, and may well be behind
        a proxy already."""
        _refuse_unauthenticated_exposure(host)

    @pytest.mark.parametrize("host", ["0.0.0.0", "::", ""])
    def test_a_wildcard_is_refused(self, host: str, monkeypatch) -> None:
        monkeypatch.delenv(_INSECURE_BIND_ENV, raising=False)

        with pytest.raises(SystemExit) as excinfo:
            _refuse_unauthenticated_exposure(host)

        # The message has to say what is exposed and how to proceed — whoever
        # hits this is deploying, and a bare refusal invites a worse workaround.
        message = str(excinfo.value)
        assert "no authentication" in message
        assert "127.0.0.1" in message
        assert _INSECURE_BIND_ENV in message

    @pytest.mark.parametrize("value", ["1", "true", "YES"])
    def test_the_switch_permits_it_loudly(self, value: str, monkeypatch, caplog) -> None:
        import logging

        monkeypatch.setenv(_INSECURE_BIND_ENV, value)

        with caplog.at_level(logging.WARNING):
            _refuse_unauthenticated_exposure("0.0.0.0")

        assert "no authentication" in caplog.text

    def test_the_switch_is_not_set_by_a_stray_value(self, monkeypatch) -> None:
        """`NLQ_ALLOW_INSECURE_BIND=0` means no."""
        monkeypatch.setenv(_INSECURE_BIND_ENV, "0")

        with pytest.raises(SystemExit):
            _refuse_unauthenticated_exposure("0.0.0.0")


def test_stdio_never_reaches_the_guard(monkeypatch) -> None:
    """Claude Desktop is the documented primary path and opens no socket at all.

    It must keep working with no configuration, so the guard is inside the
    `transport == "sse"` branch only.
    """
    from nlqueries.mcp_server import server

    monkeypatch.delenv(_INSECURE_BIND_ENV, raising=False)
    started: list[str] = []
    monkeypatch.setattr(server.mcp, "run", lambda transport: started.append(transport))

    server.main(transport="stdio", host="0.0.0.0")

    assert started == ["stdio"]
