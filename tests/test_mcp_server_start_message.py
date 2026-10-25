"""`mcp-server start --sse` announces the server only once its checks pass.

It printed "listening on ..." and then called `main`, whose checks (the cache
signing key, authentication, the wildcard-bind guard) could still refuse the
start: the terminal showed a green tick for a server that never ran.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner
from nlqueries.cli.main import cli
from nlqueries.mcp_server import server

_INSECURE_BIND_ENV = "NLQ_ALLOW_INSECURE_BIND"


def test_a_refused_start_prints_no_starting_line(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_INSECURE_BIND_ENV, raising=False)
    monkeypatch.setattr(server, "_require_authentication", lambda transport: True)
    ran: list[str] = []
    monkeypatch.setattr(server.mcp, "run", lambda transport: ran.append(transport))

    result = CliRunner().invoke(cli, ["mcp-server", "start", "--sse", "--host", "0.0.0.0"])

    assert result.exit_code != 0
    assert "starting on" not in result.output
    assert "listening on" not in result.output
    assert ran == []


def test_a_start_that_passes_its_checks_says_so_before_it_serves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(server, "_require_authentication", lambda transport: True)
    monkeypatch.setattr(
        server, "_refuse_unauthenticated_exposure", lambda host, *, authenticated=False: None
    )
    monkeypatch.setattr(server, "_build_server", lambda **kwargs: server.mcp)
    monkeypatch.setattr(server.mcp, "run", lambda transport: events.append(f"run {transport}"))

    result = CliRunner().invoke(cli, ["mcp-server", "start", "--sse", "--port", "8123"])

    assert result.exit_code == 0, result.output
    assert result.output.count("starting on http://127.0.0.1:8123/sse") == 1
    assert events == ["run sse"]


def test_main_calls_on_ready_after_the_checks_and_before_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(server.mcp, "run", lambda transport: events.append("run"))

    server.main(transport="stdio", on_ready=lambda: events.append("ready"))

    assert events == ["ready", "run"]


def test_main_does_not_call_on_ready_when_a_check_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_INSECURE_BIND_ENV, raising=False)
    monkeypatch.setattr(server, "_require_authentication", lambda transport: True)
    ready: list[bool] = []

    with pytest.raises(SystemExit):
        server.main(transport="sse", host="0.0.0.0", on_ready=lambda: ready.append(True))

    assert ready == []
