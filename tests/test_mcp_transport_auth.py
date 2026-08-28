"""
That a networked MCP transport does not start unauthenticated (SEC-05, step 2).

The refusal is deliberately a separate switch from the wildcard-bind one.
`NLQ_ALLOW_INSECURE_BIND` says "I know this port is reachable".
`NLQ_ALLOW_UNAUTHENTICATED_MCP` says "I know anyone who reaches it may run SQL
against my database". A deployment that made the first admission should not be
taken to have made the second, which is what sharing one switch would do.

stdio is untouched: the caller launched the process and already has whatever it
has. Breaking that would break every Claude Desktop install for no gain.
"""

from __future__ import annotations

import pytest
from nlqueries.mcp_server import server as mcp_server

ENV_VARS = (
    "NLQ_MCP_RESOURCE_URL",
    "NLQ_MCP_OIDC_DISCOVERY_URL",
    "NLQ_MCP_OIDC_CLIENT_ID",
    "NLQ_MCP_STATIC_TOKEN",
    "NLQ_MCP_STATIC_TOKEN_FILE",
    "NLQ_ALLOW_UNAUTHENTICATED_MCP",
    "NLQ_ALLOW_INSECURE_BIND",
    "NLQ_MCP_GRANTS_FILE",
)

GOOD_TOKEN = "a" * 32


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("transport", ["sse", "streamable-http"])
def test_a_networked_transport_refuses_without_a_verifier(transport: str) -> None:
    """Both of them. `streamable-http` was absent from the original exposure
    check, so it was the one transport with no guard at all."""
    with pytest.raises(SystemExit, match="no authentication"):
        mcp_server._require_authentication(transport)


@pytest.mark.parametrize("transport", ["sse", "streamable-http"])
def test_a_configured_verifier_satisfies_the_requirement(transport: str, monkeypatch) -> None:
    monkeypatch.setenv("NLQ_MCP_STATIC_TOKEN", GOOD_TOKEN)
    monkeypatch.setenv("NLQ_MCP_RESOURCE_URL", "https://mcp.example.com")

    assert mcp_server._require_authentication(transport) is True


def test_stdio_needs_no_verifier() -> None:
    """The local profile. Refusing here would break Claude Desktop."""
    assert mcp_server._require_authentication("stdio") is False


def test_the_switch_permits_an_unauthenticated_transport(monkeypatch, caplog) -> None:
    """An operator can still do it, and is told what they have done."""
    monkeypatch.setenv("NLQ_ALLOW_UNAUTHENTICATED_MCP", "1")

    with caplog.at_level("WARNING"):
        required = mcp_server._require_authentication("sse")

    assert required is False
    assert "no authentication" in caplog.text


def test_the_wildcard_bind_switch_does_not_permit_it(monkeypatch) -> None:
    """The two admissions are different. Sharing a switch would let one stand
    for the other."""
    monkeypatch.setenv("NLQ_ALLOW_INSECURE_BIND", "1")

    with pytest.raises(SystemExit, match="no authentication"):
        mcp_server._require_authentication("sse")


def test_a_broken_configuration_stops_the_server(monkeypatch) -> None:
    """Configured but unusable must not degrade to unauthenticated: that is the
    state this whole change exists to prevent."""
    from nlqueries.auth.mcp_verifier import McpAuthConfigError

    monkeypatch.setenv("NLQ_MCP_STATIC_TOKEN", "too-short")
    monkeypatch.setenv("NLQ_MCP_RESOURCE_URL", "https://mcp.example.com")

    with pytest.raises(McpAuthConfigError):
        mcp_server._require_authentication("sse")


def test_the_refusal_names_what_to_configure(monkeypatch) -> None:
    """An operator meeting this for the first time is mid-upgrade and needs the
    variable names, not a principle."""
    with pytest.raises(SystemExit) as raised:
        mcp_server._require_authentication("sse")

    message = str(raised.value)

    assert "NLQ_MCP_OIDC_DISCOVERY_URL" in message
    assert "NLQ_MCP_STATIC_TOKEN" in message
    assert "NLQ_ALLOW_UNAUTHENTICATED_MCP" in message


def test_authentication_without_grants_refuses_to_start(monkeypatch) -> None:
    """Deny-by-default is the code path; this combination is a misconfiguration.

    A server that authenticates correctly and then denies every call is a
    support case, not a control, so it is refused at startup with the thing to
    fix named.
    """
    monkeypatch.setenv("NLQ_MCP_STATIC_TOKEN", GOOD_TOKEN)
    monkeypatch.setenv("NLQ_MCP_RESOURCE_URL", "https://mcp.example.com")
    monkeypatch.delenv("NLQ_MCP_GRANTS_FILE", raising=False)

    with pytest.raises(SystemExit, match="NLQ_MCP_GRANTS_FILE"):
        mcp_server._resolve_authorizer(authenticated=True)


def test_an_unauthenticated_server_uses_the_local_profile() -> None:
    """stdio. The process owner goes through the same interface rather than
    round the side of it."""
    from nlqueries.auth.authorizer import LocalAuthorizer

    assert isinstance(mcp_server._resolve_authorizer(authenticated=False), LocalAuthorizer)


def test_a_grants_file_produces_an_allowlist(monkeypatch, tmp_path) -> None:
    from nlqueries.auth.authorizer import ConfigAllowlistAuthorizer

    grants = tmp_path / "grants.yaml"
    grants.write_text(
        "grants:\n  - subject: alice\n    agents: ['sales']\n    actions: ['query:execute']\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("NLQ_MCP_STATIC_TOKEN", GOOD_TOKEN)
    monkeypatch.setenv("NLQ_MCP_RESOURCE_URL", "https://mcp.example.com")
    monkeypatch.setenv("NLQ_MCP_GRANTS_FILE", str(grants))

    authorizer = mcp_server._resolve_authorizer(authenticated=True)

    assert isinstance(authorizer, ConfigAllowlistAuthorizer)
    assert authorizer.subjects == frozenset({"alice"})


def test_a_networked_server_without_a_verifier_does_not_get_the_local_profile(
    monkeypatch,
) -> None:
    """With the override set, every caller reaching the port was treated as the
    process owner and granted everything, and the audit record named the account
    the server runs as. The profile is chosen by transport now.
    """
    from nlqueries.auth.authorizer import AllowAllUnauthenticated, LocalAuthorizer

    authorizer = mcp_server._resolve_authorizer(authenticated=False, networked=True)

    assert isinstance(authorizer, AllowAllUnauthenticated)
    assert not isinstance(authorizer, LocalAuthorizer)


def test_an_unauthenticated_networked_caller_is_anonymous(monkeypatch) -> None:
    """Not the server's own user. An audit trail naming the wrong party is worse
    than one that says it does not know."""
    from nlqueries.auth.principal import Source

    resolve = mcp_server._principal_for_call(networked=True)
    principal = resolve()

    assert principal.source is Source.ANONYMOUS
    assert not principal.is_local


def test_a_stdio_caller_is_still_the_process_owner() -> None:
    resolve = mcp_server._principal_for_call(networked=False)

    assert resolve().is_local


def test_a_server_builds_with_authentication_and_a_grants_file(monkeypatch, tmp_path) -> None:
    """The configuration this control exists for, exercised end to end — no test
    covered it, so nothing would have caught an error on that path.
    """
    grants = tmp_path / "grants.yaml"
    grants.write_text(
        "grants:\n  - subject: alice\n    agents: ['sales']\n    actions: ['query:execute']\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("NLQ_MCP_STATIC_TOKEN", GOOD_TOKEN)
    monkeypatch.setenv("NLQ_MCP_RESOURCE_URL", "https://mcp.example.com")
    monkeypatch.setenv("NLQ_MCP_GRANTS_FILE", str(grants))

    server = mcp_server._build_server(
        host="127.0.0.1", port=8123, authenticated=True, networked=True
    )

    assert server._token_verifier is not None
    assert len(server._tool_manager.list_tools()) == len(mcp_server._ALL_TOOLS)
