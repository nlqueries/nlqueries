"""
Authenticating a caller on the MCP transport (SEC-05, step 2).

Nine tools were reachable without credentials, one of which runs SQL against a
configured database. The SDK rejects a request whose bearer token the verifier
declines, so what is under test here is which tokens are declined, and that a
configuration which is present but unusable stops the server rather than
falling back to none.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
from nlqueries.auth.mcp_verifier import (
    MIN_STATIC_TOKEN_LENGTH,
    McpAuthConfigError,
    OidcAccessTokenVerifier,
    StaticTokenVerifier,
    build_verifier,
    principal_from,
)
from nlqueries.auth.oidc_token import OidcClaims, OidcVerificationError
from nlqueries.auth.principal import Source

GOOD_TOKEN = "a" * MIN_STATIC_TOKEN_LENGTH
RESOURCE = "https://mcp.example.com"

ENV_VARS = (
    "NLQ_MCP_RESOURCE_URL",
    "NLQ_MCP_OIDC_DISCOVERY_URL",
    "NLQ_MCP_OIDC_CLIENT_ID",
    "NLQ_MCP_STATIC_TOKEN",
    "NLQ_MCP_STATIC_TOKEN_FILE",
    "NLQ_MCP_STATIC_SUBJECT",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """CI runs with its own environment; none of these may leak in."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _claims(sub: str = "user-123") -> OidcClaims:
    return OidcClaims(
        sub=sub,
        email="alice@example.com",
        name=None,
        given_name=None,
        family_name=None,
        picture=None,
        email_verified=True,
        raw={"sub": sub},
    )


class TestStaticToken:
    @pytest.mark.anyio
    async def test_the_configured_token_is_accepted(self) -> None:
        verifier = StaticTokenVerifier(GOOD_TOKEN, "operator", RESOURCE)

        result = await verifier.verify_token(GOOD_TOKEN)

        assert result is not None
        assert result.subject == "operator"

    @pytest.mark.anyio
    async def test_any_other_token_is_declined(self) -> None:
        verifier = StaticTokenVerifier(GOOD_TOKEN, "operator", RESOURCE)

        assert await verifier.verify_token("b" * MIN_STATIC_TOKEN_LENGTH) is None

    @pytest.mark.anyio
    async def test_a_prefix_of_the_token_is_declined(self) -> None:
        verifier = StaticTokenVerifier(GOOD_TOKEN, "operator", RESOURCE)

        assert await verifier.verify_token(GOOD_TOKEN[:-1]) is None

    @pytest.mark.anyio
    async def test_a_rejected_token_is_not_logged(self, caplog) -> None:
        """It is still a credential, and logs travel into support bundles."""
        verifier = StaticTokenVerifier(GOOD_TOKEN, "operator", RESOURCE)
        presented = "z" * MIN_STATIC_TOKEN_LENGTH

        with caplog.at_level(logging.DEBUG):
            await verifier.verify_token(presented)

        assert presented not in caplog.text

    def test_a_short_token_is_refused_at_construction(self) -> None:
        """It is the only thing between the network and the query tool."""
        with pytest.raises(McpAuthConfigError, match="shorter than"):
            StaticTokenVerifier("short", "operator", RESOURCE)


class _AcceptingOidc:
    def verify(self, token: str, client_id: str) -> OidcClaims:
        return _claims()


class _RefusingOidc:
    def verify(self, token: str, client_id: str) -> OidcClaims:
        raise OidcVerificationError("Token has expired.")


class TestOidc:
    @pytest.mark.anyio
    async def test_a_verified_token_names_its_subject(self) -> None:
        verifier = OidcAccessTokenVerifier(_AcceptingOidc(), "client-1", RESOURCE)  # type: ignore[arg-type]

        result = await verifier.verify_token("a-token")

        assert result is not None
        assert result.subject == "user-123"

    @pytest.mark.anyio
    async def test_a_token_the_verifier_refuses_is_declined(self, caplog) -> None:
        """The reason reaches the log; the token does not."""
        verifier = OidcAccessTokenVerifier(_RefusingOidc(), "client-1", RESOURCE)  # type: ignore[arg-type]

        with caplog.at_level(logging.WARNING):
            result = await verifier.verify_token("an-expired-token")

        assert result is None
        assert "an-expired-token" not in caplog.text
        assert "expired" in caplog.text


class TestConfiguration:
    def test_nothing_configured_is_not_an_error(self) -> None:
        """stdio needs no verifier. Refusing here would break Claude Desktop."""
        assert build_verifier() is None

    def test_a_static_token_without_a_resource_url_is_refused(self, monkeypatch) -> None:
        monkeypatch.setenv("NLQ_MCP_STATIC_TOKEN", GOOD_TOKEN)

        with pytest.raises(McpAuthConfigError, match="NLQ_MCP_RESOURCE_URL"):
            build_verifier()

    def test_configuring_both_kinds_is_refused(self, monkeypatch) -> None:
        """Two ways in means two things to get wrong."""
        monkeypatch.setenv("NLQ_MCP_STATIC_TOKEN", GOOD_TOKEN)
        monkeypatch.setenv("NLQ_MCP_OIDC_DISCOVERY_URL", "https://idp.example.com/x")
        monkeypatch.setenv("NLQ_MCP_RESOURCE_URL", RESOURCE)

        with pytest.raises(McpAuthConfigError, match="Configure one"):
            build_verifier()

    def test_oidc_without_a_client_id_is_refused(self, monkeypatch) -> None:
        monkeypatch.setenv("NLQ_MCP_OIDC_DISCOVERY_URL", "https://idp.example.com/x")
        monkeypatch.setenv("NLQ_MCP_RESOURCE_URL", RESOURCE)

        with pytest.raises(McpAuthConfigError, match="NLQ_MCP_OIDC_CLIENT_ID"):
            build_verifier()

    def test_a_malformed_resource_url_is_refused(self, monkeypatch) -> None:
        """Caught at startup naming the setting, rather than surfacing later as
        a validation error from inside the SDK."""
        monkeypatch.setenv("NLQ_MCP_STATIC_TOKEN", GOOD_TOKEN)
        monkeypatch.setenv("NLQ_MCP_RESOURCE_URL", "not-a-url")

        with pytest.raises(McpAuthConfigError, match="NLQ_MCP_RESOURCE_URL"):
            build_verifier()

    def test_a_static_configuration_produces_a_verifier(self, monkeypatch) -> None:
        monkeypatch.setenv("NLQ_MCP_STATIC_TOKEN", GOOD_TOKEN)
        monkeypatch.setenv("NLQ_MCP_RESOURCE_URL", RESOURCE)

        built = build_verifier()

        assert built is not None
        verifier, settings = built
        assert isinstance(verifier, StaticTokenVerifier)
        assert str(settings.resource_server_url).rstrip("/") == RESOURCE

    def test_a_token_file_is_read(self, monkeypatch, tmp_path) -> None:
        path = tmp_path / "token"
        path.write_text(GOOD_TOKEN + "\n", encoding="utf-8")
        monkeypatch.setenv("NLQ_MCP_STATIC_TOKEN_FILE", str(path))
        monkeypatch.setenv("NLQ_MCP_RESOURCE_URL", RESOURCE)

        assert build_verifier() is not None

    def test_a_token_file_that_cannot_be_read_is_refused(self, monkeypatch, tmp_path) -> None:
        """Configured but unusable is the operator's problem, and not a reason
        to fall back to no authentication."""
        monkeypatch.setenv("NLQ_MCP_STATIC_TOKEN_FILE", str(tmp_path / "missing"))

        with pytest.raises(McpAuthConfigError, match="cannot be read"):
            build_verifier()


class _StaticAccessToken:
    subject = "operator"
    client_id = "operator"
    scopes: list[str] = []
    claims: dict[str, Any] | None = None


class _OidcAccessToken:
    subject = "user-123"
    client_id = "client-1"
    scopes = ["read"]
    claims = {"groups": ["finance"]}


class TestPrincipalMapping:
    def test_a_static_access_token_maps_to_a_static_principal(self) -> None:
        principal = principal_from(_StaticAccessToken())

        assert principal.subject == "operator"
        assert principal.source is Source.STATIC_TOKEN

    def test_an_oidc_access_token_carries_its_claims(self) -> None:
        principal = principal_from(_OidcAccessToken())

        assert principal.source is Source.OIDC
        assert principal.claims["groups"] == ["finance"]
        assert principal.scopes == frozenset({"read"})
