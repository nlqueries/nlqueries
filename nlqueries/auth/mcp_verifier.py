"""
nlqueries.auth.mcp_verifier
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Turning a bearer token presented over the MCP transport into a principal.

Step 2 of SEC-05. The SDK supplies the enforcement point --
``FastMCP(token_verifier=..., auth=...)`` calls ``verify_token`` and rejects the
request when it returns ``None`` -- so what belongs here is establishing who the
caller is, not deciding what they may do. That decision is the authorizer's, and
arrives with step 3.

Two ways to be somebody, per decision C:

*OIDC* wraps the verifier hardened in #157, which refuses a token whose issuer
cannot be checked and one that names no subject. This is what a deployment with
an identity provider should use.

*A static token* exists because requiring an identity provider to run a
network-reachable server would push a single-operator install towards turning
authentication off entirely, which is the outcome worth avoiding. It
authenticates; it does not authorise. The subject it maps to is granted whatever
the authorizer grants that subject and nothing more.

A rejected token is never logged, and neither is an accepted one. The audit
record names the subject and the decision.
"""

from __future__ import annotations

import hmac
import logging
import os
from typing import Any

from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.settings import AuthSettings
from pydantic import AnyHttpUrl, ValidationError

from nlqueries.auth.oidc_token import OidcTokenVerifier, OidcVerificationError
from nlqueries.auth.principal import Principal, Source

logger = logging.getLogger(__name__)

# --- configuration -----------------------------------------------------------

#: Where clients reach this server. Required whenever authentication is on: the
#: SDK advertises it, and a token issued for a different resource should not be
#: accepted here.
RESOURCE_URL_ENV = "NLQ_MCP_RESOURCE_URL"

OIDC_DISCOVERY_ENV = "NLQ_MCP_OIDC_DISCOVERY_URL"
OIDC_CLIENT_ID_ENV = "NLQ_MCP_OIDC_CLIENT_ID"

STATIC_TOKEN_ENV = "NLQ_MCP_STATIC_TOKEN"
STATIC_TOKEN_FILE_ENV = "NLQ_MCP_STATIC_TOKEN_FILE"
STATIC_SUBJECT_ENV = "NLQ_MCP_STATIC_SUBJECT"

#: The shortest token worth calling one. A static token is the only thing
#: standing between the network and `query`, so a short one is a mistake worth
#: refusing rather than warning about.
MIN_STATIC_TOKEN_LENGTH = 32


class McpAuthConfigError(RuntimeError):
    """Raised for a configuration that cannot be honoured.

    Distinct from a rejected token: this is the operator's problem, and the
    server should not start.
    """


class OidcAccessTokenVerifier(TokenVerifier):
    """Verifies an OIDC ID token and reports the subject it names."""

    def __init__(self, verifier: OidcTokenVerifier, client_id: str, resource_url: str) -> None:
        self._verifier = verifier
        self._client_id = client_id
        self._resource_url = resource_url

    async def verify_token(self, token: str) -> AccessToken | None:
        try:
            claims = self._verifier.verify(token, self._client_id)
        except OidcVerificationError as exc:
            # The reason, never the token. A rejected token is still a
            # credential and still reaches logs.
            logger.warning("MCP token rejected: %s", exc)
            return None

        return AccessToken(
            token=token,
            client_id=self._client_id,
            scopes=[],
            subject=claims.sub,
            resource=self._resource_url,
            claims=dict(claims.raw),
        )


class StaticTokenVerifier(TokenVerifier):
    """Compares against one pre-shared token and maps it to one subject."""

    def __init__(self, token: str, subject: str, resource_url: str) -> None:
        if len(token) < MIN_STATIC_TOKEN_LENGTH:
            raise McpAuthConfigError(
                f"The static MCP token is shorter than {MIN_STATIC_TOKEN_LENGTH} "
                "characters. It is the only thing between the network and the "
                "query tool. Generate one with: openssl rand -hex 32"
            )
        self._token = token
        self._subject = subject
        self._resource_url = resource_url

    async def verify_token(self, token: str) -> AccessToken | None:
        # compare_digest, so a wrong token takes the same time as a right one
        # regardless of how much of the prefix matched.
        if not hmac.compare_digest(token, self._token):
            logger.warning("MCP token rejected: does not match the configured static token")
            return None

        return AccessToken(
            token=token,
            client_id=self._subject,
            scopes=[],
            subject=self._subject,
            resource=self._resource_url,
        )


def _static_token() -> str | None:
    """The configured static token, from a file or the environment."""
    path = os.getenv(STATIC_TOKEN_FILE_ENV, "").strip()
    if path:
        try:
            return open(path, encoding="utf-8").read().strip()  # noqa: SIM115
        except OSError as exc:
            raise McpAuthConfigError(
                f"{STATIC_TOKEN_FILE_ENV} is set to {path}, which cannot be read."
            ) from exc
    return os.getenv(STATIC_TOKEN_ENV, "").strip() or None


def _url(value: str, name: str) -> AnyHttpUrl:
    """Parse *value* as a URL, naming *name* when it is not one.

    Checked at startup rather than left to the SDK: a malformed resource URL
    should stop the server with a message that says which setting is wrong.
    """
    try:
        return AnyHttpUrl(value)
    except ValidationError as exc:
        raise McpAuthConfigError(f"{name} is not a valid URL: {value!r}") from exc


def build_verifier() -> tuple[TokenVerifier, AuthSettings] | None:
    """The verifier this deployment is configured for, or None for neither.

    Returns None only when nothing is configured. A configuration that is
    present but unusable raises instead, because the alternative is a server
    that starts and turns out to be open.
    """
    discovery = os.getenv(OIDC_DISCOVERY_ENV, "").strip()
    static = _static_token()

    if not discovery and not static:
        return None

    if discovery and static:
        raise McpAuthConfigError(
            f"Both {OIDC_DISCOVERY_ENV} and a static token are set. Configure one: "
            "two ways in means two things to get wrong."
        )

    resource_url = os.getenv(RESOURCE_URL_ENV, "").strip()
    if not resource_url:
        raise McpAuthConfigError(
            f"{RESOURCE_URL_ENV} must be set when MCP authentication is configured. "
            "It is the URL clients reach this server at, and it is what stops a "
            "token issued for somewhere else being accepted here."
        )

    # Validated here rather than at the point of use: the static path derives
    # the issuer from this value, so a malformed one reported as "issuer" would
    # name something the operator never set.
    resource = _url(resource_url, RESOURCE_URL_ENV)

    if discovery:
        client_id = os.getenv(OIDC_CLIENT_ID_ENV, "").strip()
        if not client_id:
            raise McpAuthConfigError(
                f"{OIDC_CLIENT_ID_ENV} must be set alongside {OIDC_DISCOVERY_ENV}: "
                "it is the audience a token has to be addressed to."
            )
        oidc = OidcTokenVerifier(discovery)
        verifier: TokenVerifier = OidcAccessTokenVerifier(oidc, client_id, resource_url)
        # From the provider's discovery document, so a bad value here is the
        # provider's, not the operator's.
        issuer_url = _url(oidc.issuer, f"the issuer in {discovery}")
    else:
        assert static is not None
        subject = os.getenv(STATIC_SUBJECT_ENV, "").strip() or "static-token"
        verifier = StaticTokenVerifier(static, subject, resource_url)
        # The deployment issued this token to itself; there is no third party.
        issuer_url = resource

    return verifier, AuthSettings(issuer_url=issuer_url, resource_server_url=resource)


def principal_from(access_token: Any) -> Principal:
    """The principal an authorizer will be asked about.

    Kept here rather than on the transport so there is one place that decides
    what an ``AccessToken`` means.
    """
    subject = getattr(access_token, "subject", None) or getattr(access_token, "client_id", "")
    claims = getattr(access_token, "claims", None) or {}
    source = Source.OIDC if claims else Source.STATIC_TOKEN
    return Principal(
        subject=str(subject),
        source=source,
        scopes=frozenset(getattr(access_token, "scopes", ()) or ()),
        claims=dict(claims),
    )
