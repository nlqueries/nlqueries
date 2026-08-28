"""OIDC ID token verification utility.

Verifies JWTs issued by any OIDC-compliant provider using that provider's
JWKS endpoint.  Pure utility — no FastAPI, Celery, or database dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import jwt
from jwt.algorithms import RSAAlgorithm


class OidcVerificationError(Exception):
    """Raised when token verification fails for any reason."""


@dataclass
class OidcClaims:
    sub: str
    email: str
    name: str | None
    given_name: str | None
    family_name: str | None
    picture: str | None
    email_verified: bool
    raw: dict[str, Any] = field(default_factory=dict)


class OidcTokenVerifier:
    """Verify OIDC ID tokens against a provider's JWKS endpoint.

    Usage:
        verifier = OidcTokenVerifier(
            discovery_url="https://accounts.google.com/.well-known/openid-configuration"
        )
        claims = verifier.verify(id_token, client_id="my-client-id")
    """

    def __init__(self, discovery_url: str, http_timeout: int = 10) -> None:
        """Fetch the OIDC discovery document and cache the JWKS URI.

        JWKS is NOT fetched here — it is fetched lazily on first verify() call
        and cached for 1 hour.
        """
        self._http_timeout = http_timeout
        try:
            resp = httpx.get(discovery_url, timeout=http_timeout)
            resp.raise_for_status()
            doc: dict[str, Any] = resp.json()
        except OidcVerificationError:
            raise
        except Exception as exc:
            raise OidcVerificationError(f"Failed to fetch OIDC discovery document: {exc}") from exc

        self._issuer: str = doc.get("issuer", "")
        self._jwks_uri: str = doc.get("jwks_uri", "")
        if not self._jwks_uri:
            raise OidcVerificationError("Discovery document is missing 'jwks_uri'.")
        # Refused rather than carried as an empty string. `issuer=None` does not
        # tell PyJWT to compare against nothing, it tells it not to compare, so
        # a document without this disabled the issuer check entirely and any
        # token whose JWKS could be fetched would verify (SEC-15).
        if not self._issuer:
            raise OidcVerificationError(
                "Discovery document is missing 'issuer'. Without it a token's "
                "issuer cannot be checked, so tokens minted by another provider "
                "would be accepted."
            )

        # Cache: {"keys": [...], "fetched_at": datetime}
        self._jwks_cache: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_jwks(self) -> list[dict[str, Any]]:
        """Fetch fresh JWKS from the provider and update the cache."""
        try:
            resp = httpx.get(self._jwks_uri, timeout=self._http_timeout)
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
        except OidcVerificationError:
            raise
        except Exception as exc:
            raise OidcVerificationError(f"Failed to fetch JWKS: {exc}") from exc

        keys: list[dict[str, Any]] = data.get("keys", [])
        self._jwks_cache = {"keys": keys, "fetched_at": datetime.now(UTC)}
        return keys

    def _get_keys(self) -> list[dict[str, Any]]:
        """Return cached JWKS keys, re-fetching if the cache is stale (> 60 min)."""
        if not self._jwks_cache:
            return self._fetch_jwks()
        fetched_at: datetime = self._jwks_cache["fetched_at"]
        if datetime.now(UTC) - fetched_at > timedelta(minutes=60):
            return self._fetch_jwks()
        keys: list[dict[str, Any]] = self._jwks_cache["keys"]
        return keys

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(self, id_token: str, client_id: str) -> OidcClaims:
        """Verify an OIDC ID token and return the parsed claims.

        Steps:
        1. Decode the JWT header to get the ``kid`` (key ID).
        2. Fetch the JWKS from the cached URI; find the matching key by ``kid``.
           If not found, refresh the JWKS cache once (provider may have rotated keys).
        3. Verify the JWT signature, expiry (``exp``), audience (``aud == client_id``),
           and issuer (``iss`` matches the discovery document's ``issuer``).
        4. Extract and return OidcClaims. Raises OidcVerificationError on any failure.
        """
        # Step 1 — decode header
        try:
            header = jwt.get_unverified_header(id_token)
        except jwt.exceptions.DecodeError as exc:
            raise OidcVerificationError(f"Invalid token header: {exc}") from exc

        kid: str | None = header.get("kid")

        # Step 2 — locate the signing key
        keys = self._get_keys()
        key_data = next((k for k in keys if k.get("kid") == kid), None)

        if key_data is None:
            # Provider may have rotated keys — refresh the cache once
            keys = self._fetch_jwks()
            key_data = next((k for k in keys if k.get("kid") == kid), None)
            if key_data is None:
                raise OidcVerificationError(
                    f"No JWKS key found for kid={kid!r}. "
                    "Ensure the discovery URL points to the correct provider."
                )

        # Build the public key from the JWK entry
        try:
            public_key = RSAAlgorithm.from_jwk(key_data)
        except Exception as exc:
            raise OidcVerificationError(f"Failed to parse JWKS key: {exc}") from exc

        # Step 3 — verify signature, expiry, audience, and issuer.
        # PyJWT 2.x automatically verifies exp/aud/iss when the corresponding
        # parameters are supplied — no need to set options explicitly.
        issuer_arg: str | None = self._issuer if self._issuer else None

        try:
            claims: dict[str, Any] = jwt.decode(
                id_token,
                public_key,  # type: ignore[arg-type]
                algorithms=["RS256"],
                audience=client_id,
                issuer=issuer_arg,
            )
        except jwt.ExpiredSignatureError as exc:
            raise OidcVerificationError("Token has expired.") from exc
        except jwt.InvalidAudienceError as exc:
            raise OidcVerificationError(
                f"Token audience is invalid (expected '{client_id}')."
            ) from exc
        except jwt.InvalidIssuerError as exc:
            raise OidcVerificationError(
                f"Token issuer is invalid (expected '{self._issuer}')."
            ) from exc
        except jwt.PyJWTError as exc:
            raise OidcVerificationError(f"Token verification failed: {exc}") from exc

        # Step 4 — extract claims.
        # `sub` is the identity the rest of the system authorises against. A
        # token without one authenticates nobody, and reading it with a default
        # turned that into an identity of empty string — which compares equal to
        # the next one (SEC-15).
        subject = str(claims.get("sub") or "").strip()
        if not subject:
            raise OidcVerificationError("Token has no subject ('sub') claim.")

        return OidcClaims(
            sub=subject,
            email=str(claims.get("email", "")),
            name=claims.get("name"),
            given_name=claims.get("given_name"),
            family_name=claims.get("family_name"),
            picture=claims.get("picture"),
            email_verified=bool(claims.get("email_verified", False)),
            raw=claims,
        )
