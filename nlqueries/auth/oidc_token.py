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

#: How far a provider's clock may be from this host's, in seconds, before a
#: fresh token reads as not yet valid or a current one as expired. Zero made
#: a provider one second ahead refuse every sign-in.
DEFAULT_LEEWAY_SECONDS = 60

#: The signing algorithms an ID token may use, by the JWK key type that
#: verifies them. Asymmetric only: an HMAC algorithm would be checked against
#: a published key as if it were a shared secret, and "none" signs nothing.
_ALGORITHMS_BY_KEY_TYPE: dict[str, frozenset[str]] = {
    "RSA": frozenset({"RS256", "RS384", "RS512", "PS256", "PS384", "PS512"}),
    "EC": frozenset({"ES256", "ES384", "ES512"}),
    "OKP": frozenset({"EdDSA"}),
}
_ACCEPTED_ALGORITHMS = frozenset().union(*_ALGORITHMS_BY_KEY_TYPE.values())

#: The curve each ECDSA algorithm is defined on (RFC 7518, 3.4).
_EC_CURVES = {"ES256": "P-256", "ES384": "P-384", "ES512": "P-521"}
_EDDSA_CURVES = frozenset({"Ed25519", "Ed448"})


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

    def __init__(
        self,
        discovery_url: str,
        http_timeout: int = 10,
        leeway_seconds: int = DEFAULT_LEEWAY_SECONDS,
    ) -> None:
        """Fetch the OIDC discovery document and cache the JWKS URI.

        JWKS is NOT fetched here — it is fetched lazily on first verify() call
        and cached for 1 hour. *leeway_seconds* is the clock difference allowed
        on ``exp``, ``nbf`` and ``iat``.
        """
        if leeway_seconds < 0:
            raise ValueError("leeway_seconds cannot be negative.")
        self._http_timeout = http_timeout
        self._leeway = leeway_seconds
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

    @property
    def issuer(self) -> str:
        """The issuer this verifier checks tokens against.

        Always non-empty: a discovery document without one is refused above.
        """
        return self._issuer

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

    @staticmethod
    def _signing_key(key_data: dict[str, Any], algorithm: str) -> Any:
        """The public key in *key_data*, once it is shown to fit *algorithm*.

        The token's header names its algorithm, so the key has to agree with
        it: its type, its own ``alg`` when it states one, and for EC and OKP
        keys its curve. Otherwise a token could choose how its signature is
        checked.
        """
        key_type = key_data.get("kty")
        kid = key_data.get("kid")
        if algorithm not in _ALGORITHMS_BY_KEY_TYPE.get(str(key_type), frozenset()):
            raise OidcVerificationError(
                f"Token is signed with {algorithm!r}, which its {key_type} key "
                f"(kid={kid!r}) cannot verify."
            )
        stated = key_data.get("alg")
        if stated and stated != algorithm:
            raise OidcVerificationError(
                f"Token is signed with {algorithm!r}, but its key (kid={kid!r}) is for {stated!r}."
            )
        curve = key_data.get("crv")
        if key_type == "EC" and curve != _EC_CURVES[algorithm]:
            raise OidcVerificationError(
                f"Token is signed with {algorithm!r}, which needs curve "
                f"{_EC_CURVES[algorithm]}; its key (kid={kid!r}) is on {curve!r}."
            )
        if key_type == "OKP" and curve not in _EDDSA_CURVES:
            raise OidcVerificationError(
                f"Token is signed with EdDSA; its key (kid={kid!r}) is on {curve!r}."
            )
        try:
            return jwt.PyJWK(key_data, algorithm=algorithm).key
        except Exception as exc:
            raise OidcVerificationError(f"Failed to parse JWKS key: {exc}") from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(self, id_token: str, client_id: str) -> OidcClaims:
        """Verify an OIDC ID token and return the parsed claims.

        Steps:
        1. Decode the JWT header to get the ``kid`` (key ID) and ``alg``, and
           refuse an algorithm that is not asymmetric.
        2. Fetch the JWKS from the cached URI; find the matching key by ``kid``.
           If not found, refresh the JWKS cache once (provider may have rotated keys).
           The key must fit the algorithm (see :meth:`_signing_key`).
        3. Verify the JWT signature, expiry (``exp``), audience (``aud == client_id``),
           and issuer (``iss`` matches the discovery document's ``issuer``),
           allowing the clock difference given at construction.
        4. Extract and return OidcClaims. Raises OidcVerificationError on any failure.
        """
        # Step 1 — decode header
        try:
            header = jwt.get_unverified_header(id_token)
        except jwt.exceptions.DecodeError as exc:
            raise OidcVerificationError(f"Invalid token header: {exc}") from exc

        algorithm = str(header.get("alg"))
        if algorithm not in _ACCEPTED_ALGORITHMS:
            raise OidcVerificationError(
                f"Token is signed with {algorithm!r}; only asymmetric algorithms are "
                f"accepted ({', '.join(sorted(_ACCEPTED_ALGORITHMS))})."
            )
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

        public_key = self._signing_key(key_data, algorithm)

        # Step 3 — verify signature, expiry, audience, and issuer.
        # PyJWT 2.x automatically verifies exp/aud/iss when the corresponding
        # parameters are supplied — no need to set options explicitly.
        issuer_arg: str | None = self._issuer if self._issuer else None

        try:
            claims: dict[str, Any] = jwt.decode(
                id_token,
                public_key,
                algorithms=[algorithm],
                audience=client_id,
                issuer=issuer_arg,
                leeway=self._leeway,
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

        # The email and its flag decide whose account a sign-in links to, so
        # they are read as strictly as `sub`. `str()` made `"email": null` the
        # address "None" (and `true` "True"), which every such token then
        # shared, and `bool()` read a provider's string `"false"` as verified.
        # The flag vouches for an address, so with none it is false: an empty
        # address marked verified would match every other such token.
        sent = claims.get("email")
        email = sent.strip() if isinstance(sent, str) else ""
        return OidcClaims(
            sub=subject,
            email=email,
            name=claims.get("name"),
            given_name=claims.get("given_name"),
            family_name=claims.get("family_name"),
            picture=claims.get("picture"),
            email_verified=bool(email) and _is_true(claims.get("email_verified")),
            raw=claims,
        )


def _is_true(value: Any) -> bool:
    """A boolean claim. Some providers send JSON booleans as strings, and
    ``bool("false")`` is True; anything but ``true`` or ``"true"`` is false."""
    return value is True or (isinstance(value, str) and value.strip().lower() == "true")
