"""OIDC ID token verification utility.

Verifies JWTs issued by any OIDC-compliant provider using that provider's
JWKS endpoint.  Pure utility — no FastAPI, Celery, or database dependencies.
"""

from __future__ import annotations

import ipaddress
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlsplit

import httpx
import jwt

logger = logging.getLogger(__name__)

#: The clock the refresh intervals are measured on. A name of this module's own,
#: so a test can move it without moving `time.monotonic` for everything else.
_monotonic = time.monotonic

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

#: How long fetched keys are used before they are fetched again.
_JWKS_MAX_AGE = timedelta(minutes=60)

#: At most one JWKS fetch per this many seconds is caused by a token whose key
#: id is not in the cached set, however many such tokens arrive: the key id is
#: chosen by whoever sent the token, before anything about it is verified. A
#: provider's key rotation is still picked up within one interval. The same
#: interval spaces retries when a scheduled refresh fails.
_FORCED_REFRESH_INTERVAL = 60.0

#: How long a request with no usable keys waits for another request's fetch.
#: Long enough for an ordinary fetch, short of the HTTP timeout an outage costs.
_FETCH_WAIT_SECONDS = 2.0

#: The largest discovery document or JWKS read. Both are a few kilobytes.
_MAX_DOCUMENT_BYTES = 1 << 20

#: Response headers that describe the body as sent, not as decoded.
_ENCODED_FORM_HEADERS = frozenset({"content-encoding", "content-length", "transfer-encoding"})

#: How long keys keep being used while refreshing them fails. Past this the
#: failure is raised, as it is when nothing is cached: a provider whose JWKS
#: has moved for good must not leave a revoked key accepted indefinitely.
_JWKS_MAX_STALE = timedelta(hours=24)

#: The curve each ECDSA algorithm is defined on (RFC 7518, 3.4).
_EC_CURVES = {"ES256": "P-256", "ES384": "P-384", "ES512": "P-521"}
_EDDSA_CURVES = frozenset({"Ed25519", "Ed448"})


class OidcVerificationError(Exception):
    """Raised when token verification fails for any reason."""


def _is_loopback(host: str | None) -> bool:
    if not host:
        return False
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _require_https(url: str, what: str) -> None:
    """Refuse *url* unless it is https, or http to this machine.

    The discovery document names the keys that decide who a token's bearer is,
    so whoever can alter it in transit can sign in as anyone. Plain http is
    allowed only to a loopback host, for an identity provider run locally.
    """
    parts = urlsplit(url)
    if parts.scheme == "https":
        return
    if parts.scheme == "http" and _is_loopback(parts.hostname):
        return
    raise OidcVerificationError(
        f"{what} must use https (plain http is accepted only for localhost), got {url!r}."
    )


def _get(url: str, timeout: float, what: str) -> httpx.Response:
    """GET *url*, refusing a body larger than :data:`_MAX_DOCUMENT_BYTES`.

    Streamed, so an oversized body is abandoned rather than read into memory.
    Redirects are not followed, which is httpx's default.
    """
    with httpx.stream("GET", url, timeout=timeout) as resp:
        body = bytearray()
        for chunk in resp.iter_bytes():
            body += chunk
            if len(body) > _MAX_DOCUMENT_BYTES:
                raise OidcVerificationError(
                    f"{what} at {url} is larger than {_MAX_DOCUMENT_BYTES} bytes."
                )
        # `iter_bytes` has already decompressed the body, so the headers that
        # describe the encoded form are dropped: given `content-encoding: gzip`,
        # the new response would try to decompress it a second time and fail,
        # and providers that compress (Google does) would verify nothing.
        headers = [
            (name, value)
            for name, value in resp.headers.items()
            if name.lower() not in _ENCODED_FORM_HEADERS
        ]
        return httpx.Response(
            resp.status_code, headers=headers, content=bytes(body), request=resp.request
        )


@dataclass
class OidcClaims:
    """The verified claims of an ID token.

    ``email`` is empty when the token carried no usable address (none, or a
    claim that is not a string); an empty address must never be matched
    against an account. ``email_verified`` is never true without an address.
    """

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

        The discovery URL and the ``jwks_uri`` it names must be https (or http
        to localhost), and the document's ``issuer`` must be the one the URL
        was derived from (OpenID Connect Discovery 1.0, section 4.3).
        """
        if leeway_seconds < 0:
            raise ValueError("leeway_seconds cannot be negative.")
        self._http_timeout = http_timeout
        self._leeway = leeway_seconds
        _require_https(discovery_url, "The OIDC discovery URL")
        try:
            resp = _get(discovery_url, timeout=http_timeout, what="The OIDC discovery document")
            resp.raise_for_status()
            doc: Any = resp.json()
        except OidcVerificationError:
            raise
        except Exception as exc:
            raise OidcVerificationError(f"Failed to fetch OIDC discovery document: {exc}") from exc
        if not isinstance(doc, dict):
            raise OidcVerificationError("The OIDC discovery document is not a JSON object.")

        issuer = doc.get("issuer", "")
        jwks_uri = doc.get("jwks_uri", "")
        self._issuer: str = issuer if isinstance(issuer, str) else ""
        self._jwks_uri: str = jwks_uri if isinstance(jwks_uri, str) else ""
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
        # The issuer has to be the one this URL belongs to. Otherwise a
        # document served from one place could name any issuer, and tokens
        # minted by that issuer would verify here.
        expected = self._issuer.rstrip("/") + "/.well-known/openid-configuration"
        if discovery_url.rstrip("/") != expected:
            raise OidcVerificationError(
                f"The discovery document's issuer {self._issuer!r} does not match the "
                f"discovery URL {discovery_url!r}; it should be served at {expected!r}."
            )
        _require_https(self._jwks_uri, "The provider's jwks_uri")

        # Cache: {"keys": [...], "fetched_at": datetime}
        self._jwks_cache: dict[str, Any] = {}
        # Serialises fetches, so concurrent requests needing fresh keys wait
        # for one fetch rather than each making their own.
        self._fetch_lock = threading.Lock()
        # time.monotonic() of the last fetch caused by an unknown key id.
        self._last_forced_refresh = float("-inf")
        # time.monotonic() before which a failed fetch is not retried, and why
        # the last one failed.
        self._retry_after = float("-inf")
        self._last_failure = ""

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
            resp = _get(self._jwks_uri, timeout=self._http_timeout, what="The JWKS")
            resp.raise_for_status()
            data: Any = resp.json()
        except OidcVerificationError:
            raise
        except Exception as exc:
            raise OidcVerificationError(f"Failed to fetch JWKS: {exc}") from exc

        # A reply that is not a key set is a failed fetch, not an empty set.
        # Cached as empty, a provider or proxy answering 200 with `{}` or a
        # maintenance page replaced the working keys with none, refused every
        # token until the next refresh, and said nothing; raised, it goes
        # through the same grace and retry spacing as a connection error.
        sent = data.get("keys") if isinstance(data, dict) else None
        if not isinstance(sent, list):
            raise OidcVerificationError(
                "Failed to fetch JWKS: the reply is not a JSON object with a 'keys' list."
            )
        keys: list[dict[str, Any]] = [k for k in sent if isinstance(k, dict)]
        # And a key set with no keys in it, or none that is an object: cached,
        # it would replace the working keys with none just the same.
        if not keys:
            raise OidcVerificationError(
                "Failed to fetch JWKS: the reply's 'keys' list has no keys."
            )
        self._jwks_cache = {"keys": keys, "fetched_at": datetime.now(UTC)}
        return keys

    def _fresh(self) -> bool:
        return bool(self._jwks_cache) and (
            datetime.now(UTC) - self._jwks_cache["fetched_at"] <= _JWKS_MAX_AGE
        )

    def _usable(self) -> bool:
        return bool(self._jwks_cache) and (
            datetime.now(UTC) - self._jwks_cache["fetched_at"] <= _JWKS_MAX_STALE
        )

    def _cached(self) -> list[dict[str, Any]]:
        keys: list[dict[str, Any]] = self._jwks_cache.get("keys", [])
        return keys

    def _refused_until_retry(self) -> OidcVerificationError:
        return OidcVerificationError(
            f"Failed to fetch JWKS; not retried until {_FORCED_REFRESH_INTERVAL:.0f} s "
            f"after the last attempt: {self._last_failure}"
        )

    def _get_keys(self) -> list[dict[str, Any]]:
        """Return cached JWKS keys, re-fetching if the cache is stale (> 60 min).

        At most one fetch is attempted per :data:`_FORCED_REFRESH_INTERVAL`
        after a failure, whatever is cached, so a provider that is down costs
        each request nothing rather than a wait on the network. Until the next
        attempt, keys younger than :data:`_JWKS_MAX_STALE` stay in use; with
        nothing usable the last failure is raised at once.

        While another request is refreshing, usable cached keys are returned
        rather than waited behind. Only with nothing usable does a request wait
        for that fetch, since it cannot verify anything without it.
        """
        if self._fresh():
            return self._cached()
        usable = self._usable()
        if _monotonic() < self._retry_after:
            if usable:
                return self._cached()
            raise self._refused_until_retry()
        if usable:
            if not self._fetch_lock.acquire(blocking=False):
                return self._cached()
        elif not self._fetch_lock.acquire(timeout=_FETCH_WAIT_SECONDS):
            # Nothing usable, and another request's fetch has not finished:
            # waiting longer would hold a worker thread, shared with the rest of
            # the server, for up to the HTTP timeout, during an outage once per
            # retry interval -- and a fetch that then fails refuses it anyway.
            raise OidcVerificationError(
                f"The provider's signing keys are still being fetched after "
                f"{_FETCH_WAIT_SECONDS:.0f} s; refused rather than queued behind it."
            )
        try:
            # Another request may have refreshed, or failed to, meanwhile.
            if self._fresh():
                return self._cached()
            if _monotonic() < self._retry_after:
                if usable:
                    return self._cached()
                raise self._refused_until_retry()
            try:
                return self._fetch_jwks()
            except OidcVerificationError as exc:
                self._retry_after = _monotonic() + _FORCED_REFRESH_INTERVAL
                self._last_failure = str(exc)
                if not usable:
                    raise
                logger.warning(
                    "Refreshing the OIDC signing keys failed; using the cached keys "
                    "and retrying in %.0f s: %s",
                    _FORCED_REFRESH_INTERVAL,
                    exc,
                )
                return self._cached()
        finally:
            self._fetch_lock.release()

    def _refresh_for_unknown_kid(self) -> list[dict[str, Any]]:
        """The keys after fetching for a key id not in the cache, at most once
        per interval.

        Within the interval the cached keys are returned without a fetch, and
        without waiting on the lock: the key id comes from an unverified token,
        so honouring every one would let anybody make this server fetch the
        JWKS on every request, and making them wait would let them hold every
        worker thread for the length of the one fetch allowed. They are the
        cached keys as they are now, so once a rotation's fetch has completed
        later requests find the new key; one arriving while it is in flight is
        refused and retries. An invented key id is never among them.
        """
        if _monotonic() - self._last_forced_refresh < _FORCED_REFRESH_INTERVAL:
            return self._cached()
        # Never waits: a fetch already in flight, scheduled or forced, means
        # this request is refused with the keys cached now and retries.
        if not self._fetch_lock.acquire(blocking=False):
            return self._cached()
        try:
            now = _monotonic()
            if now - self._last_forced_refresh < _FORCED_REFRESH_INTERVAL:
                return self._cached()
            self._last_forced_refresh = now
            return self._fetch_jwks()
        finally:
            self._fetch_lock.release()

    @staticmethod
    def _key_for(keys: list[dict[str, Any]], kid: str | None) -> dict[str, Any] | None:
        """The signing key named *kid*. A key published for encryption
        (``use: enc``) is not one, even when its id matches."""
        return next(
            (k for k in keys if k.get("kid") == kid and k.get("use", "sig") == "sig"),
            None,
        )

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
        2. Fetch the JWKS from the cached URI; find the matching signing key by
           ``kid``. If not found, refresh the JWKS (the provider may have
           rotated keys), at most once per :data:`_FORCED_REFRESH_INTERVAL`.
           The key must fit the algorithm (see :meth:`_signing_key`).
        3. Verify the JWT signature, expiry (``exp``), audience (``aud``
           contains ``client_id``, and ``azp`` is ``client_id`` when present,
           and present when ``aud`` names several), and issuer (``iss``
           matches the discovery document's ``issuer``), allowing the clock
           difference given at construction.
           ``exp``, ``iss`` and ``aud`` must be present.
        4. Extract and return OidcClaims. Raises OidcVerificationError on any failure.
        """
        # Step 1 — decode header. Any invalid-token error, not only a decode
        # error: the header comes from whoever sent the token (PyJWT raises
        # InvalidTokenError for a `kid` that is not a string), and an exception
        # escaping here is a 500 and a traceback per request.
        try:
            header = jwt.get_unverified_header(id_token)
        except jwt.exceptions.InvalidTokenError as exc:
            raise OidcVerificationError(f"Invalid token header: {exc}") from exc

        algorithm = str(header.get("alg"))
        if algorithm not in _ACCEPTED_ALGORITHMS:
            raise OidcVerificationError(
                f"Token is signed with {algorithm!r}; only asymmetric algorithms are "
                f"accepted ({', '.join(sorted(_ACCEPTED_ALGORITHMS))})."
            )
        # PyJWT has already refused a `kid` that is not a string, with an
        # InvalidTokenError other than DecodeError, caught above.
        kid: str | None = header.get("kid")

        # Step 2 — locate the signing key
        keys = self._get_keys()
        key_data = self._key_for(keys, kid)

        if key_data is None:
            # The provider may have rotated its keys: refresh, unless a refresh
            # for an unknown key id was made within the last interval.
            key_data = self._key_for(self._refresh_for_unknown_kid(), kid)
            if key_data is None:
                raise OidcVerificationError(
                    f"No JWKS key found for kid={kid!r}. "
                    "Ensure the discovery URL points to the correct provider."
                )

        public_key = self._signing_key(key_data, algorithm)

        # Step 3 — verify signature, expiry, audience, and issuer.
        # PyJWT checks `exp`, `aud` and `iss` against the arguments given, but
        # only when the token carries them: a token without `exp` would never
        # expire. So all three are required. `sub` is checked below, with a
        # message of its own.
        issuer_arg: str | None = self._issuer if self._issuer else None

        try:
            claims: dict[str, Any] = jwt.decode(
                id_token,
                public_key,
                algorithms=[algorithm],
                audience=client_id,
                issuer=issuer_arg,
                leeway=self._leeway,
                options={"require": ["exp", "iss", "aud"]},
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

        # `azp` names the party the token was issued to (OpenID Connect Core
        # 1.0, section 3.1.3.7). When present it must be this client, even if
        # the audience is this client alone: a provider can issue a token
        # audienced to one client at another's request, and that token is the
        # requester's, not this client's. A token addressed to several
        # audiences must carry it, since the audience alone does not say whose
        # it is.
        audience = claims.get("aud")
        authorized_party = claims.get("azp")
        several = isinstance(audience, list) and len(audience) > 1
        if (authorized_party is not None or several) and authorized_party != client_id:
            raise OidcVerificationError(f"Token was not issued to '{client_id}' (azp).")

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
