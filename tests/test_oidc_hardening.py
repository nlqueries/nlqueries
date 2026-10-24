"""How the OIDC verifier treats its provider and the tokens it is handed.

``tests/test_oidc_token.py`` covers verifying a good token and the usual
refusals. These cover what the verifier will fetch, how often, from where, and
which claims a token must carry.
"""

from __future__ import annotations

import gzip
import json
import threading
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
from nlqueries.auth import oidc_token
from nlqueries.auth.oidc_token import OidcTokenVerifier, OidcVerificationError

from tests.test_oidc_token import (
    CLIENT_ID,
    DISCOVERY_DOC,
    DISCOVERY_URL,
    ISSUER,
    JWKS_URI,
    _forge_header,
    _generate_rsa_key_pair,
    _make_jwks,
    _make_token,
    _mock_http_response,
)


class _Provider:
    """A stand-in for the provider's two endpoints that counts JWKS fetches."""

    def __init__(self, jwks: dict[str, Any], discovery: dict[str, Any] | None = None) -> None:
        self.jwks = jwks
        self.discovery = discovery or DISCOVERY_DOC
        self.jwks_fetches = 0
        self.jwks_error: Exception | None = None

    def get(self, url: str, **kwargs: Any) -> MagicMock:
        if "openid-configuration" in url:
            return _mock_http_response(self.discovery)
        self.jwks_fetches += 1
        if self.jwks_error is not None:
            raise self.jwks_error
        return _mock_http_response(self.jwks)


class _Clock:
    """``time.monotonic`` under the test's control."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    c = _Clock()
    monkeypatch.setattr(oidc_token, "_monotonic", c)
    return c


def _verifier(provider: _Provider, url: str = DISCOVERY_URL) -> OidcTokenVerifier:
    with patch.object(oidc_token, "_get", side_effect=provider.get):
        return OidcTokenVerifier(url)


def _verify(verifier: OidcTokenVerifier, provider: _Provider, token: str) -> Any:
    with patch.object(oidc_token, "_get", side_effect=provider.get):
        return verifier.verify(token, CLIENT_ID)


# ---------------------------------------------------------------------------
# Unknown key ids
# ---------------------------------------------------------------------------


class TestUnknownKeyIds:
    def test_unknown_key_ids_cause_at_most_one_refresh_per_interval(self, clock: _Clock) -> None:
        """The key id is read from a token nothing has verified yet. Refreshing
        for every unknown one let any client make the server fetch the JWKS on
        every request."""
        _, public_key = _generate_rsa_key_pair()
        stranger, _ = _generate_rsa_key_pair()
        provider = _Provider(_make_jwks(public_key))
        verifier = _verifier(provider)

        for _ in range(20):
            token = _make_token(stranger, kid=uuid.uuid4().hex)
            with pytest.raises(OidcVerificationError, match="No JWKS key found"):
                _verify(verifier, provider, token)

        # The first fetch fills the cache; one more is the single refresh.
        assert provider.jwks_fetches == 2

    def test_a_rotated_key_is_picked_up_after_the_interval(self, clock: _Clock) -> None:
        old_private, old_public = _generate_rsa_key_pair()
        new_private, new_public = _generate_rsa_key_pair()
        provider = _Provider(_make_jwks(old_public, kid="old"))
        verifier = _verifier(provider)
        _verify(verifier, provider, _make_token(old_private, kid="old"))

        # The provider rotates. The first unknown id refreshes at once.
        provider.jwks = _make_jwks(new_public, kid="new")
        assert _verify(verifier, provider, _make_token(new_private, kid="new")).sub == "user-abc"

        # A second rotation within the interval is not fetched for...
        newer_private, newer_public = _generate_rsa_key_pair()
        provider.jwks = _make_jwks(newer_public, kid="newer")
        with pytest.raises(OidcVerificationError):
            _verify(verifier, provider, _make_token(newer_private, kid="newer"))

        # ...and is once the interval has passed.
        clock.now += oidc_token._FORCED_REFRESH_INTERVAL
        assert _verify(verifier, provider, _make_token(newer_private, kid="newer")).sub

    def test_requests_queued_behind_a_rotation_get_the_new_key(self, clock: _Clock) -> None:
        """At a rotation, concurrent requests bearing the new key id queue on the
        fetch lock behind the one that fetches. Within the interval they must be
        given the keys that fetch brought in, not refused."""
        old_private, old_public = _generate_rsa_key_pair()
        new_private, new_public = _generate_rsa_key_pair()
        provider = _Provider(_make_jwks(old_public, kid="old"))
        verifier = _verifier(provider)
        _verify(verifier, provider, _make_token(old_private, kid="old"))

        # The first request to see the new kid fetched it a moment ago.
        provider.jwks = _make_jwks(new_public, kid="new")
        with patch.object(oidc_token, "_get", side_effect=provider.get):
            verifier._refresh_for_unknown_kid()
        fetches = provider.jwks_fetches

        # A request that read the key list before that fetch now asks again.
        with patch.object(oidc_token, "_get", side_effect=provider.get):
            keys = verifier._refresh_for_unknown_kid()

        assert verifier._key_for(keys, "new") is not None
        assert provider.jwks_fetches == fetches

    def test_a_known_key_needs_no_fetch(self, clock: _Clock) -> None:
        private_key, public_key = _generate_rsa_key_pair()
        provider = _Provider(_make_jwks(public_key))
        verifier = _verifier(provider)
        for _ in range(5):
            _verify(verifier, provider, _make_token(private_key))
        assert provider.jwks_fetches == 1


def _finishes(call: Any, seconds: float = 2.0) -> tuple[bool, Any]:
    """Run *call* in a thread; whether it returned within *seconds*, and what."""
    result: list[Any] = []

    def run() -> None:
        try:
            result.append(call())
        except Exception as exc:  # noqa: BLE001 - reported to the caller
            result.append(exc)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(seconds)
    return (not worker.is_alive(), result[0] if result else None)


class TestFetchAttemptsAreBounded:
    """A provider that is down costs one attempt per interval, in every state."""

    def test_with_nothing_cached_a_failure_is_not_retried_within_the_interval(
        self, clock: _Clock
    ) -> None:
        private_key, public_key = _generate_rsa_key_pair()
        provider = _Provider(_make_jwks(public_key))
        provider.jwks_error = httpx.ConnectError("down")
        verifier = _verifier(provider)

        for _ in range(5):
            with pytest.raises(OidcVerificationError, match="Failed to fetch JWKS"):
                _verify(verifier, provider, _make_token(private_key))
        assert provider.jwks_fetches == 1

        clock.now += oidc_token._FORCED_REFRESH_INTERVAL
        provider.jwks_error = None
        assert _verify(verifier, provider, _make_token(private_key)).sub == "user-abc"
        assert provider.jwks_fetches == 2

    def test_past_the_stale_limit_a_failure_is_not_retried_within_the_interval(
        self, clock: _Clock
    ) -> None:
        private_key, public_key = _generate_rsa_key_pair()
        provider = _Provider(_make_jwks(public_key))
        verifier = _verifier(provider)
        _verify(verifier, provider, _make_token(private_key))
        verifier._jwks_cache["fetched_at"] = (
            datetime.now(UTC) - oidc_token._JWKS_MAX_STALE - timedelta(minutes=1)
        )
        provider.jwks_error = httpx.ConnectError("down")

        for _ in range(5):
            with pytest.raises(OidcVerificationError):
                _verify(verifier, provider, _make_token(private_key))
        assert provider.jwks_fetches == 2

    def test_a_request_with_usable_keys_does_not_wait_behind_a_refresh(self, clock: _Clock) -> None:
        private_key, public_key = _generate_rsa_key_pair()
        provider = _Provider(_make_jwks(public_key))
        verifier = _verifier(provider)
        _verify(verifier, provider, _make_token(private_key))
        verifier._jwks_cache["fetched_at"] = datetime.now(UTC) - timedelta(hours=2)

        # Another request is part-way through the refresh.
        with verifier._fetch_lock:
            done, keys = _finishes(verifier._get_keys)

        assert done and verifier._key_for(keys, "test-kid-1") is not None

    def test_an_unknown_key_id_within_the_interval_does_not_wait_on_the_lock(
        self, clock: _Clock
    ) -> None:
        """Waiting would let invented key ids hold every worker thread for the
        length of the one fetch allowed."""
        _, public_key = _generate_rsa_key_pair()
        provider = _Provider(_make_jwks(public_key))
        verifier = _verifier(provider)
        verifier._last_forced_refresh = clock.now

        with verifier._fetch_lock:
            done, _ = _finishes(verifier._refresh_for_unknown_kid)

        assert done
        assert provider.jwks_fetches == 0

    def test_an_unknown_key_id_does_not_wait_behind_a_fetch_in_flight(self, clock: _Clock) -> None:
        """With the interval elapsed but another fetch holding the lock (a
        scheduled refresh, say), the request is refused now rather than
        occupying a worker thread for the length of that fetch."""
        _, public_key = _generate_rsa_key_pair()
        provider = _Provider(_make_jwks(public_key))
        verifier = _verifier(provider)
        verifier._last_forced_refresh = clock.now - oidc_token._FORCED_REFRESH_INTERVAL

        with verifier._fetch_lock:
            done, _ = _finishes(verifier._refresh_for_unknown_kid)

        assert done
        assert provider.jwks_fetches == 0


class TestScheduledRefresh:
    def _stale(self, verifier: OidcTokenVerifier) -> None:
        verifier._jwks_cache["fetched_at"] = datetime.now(UTC) - timedelta(hours=2)

    def test_a_failed_refresh_keeps_the_cached_keys(self, clock: _Clock) -> None:
        """A provider that is down for a moment should not turn every request
        into a wait for it, nor sign everybody out."""
        private_key, public_key = _generate_rsa_key_pair()
        provider = _Provider(_make_jwks(public_key))
        verifier = _verifier(provider)
        _verify(verifier, provider, _make_token(private_key))

        self._stale(verifier)
        provider.jwks_error = httpx.ConnectError("down")
        for _ in range(3):
            assert _verify(verifier, provider, _make_token(private_key)).sub == "user-abc"
        # One attempt, then no retry until the interval has passed.
        assert provider.jwks_fetches == 2

        clock.now += oidc_token._FORCED_REFRESH_INTERVAL
        _verify(verifier, provider, _make_token(private_key))
        assert provider.jwks_fetches == 3

    @pytest.mark.parametrize(
        "reply",
        # The last two are key sets with no key in them: cached, they replaced
        # the working keys with none just the same.
        [{}, [], {"keys": "none"}, {"keys": None}, {"keys": []}, {"keys": [1]}],
    )
    def test_a_reply_that_is_not_a_key_set_keeps_the_cached_keys(
        self, reply: Any, clock: _Clock
    ) -> None:
        """A provider or proxy answering 200 with something other than a key
        set is a failed refresh. Cached as an empty set, it refused every token
        for up to an hour and logged nothing."""
        private_key, public_key = _generate_rsa_key_pair()
        provider = _Provider(_make_jwks(public_key))
        verifier = _verifier(provider)
        _verify(verifier, provider, _make_token(private_key))

        self._stale(verifier)
        provider.jwks = reply
        assert _verify(verifier, provider, _make_token(private_key)).sub == "user-abc"
        # Spaced like any other failure: no second attempt within the interval.
        _verify(verifier, provider, _make_token(private_key))
        assert provider.jwks_fetches == 2

    def test_with_nothing_usable_a_request_does_not_queue_behind_a_slow_fetch(
        self, clock: _Clock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Another request's fetch holds the lock; one with nothing usable waits
        a short, bounded time and is refused, not parked on a worker thread for
        the HTTP timeout."""
        import time

        private_key, public_key = _generate_rsa_key_pair()
        provider = _Provider(_make_jwks(public_key))
        verifier = _verifier(provider)
        monkeypatch.setattr(oidc_token, "_FETCH_WAIT_SECONDS", 0.2)

        assert verifier._fetch_lock.acquire()  # the fetch in flight
        try:
            started = time.monotonic()
            with pytest.raises(OidcVerificationError, match="still being fetched"):
                _verify(verifier, provider, _make_token(private_key))
            assert time.monotonic() - started < 5
        finally:
            verifier._fetch_lock.release()
        assert provider.jwks_fetches == 0  # it never fetched for itself

    def test_with_nothing_cached_a_reply_that_is_not_a_key_set_is_refused(
        self, clock: _Clock
    ) -> None:
        private_key, _ = _generate_rsa_key_pair()
        provider = _Provider({})
        verifier = _verifier(provider)
        with pytest.raises(OidcVerificationError, match="'keys' list"):
            _verify(verifier, provider, _make_token(private_key))

    def test_keys_past_the_stale_limit_are_not_used(self, clock: _Clock) -> None:
        """A JWKS that has moved for good must not leave the last keys seen,
        revoked or not, accepted indefinitely."""
        private_key, public_key = _generate_rsa_key_pair()
        provider = _Provider(_make_jwks(public_key))
        verifier = _verifier(provider)
        _verify(verifier, provider, _make_token(private_key))

        verifier._jwks_cache["fetched_at"] = (
            datetime.now(UTC) - oidc_token._JWKS_MAX_STALE - timedelta(minutes=1)
        )
        provider.jwks_error = httpx.ConnectError("moved")
        with pytest.raises(OidcVerificationError, match="Failed to fetch JWKS"):
            _verify(verifier, provider, _make_token(private_key))

    def test_with_nothing_cached_a_failed_fetch_is_refused(self, clock: _Clock) -> None:
        private_key, public_key = _generate_rsa_key_pair()
        provider = _Provider(_make_jwks(public_key))
        provider.jwks_error = httpx.ConnectError("down")
        verifier = _verifier(provider)
        with pytest.raises(OidcVerificationError, match="Failed to fetch JWKS"):
            _verify(verifier, provider, _make_token(private_key))


# ---------------------------------------------------------------------------
# Where keys come from
# ---------------------------------------------------------------------------


class TestProviderEndpoints:
    def test_a_plain_http_discovery_url_is_refused(self) -> None:
        provider = _Provider({"keys": []})
        with pytest.raises(OidcVerificationError, match="https"):
            _verifier(provider, "http://accounts.example.com/.well-known/openid-configuration")
        # Refused before anything is fetched.
        assert provider.jwks_fetches == 0

    @pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "[::1]"])
    def test_plain_http_to_this_machine_is_allowed(self, host: str) -> None:
        issuer = f"http://{host}:8080/realms/dev"
        provider = _Provider(
            {"keys": []}, {"issuer": issuer, "jwks_uri": f"{issuer}/protocol/openid-connect/certs"}
        )
        assert _verifier(provider, f"{issuer}/.well-known/openid-configuration").issuer == issuer

    def test_a_plain_http_jwks_uri_is_refused(self) -> None:
        provider = _Provider(
            {"keys": []}, {"issuer": ISSUER, "jwks_uri": "http://accounts.example.com/jwks"}
        )
        with pytest.raises(OidcVerificationError, match="jwks_uri must use https"):
            _verifier(provider)

    def test_an_issuer_that_does_not_match_the_url_is_refused(self) -> None:
        """OpenID Connect Discovery 4.3: the issuer is the one the discovery URL
        was formed from. Otherwise the document could name any issuer."""
        provider = _Provider(
            {"keys": []}, {"issuer": "https://other.example.com", "jwks_uri": JWKS_URI}
        )
        with pytest.raises(OidcVerificationError, match="does not match"):
            _verifier(provider)

    def test_an_issuer_with_a_trailing_slash_matches(self) -> None:
        provider = _Provider({"keys": []}, {"issuer": ISSUER + "/", "jwks_uri": JWKS_URI})
        assert _verifier(provider).issuer == ISSUER + "/"

    def test_a_document_that_is_not_an_object_is_refused(self) -> None:
        provider = _Provider({"keys": []})
        provider.discovery = ["not", "an", "object"]  # type: ignore[assignment]
        with pytest.raises(OidcVerificationError, match="not a JSON object"):
            _verifier(provider)

    def test_a_key_published_for_encryption_is_not_used_to_verify(self, clock: _Clock) -> None:
        private_key, public_key = _generate_rsa_key_pair()
        jwks = _make_jwks(public_key)
        jwks["keys"][0]["use"] = "enc"
        provider = _Provider(jwks)
        verifier = _verifier(provider)
        with pytest.raises(OidcVerificationError, match="No JWKS key found"):
            _verify(verifier, provider, _make_token(private_key))

    def test_a_key_that_states_no_use_is_still_a_signing_key(self, clock: _Clock) -> None:
        private_key, public_key = _generate_rsa_key_pair()
        jwks = _make_jwks(public_key)
        del jwks["keys"][0]["use"]
        provider = _Provider(jwks)
        assert _verify(_verifier(provider), provider, _make_token(private_key)).sub


def _served(response: httpx.Response) -> Any:
    """``httpx.stream`` answered by *response* through httpx's own transport,
    so headers, decompression and streaming behave as they do on the wire."""
    client = httpx.Client(transport=httpx.MockTransport(lambda request: response))

    def stream(method: str, url: str, **kwargs: Any) -> Any:
        kwargs.pop("timeout", None)
        return client.stream(method, url, **kwargs)

    return patch.object(oidc_token.httpx, "stream", side_effect=stream)


class TestFetchingDocuments:
    def test_a_document_over_the_limit_is_refused(self) -> None:
        big = b" " * (oidc_token._MAX_DOCUMENT_BYTES + 1)
        with (
            _served(httpx.Response(200, content=big)),
            pytest.raises(OidcVerificationError, match="larger than"),
        ):
            oidc_token._get(JWKS_URI, timeout=1, what="The JWKS")

    def test_a_document_within_the_limit_is_returned_whole(self) -> None:
        body = json.dumps({"keys": []}).encode()
        with _served(httpx.Response(200, content=body)):
            resp = oidc_token._get(JWKS_URI, timeout=1, what="The JWKS")
        assert resp.json() == {"keys": []}

    def test_a_compressed_document_is_read(self) -> None:
        """Providers compress their replies (Google does), and httpx asks them
        to. Rebuilt with `content-encoding` still set, the already-decoded body
        was decompressed a second time and nothing verified."""
        body = json.dumps({"keys": [{"kid": "k1"}]}).encode()
        compressed = httpx.Response(
            200,
            headers={"content-encoding": "gzip", "content-type": "application/json"},
            content=gzip.compress(body),
        )
        with _served(compressed):
            resp = oidc_token._get(JWKS_URI, timeout=1, what="The JWKS")
        assert resp.json() == {"keys": [{"kid": "k1"}]}

    def test_the_size_limit_applies_to_the_decompressed_body(self) -> None:
        """A small compressed reply can expand to far more than it sent."""
        big = b" " * (oidc_token._MAX_DOCUMENT_BYTES + 1)
        compressed = httpx.Response(
            200, headers={"content-encoding": "gzip"}, content=gzip.compress(big)
        )
        with _served(compressed), pytest.raises(OidcVerificationError, match="larger than"):
            oidc_token._get(JWKS_URI, timeout=1, what="The JWKS")


# ---------------------------------------------------------------------------
# Claims a token must carry
# ---------------------------------------------------------------------------


def _sign(private_key: Any, claims: dict[str, Any]) -> str:
    import jwt

    return jwt.encode(claims, private_key, algorithm="RS256", headers={"kid": "test-kid-1"})


def _claims(**overrides: Any) -> dict[str, Any]:
    now = int(datetime.now(UTC).timestamp())
    base: dict[str, Any] = {
        "sub": "user-abc",
        "aud": CLIENT_ID,
        "iss": ISSUER,
        "iat": now,
        "exp": now + 3600,
    }
    base.update(overrides)
    return {k: v for k, v in base.items() if v is not None}


class TestRequiredClaims:
    @pytest.mark.parametrize("missing", ["exp", "iss", "aud"])
    def test_a_token_without_the_claim_is_refused(self, missing: str, clock: _Clock) -> None:
        """PyJWT checks these only when present; a token without `exp` never expired."""
        private_key, public_key = _generate_rsa_key_pair()
        provider = _Provider(_make_jwks(public_key))
        token = _sign(private_key, _claims(**{missing: None}))
        with pytest.raises(OidcVerificationError):
            _verify(_verifier(provider), provider, token)

    def test_a_multi_audience_token_issued_to_another_party_is_refused(self, clock: _Clock) -> None:
        private_key, public_key = _generate_rsa_key_pair()
        provider = _Provider(_make_jwks(public_key))
        token = _sign(private_key, _claims(aud=[CLIENT_ID, "other-app"], azp="other-app"))
        with pytest.raises(OidcVerificationError, match="azp"):
            _verify(_verifier(provider), provider, token)

    def test_a_multi_audience_token_issued_to_this_client_is_accepted(self, clock: _Clock) -> None:
        private_key, public_key = _generate_rsa_key_pair()
        provider = _Provider(_make_jwks(public_key))
        token = _sign(private_key, _claims(aud=[CLIENT_ID, "other-app"], azp=CLIENT_ID))
        assert _verify(_verifier(provider), provider, token).sub == "user-abc"

    def test_a_single_audience_token_issued_to_another_client_is_refused(
        self, clock: _Clock
    ) -> None:
        """A provider can issue a token audienced to this client at another
        client's request (Google's cross-client identity does). `azp` says
        whose it is, and it is not this client's."""
        private_key, public_key = _generate_rsa_key_pair()
        provider = _Provider(_make_jwks(public_key))
        token = _sign(private_key, _claims(aud=CLIENT_ID, azp="other-app"))
        with pytest.raises(OidcVerificationError, match="azp"):
            _verify(_verifier(provider), provider, token)

    def test_a_multi_audience_token_without_azp_is_refused(self, clock: _Clock) -> None:
        private_key, public_key = _generate_rsa_key_pair()
        provider = _Provider(_make_jwks(public_key))
        token = _sign(private_key, _claims(aud=[CLIENT_ID, "other-app"]))
        with pytest.raises(OidcVerificationError, match="azp"):
            _verify(_verifier(provider), provider, token)

    def test_a_single_audience_token_naming_this_client_as_azp_is_accepted(
        self, clock: _Clock
    ) -> None:
        private_key, public_key = _generate_rsa_key_pair()
        provider = _Provider(_make_jwks(public_key))
        token = _sign(private_key, _claims(azp=CLIENT_ID))
        assert _verify(_verifier(provider), provider, token).sub == "user-abc"

    def test_a_single_audience_list_needs_no_azp(self, clock: _Clock) -> None:
        private_key, public_key = _generate_rsa_key_pair()
        provider = _Provider(_make_jwks(public_key))
        token = _sign(private_key, _claims(aud=[CLIENT_ID]))
        assert _verify(_verifier(provider), provider, token).sub == "user-abc"


class TestMalformedHeaders:
    def test_a_key_id_that_is_not_a_string_is_refused_cleanly(self, clock: _Clock) -> None:
        """An exception other than OidcVerificationError reached the transport
        as a 500, with a traceback logged per request."""
        private_key, public_key = _generate_rsa_key_pair()
        provider = _Provider(_make_jwks(public_key))
        token = _forge_header(_make_token(private_key), kid=123)
        with pytest.raises(OidcVerificationError, match="Invalid token header"):
            _verify(_verifier(provider), provider, token)
