"""Tests for nlqueries.auth.oidc_token."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm
from nlqueries.auth.oidc_token import OidcClaims, OidcTokenVerifier, OidcVerificationError

# ---------------------------------------------------------------------------
# Test fixtures / helpers
# ---------------------------------------------------------------------------

DISCOVERY_URL = "https://accounts.example.com/.well-known/openid-configuration"
JWKS_URI = "https://accounts.example.com/.well-known/jwks.json"
ISSUER = "https://accounts.example.com"
CLIENT_ID = "my-client-id"

DISCOVERY_DOC: dict[str, Any] = {
    "issuer": ISSUER,
    "jwks_uri": JWKS_URI,
}


def _generate_rsa_key_pair() -> tuple[Any, Any]:
    """Return (private_key, public_key) for test RS256 signing."""
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    return private_key, private_key.public_key()


def _private_key_pem(private_key: Any) -> bytes:
    result: bytes = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return result


def _make_jwks(public_key: Any, kid: str = "test-kid-1") -> dict[str, Any]:
    """Build a JWKS dict from an RSA public key."""
    jwk: dict[str, Any] = json.loads(RSAAlgorithm.to_jwk(public_key))
    jwk["kid"] = kid
    jwk["use"] = "sig"
    jwk["alg"] = "RS256"
    return {"keys": [jwk]}


def _make_token(
    private_key: Any,
    kid: str = "test-kid-1",
    sub: str = "user-abc",
    email: str = "alice@example.com",
    client_id: str = CLIENT_ID,
    issuer: str = ISSUER,
    exp_delta_seconds: int = 3600,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    """Sign and return a minimal OIDC ID token JWT."""
    now = datetime.now(UTC)
    payload: dict[str, Any] = {
        "sub": sub,
        "email": email,
        "name": "Alice Example",
        "given_name": "Alice",
        "family_name": "Example",
        "picture": "https://example.com/alice.jpg",
        "email_verified": True,
        "aud": client_id,
        "iss": issuer,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(seconds=exp_delta_seconds)).timestamp()),
    }
    if extra_claims:
        payload.update(extra_claims)
    return jwt.encode(
        payload,
        _private_key_pem(private_key),
        algorithm="RS256",
        headers={"kid": kid},
    )


def _mock_http_response(data: dict[str, Any], status_code: int = 200) -> MagicMock:
    m = MagicMock()
    m.status_code = status_code
    m.json.return_value = data
    m.raise_for_status = MagicMock()
    return m


def _make_url_router(
    discovery_response: dict[str, Any],
    jwks_responses: list[dict[str, Any]],
) -> Any:
    """Return a side_effect function that routes httpx.get calls.

    The first call to any URL containing 'openid-configuration' returns the
    discovery document.  Subsequent calls (to the JWKS URI) return
    ``jwks_responses`` in order.
    """
    jwks_iter = iter(jwks_responses)

    def _get(url: str, **kwargs: Any) -> MagicMock:
        if "openid-configuration" in url:
            return _mock_http_response(discovery_response)
        return _mock_http_response(next(jwks_iter))

    return _get


# ---------------------------------------------------------------------------
# Test: verify_valid_token_returns_claims
# ---------------------------------------------------------------------------


class TestVerifyValidToken:
    def test_verify_valid_token_returns_claims(self) -> None:
        """Generate RS256 JWT; mock JWKS response; assert email and sub extracted."""
        private_key, public_key = _generate_rsa_key_pair()
        token = _make_token(private_key)
        jwks = _make_jwks(public_key)

        side_effect = _make_url_router(DISCOVERY_DOC, [jwks])

        with patch("nlqueries.auth.oidc_token.httpx.get", side_effect=side_effect):
            verifier = OidcTokenVerifier(DISCOVERY_URL)
            claims = verifier.verify(token, CLIENT_ID)

        assert claims.email == "alice@example.com"
        assert claims.sub == "user-abc"
        assert claims.name == "Alice Example"
        assert claims.given_name == "Alice"
        assert claims.family_name == "Example"
        assert claims.picture == "https://example.com/alice.jpg"
        assert claims.email_verified is True
        assert isinstance(claims, OidcClaims)
        assert claims.raw["sub"] == "user-abc"

    def test_verify_returns_oidc_claims_instance(self) -> None:
        """verify() always returns an OidcClaims dataclass."""
        private_key, public_key = _generate_rsa_key_pair()
        token = _make_token(private_key)
        jwks = _make_jwks(public_key)

        side_effect = _make_url_router(DISCOVERY_DOC, [jwks])

        with patch("nlqueries.auth.oidc_token.httpx.get", side_effect=side_effect):
            verifier = OidcTokenVerifier(DISCOVERY_URL)
            claims = verifier.verify(token, CLIENT_ID)

        assert isinstance(claims, OidcClaims)

    def test_optional_claims_can_be_none(self) -> None:
        """Tokens without name/picture still parse; optional fields are None."""
        private_key, public_key = _generate_rsa_key_pair()
        # Don't include optional claims
        now = datetime.now(UTC)
        payload: dict[str, Any] = {
            "sub": "user-min",
            "email": "min@example.com",
            "email_verified": False,
            "aud": CLIENT_ID,
            "iss": ISSUER,
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(seconds=3600)).timestamp()),
        }
        token = jwt.encode(
            payload,
            _private_key_pem(private_key),
            algorithm="RS256",
            headers={"kid": "test-kid-1"},
        )
        jwks = _make_jwks(public_key)
        side_effect = _make_url_router(DISCOVERY_DOC, [jwks])

        with patch("nlqueries.auth.oidc_token.httpx.get", side_effect=side_effect):
            verifier = OidcTokenVerifier(DISCOVERY_URL)
            claims = verifier.verify(token, CLIENT_ID)

        assert claims.sub == "user-min"
        assert claims.name is None
        assert claims.picture is None
        assert claims.email_verified is False


# ---------------------------------------------------------------------------
# Test: verify_expired_token_raises_error
# ---------------------------------------------------------------------------


class TestVerifyExpiredToken:
    def test_verify_expired_token_raises_error(self) -> None:
        """Expired token raises OidcVerificationError."""
        private_key, public_key = _generate_rsa_key_pair()
        # exp in the past
        token = _make_token(private_key, exp_delta_seconds=-1)
        jwks = _make_jwks(public_key)

        side_effect = _make_url_router(DISCOVERY_DOC, [jwks])

        with patch("nlqueries.auth.oidc_token.httpx.get", side_effect=side_effect):
            verifier = OidcTokenVerifier(DISCOVERY_URL)
            with pytest.raises(OidcVerificationError, match="expired"):
                verifier.verify(token, CLIENT_ID)


# ---------------------------------------------------------------------------
# Test: verify_wrong_audience_raises_error
# ---------------------------------------------------------------------------


class TestVerifyWrongAudience:
    def test_verify_wrong_audience_raises_error(self) -> None:
        """Token with wrong audience raises OidcVerificationError."""
        private_key, public_key = _generate_rsa_key_pair()
        token = _make_token(private_key, client_id="wrong-client")
        jwks = _make_jwks(public_key)

        side_effect = _make_url_router(DISCOVERY_DOC, [jwks])

        with patch("nlqueries.auth.oidc_token.httpx.get", side_effect=side_effect):
            verifier = OidcTokenVerifier(DISCOVERY_URL)
            with pytest.raises(OidcVerificationError, match="audience"):
                verifier.verify(token, CLIENT_ID)


# ---------------------------------------------------------------------------
# Test: JWKS cache refreshed when kid not found
# ---------------------------------------------------------------------------


class TestJwksCacheRefresh:
    def test_jwks_cache_refreshed_when_kid_not_found(self) -> None:
        """First JWKS response lacks the correct kid; refresh returns it.

        Assert JWKS endpoint is fetched exactly twice.
        """
        private_key, public_key = _generate_rsa_key_pair()
        token = _make_token(private_key, kid="correct-kid")
        correct_jwks = _make_jwks(public_key, kid="correct-kid")

        # First JWKS response: wrong kid
        wrong_jwks: dict[str, Any] = {
            "keys": [
                {
                    "kid": "other-kid",
                    "kty": "RSA",
                    "alg": "RS256",
                    "use": "sig",
                    "n": "sGmSHodp",
                    "e": "AQAB",
                }
            ]
        }

        jwks_call_count = 0

        def _get(url: str, **kwargs: Any) -> MagicMock:
            nonlocal jwks_call_count
            if "openid-configuration" in url:
                return _mock_http_response(DISCOVERY_DOC)
            jwks_call_count += 1
            if jwks_call_count == 1:
                return _mock_http_response(wrong_jwks)
            return _mock_http_response(correct_jwks)

        with patch("nlqueries.auth.oidc_token.httpx.get", side_effect=_get):
            verifier = OidcTokenVerifier(DISCOVERY_URL)
            claims = verifier.verify(token, CLIENT_ID)

        assert jwks_call_count == 2, (
            f"Expected JWKS to be fetched exactly twice; got {jwks_call_count}"
        )
        assert claims.sub == "user-abc"
        assert claims.email == "alice@example.com"

    def test_jwks_cache_used_on_second_verify(self) -> None:
        """Second verify() call reuses the JWKS cache (no extra HTTP call)."""
        private_key, public_key = _generate_rsa_key_pair()
        token = _make_token(private_key)
        jwks = _make_jwks(public_key)

        call_log: list[str] = []

        def _get(url: str, **kwargs: Any) -> MagicMock:
            call_log.append(url)
            if "openid-configuration" in url:
                return _mock_http_response(DISCOVERY_DOC)
            return _mock_http_response(jwks)

        with patch("nlqueries.auth.oidc_token.httpx.get", side_effect=_get):
            verifier = OidcTokenVerifier(DISCOVERY_URL)
            verifier.verify(token, CLIENT_ID)
            verifier.verify(token, CLIENT_ID)  # second call — cache should be used

        # 1 discovery + 1 JWKS (not 2 JWKS)
        jwks_calls = [u for u in call_log if "openid-configuration" not in u]
        assert len(jwks_calls) == 1

    def test_kid_not_found_after_refresh_raises_error(self) -> None:
        """If the kid is still absent after a JWKS refresh, raise OidcVerificationError."""
        private_key, _ = _generate_rsa_key_pair()
        token = _make_token(private_key, kid="missing-kid")

        # Both JWKS responses use wrong kid
        wrong_jwks: dict[str, Any] = {
            "keys": [{"kid": "other-kid", "kty": "RSA", "n": "x", "e": "AQAB"}]
        }

        def _get(url: str, **kwargs: Any) -> MagicMock:
            if "openid-configuration" in url:
                return _mock_http_response(DISCOVERY_DOC)
            return _mock_http_response(wrong_jwks)

        with patch("nlqueries.auth.oidc_token.httpx.get", side_effect=_get):
            verifier = OidcTokenVerifier(DISCOVERY_URL)
            with pytest.raises(OidcVerificationError, match="No JWKS key found"):
                verifier.verify(token, CLIENT_ID)


# ---------------------------------------------------------------------------
# Test: discovery document errors
# ---------------------------------------------------------------------------


class TestDiscoveryErrors:
    def test_missing_jwks_uri_raises_error(self) -> None:
        """Discovery doc without 'jwks_uri' raises OidcVerificationError at init."""
        bad_doc = {"issuer": ISSUER}  # no jwks_uri

        with (
            patch(
                "nlqueries.auth.oidc_token.httpx.get",
                return_value=_mock_http_response(bad_doc),
            ),
            pytest.raises(OidcVerificationError, match="jwks_uri"),
        ):
            OidcTokenVerifier(DISCOVERY_URL)

    def test_discovery_http_error_raises_oidc_error(self) -> None:
        """HTTP failure on discovery fetch raises OidcVerificationError."""
        import httpx as _httpx

        with (
            patch(
                "nlqueries.auth.oidc_token.httpx.get",
                side_effect=_httpx.ConnectError("connection refused"),
            ),
            pytest.raises(OidcVerificationError, match="discovery"),
        ):
            OidcTokenVerifier(DISCOVERY_URL)
