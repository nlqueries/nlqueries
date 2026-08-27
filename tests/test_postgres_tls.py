"""
Resolution and description of the PostgreSQL TLS mode.

The behaviour these encode was measured against PostgreSQL 16 and libpq, using
servers presenting a correct certificate, one for a different hostname, one
signed by an untrusted CA, and an expired one. The measured matrix is recorded
in ``nlqueries/connectors/postgres_tls``.
"""

from __future__ import annotations

from nlqueries.connectors.postgres_tls import TlsPosture, describe, resolve_ssl_mode

_CA = "/etc/ssl/db-ca.crt"


class TestResolveSslMode:
    def test_no_certificate_and_no_mode_requires_encryption(self) -> None:
        """Without a root certificate, verify-full cannot be satisfied, so the
        mode that refuses an unencrypted server is used."""
        assert resolve_ssl_mode({}) == "require"

    def test_a_root_certificate_selects_verify_full(self) -> None:
        """A caller that configures a root certificate has supplied everything
        verify-full needs; the only additional check is the hostname."""
        assert resolve_ssl_mode({"ssl_ca_cert": _CA}) == "verify-full"

    def test_an_explicit_mode_is_honoured(self) -> None:
        assert resolve_ssl_mode({"ssl_mode": "verify-ca", "ssl_ca_cert": _CA}) == "verify-ca"
        assert resolve_ssl_mode({"ssl_mode": "disable"}) == "disable"

    def test_an_explicit_mode_overrides_the_certificate_default(self) -> None:
        assert resolve_ssl_mode({"ssl_mode": "require", "ssl_ca_cert": _CA}) == "require"

    def test_an_empty_mode_is_treated_as_absent(self) -> None:
        assert resolve_ssl_mode({"ssl_mode": ""}) == "require"
        assert resolve_ssl_mode({"ssl_mode": None}) == "require"


class TestPosture:
    def test_verify_full_verifies_both(self) -> None:
        posture = TlsPosture(ssl_mode="verify-full", has_root_certificate=True)

        assert posture.verifies_certificate_chain
        assert posture.verifies_hostname
        assert posture.is_fully_verified
        assert posture.concerns == ()

    def test_require_without_a_certificate_verifies_nothing(self) -> None:
        """Measured: this accepts a certificate signed by an untrusted CA and an
        expired certificate. The channel is encrypted to whichever server
        answers."""
        posture = TlsPosture(ssl_mode="require", has_root_certificate=False)

        assert posture.is_encrypted
        assert not posture.verifies_certificate_chain
        assert not posture.is_fully_verified
        assert "not verified" in posture.summary()

    def test_require_with_a_certificate_verifies_the_chain(self) -> None:
        """libpq documents that sslmode=require with a valid root certificate
        behaves as verify-ca, and the measurement confirms it: the untrusted CA
        and the expired certificate are both refused."""
        posture = TlsPosture(ssl_mode="require", has_root_certificate=True)

        assert posture.verifies_certificate_chain
        assert not posture.verifies_hostname

    def test_verify_ca_does_not_check_the_hostname(self) -> None:
        """Measured: a certificate naming a different host is accepted."""
        posture = TlsPosture(ssl_mode="verify-ca", has_root_certificate=True)

        assert posture.verifies_certificate_chain
        assert not posture.verifies_hostname
        assert any("names this host" in c for c in posture.concerns)

    def test_the_unencrypted_modes_report_that_first(self) -> None:
        """When the connection may be unencrypted, that is the only finding
        worth reporting: certificate verification does not apply."""
        for mode in ("disable", "allow", "prefer"):
            posture = TlsPosture(ssl_mode=mode, has_root_certificate=True)

            assert not posture.is_encrypted
            assert len(posture.concerns) == 1
            assert "unencrypted" in posture.concerns[0]

    def test_describe_reads_the_credentials(self) -> None:
        assert describe({"ssl_ca_cert": _CA}).ssl_mode == "verify-full"
        assert describe({}).ssl_mode == "require"
        assert not describe({}).has_root_certificate
        assert describe({"ssl_ca_cert": _CA}).has_root_certificate
