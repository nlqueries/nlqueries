# nlqueries-core — OSS (BSL 1.1)
"""
nlqueries.connectors.postgres_tls
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Resolution and description of the TLS mode used for a PostgreSQL connection.

Measured against PostgreSQL 16 and libpq, using four servers presenting a
correct certificate, a certificate for a different hostname, one signed by an
untrusted CA, and an expired one:

===============  ==========  ==========  ===============
sslmode          rogue CA    expired     wrong hostname
===============  ==========  ==========  ===============
require          accepted    accepted    accepted
require + CA     refused     refused     accepted
verify-ca + CA   refused     refused     accepted
verify-full + CA refused     refused     refused
===============  ==========  ==========  ===============

Two results determine what this module does.

``require`` without a root certificate performs no verification: it establishes
an encrypted channel to whichever server answers, which does not exclude an
attacker positioned between the client and the database.

``sslrootcert`` is not ignored under ``require``. libpq documents that
``sslmode=require`` with a valid root certificate behaves as ``verify-ca``, and
the measurements above confirm it. The remaining difference between that and
``verify-full`` is hostname verification.

A caller that supplies a root certificate has therefore configured everything
``verify-full`` requires. This module selects ``verify-full`` in that case unless
an explicit mode is given, and describes the resulting posture for
``nlqueries health``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Modes that establish an encrypted channel without authenticating the server.
UNVERIFIED_MODES = frozenset({"require"})

#: Modes that permit an unencrypted connection.
UNENCRYPTED_MODES = frozenset({"disable", "allow", "prefer"})

#: The mode applied when no root certificate is configured and none is given.
#: Retained from the change that removed ``prefer``: it refuses a server with no
#: TLS, which ``prefer`` accepted silently.
DEFAULT_WITHOUT_CA = "require"

#: The mode applied when a root certificate is configured and no mode is given.
DEFAULT_WITH_CA = "verify-full"


def resolve_ssl_mode(credentials: dict[str, Any]) -> str:
    """Return the ``sslmode`` to use for *credentials*.

    An explicit ``ssl_mode`` is always honoured. Otherwise the presence of
    ``ssl_ca_cert`` selects the default: a caller that has configured a root
    certificate has supplied everything ``verify-full`` requires, and the only
    additional check it performs is that the certificate names the host being
    connected to.
    """
    explicit = credentials.get("ssl_mode")
    if explicit:
        return str(explicit)
    if credentials.get("ssl_ca_cert"):
        return DEFAULT_WITH_CA
    return DEFAULT_WITHOUT_CA


@dataclass(frozen=True)
class TlsPosture:
    """The effective TLS configuration of a connection, and its consequences."""

    ssl_mode: str
    has_root_certificate: bool

    @property
    def is_encrypted(self) -> bool:
        return self.ssl_mode not in UNENCRYPTED_MODES

    @property
    def verifies_certificate_chain(self) -> bool:
        """True when the server's certificate is checked against a trusted CA.

        ``require`` qualifies only when a root certificate is configured, in
        which case libpq applies the same checks as ``verify-ca``.
        """
        if self.ssl_mode in ("verify-ca", "verify-full"):
            return True
        return self.ssl_mode == "require" and self.has_root_certificate

    @property
    def verifies_hostname(self) -> bool:
        return self.ssl_mode == "verify-full"

    @property
    def concerns(self) -> tuple[str, ...]:
        """One entry per property of the connection that is not verified."""
        if not self.is_encrypted:
            return (f"ssl_mode is '{self.ssl_mode}', which permits an unencrypted connection",)

        found: list[str] = []
        if not self.verifies_certificate_chain:
            found.append(
                f"ssl_mode is '{self.ssl_mode}' with no ssl_ca_cert, so the server's "
                f"certificate is not verified and the connection is encrypted to "
                f"whichever server answers"
            )
        elif not self.verifies_hostname:
            found.append(
                f"ssl_mode is '{self.ssl_mode}', which verifies the certificate chain "
                f"but not that the certificate names this host"
            )
        return tuple(found)

    @property
    def is_fully_verified(self) -> bool:
        return not self.concerns

    def summary(self) -> str:
        """A single line for logs and the health report."""
        if not self.concerns:
            return f"ssl_mode '{self.ssl_mode}': chain and hostname verified"
        return "; ".join(self.concerns)


def describe(credentials: dict[str, Any]) -> TlsPosture:
    """Describe the TLS posture *credentials* will produce."""
    return TlsPosture(
        ssl_mode=resolve_ssl_mode(credentials),
        has_root_certificate=bool(credentials.get("ssl_ca_cert")),
    )
