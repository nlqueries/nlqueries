"""
Unit tests for PostgresConnector SSL connect_args assembly.

These tests mock ``create_engine`` and require no live database or Docker.
They verify that ``PostgresConnector.connect()`` correctly forwards SSL
credential keys to psycopg2's ``connect_args``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import sqlalchemy as sa
from nlqueries.connectors.postgres import PostgresConnector

from tests.conftest import granted

_BASE_CREDS: dict[str, Any] = {
    "host": "db.example.com",
    "port": 5432,
    "database": "mydb",
    "user": "alice",
    "password": "test-password",
}


def _connect_and_capture(extra_creds: dict[str, Any]) -> dict[str, Any]:
    """Connect with merged credentials and return the captured connect_args."""
    captured: dict[str, Any] = {}

    def fake_create_engine(url: object, **kwargs: Any) -> sa.Engine:
        captured.update(kwargs)
        # A real Engine rather than a MagicMock: connect() attaches the
        # identity-check listener to it, and SQLAlchemy's event system cannot
        # bind to a mock. Nothing here opens a connection, so the URL only has
        # to be one SQLAlchemy will parse.
        return sa.create_engine("sqlite://")

    with patch("nlqueries.connectors.postgres.create_engine", side_effect=fake_create_engine):
        connector = granted(PostgresConnector())
        connector.connect({**_BASE_CREDS, **extra_creds})

    result: dict[str, Any] = captured.get("connect_args", {})
    return result


def test_connect_requires_tls_by_default() -> None:
    """Reverses an earlier default, so the reasoning is recorded.

    psycopg2's `prefer` tries TLS and takes a plaintext session when the server
    offers none — silently, with nothing to show that the credentials and every
    row of every result crossed the network in the clear. `require` removes the
    downgrade; a server with no TLS now fails to connect rather than quietly
    succeeding in the open.
    """
    args = _connect_and_capture({})
    assert args["sslmode"] == "require"


def test_plaintext_is_still_possible_but_has_to_be_asked_for() -> None:
    """The previous behaviour remains available but must be stated.

    `ssl_mode: disable` records in the connector's configuration what the old
    default did implicitly.
    """
    args = _connect_and_capture({"ssl_mode": "disable"})
    assert args["sslmode"] == "disable"


def test_verify_full_is_propagated() -> None:
    """`require` encrypts but verifies nothing — it stops eavesdropping, not an
    active man-in-the-middle. This is the setting production wants."""
    args = _connect_and_capture({"ssl_mode": "verify-full", "ssl_ca_cert": "/tmp/ca.pem"})
    assert args["sslmode"] == "verify-full"
    assert args["sslrootcert"] == "/tmp/ca.pem"


def test_connect_propagates_ssl_mode() -> None:
    args = _connect_and_capture({"ssl_mode": "require"})
    assert args["sslmode"] == "require"


def test_connect_sets_sslrootcert_when_provided() -> None:
    args = _connect_and_capture({"ssl_ca_cert": "/tmp/ca.pem"})
    assert args["sslrootcert"] == "/tmp/ca.pem"


def test_connect_omits_sslrootcert_when_not_provided() -> None:
    args = _connect_and_capture({})
    assert "sslrootcert" not in args


def test_connect_sets_client_cert_and_key() -> None:
    args = _connect_and_capture(
        {"ssl_client_cert": "/tmp/client.crt", "ssl_client_key": "/tmp/client.key"}
    )
    assert args["sslcert"] == "/tmp/client.crt"
    assert args["sslkey"] == "/tmp/client.key"
