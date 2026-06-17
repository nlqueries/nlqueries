"""
Unit tests for PostgresConnector SSL connect_args assembly.

These tests mock ``create_engine`` and require no live database or Docker.
They verify that ``PostgresConnector.connect()`` correctly forwards SSL
credential keys to psycopg2's ``connect_args``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from nlqueries.connectors.postgres import PostgresConnector

_BASE_CREDS: dict[str, Any] = {
    "host": "db.example.com",
    "port": 5432,
    "database": "mydb",
    "user": "alice",
    "password": "s3cr3t",
}


def _connect_and_capture(extra_creds: dict[str, Any]) -> dict[str, Any]:
    """Connect with merged credentials and return the captured connect_args."""
    captured: dict[str, Any] = {}

    def fake_create_engine(url: object, **kwargs: Any) -> MagicMock:
        captured.update(kwargs)
        return MagicMock()

    with patch("nlqueries.connectors.postgres.create_engine", side_effect=fake_create_engine):
        connector = PostgresConnector()
        connector.connect({**_BASE_CREDS, **extra_creds})

    result: dict[str, Any] = captured.get("connect_args", {})
    return result


def test_connect_sets_sslmode_prefer_by_default() -> None:
    args = _connect_and_capture({})
    assert args["sslmode"] == "prefer"


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
