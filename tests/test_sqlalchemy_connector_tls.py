"""The generic connector must not accept a TLS setting and then ignore it.

`SQLAlchemyConnector` takes a whole SQLAlchemy URL, so it has none of the
per-field TLS handling `PostgresConnector` grew. Until the loader started
passing the URL through it could not be opened on the agent or CLI paths at
all, so this never bit; now that it can be opened, an operator who sets
`ssl_mode` or `ssl_ca_cert` on a `db_type: sqlalchemy` connector would have had
those values stored, delivered, and silently discarded -- connecting under
libpq's `prefer` default, which falls back to plaintext without saying so.

That is the same silent downgrade `PostgresConnector` deliberately removed. The
rule here is that a configured setting is either applied or refused, never
dropped.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from nlqueries.connectors.sqlalchemy_connector import SQLAlchemyConnector

_PG = "postgresql://user:pw@db.internal:5432/shop"


@pytest.fixture
def engine_args():
    """Capture what connect() hands create_engine."""
    seen: dict[str, Any] = {}

    def _fake(url, **kwargs):
        seen["url"] = url
        seen.update(kwargs)
        return object()

    with patch("nlqueries.connectors.sqlalchemy_connector.create_engine", side_effect=_fake):
        yield seen


def test_the_baseline_connects_at_all(engine_args) -> None:
    """Canary. Without it every assertion below could pass over a connect() that
    never reached create_engine."""
    SQLAlchemyConnector().connect({"url": _PG})
    assert engine_args["url"] == _PG
    assert engine_args["pool_pre_ping"] is True


def test_no_tls_settings_means_no_connect_args(engine_args) -> None:
    """An operator who configured nothing gets exactly the behaviour they had:
    the URL decides, and nothing is injected behind it."""
    SQLAlchemyConnector().connect({"url": _PG})
    assert engine_args["connect_args"] == {}


def test_ssl_mode_reaches_the_driver(engine_args) -> None:
    SQLAlchemyConnector().connect({"url": _PG, "ssl_mode": "verify-full"})
    assert engine_args["connect_args"] == {"sslmode": "verify-full"}


def test_certificate_material_reaches_the_driver(engine_args) -> None:
    SQLAlchemyConnector().connect(
        {
            "url": _PG,
            "ssl_mode": "verify-full",
            "ssl_ca_cert": "/etc/ssl/ca.pem",
            "ssl_client_cert": "/etc/ssl/client.crt",
            "ssl_client_key": "/etc/ssl/client.key",
        }
    )
    assert engine_args["connect_args"] == {
        "sslmode": "verify-full",
        "sslrootcert": "/etc/ssl/ca.pem",
        "sslcert": "/etc/ssl/client.crt",
        "sslkey": "/etc/ssl/client.key",
    }


@pytest.mark.parametrize(
    "url",
    [
        "postgresql+psycopg://user:pw@db.internal/shop",  # psycopg 3
        "postgresql://user:pw@db.internal/shop",  # bare -> psycopg2
    ],
)
def test_the_libpq_drivers_are_recognised(engine_args, url: str) -> None:
    SQLAlchemyConnector().connect({"url": url, "ssl_mode": "require"})
    assert engine_args["connect_args"] == {"sslmode": "require"}


@pytest.mark.parametrize(
    "url",
    [
        "mysql+pymysql://user:pw@db.internal/shop",  # different parameter names
        "postgresql+pg8000://user:pw@db.internal/shop",  # PostgreSQL, but ssl_context
        "sqlite:////data/db.sqlite",  # no network at all
    ],
)
def test_a_setting_that_cannot_be_applied_is_refused_not_ignored(engine_args, url: str) -> None:
    """The point of the change. Dropping these silently is how an operator ends
    up with a plaintext session they believe is verified."""
    with pytest.raises(ValueError, match="cannot apply"):
        SQLAlchemyConnector().connect({"url": url, "ssl_ca_cert": "/etc/ssl/ca.pem"})
    assert "connect_args" not in engine_args, "the engine must not be built after a refusal"


def test_those_drivers_still_work_without_tls_settings(engine_args) -> None:
    """The refusal is scoped to settings that were actually configured. A MySQL
    or SQLite URL with no TLS keys is untouched, as it was before."""
    SQLAlchemyConnector().connect({"url": "mysql+pymysql://user:pw@db.internal/shop"})
    assert engine_args["connect_args"] == {}
