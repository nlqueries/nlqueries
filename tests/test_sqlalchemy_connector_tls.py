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


def test_a_root_certificate_alone_selects_verify_full(engine_args) -> None:
    """The case a pure key rename gets wrong, and the reason this needs the
    resolver rather than a mapping.

    `ssl_ca_cert` with no `ssl_mode` is a deliberate combination, not an
    oversight: the GCP Cloud SQL IAM provider leaves the mode unset when a root
    certificate is configured precisely because it expects the connector to
    choose. Renaming keys alone yields `sslrootcert` under libpq's `prefer`, so
    the operator supplies a CA and gets a session that falls back to plaintext
    and verifies nothing -- the failure this whole change exists to remove.
    """
    SQLAlchemyConnector().connect({"url": _PG, "ssl_ca_cert": "/etc/ssl/ca.pem"})
    assert engine_args["connect_args"] == {
        "sslmode": "verify-full",
        "sslrootcert": "/etc/ssl/ca.pem",
    }


def test_no_certificate_and_no_mode_still_insists_on_encryption(engine_args) -> None:
    """The resolver's other default, held here so a future edit cannot quietly
    reintroduce `prefer` for a connector that configured a client certificate
    but no mode."""
    SQLAlchemyConnector().connect({"url": _PG, "ssl_client_cert": "/etc/ssl/client.crt"})
    assert engine_args["connect_args"]["sslmode"] == "require"


def test_an_explicit_mode_is_still_honoured(engine_args) -> None:
    """Canary for the two above: the resolver must not override what the
    operator actually asked for, including a deliberately weaker mode."""
    SQLAlchemyConnector().connect(
        {"url": _PG, "ssl_mode": "disable", "ssl_ca_cert": "/etc/ssl/ca.pem"}
    )
    assert engine_args["connect_args"]["sslmode"] == "disable"


def test_a_url_that_already_sets_tls_is_not_silently_overruled(engine_args) -> None:
    """`create_engine` unions the URL's parameters with `connect_args` and lets
    `connect_args` win, so a stricter URL setting would be discarded with no
    indication. The documented advice for an unmapped driver is to put the
    posture in the query string, which makes both being present something an
    operator may reasonably do."""
    with pytest.raises(ValueError, match="will not silently overrule the URL"):
        SQLAlchemyConnector().connect({"url": _PG + "?sslmode=verify-full", "ssl_mode": "require"})
    assert "connect_args" not in engine_args, "the engine must not be built after a refusal"


def test_a_url_with_tls_and_no_configured_settings_is_left_alone(engine_args) -> None:
    """Canary for the refusal: it fires on a clash, not on the mere presence of
    TLS in the URL. Expressing the posture there is the documented route."""
    SQLAlchemyConnector().connect({"url": _PG + "?sslmode=verify-full"})
    assert engine_args["connect_args"] == {}


# ---------------------------------------------------------------------------
# Follow-up to the review on #169
# ---------------------------------------------------------------------------


def test_a_url_parameter_this_connector_would_not_set_is_not_a_clash(engine_args) -> None:
    """The refusal exists to stop a silent overrule, so it must fire only where
    one would happen. `?sslrootcert=` beside a configured `ssl_mode` replaces
    nothing — the two name different parameters — and refusing it turned a valid
    split configuration into an error whose message listed two sets that did not
    overlap."""
    SQLAlchemyConnector().connect(
        {"url": _PG + "?sslrootcert=/etc/ssl/ca.pem", "ssl_mode": "verify-full"}
    )
    assert engine_args["connect_args"] == {"sslmode": "verify-full"}


def test_the_same_parameter_from_both_sides_is_still_refused(engine_args) -> None:
    """Canary for the above: narrowing the check must not disarm it."""
    with pytest.raises(ValueError, match="will not silently overrule the URL"):
        SQLAlchemyConnector().connect({"url": _PG + "?sslmode=verify-full", "ssl_mode": "require"})


def test_a_resolved_posture_is_reported(engine_args) -> None:
    """`nlqueries health` reads `connector.tls`. This connector resolved a
    definite mode and then discarded it, so a `sqlalchemy` entry running at
    `require` with no certificate — encrypted, verifying nothing — reported no
    concern, while an identical `postgres` entry reported one."""
    connector = SQLAlchemyConnector()
    connector.connect({"url": _PG, "ssl_mode": "require"})
    assert connector.tls is not None
    assert connector.tls.ssl_mode == "require"
    assert connector.tls.concerns, "require without a CA verifies nothing and should say so"


def test_a_verified_posture_reports_no_concern(engine_args) -> None:
    """Canary: the posture is reported, not merely a complaint. A connector that
    always had concerns would satisfy the test above."""
    connector = SQLAlchemyConnector()
    connector.connect({"url": _PG, "ssl_ca_cert": "/etc/ssl/ca.pem"})
    assert connector.tls is not None
    assert connector.tls.ssl_mode == "verify-full"
    assert not connector.tls.concerns


def test_a_url_only_posture_is_read_from_the_url(engine_args) -> None:
    """The URL decides here, and it is readable, so it is reported. Silence would
    be the honest answer only if the posture were unknowable, and it is not."""
    connector = SQLAlchemyConnector()
    connector.connect({"url": _PG + "?sslmode=verify-full&sslrootcert=/etc/ssl/ca.pem"})
    assert connector.tls is not None
    assert connector.tls.ssl_mode == "verify-full"
    assert not connector.tls.concerns


def test_a_split_configuration_is_described_from_both_halves(engine_args) -> None:
    """The case the narrowed clash check created.

    `?sslrootcert=` in the URL with `ssl_mode: require` in the credentials
    connects with libpq verifying the chain against that certificate. Read from
    the credentials alone the posture is `require` with no root certificate, and
    `nlqueries health` would report the connection as "encrypted to whichever
    server answers" — a confident wrong answer about a connection that is in
    fact verified.
    """
    connector = SQLAlchemyConnector()
    connector.connect({"url": _PG + "?sslrootcert=/etc/ssl/ca.pem", "ssl_mode": "require"})
    assert connector.tls is not None
    assert connector.tls.ssl_mode == "require"
    assert connector.tls.has_root_certificate, "the URL's certificate is part of the posture"
    # `require` with a root certificate verifies the chain but not the hostname,
    # so there is still a concern -- the accurate one. What must not appear is
    # the report for a connection with no certificate at all, which is what
    # reading the credentials alone produced.
    assert connector.tls.verifies_certificate_chain
    assert not any("no ssl_ca_cert" in c for c in connector.tls.concerns)


def test_a_client_certificate_beside_a_url_root_certificate_is_also_joined(
    engine_args,
) -> None:
    """The same shape with the mode arriving by resolution rather than by
    configuration: a client certificate alone resolves to `require`, and the
    URL's root certificate is what makes the chain verified."""
    connector = SQLAlchemyConnector()
    connector.connect(
        {"url": _PG + "?sslrootcert=/etc/ssl/ca.pem", "ssl_client_cert": "/etc/ssl/c.crt"}
    )
    assert connector.tls is not None
    assert connector.tls.has_root_certificate


def test_a_connection_with_no_mode_anywhere_reports_nothing(engine_args) -> None:
    """None, not a guess. With no mode on either side libpq applies its own
    `prefer` default, which is not what `resolve_ssl_mode` would have chosen, so
    describing these settings would report a posture the connection is not
    running under."""
    connector = SQLAlchemyConnector()
    connector.connect({"url": _PG})
    assert connector.tls is None


def test_the_posture_is_unset_before_connect() -> None:
    assert SQLAlchemyConnector().tls is None
