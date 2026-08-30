"""The loader must hand a connector the configuration it was given.

``open_connector_for_agent`` built the credentials dict for
``DatabaseConnector.connect`` from a hardcoded list of keys, and silently
dropped everything the list did not name. What it dropped included every TLS
key a connector reads: ``ssl_mode``, ``ssl_ca_cert``, ``ssl_client_cert`` and
``ssl_client_key``.

That has two consequences, in opposite directions, and neither is visible from
the connector:

* a database with no TLS cannot be reached at all, because with no ``ssl_mode``
  the connector resolves the ``require`` default and libpq refuses the server;
* a connector configured to verify a private CA is opened *without* the CA, so
  it resolves ``require`` rather than ``verify-full`` -- an encrypted session
  that authenticates nothing, which is the posture the operator configured
  against.

These tests assert the property (the configuration reaches ``connect``), not the
current key list, so a connector that learns a new key is covered without
anybody remembering to extend a test.
"""

from __future__ import annotations

from typing import Any

import pytest
import yaml
from nlqueries import config
from nlqueries.connectors import loader

_CONNECTOR_ID = "postgres:localhost:db"


@pytest.fixture(autouse=True)
def _clean_cache():
    loader.invalidate_connector_cache()
    yield
    loader.invalidate_connector_cache()


@pytest.fixture
def seen(monkeypatch):
    """Capture the credentials dict the connector is actually handed."""
    captured: dict[str, Any] = {}

    class _Connector:
        def connect(self, credentials: dict[str, Any]) -> None:
            captured.clear()
            captured.update(credentials)

    monkeypatch.setitem(loader.CONNECTOR_REGISTRY, "postgres", _Connector)
    monkeypatch.setattr(config, "CONNECTOR_CACHE_ENABLED", False)
    return captured


@pytest.fixture
def write_config(tmp_path, monkeypatch):
    path = tmp_path / "connectors.yaml"
    monkeypatch.setattr(config, "CONNECTORS_FILE", path)

    def _write(**extra: Any) -> None:
        path.write_text(
            yaml.safe_dump(
                {
                    _CONNECTOR_ID: {
                        "db_type": "postgres",
                        "url": "postgresql://user:secret@localhost:5432/db",
                        **extra,
                    }
                }
            )
        )

    return _write


def _open(seen: dict[str, Any]) -> dict[str, Any]:
    from nlqueries.execution import ExecutionPolicy

    assert (
        loader.open_connector_for_agent(_CONNECTOR_ID, ExecutionPolicy.execute_read_only())
        is not None
    ), "the connector failed to open at all -- the rest of this test proves nothing"
    return seen


def test_the_baseline_config_still_reaches_the_connector(write_config, seen) -> None:
    """Canary. Without this an empty `seen` would satisfy every assertion below
    by vacuous truth, and the suite would report success over a loader that
    passes nothing at all."""
    write_config()
    creds = _open(seen)
    assert creds["host"] == "localhost"
    assert creds["port"] == 5432
    assert creds["database"] == "db"
    assert creds["user"] == "user"
    assert creds["password"] == "secret"


def test_ssl_mode_reaches_the_connector(write_config, seen) -> None:
    """The e2e defect: `ssl_mode: disable` was configured, honoured when the
    connector was created and tested, and then dropped on the path that
    executes queries -- so every query failed with "server does not support
    SSL, but SSL was required"."""
    write_config(ssl_mode="disable")
    assert _open(seen)["ssl_mode"] == "disable"


def test_certificate_material_reaches_the_connector(write_config, seen) -> None:
    """The security direction of the same bug. Dropping `ssl_ca_cert` does not
    fail the connection -- it downgrades `verify-full` to `require`, and the
    session comes up encrypted but unverified."""
    write_config(
        ssl_ca_cert="/etc/ssl/rds-global-bundle.pem",
        ssl_client_cert="/etc/ssl/client.crt",
        ssl_client_key="/etc/ssl/client.key",
    )
    creds = _open(seen)
    assert creds["ssl_ca_cert"] == "/etc/ssl/rds-global-bundle.pem"
    assert creds["ssl_client_cert"] == "/etc/ssl/client.crt"
    assert creds["ssl_client_key"] == "/etc/ssl/client.key"


def test_a_key_no_connector_reads_today_is_still_passed(write_config, seen) -> None:
    """The property, stated directly: the loader is not an allow-list. A key it
    has never heard of reaches `connect`, so the next connector option works
    without a change here -- which is the failure this whole file exists for."""
    write_config(some_option_added_later="value")
    assert _open(seen)["some_option_added_later"] == "value"


def test_entry_bookkeeping_is_not_passed_as_credentials(write_config, seen) -> None:
    """The other half of the property: keys that describe the entry rather than
    the connection stay out."""
    write_config(password_storage="plaintext")
    creds = _open(seen)
    assert "db_type" not in creds
    assert "password_storage" not in creds


def test_the_url_is_passed_because_a_connector_connects_with_it(write_config, seen) -> None:
    """`url` is deliberately not entry bookkeeping. SQLAlchemyConnector, which
    backs the `sqlalchemy` db_type, reads `credentials["url"]` and raises
    ValueError without it -- and `open_connector_for_agent` turns that into a
    silent None. Withholding the URL would leave that connector unopenable."""
    write_config()
    assert _open(seen)["url"] == "postgresql://user:secret@localhost:5432/db"


def test_the_url_still_wins_for_the_fields_it_carries(write_config, seen) -> None:
    """`_get_full_url` resolves a keychain password into the URL, so the URL --
    not the raw entry -- is authoritative for host/port/database/user/password.
    A stale duplicate in the entry must not override it."""
    write_config(host="stale-host", port=9999, database="stale-db", user="stale-user")
    creds = _open(seen)
    assert creds["host"] == "localhost"
    assert creds["port"] == 5432
    assert creds["database"] == "db"
    assert creds["user"] == "user"


def test_the_url_carries_the_keychain_password_not_the_stored_one(
    write_config, seen, monkeypatch
) -> None:
    """Passing the URL through is only safe because it is the *resolved* one.

    When the password lives in the OS keychain the stored URL holds whatever
    stale value was written with it, and `_get_full_url` injects the real one.
    Handing over `cfg["url"]` instead would give SQLAlchemyConnector a URL that
    cannot authenticate, in the one case the keychain exists to serve."""
    import sys
    import types

    fake = types.ModuleType("keyring")
    fake.get_password = lambda service, name: "from-keychain"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "keyring", fake)

    write_config(password_storage="keychain")
    creds = _open(seen)
    assert creds["password"] == "from-keychain"
    assert "from-keychain" in creds["url"]
    assert "secret" not in creds["url"]
