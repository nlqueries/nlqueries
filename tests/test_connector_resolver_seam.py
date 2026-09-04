"""The query path must resolve a connector class the same way every other path does.

``CONNECTOR_REGISTRY`` holds one class per db-type, which is the whole answer
for the connectors this package ships and not the whole answer for a deployment
that adds an authentication method to one of them. A key-pair or token sign-in
differs from a password sign-in only in the driver handshake, so the natural
implementation is a subclass overriding ``connect()`` -- and a registry keyed by
db-type has nowhere to put a second class for the same type.

Before the seam, a deployment could select its own class everywhere it
controlled the call and still lose it on the query path, because
``open_connector_for_agent`` read the registry directly. The connector created
green, tested green, and then answered every question through a class that could
not perform its sign-in. That is the same shape as the TLS defect this area has
been fixing: a setting honoured on the paths that connect from a request body
and dropped on the one that connects from the projected file.

These tests are therefore about ``open_connector_for_agent`` -- the class that is
actually *connected* -- rather than about the helper's return value, which could
be right while the loader went on ignoring it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
import yaml
from nlqueries import config, connectors
from nlqueries.connectors import loader
from nlqueries.execution import ExecutionPolicy

_AGENT = "postgres:localhost:db"


class _Registered:
    """Stands in for the class the registry holds for a db-type."""

    def __init__(self) -> None:
        self.credentials: dict[str, Any] | None = None

    def connect(self, credentials: dict[str, Any]) -> None:
        self.credentials = credentials


class _Resolved(_Registered):
    """Stands in for the subclass a deployment resolves for an auth method."""


@pytest.fixture(autouse=True)
def _no_resolver_leaks():
    """A resolver is process-wide, so a test that installs one and does not
    remove it would silently re-route every connector opened afterwards -- in
    this file and in every other."""
    connectors.set_connector_resolver(None)
    loader.invalidate_connector_cache()
    yield
    connectors.set_connector_resolver(None)
    loader.invalidate_connector_cache()


@pytest.fixture
def connectors_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "connectors.yaml"
    monkeypatch.setattr(config, "CONNECTORS_FILE", path)
    monkeypatch.setattr(config, "CONNECTOR_CACHE_ENABLED", False)
    return path


@pytest.fixture
def registered(monkeypatch: pytest.MonkeyPatch) -> type[_Registered]:
    monkeypatch.setitem(loader.CONNECTOR_REGISTRY, "postgres", _Registered)
    return _Registered


def _write(path: Path, **entry: Any) -> None:
    base = {"db_type": "postgres", "url": "postgresql://user:hunter2@localhost:5432/db"}
    base.update(entry)
    path.write_text(yaml.safe_dump({_AGENT: base}))


def _opened(agent: str = _AGENT):
    """The connector object the loader built, unwrapped from its policy wrapper.

    The loader returns a ``PermittedConnector``, so ``type(...)`` on the return
    value is the wrapper for every case and would make each assertion below pass
    or fail together regardless of what was resolved.
    """
    permitted = loader.open_connector_for_agent(agent, ExecutionPolicy.execute_read_only())
    return None if permitted is None else permitted._inner  # noqa: SLF001


# ---------------------------------------------------------------------------
# The default: nothing installed, nothing changed
# ---------------------------------------------------------------------------


def test_with_no_resolver_the_registry_class_is_built(connectors_file, registered) -> None:
    """The baseline every other test here is measured against.

    Without this, a seam that resolved nothing at all -- or one that returned the
    subclass unconditionally -- would satisfy the assertions below just as well.
    """
    _write(connectors_file)
    assert type(_opened()) is _Registered


def test_the_helper_and_the_loader_agree_by_default(registered) -> None:
    assert connectors.connector_class_for("postgres") is _Registered
    assert connectors.connector_class_for("postgres", {}) is _Registered


def test_an_unresolvable_db_type_is_still_none(connectors_file) -> None:
    """Unchanged behaviour, and the reason callers keep reporting it themselves:
    the seam returns ``None`` for a type nothing resolves, exactly as
    ``CONNECTOR_REGISTRY.get`` did."""
    assert connectors.connector_class_for("teradata", {"db_type": "teradata"}) is None


# ---------------------------------------------------------------------------
# An installed resolver
# ---------------------------------------------------------------------------


def test_the_resolved_class_is_the_one_connected(connectors_file, registered) -> None:
    """The property that matters. Not "the resolver was called" and not "the
    helper returned the subclass" -- the object the loader hands back, having
    called ``connect()`` on it, must be the resolved one."""
    connectors.set_connector_resolver(lambda db_type, cfg: _Resolved)
    _write(connectors_file)

    opened = _opened()
    assert type(opened) is _Resolved
    assert opened.credentials is not None, "the resolved class must have been connected"


def test_the_resolver_sees_the_entry_it_must_decide_from(connectors_file, registered) -> None:
    """The whole point of passing *cfg*: the deciding field is one the registry
    knows nothing about, so a seam that passed only the db-type would be unable
    to express the selection it exists for."""
    seen: list[tuple[str, dict[str, Any]]] = []

    def resolver(db_type: str, cfg: Any) -> type | None:
        seen.append((db_type, dict(cfg)))
        return _Resolved if cfg.get("auth_method") == "keypair" else None

    connectors.set_connector_resolver(resolver)
    _write(connectors_file, auth_method="keypair")

    assert type(_opened()) is _Resolved
    assert seen and seen[0][0] == "postgres"
    assert seen[0][1]["auth_method"] == "keypair"


def test_a_resolver_with_no_opinion_falls_through_to_the_registry(
    connectors_file, registered
) -> None:
    """``None`` means "no opinion", not "no connector". A resolver that answers
    for one auth method must leave every other connector in the deployment
    opening exactly as it did before."""
    connectors.set_connector_resolver(
        lambda db_type, cfg: _Resolved if cfg.get("auth_method") == "keypair" else None
    )
    _write(connectors_file, auth_method="password")

    assert type(_opened()) is _Registered


def test_removing_the_resolver_restores_the_registry(connectors_file, registered) -> None:
    connectors.set_connector_resolver(lambda db_type, cfg: _Resolved)
    _write(connectors_file)
    assert type(_opened()) is _Resolved

    connectors.set_connector_resolver(None)
    assert type(_opened()) is _Registered


# ---------------------------------------------------------------------------
# A resolver that misbehaves
# ---------------------------------------------------------------------------


def test_a_raising_resolver_does_not_take_the_query_path_down(
    connectors_file, registered, caplog
) -> None:
    """It is consulted for every connector, including the ones it has no opinion
    about. A fault in it must not stop a password-authenticated Postgres
    connector that would otherwise have opened -- that would turn an extension
    bug into a total outage."""

    def resolver(db_type: str, cfg: Any) -> type | None:
        raise RuntimeError("resolver is broken")

    connectors.set_connector_resolver(resolver)
    _write(connectors_file)

    with caplog.at_level(logging.WARNING, logger=connectors.logger.name):
        assert type(_opened()) is _Registered
    assert "connector resolver raised" in caplog.text
    assert "resolver is broken" in caplog.text, "the resolver's own words are the diagnosis"


def test_a_working_resolver_logs_no_warning(connectors_file, registered, caplog) -> None:
    """Negative control for the test above, which would pass just as well
    against a seam that warned on every call -- and a warning on every connector
    open is one people learn to filter out, taking the real one with it."""
    connectors.set_connector_resolver(lambda db_type, cfg: _Resolved)
    _write(connectors_file)

    with caplog.at_level(logging.WARNING, logger=connectors.logger.name):
        assert type(_opened()) is _Resolved
    assert caplog.records == []


# ---------------------------------------------------------------------------
# The cache
# ---------------------------------------------------------------------------


def test_a_changed_auth_method_is_re_resolved_through_the_cache(
    connectors_file, registered, monkeypatch
) -> None:
    """Resolution happens on a cache miss, so the cache is what decides whether a
    connector switched to key-pair authentication actually starts using it.

    The configuration entry is part of the cache fingerprint, so changing
    ``auth_method`` is a miss by itself and the rebuild goes through the
    resolver. This is the other half of the seam's documented "install it before
    the first connector is opened" caveat: settings that do not change are served
    from the cache and not re-resolved.
    """
    monkeypatch.setattr(config, "CONNECTOR_CACHE_ENABLED", True)
    connectors.set_connector_resolver(
        lambda db_type, cfg: _Resolved if cfg.get("auth_method") == "keypair" else None
    )

    _write(connectors_file, auth_method="password")
    assert type(_opened()) is _Registered

    _write(connectors_file, auth_method="keypair")
    assert type(_opened()) is _Resolved, "the entry changed, so the cached class must not be reused"
