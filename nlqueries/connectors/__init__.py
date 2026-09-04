# nlqueries-core — OSS (BSL 1.1)
# This package must NEVER import from the enterprise layer.

import logging
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

from nlqueries.connectors.base import (
    POLICY_COLUMN,
    POLICY_ROW,
    ColumnSpec,
    DatabaseConnector,
    QueryRecord,
    QueryResult,
    SchemaSpec,
    SecurityPolicy,
    SecurityPolicyReport,
    TableSpec,
)
from nlqueries.connectors.postgres import PostgresConnector
from nlqueries.connectors.sqlalchemy_connector import SQLAlchemyConnector
from nlqueries.connectors.sqlite import SQLiteConnector

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Connector registry
#
# Maps a db-type identifier (as used by the CLI / connector configs) to its
# DatabaseConnector implementation. New connectors register themselves here.
#
# Connectors with optional driver dependencies are registered lazily below so
# that a missing optional extra never prevents the core package from importing.
# The generic SQLAlchemy connector needs no extra (SQLAlchemy is a base dep), so
# it's registered here; the URL's own driver is what must be installed at use.
# ---------------------------------------------------------------------------
CONNECTOR_REGISTRY: dict[str, type[DatabaseConnector]] = {
    "postgres": PostgresConnector,
    "sqlalchemy": SQLAlchemyConnector,
    # SQLite ships with Python (stdlib ``sqlite3``), so it's always available —
    # registered eagerly like postgres rather than behind an optional extra.
    "sqlite": SQLiteConnector,
}


def _register_optional_connectors() -> None:
    """Register connectors that require optional driver extras."""
    try:
        from nlqueries.connectors.snowflake import SnowflakeConnector  # noqa: PLC0415

        CONNECTOR_REGISTRY["snowflake"] = SnowflakeConnector
    except Exception:  # noqa: BLE001
        pass

    try:
        from nlqueries.connectors.bigquery import BigQueryConnector  # noqa: PLC0415

        CONNECTOR_REGISTRY["bigquery"] = BigQueryConnector
    except Exception:  # noqa: BLE001
        pass

    try:
        from nlqueries.connectors.redshift import RedshiftConnector  # noqa: PLC0415

        CONNECTOR_REGISTRY["redshift"] = RedshiftConnector
    except Exception:  # noqa: BLE001
        pass

    try:
        from nlqueries.connectors.mssql import MSSQLConnector  # noqa: PLC0415

        CONNECTOR_REGISTRY["mssql"] = MSSQLConnector
    except Exception:  # noqa: BLE001
        pass

    try:
        from nlqueries.connectors.duckdb import DuckDBConnector  # noqa: PLC0415

        CONNECTOR_REGISTRY["duckdb"] = DuckDBConnector
    except Exception:  # noqa: BLE001
        pass


_register_optional_connectors()


# ---------------------------------------------------------------------------
# Connector class resolution
#
# ``CONNECTOR_REGISTRY`` maps a db-type to one class, which is the whole answer
# for every connector this package ships. It is not the whole answer for a
# deployment that adds authentication methods on top of one of them: the driver
# handshake for a key-pair or token sign-in differs from a password sign-in
# while the schema, history and execution behaviour are identical, so the
# natural implementation is a subclass overriding ``connect()`` alone -- and a
# registry keyed by db-type has nowhere to put a second class for the same type.
#
# The consequence was a split path. A deployment that resolved its own class at
# the sites it controlled still went through ``open_connector_for_agent`` on the
# query path, which reads the registry directly, so a connector could be created
# and tested successfully and then answer every question through the wrong class
# -- the same shape as the TLS defect this module's neighbours were fixing, and
# invisible for the same reason.
#
# So resolution is a named seam rather than a dict lookup. The default is
# exactly the previous behaviour, and an installed resolver is consulted first
# with the connector's own configuration entry, which is where an
# authentication method is recorded.
# ---------------------------------------------------------------------------

#: What :func:`set_connector_resolver` accepts: the db-type and the connector's
#: configuration entry, returning the class to build or ``None`` to defer to the
#: registry.
ConnectorResolver = Callable[[str, Mapping[str, Any]], "type[DatabaseConnector] | None"]

_resolver: ConnectorResolver | None = None

#: Handed to a resolver when there is no entry, so it never receives ``None``.
_NO_ENTRY: Mapping[str, Any] = MappingProxyType({})


def set_connector_resolver(resolver: ConnectorResolver | None) -> None:
    """Install *resolver* as the first choice for connector-class resolution.

    ``None`` removes it. Process-wide and idempotent: the last call wins, and
    installing the same resolver twice is not an error.

    **Every change here is subject to the connector cache, in both directions.**
    Resolution happens once per cache fingerprint, and the fingerprint covers the
    configuration entry and the password -- not which resolver was installed. So
    with ``CONNECTOR_CACHE_ENABLED``:

    * A resolver installed *late* does not reach a connector already built and
      cached under the registry class.
    * Removing or replacing one does not reach a connector already built and
      cached under the previous resolver's class; it keeps being served under
      that class until its entry changes or the TTL expires.

    A resolver that *raises* is the exception to the exception: the fallback taken
    in its place is deliberately not cached, so a transient fault corrects itself
    on the next attempt rather than being held for the TTL.

    A connector whose settings change is rebuilt by itself, because the entry is
    in the fingerprint. Nothing else is. Call
    :func:`nlqueries.connectors.loader.invalidate_connector_cache` around any
    change to resolution that has to take effect immediately -- installing,
    removing or replacing -- which is the same reason a rotated credential calls
    it rather than waiting for the TTL.
    """
    global _resolver
    _resolver = resolver


def connector_class_for(
    db_type: str, cfg: Mapping[str, Any] | None = None
) -> type[DatabaseConnector] | None:
    """The connector class to build for *db_type*, given its entry *cfg*.

    *cfg* is the connector's configuration entry as it appears in
    ``CONNECTORS_FILE`` -- ``db_type``, ``url``, and whatever else was recorded
    for it, which is where a deployment records an authentication method. It is
    what every call site in this package passes, including ``nlqueries connect``,
    which has no entry yet and builds an entry-shaped mapping rather than handing
    over its own options: a resolver should never have to ask which caller it is
    serving. Note that an entry has no discrete ``password`` -- the password is
    inside ``url``.

    An installed resolver is asked first and its answer wins; ``None`` from it
    means "no opinion" and falls through to :data:`CONNECTOR_REGISTRY`, which is
    also the whole of the behaviour when no resolver is installed. Returns
    ``None`` for a db-type nothing resolves, exactly as ``CONNECTOR_REGISTRY.get``
    does, so callers keep reporting that themselves.

    A resolver that raises is logged and ignored rather than propagated. It sits
    on the query path of every connector, including the ones it has no opinion
    about, so a fault in the extension must not be able to take down a
    password-authenticated Postgres connector that would otherwise have opened.
    """
    return _resolve(db_type, cfg)[0]


def _resolve(
    db_type: str, cfg: Mapping[str, Any] | None = None
) -> tuple[type[DatabaseConnector] | None, bool]:
    """As :func:`connector_class_for`, plus whether resolution *degraded*.

    Package-internal, and deliberately not the public form: only the loader has a
    cache to keep the flag out of, and every other caller wants a class rather
    than a pair. :func:`connector_class_for` is this with the flag dropped.

    Degraded means the resolver raised and the registry answered in its place, so
    the class is a fallback rather than a decision. The caller needs to know
    because a fallback must not be cached: the connector cache is keyed on the
    entry and the password, so a resolver that fails for a moment -- one
    consulting a configuration service, say -- would otherwise pin the registry
    class for the whole TTL of every connector the fallback can still open. One
    transient fault would reinstate exactly the split this seam removes, and the
    only trace would be a single warning.

    Declining to resolve is not degraded. A resolver returning ``None`` has
    answered, and its answer is "the registry", which is as cacheable as any
    other.
    """
    if _resolver is not None:
        try:
            # Read-only, because the loader passes the entry it has already
            # fingerprinted and will shortly build credentials from. A resolver
            # that added a default or rewrote `url` would change the credentials
            # without changing the fingerprint, so the connector would be cached
            # under a description of a configuration it was not built from. The
            # `Mapping` annotation states that intention; this enforces it, and a
            # resolver that tries is reported like any other misbehaving one.
            resolved = _resolver(db_type, MappingProxyType(dict(cfg)) if cfg else _NO_ENTRY)
        except Exception:  # noqa: BLE001 -- see the docstring
            logger.warning(
                "The installed connector resolver raised for db_type %r; falling back to "
                "the registry. The connector will be opened with the class registered for "
                "its type, which is wrong for any authentication method the resolver "
                "exists to select. It is not cached, so the next attempt resolves again.",
                db_type,
                exc_info=True,
            )
            return CONNECTOR_REGISTRY.get(db_type), True
        if resolved is not None:
            return resolved, False
    return CONNECTOR_REGISTRY.get(db_type), False


__all__ = [
    "ColumnSpec",
    "DatabaseConnector",
    "PostgresConnector",
    "QueryRecord",
    "QueryResult",
    "SchemaSpec",
    "SQLiteConnector",
    "TableSpec",
    "CONNECTOR_REGISTRY",
    "ConnectorResolver",
    "connector_class_for",
    "set_connector_resolver",
    "POLICY_ROW",
    "POLICY_COLUMN",
    "SecurityPolicy",
    "SecurityPolicyReport",
]
