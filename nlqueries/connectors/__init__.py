# nlqueries-core — OSS (BSL 1.1)
# This package must NEVER import from the enterprise layer.

import logging
from collections.abc import Callable, Mapping
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


def set_connector_resolver(resolver: ConnectorResolver | None) -> None:
    """Install *resolver* as the first choice for connector-class resolution.

    ``None`` removes it, restoring the registry-only behaviour. Process-wide and
    idempotent: the last call wins, and installing the same resolver twice is
    not an error.

    **Install it before the first connector is opened.** Resolution happens once
    per cache fingerprint, so a connector already built and cached under the
    registry class is served from the cache and not re-resolved. The
    configuration entry is part of that fingerprint, so a connector whose
    settings change is rebuilt through the new resolver by itself; a resolver
    installed late against unchanged settings is not. Call
    :func:`nlqueries.connectors.loader.invalidate_connector_cache` if that
    ordering cannot be guaranteed.
    """
    global _resolver
    _resolver = resolver


def connector_class_for(
    db_type: str, cfg: Mapping[str, Any] | None = None
) -> type[DatabaseConnector] | None:
    """The connector class to build for *db_type*, given its entry *cfg*.

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
    if _resolver is not None:
        try:
            resolved = _resolver(db_type, cfg or {})
        except Exception:  # noqa: BLE001 -- see the docstring
            logger.warning(
                "The installed connector resolver raised for db_type %r; falling back to "
                "the registry. The connector will be opened with the class registered for "
                "its type, which is wrong for any authentication method the resolver "
                "exists to select.",
                db_type,
                exc_info=True,
            )
        else:
            if resolved is not None:
                return resolved
    return CONNECTOR_REGISTRY.get(db_type)


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
