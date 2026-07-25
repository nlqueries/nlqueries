# nlqueries-core — OSS (BSL 1.1)
# This package must NEVER import from the enterprise layer.

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
    "POLICY_ROW",
    "POLICY_COLUMN",
    "SecurityPolicy",
    "SecurityPolicyReport",
]
