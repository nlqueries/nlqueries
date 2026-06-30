# nlqueries-core — OSS (BSL 1.1)
# This package must NEVER import from the enterprise layer.

from nlqueries.connectors.base import (
    ColumnSpec,
    DatabaseConnector,
    QueryRecord,
    QueryResult,
    SchemaSpec,
    TableSpec,
)
from nlqueries.connectors.bigquery import BigQueryConnector
from nlqueries.connectors.postgres import PostgresConnector
from nlqueries.connectors.snowflake import SnowflakeConnector

# ---------------------------------------------------------------------------
# Connector registry
#
# Maps a db-type identifier (as used by the CLI / connector configs) to its
# DatabaseConnector implementation. New connectors register themselves here.
#
# Tier-1 connectors with optional driver dependencies (redshift, mssql, duckdb)
# are registered lazily below so that a missing optional extra never prevents
# the core package from importing.
# ---------------------------------------------------------------------------
CONNECTOR_REGISTRY: dict[str, type[DatabaseConnector]] = {
    "postgres": PostgresConnector,
    "snowflake": SnowflakeConnector,
    "bigquery": BigQueryConnector,
}


def _register_optional_connectors() -> None:
    """Register connectors that require optional driver extras."""
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
    "BigQueryConnector",
    "ColumnSpec",
    "DatabaseConnector",
    "PostgresConnector",
    "QueryRecord",
    "QueryResult",
    "SchemaSpec",
    "SnowflakeConnector",
    "TableSpec",
    "CONNECTOR_REGISTRY",
]
