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
from nlqueries.connectors.postgres import PostgresConnector

# ---------------------------------------------------------------------------
# Connector registry
#
# Maps a db-type identifier (as used by the CLI / connector configs) to its
# DatabaseConnector implementation. New connectors register themselves here.
#
# Connectors with optional driver dependencies are registered lazily below so
# that a missing optional extra never prevents the core package from importing.
# ---------------------------------------------------------------------------
CONNECTOR_REGISTRY: dict[str, type[DatabaseConnector]] = {
    "postgres": PostgresConnector,
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
    "TableSpec",
    "CONNECTOR_REGISTRY",
]
