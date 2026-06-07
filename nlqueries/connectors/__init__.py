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
# ---------------------------------------------------------------------------
CONNECTOR_REGISTRY: dict[str, type[DatabaseConnector]] = {
    "postgres": PostgresConnector,
    "snowflake": SnowflakeConnector,
    "bigquery": BigQueryConnector,
}

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
