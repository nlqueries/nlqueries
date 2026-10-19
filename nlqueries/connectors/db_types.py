# nlqueries-core — OSS (BSL 1.1)
"""
nlqueries.connectors.db_types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The db-type spellings that name a connector registered under another name.

``nlqueries connect`` accepts ``postgresql`` and ``mysql`` and stores the type
exactly as given, but the registry and the capabilities table are keyed by
connector: ``postgres``, and ``sqlalchemy`` for MySQL, which runs on the generic
connector. Every lookup by db type canonicalises it here first, so a connector
registered under either spelling can be opened again and described.

A module of its own so that the registry and the capabilities table can both
import it without importing each other.
"""

from __future__ import annotations

#: Spelling as stored -> the name the connector is registered under.
DB_TYPE_ALIASES: dict[str, str] = {
    "postgresql": "postgres",
    # `connect mysql` composes a `mysql+pymysql://` URL, which is all the generic
    # connector needs; it already knows MySQL's read-only and timeout rules.
    # There is no dedicated MySQL connector, and no query history source for it.
    "mysql": "sqlalchemy",
}


def canonical_db_type(db_type: str) -> str:
    """*db_type*, lower-cased, under the name its connector is registered as.

    Callers look an exact name up first, so a connector later registered
    under one of these spellings is used rather than shadowed.
    """
    lowered = db_type.lower()
    return DB_TYPE_ALIASES.get(lowered, lowered)
