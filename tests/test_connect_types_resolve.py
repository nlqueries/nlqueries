"""Every type `nlqueries connect` accepts must open again later.

`connect` stores the db type exactly as given, and `query`, `export-kb` and
`process-history` reopen the connector through `connector_class_for`. `mysql`
and `postgresql` were accepted by the first and unknown to the second, so a
connector registered under either name connected once and never again.
"""

from __future__ import annotations

import dataclasses

import pytest
from nlqueries.cli.main import _DB_SCHEMES
from nlqueries.connectors import CONNECTOR_REGISTRY, connector_class_for
from nlqueries.connectors.capabilities import for_dialect
from nlqueries.connectors.postgres import PostgresConnector
from nlqueries.connectors.sqlalchemy_connector import SQLAlchemyConnector
from nlqueries.execution import ExecutionPolicy

#: Registered only when their optional extra imports; absent otherwise.
_OPTIONAL = {"snowflake", "bigquery", "redshift", "mssql", "duckdb"}


@pytest.mark.parametrize("db_type", sorted(_DB_SCHEMES))
def test_every_type_connect_accepts_opens_again(db_type: str) -> None:
    if db_type in _OPTIONAL and db_type not in CONNECTOR_REGISTRY:
        pytest.skip(f"the {db_type} extra is not installed")
    assert connector_class_for(db_type, {"db_type": db_type}) is not None


@pytest.mark.parametrize(
    ("spelling", "registered"), [("postgresql", "postgres"), ("mysql", "sqlalchemy")]
)
def test_an_alias_is_described_by_its_connectors_capabilities(
    spelling: str, registered: str
) -> None:
    """`nlqueries health` reads capabilities by the stored type; an alias that
    resolved to a connector but had no description would report as unknown."""
    assert for_dialect(spelling) is for_dialect(registered) is not None


def test_a_resolver_is_asked_with_the_type_as_stored() -> None:
    """Its contract names the db type, and its entry carries the stored
    spelling; canonicalising only the registry lookup keeps the two the same."""
    from nlqueries.connectors import set_connector_resolver

    seen: list[str] = []

    def _resolver(db_type: str, cfg: object) -> None:
        seen.append(db_type)
        return None

    set_connector_resolver(_resolver)
    try:
        cls = connector_class_for("mysql", {"db_type": "mysql"})
    finally:
        set_connector_resolver(None)

    assert seen == ["mysql"]
    assert cls is SQLAlchemyConnector


def test_a_connector_registered_under_an_alias_is_not_shadowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If a dedicated MySQL connector is ever registered as `mysql`, it must
    be the one used and described, not the generic connector the alias names."""
    from nlqueries.connectors import capabilities

    class DedicatedMySQL(SQLAlchemyConnector):
        pass

    # A distinct object: the alias would otherwise return the same one.
    dedicated = dataclasses.replace(capabilities.CAPABILITIES["sqlalchemy"], dialect="mysql")
    monkeypatch.setitem(CONNECTOR_REGISTRY, "mysql", DedicatedMySQL)
    monkeypatch.setitem(capabilities.CAPABILITIES, "mysql", dedicated)

    assert connector_class_for("mysql", {"db_type": "mysql"}) is DedicatedMySQL
    assert for_dialect("mysql") is dedicated


def test_mysql_runs_on_the_generic_connector() -> None:
    assert connector_class_for("mysql", {"db_type": "mysql"}) is SQLAlchemyConnector


def test_postgresql_is_the_postgres_connector() -> None:
    assert connector_class_for("postgresql", {"db_type": "postgresql"}) is PostgresConnector


def test_a_mysql_typed_entry_connects_and_answers(tmp_path) -> None:
    """What the loader does with a stored `mysql` entry, against SQLite through
    the same generic connector (no MySQL server in the test environment)."""
    cls = connector_class_for("mysql", {"db_type": "mysql"})
    assert cls is not None
    connector = cls()
    connector.connect({"url": f"sqlite:///{tmp_path / 'shop.db'}"})
    connector.bind_execution_policy(ExecutionPolicy.execute_read_only())

    result = connector.execute_query("SELECT 1 AS one")

    assert result.error is None
    assert [list(row) for row in result.rows] == [[1]]
