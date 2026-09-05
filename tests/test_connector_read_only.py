"""What each SQLAlchemy-backed connector sends around a query, and what it never sends.

`docs/contributing.md` says every connector enforces read-only execution. Four
did -- Postgres and Redshift through `SET TRANSACTION READ ONLY`, SQLite through
`mode=ro`, DuckDB through `read_only=True` -- and four did not. MSSQL and the
generic SQLAlchemy connector used `engine.begin()`, which *commits* when its
block exits without an exception.

That matters because of what the layer above cannot see. Every validator in front
of a connector asks whether the statement's root node is a `Select`, and
`SELECT some_volatile_function(...)` satisfies that while writing. An audit
reproduced exactly that through the Postgres connector twice, eight weeks apart,
and `engine.begin()` committed it both times. The same statement through MSSQL or
through the generic connector -- which serves MySQL and every URL without a
dedicated class -- had the same ending.

CI provisions neither SQL Server nor MySQL, so these assert the sequence sent to
the driver rather than the database's response to it, in the same manner as
`tests/test_redshift_guards.py`. What they hold is the ordering (`SET TRANSACTION
READ ONLY` before the query, where the dialect has one) and the absence of a
commit on the success path, which is the property that was missing.
"""

from __future__ import annotations

from typing import Any

import pytest
from nlqueries.connectors.mssql import MSSQLConnector
from nlqueries.connectors.sqlalchemy_connector import SQLAlchemyConnector

from tests.conftest import granted


class _Result:
    """Enough of a SQLAlchemy CursorResult for the collection path."""

    returns_rows = True

    def keys(self) -> list[str]:
        return ["n"]

    def fetchmany(self, size: int) -> list[Any]:
        return []

    def fetchall(self) -> list[Any]:
        return []

    def __iter__(self) -> Any:
        return iter(())


class _Dialect:
    def __init__(self, name: str) -> None:
        self.name = name
        self._is_mariadb = False


class _Engine:
    def __init__(self, dialect_name: str) -> None:
        self.dialect = _Dialect(dialect_name)


class _Connection:
    """Records statements, commits and rollbacks in the order they happen."""

    def __init__(self, dialect_name: str) -> None:
        self.statements: list[str] = []
        self.commits = 0
        self.rollbacks = 0
        self.engine = _Engine(dialect_name)

    def execute(self, clause: Any) -> _Result:
        self.statements.append(str(clause))
        return _Result()

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1

    def __enter__(self) -> _Connection:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class _FakeEngine:
    """An engine whose `begin()` would commit, so a test notices if it is used."""

    def __init__(self, dialect_name: str) -> None:
        self.conn = _Connection(dialect_name)
        self.dialect = self.conn.engine.dialect

    def connect(self) -> _Connection:
        return self.conn

    def begin(self) -> _Connection:  # pragma: no cover - reaching this is the failure
        raise AssertionError(
            "engine.begin() commits when its block exits; execution must use connect()"
        )


def _mssql(dialect_name: str = "mssql") -> tuple[MSSQLConnector, _Connection]:
    connector = granted(MSSQLConnector())
    engine = _FakeEngine(dialect_name)
    connector._engine = engine  # noqa: SLF001
    return connector, engine.conn


def _generic(dialect_name: str) -> tuple[SQLAlchemyConnector, _Connection]:
    connector = granted(SQLAlchemyConnector())
    engine = _FakeEngine(dialect_name)
    connector._engine = engine  # noqa: SLF001
    return connector, engine.conn


# ---------------------------------------------------------------------------
# The property that was missing: nothing is ever committed
# ---------------------------------------------------------------------------


def test_mssql_never_commits_a_statement_that_succeeded() -> None:
    """The success path is the one that mattered.

    A statement that fails was never committed anyway; the gap was a statement
    that *worked* and was therefore made permanent by `begin()`.
    """
    connector, conn = _mssql()

    result = connector.execute_query("SELECT 1")

    assert result.error is None, "the query itself must still succeed"
    assert conn.commits == 0
    assert conn.rollbacks == 1


def test_mssql_rolls_back_when_the_statement_fails_too() -> None:
    connector, conn = _mssql()

    def _boom(clause: Any) -> _Result:
        conn.statements.append(str(clause))
        raise RuntimeError("driver said no")

    conn.execute = _boom  # type: ignore[method-assign]
    result = connector.execute_query("SELECT 1")

    assert result.error is not None
    assert conn.commits == 0
    assert conn.rollbacks == 1


@pytest.mark.parametrize("dialect", ["mysql", "mariadb", "postgresql", "sqlite", "oracle"])
def test_the_generic_connector_never_commits(dialect: str) -> None:
    """Whatever the URL turns out to be. This connector serves MySQL and every
    engine nobody wrote a dedicated class for, so the guard cannot depend on
    recognising the dialect."""
    connector, conn = _generic(dialect)

    result = connector.execute_query("SELECT 1")

    assert result.error is None, dialect
    assert conn.commits == 0, dialect
    assert conn.rollbacks == 1, dialect


# ---------------------------------------------------------------------------
# The read-only transaction, where the dialect has one
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dialect", ["postgresql", "redshift"])
def test_the_transaction_is_made_read_only_before_the_query(dialect: str) -> None:
    """First statement in the transaction, which is the only place it can be set.

    Broader than any validator can be, because it applies to what the statement
    does rather than to how it is spelled: DML and DDL anywhere in the call graph.
    """
    connector, conn = _generic(dialect)

    connector.execute_query("SELECT 1")

    assert conn.statements[0] == "SET TRANSACTION READ ONLY", dialect
    assert conn.statements[-1] == "SELECT 1", dialect


@pytest.mark.parametrize("dialect", ["mysql", "mariadb"])
def test_mysql_is_left_to_the_rollback_and_not_sent_a_statement_that_errors(
    dialect: str,
) -> None:
    """Deliberately absent, and asserted so nobody adds it back untested.

    MySQL's form is `SET SESSION TRANSACTION READ ONLY`, which configures
    *subsequent* transactions and is refused with error 1568 inside an open one.
    SQLAlchemy emits BEGIN on the first execute, so by the time a connector can
    send anything the transaction is already open: issuing it would turn every
    MySQL query into an error, on top of a rollback that already prevents the
    write. `START TRANSACTION READ ONLY` is the right statement and is not
    something a caller can issue while SQLAlchemy owns the transaction.
    """
    connector, conn = _generic(dialect)

    connector.execute_query("SELECT 1")

    assert not any("READ ONLY" in statement for statement in conn.statements), (
        f"{dialect}: a read-only statement was sent that MySQL refuses inside a transaction"
    )
    assert conn.rollbacks == 1, f"{dialect}: the rollback is the whole guard here"


def test_mssql_is_not_sent_a_read_only_statement() -> None:
    """SQL Server has no read-only transaction to ask for --
    `ApplicationIntent=ReadOnly` is availability-group routing, not a permission.
    Its DDL is transactional, so the rollback undoes a `DROP` as well as DML."""
    connector, conn = _mssql()

    connector.execute_query("SELECT 1")

    assert conn.statements == ["SELECT 1"]
    assert conn.rollbacks == 1


class _RollbackFails(_Connection):
    """A connection whose rollback raises, as one dropped mid-statement would."""

    def rollback(self) -> None:
        self.rollbacks += 1
        raise RuntimeError("connection is closed")


def _with_failing_rollback(
    connector: MSSQLConnector | SQLAlchemyConnector, dialect_name: str
) -> _Connection:
    engine = _FakeEngine(dialect_name)
    engine.conn = _RollbackFails(dialect_name)
    connector._engine = engine  # noqa: SLF001
    return engine.conn


@pytest.mark.parametrize(
    ("make", "dialect"),
    [
        (lambda: granted(MSSQLConnector()), "mssql"),
        (lambda: granted(SQLAlchemyConnector()), "postgresql"),
    ],
    ids=["mssql", "sqlalchemy"],
)
def test_a_failing_rollback_does_not_become_the_result(
    make: Any, dialect: str, caplog: pytest.LogCaptureFixture
) -> None:
    """The rollback runs in a `finally`, so raising there would rewrite the outcome.

    A connection dropped after a cancelled or timed-out statement is the case
    that actually produces this. On the success path it would report a query
    that worked as an error; on the failure path it would replace the driver's
    message -- the one the caller needs to see -- with the rollback's.

    Nothing is committed either way, which is why swallowing it is safe rather
    than merely convenient: the connection is reset when the pool takes it back,
    and a connection that cannot be reached has no transaction left to commit.
    """
    import logging

    connector = make()
    conn = _with_failing_rollback(connector, dialect)

    with caplog.at_level(logging.WARNING):
        result = connector.execute_query("SELECT 1")

    assert conn.rollbacks == 1, "the rollback was not attempted"
    assert result.error is None, (
        f"a failing rollback was reported as the query's error: {result.error}"
    )
    assert any("rollback" in r.getMessage().lower() for r in caplog.records), (
        "the rollback failure was swallowed without a trace"
    )
