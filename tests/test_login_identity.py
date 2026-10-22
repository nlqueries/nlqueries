"""Over-privileged logins for MySQL, SQL Server and Snowflake.

Each engine's grants are read into a :class:`LoginIdentity` that ``nlqueries
health`` reports, as the PostgreSQL connector's identity already is. These run
against fake catalog results: a read-only login reports nothing, a login with
write, DDL or administrative rights is flagged, and a check that cannot
complete says so rather than passing.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

import pytest
from nlqueries.connectors import mssql_identity, mysql_identity, snowflake_identity
from nlqueries.connectors.login_identity import LoginIdentity


class _Cursor:
    """A DBAPI cursor that answers each statement from a script."""

    def __init__(self, answers: dict[str, Any], description: Any = None) -> None:
        self._answers = answers
        self._rows: list[tuple[Any, ...]] = []
        self.description = description
        self.executed: list[str] = []
        self.closed = False

    def execute(self, sql: str) -> None:
        self.executed.append(sql)
        for start, rows in self._answers.items():
            if sql.startswith(start):
                if isinstance(rows, Exception):
                    raise rows
                self._rows = list(rows)
                return
        raise AssertionError(f"unscripted statement: {sql}")

    def fetchone(self) -> Any:
        return self._rows[0] if self._rows else None

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._rows

    def close(self) -> None:
        self.closed = True


def _connection(cursor: _Cursor) -> Any:
    return SimpleNamespace(cursor=lambda: cursor)


class _UnbufferedCursor(_Cursor):
    """mysql-connector-python's default cursor: a result has to be read to its
    end before the next statement, and ``fetchone`` does not read the end even
    of a one-row result."""

    def __init__(self, answers: dict[str, Any]) -> None:
        super().__init__(answers)
        self._unread = False

    def execute(self, sql: str) -> None:
        if self._unread:
            raise RuntimeError("Unread result found")
        super().execute(sql)
        self._unread = True

    def fetchall(self) -> list[tuple[Any, ...]]:
        self._unread = False
        return super().fetchall()


# --- the shared record ----------------------------------------------------------


def test_a_login_with_no_concerns_is_least_privilege() -> None:
    identity = LoginIdentity("mysql", "reader@%", "shop")
    assert identity.is_least_privilege
    assert identity.summary() == "reader@%@shop: least privilege"


def test_what_could_not_be_read_is_reported_with_the_result() -> None:
    identity = LoginIdentity("mssql", "reader", "shop", unread=("server roles (denied)",))
    assert identity.summary() == (
        "reader@shop: nothing beyond reading in what was read (not read: server roles (denied))"
    )
    assert not identity.is_least_privilege  # a caveat, not a pass


def test_health_warns_on_a_check_read_only_in_part() -> None:
    from nlqueries.cli.main import _identity_check

    partial = LoginIdentity("mssql", "reader", "shop", unread=("server roles (denied)",))
    assert _identity_check("db", partial).status == "warn"
    assert _identity_check("db", LoginIdentity("mssql", "reader", "shop")).status == "ok"


def test_an_undetermined_check_is_not_least_privilege() -> None:
    identity = LoginIdentity.undetermined("mysql", "access denied")
    assert not identity.is_least_privilege
    assert identity.summary() == "identity could not be determined: access denied"


# --- MySQL ----------------------------------------------------------------------


class TestMySQL:
    def test_a_read_only_login_reports_nothing(self) -> None:
        grants = [
            "GRANT USAGE ON *.* TO `reader`@`%`",
            "GRANT SELECT, SHOW VIEW ON `shop`.* TO `reader`@`%`",
            "GRANT SELECT (`id`, `name`) ON `shop`.`customers` TO `reader`@`%`",
        ]
        assert mysql_identity.concerns_from_grants(grants) == ()

    @pytest.mark.parametrize(
        ("grant", "expected"),
        [
            (
                "GRANT SELECT, INSERT, UPDATE ON `shop`.* TO `app`@`%`",
                "holds INSERT, UPDATE on `shop`.*",
            ),
            (
                "GRANT CREATE, DROP, ALTER ON `shop`.* TO `app`@`%`",
                "holds CREATE, DROP, ALTER on `shop`.*",
            ),
            ("GRANT FILE ON *.* TO `app`@`%`", "holds FILE on *.*"),
            ("GRANT ALL PRIVILEGES ON *.* TO `root`@`localhost`", "holds ALL PRIVILEGES on *.*"),
            (
                "GRANT SYSTEM_VARIABLES_ADMIN ON *.* TO `app`@`%`",
                "holds SYSTEM_VARIABLES_ADMIN on *.*",
            ),
            ("GRANT EXECUTE ON `shop`.* TO `app`@`%`", "holds EXECUTE on `shop`.*"),
        ],
    )
    def test_write_ddl_and_admin_rights_are_flagged(self, grant: str, expected: str) -> None:
        assert mysql_identity.concerns_from_grants([grant]) == (expected,)

    def test_grant_option_is_flagged_even_on_read_privileges(self) -> None:
        concerns = mysql_identity.concerns_from_grants(
            ["GRANT SELECT ON `shop`.* TO `reader`@`%` WITH GRANT OPTION"]
        )
        assert concerns == ("can grant its privileges on `shop`.* to others",)

    def test_a_role_grant_is_flagged_because_it_is_not_expanded(self) -> None:
        concerns = mysql_identity.concerns_from_grants(["GRANT `analyst`@`%` TO `reader`@`%`"])
        assert concerns == ("is granted `analyst`@`%`, whose privileges this check does not read",)

    def test_inspect_reads_the_login_and_its_grants(self) -> None:
        cursor = _Cursor(
            {
                "SELECT CURRENT_USER()": [("app@%", "shop")],
                "SHOW GRANTS": [("GRANT SELECT, DELETE ON `shop`.* TO `app`@`%`",)],
            }
        )
        identity = mysql_identity.inspect_identity(_connection(cursor))
        assert identity.user == "app@%" and identity.database == "shop"
        assert identity.concerns == ("holds DELETE on `shop`.*",)
        assert cursor.closed

    def test_an_unbuffered_cursor_reads_both_statements(self) -> None:
        cursor = _UnbufferedCursor(
            {
                "SELECT CURRENT_USER()": [("app@%", "shop")],
                "SHOW GRANTS": [("GRANT SELECT ON `shop`.* TO `app`@`%`",)],
            }
        )
        identity = mysql_identity.inspect_identity(_connection(cursor))
        assert identity.undetermined_reason is None
        assert identity.is_least_privilege

    def test_a_failed_read_is_undetermined_not_clean(self) -> None:
        cursor = _Cursor(
            {"SELECT CURRENT_USER()": [("app@%", "shop")], "SHOW GRANTS": RuntimeError("denied")}
        )
        identity = mysql_identity.inspect_identity(_connection(cursor))
        assert identity.undetermined_reason == "denied"
        assert not identity.is_least_privilege


# --- SQL Server -----------------------------------------------------------------


_MSSQL_READER = {
    "server_roles": ["public"],
    "database_roles": ["public", "db_datareader"],
    "database_permissions": ["CONNECT", "SELECT", "VIEW DEFINITION", "SHOWPLAN"],
    "server_permissions": ["CONNECT SQL", "VIEW ANY DATABASE"],
}


class TestSQLServer:
    def test_a_read_only_login_reports_nothing(self) -> None:
        assert mssql_identity.concerns_from(**_MSSQL_READER) == ()

    def test_what_public_holds_by_default_is_not_flagged(self) -> None:
        """Every user's effective permissions since SQL Server 2016."""
        concerns = mssql_identity.concerns_from(
            **{
                **_MSSQL_READER,
                "database_permissions": [
                    *_MSSQL_READER["database_permissions"],
                    "VIEW ANY COLUMN ENCRYPTION KEY DEFINITION",
                    "VIEW ANY COLUMN MASTER KEY DEFINITION",
                ],
            }
        )
        assert concerns == ()

    def test_what_a_listed_view_permission_implies_is_not_flagged(self) -> None:
        """SQL Server 2022 reports the narrower permissions a grant implies."""
        concerns = mssql_identity.concerns_from(
            **{
                **_MSSQL_READER,
                "database_permissions": [
                    *_MSSQL_READER["database_permissions"],
                    "VIEW DATABASE STATE",
                    "VIEW DATABASE SECURITY STATE",
                    "VIEW SECURITY DEFINITION",
                ],
                "server_permissions": [
                    *_MSSQL_READER["server_permissions"],
                    "VIEW SERVER STATE",
                    "VIEW SERVER SECURITY STATE",
                ],
            }
        )
        assert concerns == ()

    def test_reading_the_server_error_log_is_reported(self) -> None:
        """Not on the list, since nothing listed is known to imply it."""
        concerns = mssql_identity.concerns_from(
            **{
                **_MSSQL_READER,
                "server_permissions": [*_MSSQL_READER["server_permissions"], "VIEW ANY ERROR LOG"],
            }
        )
        assert concerns == ("holds VIEW ANY ERROR LOG on the server",)

    def test_a_long_list_of_permissions_is_cut_short(self) -> None:
        many = [f"ALTER ANY THING {n:02}" for n in range(12)]
        (concern,) = mssql_identity.concerns_from(
            **{
                **_MSSQL_READER,
                "database_permissions": [*_MSSQL_READER["database_permissions"], *many],
            }
        )
        assert concern == f"holds {', '.join(many[:8])} and 4 more on the database"

    def test_control_is_named_for_what_it_implies(self) -> None:
        """A db_owner's effective permissions run past a hundred; CONTROL says it."""
        owner = ["CONTROL", "ALTER", "DELETE", "INSERT", *(f"ALTER ANY X{n}" for n in range(100))]
        server = ["CONTROL SERVER", "SHUTDOWN", "ALTER ANY LOGIN"]
        concerns = mssql_identity.concerns_from(
            **{
                **_MSSQL_READER,
                "database_permissions": [*_MSSQL_READER["database_permissions"], *owner],
                "server_permissions": [*_MSSQL_READER["server_permissions"], *server],
            }
        )
        assert concerns == (
            "holds CONTROL and 103 more on the database",
            "holds CONTROL SERVER and 2 more on the server",
        )

    def test_server_and_database_roles_are_flagged(self) -> None:
        concerns = mssql_identity.concerns_from(
            **{
                **_MSSQL_READER,
                "server_roles": ["public", "sysadmin"],
                "database_roles": ["public", "db_owner", "db_datawriter"],
            }
        )
        assert concerns == (
            "is a member of server role sysadmin",
            "is a member of database role db_datawriter, db_owner",
        )

    def test_effective_permissions_catch_what_roles_do_not_name(self) -> None:
        """A custom role or a direct grant shows up only in the permissions."""
        concerns = mssql_identity.concerns_from(
            **{
                **_MSSQL_READER,
                "database_permissions": [*_MSSQL_READER["database_permissions"], "INSERT", "ALTER"],
                "server_permissions": [
                    *_MSSQL_READER["server_permissions"],
                    "ADMINISTER BULK OPERATIONS",
                ],
            }
        )
        assert concerns == (
            "holds ALTER, INSERT on the database",
            "holds ADMINISTER BULK OPERATIONS on the server",
        )

    def test_inspect_reads_roles_and_permissions(self) -> None:
        cursor = _Cursor(
            {
                mssql_identity.LOGIN_SQL: [("app", "shop")],
                mssql_identity.SERVER_ROLES_SQL: [("public",)],
                mssql_identity.DATABASE_ROLES_SQL: [("public",), ("db_datawriter",)],
                mssql_identity.DATABASE_PERMISSIONS_SQL: [("CONNECT",), ("SELECT",), ("UPDATE",)],
                mssql_identity.SERVER_PERMISSIONS_SQL: [("CONNECT SQL",)],
            }
        )
        identity = mssql_identity.inspect_identity(_connection(cursor))
        assert identity.user == "app" and identity.database == "shop"
        assert identity.concerns == (
            "is a member of database role db_datawriter",
            "holds UPDATE on the database",
        )

    def test_no_permissions_back_is_undetermined(self) -> None:
        cursor = _Cursor(
            {
                mssql_identity.LOGIN_SQL: [("app", "shop")],
                mssql_identity.SERVER_ROLES_SQL: [],
                mssql_identity.DATABASE_ROLES_SQL: [],
                mssql_identity.DATABASE_PERMISSIONS_SQL: [],
                mssql_identity.SERVER_PERMISSIONS_SQL: [],
            }
        )
        identity = mssql_identity.inspect_identity(_connection(cursor))
        assert identity.undetermined_reason is not None

    def test_an_unreadable_server_scope_leaves_the_database_result(self) -> None:
        """Azure SQL, from a user database: the server reads can fail alone."""
        cursor = _Cursor(
            {
                mssql_identity.LOGIN_SQL: [("app", "shop")],
                mssql_identity.SERVER_ROLES_SQL: RuntimeError("no server_principals here"),
                mssql_identity.DATABASE_ROLES_SQL: [("public",)],
                mssql_identity.DATABASE_PERMISSIONS_SQL: [("CONNECT",), ("INSERT",)],
                mssql_identity.SERVER_PERMISSIONS_SQL: RuntimeError("no server scope"),
            }
        )
        identity = mssql_identity.inspect_identity(_connection(cursor))
        assert identity.undetermined_reason is None
        assert identity.concerns == ("holds INSERT on the database",)
        assert identity.unread == (
            "server roles (no server_principals here)",
            "server permissions (no server scope)",
        )

    def test_each_server_read_stands_on_its_own(self) -> None:
        cursor = _Cursor(
            {
                mssql_identity.LOGIN_SQL: [("app", "shop")],
                mssql_identity.SERVER_ROLES_SQL: RuntimeError("denied"),
                mssql_identity.DATABASE_ROLES_SQL: [("public",)],
                mssql_identity.DATABASE_PERMISSIONS_SQL: [("CONNECT",)],
                mssql_identity.SERVER_PERMISSIONS_SQL: [("CONNECT SQL",), ("ALTER ANY LOGIN",)],
            }
        )
        identity = mssql_identity.inspect_identity(_connection(cursor))
        assert identity.concerns == ("holds ALTER ANY LOGIN on the server",)
        assert identity.unread == ("server roles (denied)",)

    def test_an_unreadable_database_scope_is_undetermined(self) -> None:
        cursor = _Cursor(
            {
                mssql_identity.LOGIN_SQL: [("app", "shop")],
                mssql_identity.SERVER_ROLES_SQL: [("public",)],
                mssql_identity.DATABASE_ROLES_SQL: RuntimeError("denied"),
                mssql_identity.DATABASE_PERMISSIONS_SQL: [("CONNECT",)],
                mssql_identity.SERVER_PERMISSIONS_SQL: [("CONNECT SQL",)],
            }
        )
        identity = mssql_identity.inspect_identity(_connection(cursor))
        assert identity.undetermined_reason == "denied"

    def test_a_contained_user_is_named(self) -> None:
        """No server login, so ``SUSER_SNAME()`` is NULL; the user name stands in."""
        assert "COALESCE(SUSER_SNAME(), USER_NAME())" in mssql_identity.LOGIN_SQL
        cursor = _Cursor(
            {
                mssql_identity.LOGIN_SQL: [(None, "shop")],
                mssql_identity.SERVER_ROLES_SQL: [],
                mssql_identity.DATABASE_ROLES_SQL: [("public",)],
                mssql_identity.DATABASE_PERMISSIONS_SQL: [("CONNECT",)],
                mssql_identity.SERVER_PERMISSIONS_SQL: [],
            }
        )
        identity = mssql_identity.inspect_identity(_connection(cursor))
        assert "None" not in identity.summary()


# --- Snowflake ------------------------------------------------------------------


def _grants(table: dict[str, list[tuple[str, str, str]]]) -> Any:
    return lambda role: table.get(role, [])


class TestSnowflake:
    def test_a_read_only_role_reports_nothing(self) -> None:
        roles, grants, truncated = snowflake_identity.walk_roles(
            "READER",
            _grants(
                {
                    "READER": [
                        ("USAGE", "WAREHOUSE", "WH"),
                        ("USAGE", "DATABASE", "SHOP"),
                        ("SELECT", "TABLE", "SHOP.PUBLIC.ORDERS"),
                    ]
                }
            ),
        )
        assert snowflake_identity.concerns_from(roles, grants, truncated) == ()

    def test_privileges_through_an_inherited_role_are_flagged(self) -> None:
        roles, grants, truncated = snowflake_identity.walk_roles(
            "READER",
            _grants(
                {
                    "READER": [
                        ("USAGE", "ROLE", "LOADER"),
                        ("SELECT", "TABLE", "SHOP.PUBLIC.ORDERS"),
                    ],
                    "LOADER": [
                        ("INSERT", "TABLE", "SHOP.PUBLIC.ORDERS"),
                        ("INSERT", "TABLE", "SHOP.PUBLIC.ITEMS"),
                    ],
                }
            ),
        )
        assert roles == ["READER", "LOADER"]
        assert snowflake_identity.concerns_from(roles, grants, truncated) == (
            "holds INSERT on 2 table(s)",
        )

    def test_an_admin_role_in_reach_is_flagged(self) -> None:
        roles, grants, truncated = snowflake_identity.walk_roles(
            "ANALYST", _grants({"ANALYST": [("USAGE", "ROLE", "SYSADMIN")]})
        )
        assert snowflake_identity.concerns_from(roles, grants, truncated) == ("can use SYSADMIN",)

    def test_the_walk_survives_a_cycle_and_stops_at_its_cap(self, monkeypatch) -> None:
        cycle = _grants({"A": [("USAGE", "ROLE", "B")], "B": [("USAGE", "ROLE", "A")]})
        assert snowflake_identity.walk_roles("A", cycle)[0] == ["A", "B"]

        monkeypatch.setattr(snowflake_identity, "MAX_ROLES", 3)
        chain = _grants({f"R{i}": [("USAGE", "ROLE", f"R{i + 1}")] for i in range(10)})
        roles, grants, truncated = snowflake_identity.walk_roles("R0", chain)
        assert roles == ["R0", "R1", "R2"] and truncated
        assert snowflake_identity.concerns_from(roles, grants, truncated) == (
            "inherits more than 3 roles; the rest were not read",
        )

    def test_inspect_walks_from_the_current_role(self) -> None:
        description = [("created_on",), ("privilege",), ("granted_on",), ("name",)]
        cursor = _Cursor(
            {
                "SELECT CURRENT_USER()": [("APP", "READER", "SHOP")],
                'SHOW GRANTS TO ROLE "READER"': [("t", "USAGE", "ROLE", "WRITER")],
                'SHOW GRANTS TO ROLE "WRITER"': [("t", "DELETE", "TABLE", "SHOP.PUBLIC.ORDERS")],
            },
            description=description,
        )
        identity = snowflake_identity.inspect_identity(_connection(cursor))
        assert identity.user == "APP (role READER)" and identity.database == "SHOP"
        assert identity.concerns == ("holds DELETE on 1 table(s)",)

    def test_a_role_name_is_quoted(self) -> None:
        assert snowflake_identity._quoted('we"ird') == '"we""ird"'


# --- the connectors expose it ---------------------------------------------------


def test_the_mssql_connector_reads_its_identity_on_connect(monkeypatch) -> None:
    import sys
    from unittest.mock import MagicMock

    from nlqueries.connectors import mssql as mssql_module

    # pymssql is an optional extra; a stand-in is enough to reach the wiring.
    monkeypatch.setitem(sys.modules, "pymssql", MagicMock(__version__="2.4.5"))
    monkeypatch.setattr(mssql_module, "create_engine", MagicMock())
    seen: list[Any] = []
    monkeypatch.setattr(
        mssql_module, "watch_engine", lambda engine, inspect, store: seen.append((inspect, store))
    )
    connector = mssql_module.MSSQLConnector()
    connector.connect({"host": "db", "database": "shop", "user": "u", "password": "p"})

    ((inspect, store),) = seen
    assert inspect is mssql_identity.inspect_identity
    store(LoginIdentity("mssql", "u", "shop", ("holds INSERT on the database",)))
    assert connector.identity is not None and connector.identity.concerns


@pytest.mark.parametrize(
    ("url", "watched"),
    [("mysql+pymysql://u:p@db/shop", True), ("postgresql+psycopg2://u:p@db/shop", False)],
)
def test_the_generic_connector_reads_a_mysql_identity_only(
    monkeypatch, url: str, watched: bool
) -> None:
    from nlqueries.connectors import sqlalchemy_connector as sac

    seen: list[Any] = []
    monkeypatch.setattr(sac, "watch_engine", lambda engine, inspect, store: seen.append(inspect))
    monkeypatch.setattr(sac, "create_engine", lambda *a, **k: SimpleNamespace())
    sac.SQLAlchemyConnector().connect({"url": url})

    assert seen == ([mysql_identity.inspect_identity] if watched else [])


def test_the_snowflake_connector_reads_its_identity_once(monkeypatch) -> None:
    from nlqueries.connectors import snowflake as snowflake_module

    calls: list[Any] = []

    def inspect(connection: Any) -> LoginIdentity:
        calls.append(connection)
        return LoginIdentity("snowflake", "APP (role R)", "SHOP")

    monkeypatch.setattr(snowflake_module, "inspect_snowflake_identity", inspect)
    connector = snowflake_module.SnowflakeConnector()
    assert connector.identity is None  # nothing to read before connect

    connector._connection = object()
    first = connector.identity
    second = connector.identity
    assert first is second is not None
    assert len(calls) == 1


def test_the_snowflake_identity_does_not_wait_out_a_running_query(monkeypatch) -> None:
    from nlqueries.connectors import snowflake as snowflake_module

    monkeypatch.setattr(snowflake_module, "_IDENTITY_LOCK_WAIT_SECONDS", 0.05)
    monkeypatch.setattr(
        snowflake_module,
        "inspect_snowflake_identity",
        lambda connection: LoginIdentity("snowflake", "APP (role R)", "SHOP"),
    )
    connector = snowflake_module.SnowflakeConnector()
    connector._connection = object()

    connector._txn_lock.acquire()  # a query holds the session
    started = time.monotonic()
    try:
        busy = connector.identity
    finally:
        connector._txn_lock.release()
    assert time.monotonic() - started < 2  # the configured wait, not another
    assert busy is not None and busy.undetermined_reason is not None
    assert "busy" in busy.undetermined_reason

    # Not kept: once the session is free, the next read reads it.
    assert connector.identity == LoginIdentity("snowflake", "APP (role R)", "SHOP")


# --- the pool event, on a real engine --------------------------------------------


def test_the_watcher_reads_each_physical_connection_the_pool_opens() -> None:
    """The listener runs inside connection creation, so it is run here for real."""
    from nlqueries.connectors.login_identity import watch_engine
    from sqlalchemy import create_engine, text
    from sqlalchemy.pool import NullPool

    engine = create_engine("sqlite://", poolclass=NullPool)
    stored: list[LoginIdentity] = []

    def inspect(dbapi_connection: Any) -> LoginIdentity:
        # The raw DBAPI connection, usable for the reader's own statements.
        cursor = dbapi_connection.cursor()
        cursor.execute("SELECT 'reader', 'main'")
        user, database = cursor.fetchall()[0]
        cursor.close()
        return LoginIdentity("sqlite", user, database)

    watch_engine(engine, inspect, stored.append)
    for _ in range(2):
        with engine.connect() as connection:
            assert connection.execute(text("SELECT 1")).scalar() == 1

    assert stored == [LoginIdentity("sqlite", "reader", "main")] * 2
