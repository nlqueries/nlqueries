"""
Classification of a PostgreSQL login's privileges.

These tests run without a database. The equivalent checks against a real server
are in ``tests/security/test_postgres_identity_lab.py``. This file covers the
classification rules, which are therefore verified in CI regardless of whether
Docker is available.
"""

from __future__ import annotations

from nlqueries.connectors.postgres_identity import (
    DANGEROUS_PREDEFINED_ROLES,
    PostgresIdentity,
    inspect_identity,
)


def _identity(**overrides: object) -> PostgresIdentity:
    """A least-privilege identity, unless a test says otherwise."""
    base: dict[str, object] = {
        "user": "sec_reader",
        "session_user": "sec_reader",
        "database": "shop",
        "search_path": "pg_catalog, app",
        "default_transaction_read_only": True,
        "is_superuser": False,
        "may_create_database": False,
        "may_create_role": False,
        "bypasses_row_level_security": False,
        "may_replicate": False,
        "predefined_roles": (),
    }
    base.update(overrides)
    return PostgresIdentity(**base)  # type: ignore[arg-type]


def test_a_least_privilege_identity_has_no_concerns() -> None:
    assert _identity().is_least_privilege
    assert _identity().concerns == ()


def test_an_undetermined_identity_is_not_least_privilege() -> None:
    """An incomplete check must be distinguishable from a completed one that
    found no concerns."""
    unknown = PostgresIdentity.undetermined("connection reset")

    assert not unknown.is_least_privilege
    assert unknown.undetermined_reason == "connection reset"
    assert "could not be determined" in unknown.summary()


def test_inspect_identity_reports_a_failure_instead_of_raising() -> None:
    """A failure of the check must not prevent the connector from opening."""

    class Unusable:
        def cursor(self) -> object:
            raise RuntimeError("server closed the connection unexpectedly")

    identity = inspect_identity(Unusable())

    assert not identity.is_least_privilege
    assert "server closed the connection" in str(identity.undetermined_reason)


def test_inspect_identity_reports_a_missing_role_row() -> None:
    """A `current_user` with no `pg_roles` row is reported as undetermined
    rather than as an absence of findings."""

    class NoRow:
        def cursor(self) -> object:
            return self

        def execute(self, _sql: str) -> None:
            return None

        def fetchone(self) -> None:
            return None

        def close(self) -> None:
            return None

    identity = inspect_identity(NoRow())

    assert not identity.is_least_privilege
    assert "pg_roles" in str(identity.undetermined_reason)


class TestConcerns:
    def test_superuser(self) -> None:
        assert "superuser" in _identity(is_superuser=True).concerns[0]

    def test_bypassrls(self) -> None:
        found = _identity(bypasses_row_level_security=True).concerns
        assert any("BYPASSRLS" in c for c in found)

    def test_createrole(self) -> None:
        found = _identity(may_create_role=True).concerns
        assert any("CREATEROLE" in c for c in found)

    def test_replication(self) -> None:
        found = _identity(may_replicate=True).concerns
        assert any("REPLICATION" in c for c in found)

    def test_each_dangerous_predefined_role_is_reported(self) -> None:
        for role in DANGEROUS_PREDEFINED_ROLES:
            found = _identity(predefined_roles=(role,)).concerns
            assert any(role in c for c in found), role

    def test_pg_database_owner_is_not_a_concern(self) -> None:
        """`pg_database_owner` is held by the owner of any database and
        confers no additional privilege."""
        assert _identity(predefined_roles=("pg_database_owner",)).is_least_privilege

    def test_an_unknown_pg_role_is_not_a_concern(self) -> None:
        """The query enumerates every `pg_` role the login belongs to,
        including those added by later server versions. Only the roles in
        `DANGEROUS_PREDEFINED_ROLES` are reported as concerns."""
        assert _identity(predefined_roles=("pg_monitor", "pg_signal_backend")).is_least_privilege

    def test_concerns_are_ordered_with_superuser_first(self) -> None:
        found = _identity(is_superuser=True, may_create_role=True).concerns
        assert "superuser" in found[0]
