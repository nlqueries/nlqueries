"""
The recorded capabilities of each connector.

The table in ``nlqueries.connectors.capabilities`` states what each connector
enforces. These tests keep it aligned with the connectors that actually exist,
so that a connector added later cannot be left undescribed.
"""

from __future__ import annotations

from nlqueries.connectors import CONNECTOR_REGISTRY
from nlqueries.connectors.capabilities import CAPABILITIES, for_dialect


def test_every_registered_connector_has_an_entry() -> None:
    """A connector added without an entry would report as unknown at runtime.

    Failing here instead names the omission at the point it is introduced.
    """
    missing = sorted(set(CONNECTOR_REGISTRY) - set(CAPABILITIES))

    assert not missing, f"no capabilities recorded for: {', '.join(missing)}"


def test_no_entry_describes_a_connector_that_does_not_exist() -> None:
    """An entry for a removed connector would state something about nothing."""
    unknown = sorted(set(CAPABILITIES) - set(CONNECTOR_REGISTRY))

    assert not unknown, f"capabilities recorded for absent connectors: {', '.join(unknown)}"


def test_an_unrecorded_dialect_returns_none() -> None:
    """None is reported as a failure by the health check: an absent entry is
    not a statement that the dialect is safe."""
    assert for_dialect("nothing-like-this") is None


def test_lookup_is_case_insensitive() -> None:
    assert for_dialect("PostgreSQL".lower()[:8]) is not None
    assert for_dialect("POSTGRES") is for_dialect("postgres")


class TestTheRecordedFacts:
    """These assert what the connectors do today.

    Changing a connector's behaviour should change its entry, and these fail
    until it does.
    """

    def test_the_three_verified_dialects_are_the_ones_with_tests(self) -> None:
        """`verified_here` means exercised against a *real* engine.

        MSSQL, Snowflake and the generic connector gained read-only mechanisms
        without joining this set, and that is correct: their tests assert the
        statements sent to a fake driver, because CI provisions neither SQL
        Server nor Snowflake nor MySQL. Redshift has been in the same position
        all along. Marking them verified would suppress the concern
        `nlqueries health` reports -- that the mechanism is not exercised by any
        test here -- which is true and worth an operator seeing.
        """
        verified = {name for name, c in CAPABILITIES.items() if c.verified_here}

        assert verified == {"postgres", "sqlite", "duckdb"}

    def test_the_dialects_that_enforce_read_only(self) -> None:
        """Everything except BigQuery applies something of its own.

        The mechanisms are not equivalent, which is the point of recording them
        as prose rather than a boolean: Postgres and Redshift refuse a write
        outright, SQLite and DuckDB open the file read-only, and MSSQL, Snowflake
        and the generic connector run in a transaction that is never committed.
        The last three undo a write rather than refusing it, and Snowflake's DDL
        is not transactional at all, so a CREATE there still stands.

        BigQuery is the one that records nothing, and deliberately: a query job
        cannot be rolled back, and the statement type is only readable after the
        job has run. Recording that as a mechanism would tell an operator they
        were protected when nothing was prevented.

        Only three are in the verified set, which is a different question --
        see the test above.
        """
        enforcing = {name for name, c in CAPABILITIES.items() if c.enforces_read_only}

        assert enforcing == {
            "postgres",
            "sqlite",
            "duckdb",
            "redshift",
            "mssql",
            "snowflake",
            "sqlalchemy",
        }
        assert not CAPABILITIES["bigquery"].enforces_read_only

    def test_redshift_enforces_both_controls(self) -> None:
        """Measured against Redshift Serverless: a write is refused with
        SQLSTATE 25006, and a query over budget is cancelled with 57014."""
        caps = CAPABILITIES["redshift"]

        assert caps.enforces_read_only
        assert caps.enforces_statement_timeout

    def test_redshift_is_not_recorded_as_verified_here(self) -> None:
        """No test in this repository reaches a cluster, so the entry still
        reports that its mechanism is unexercised by CI."""
        caps = CAPABILITIES["redshift"]

        assert not caps.verified_here
        assert any("not exercised by any test" in c for c in caps.concerns)

    def test_a_dialect_without_read_only_says_so_first(self) -> None:
        """BigQuery is the only one left that has to lead with this.

        The other three gained a mechanism, so their first concern is now
        something else -- and saying "no read-only mechanism" for them would be
        false. What they must still report is that the mechanism is unverified
        here, which the test below covers.
        """
        assert "no read-only mechanism" in CAPABILITIES["bigquery"].concerns[0]

    def test_a_dialect_with_an_unverified_mechanism_says_that_instead(self) -> None:
        """The concern an operator needs once a mechanism exists but no test in
        this repository reaches a real engine to prove it works."""
        for name in ("mssql", "snowflake", "sqlalchemy", "redshift"):
            caps = CAPABILITIES[name]
            assert caps.enforces_read_only, name
            assert any("not exercised by any test" in c for c in caps.concerns), name
            assert not any("no read-only mechanism" in c for c in caps.concerns), name

    def test_every_entry_states_what_the_operator_must_do(self) -> None:
        for name, caps in CAPABILITIES.items():
            assert caps.operator_requirement.strip(), name
