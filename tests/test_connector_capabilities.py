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
        verified = {name for name, c in CAPABILITIES.items() if c.verified_here}

        assert verified == {"postgres", "sqlite", "duckdb"}

    def test_only_the_verified_dialects_enforce_read_only(self) -> None:
        """No other connector applies a read-only mechanism of its own."""
        enforcing = {name for name, c in CAPABILITIES.items() if c.enforces_read_only}

        assert enforcing == {"postgres", "sqlite", "duckdb"}

    def test_redshift_enforces_neither_control(self) -> None:
        """Its timeout_seconds is accepted for interface parity and ignored."""
        caps = CAPABILITIES["redshift"]

        assert not caps.enforces_read_only
        assert not caps.enforces_statement_timeout

    def test_a_dialect_without_read_only_says_so_first(self) -> None:
        for name in ("mssql", "redshift", "snowflake", "bigquery", "sqlalchemy"):
            assert "no read-only mechanism" in CAPABILITIES[name].concerns[0], name

    def test_every_entry_states_what_the_operator_must_do(self) -> None:
        for name, caps in CAPABILITIES.items():
            assert caps.operator_requirement.strip(), name
