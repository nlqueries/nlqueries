"""
Whether feedback nobody can be held to becomes a trusted example (SEC-10).

Promotion turns a (question, SQL) pair into an exemplar that
`prompt_assembly` retrieves -- filtered on `verified == True` -- and puts in
front of the model for every later question on that agent. What qualifies a
record for promotion is its rating, and the rating is supplied by whoever
submitted the record.

Over the MCP transport that is anyone who can reach the server, because it has
no authentication (SEC-05). So a party with no account could submit a pair,
rate it themselves, and have it steer generation for everybody once an operator
ran the promotion.

Provenance is recorded on the record and promotion is limited to sources that
name someone, unless an operator asks for the rest explicitly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from nlqueries.feedback.models import (
    ATTRIBUTED_SOURCES,
    SOURCE_API,
    SOURCE_CLI,
    SOURCE_MCP,
    SOURCE_UNKNOWN,
    QueryFeedback,
)
from nlqueries.feedback.promoter import promote_feedback
from nlqueries.feedback.store import load_feedback, record_feedback

AGENT = "agent-under-test"


@pytest.fixture(autouse=True)
def _isolated_feedback_dir(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr("nlqueries.config.FEEDBACK_DIR", tmp_path / "feedback")
    return tmp_path


def _write(rating: str, source: str, sql: str = "SELECT 1") -> None:
    record_feedback(
        QueryFeedback(
            question=f"question for {source} {sql}",
            generated_sql=sql,
            rating=rating,
            agent_id=AGENT,
            source=source,
        )
    )


def _promoted() -> list[str]:
    """The SQL a real run would promote, via the dry run that shares its
    selection."""
    pending = promote_feedback(AGENT, dry_run=True)
    assert isinstance(pending, list)
    return [p["sql"] for p in pending]


class TestProvenanceIsRecorded:
    def test_a_source_survives_the_round_trip(self) -> None:
        _write("up", SOURCE_CLI)

        assert load_feedback(AGENT)[0].source == SOURCE_CLI

    def test_a_record_written_before_this_field_loads_as_unknown(self, tmp_path: Path) -> None:
        """Existing feedback files have no source. Unknown, not trusted."""
        directory = tmp_path / "feedback"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{AGENT}.jsonl").write_text(
            json.dumps(
                {
                    "question": "an older record",
                    "generated_sql": "SELECT 1",
                    "rating": "up",
                    "agent_id": AGENT,
                    "timestamp": "2026-01-01T00:00:00+00:00",
                }
            )
            + "\n",
            encoding="utf-8",
        )

        assert load_feedback(AGENT)[0].source == SOURCE_UNKNOWN

    def test_the_mcp_transport_is_not_an_attributed_source(self) -> None:
        """The property the gate depends on, asserted directly so that adding
        it to the set is a deliberate act and not a passing test."""
        assert SOURCE_MCP not in ATTRIBUTED_SOURCES
        assert SOURCE_UNKNOWN not in ATTRIBUTED_SOURCES
        assert SOURCE_CLI in ATTRIBUTED_SOURCES
        assert SOURCE_API in ATTRIBUTED_SOURCES


class TestPromotionGate:
    def test_anonymous_feedback_is_not_promoted(self) -> None:
        """SEC-10. The submitter chose both the pair and the rating."""
        _write("up", SOURCE_MCP, sql="SELECT secrets FROM payroll")

        assert _promoted() == []

    def test_feedback_with_no_recorded_origin_is_not_promoted(self) -> None:
        _write("up", SOURCE_UNKNOWN, sql="SELECT 1")

        assert _promoted() == []

    def test_attributed_feedback_is_still_promoted(self) -> None:
        """The control. A gate that refused everything would also pass the test
        above."""
        _write("up", SOURCE_CLI, sql="SELECT 2")

        assert _promoted() == ["SELECT 2"]

    def test_an_operator_can_promote_anonymous_feedback_deliberately(self) -> None:
        _write("up", SOURCE_MCP, sql="SELECT 3")

        pending = promote_feedback(AGENT, dry_run=True, include_unattributed=True)

        assert isinstance(pending, list)
        assert [p["sql"] for p in pending] == ["SELECT 3"]

    def test_a_negative_rating_is_still_never_promoted(self) -> None:
        _write("down", SOURCE_CLI, sql="SELECT 4")

        assert _promoted() == []

    def test_the_gate_does_not_change_which_attributed_records_qualify(self) -> None:
        """Mixed input: only the attributed positive one comes through."""
        _write("up", SOURCE_CLI, sql="SELECT 5")
        _write("up", SOURCE_MCP, sql="SELECT 6")
        _write("down", SOURCE_CLI, sql="SELECT 7")
        _write("up", SOURCE_UNKNOWN, sql="SELECT 8")

        assert _promoted() == ["SELECT 5"]
