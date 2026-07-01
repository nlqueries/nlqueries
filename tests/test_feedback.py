"""Tests for nlqueries.feedback — round-trip JSONL read/write."""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner
from nlqueries.feedback.models import QueryFeedback
from nlqueries.feedback.store import load_feedback, record_feedback

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fb(
    agent_id: str = "agent1",
    question: str = "How many orders?",
    generated_sql: str = "SELECT COUNT(*) FROM orders",
    rating: str = "up",
    corrected_sql: str | None = None,
) -> QueryFeedback:
    return QueryFeedback(
        question=question,
        generated_sql=generated_sql,
        rating=rating,
        agent_id=agent_id,
        corrected_sql=corrected_sql,
        timestamp=datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC),
    )


# ---------------------------------------------------------------------------
# QueryFeedback model
# ---------------------------------------------------------------------------


class TestQueryFeedbackModel:
    def test_valid_up_rating(self) -> None:
        fb = _make_fb(rating="up")
        assert fb.rating == "up"

    def test_valid_down_rating(self) -> None:
        fb = _make_fb(rating="down")
        assert fb.rating == "down"

    def test_invalid_rating_raises(self) -> None:
        with pytest.raises(ValueError, match="rating must be"):
            QueryFeedback(
                question="q",
                generated_sql="SELECT 1",
                rating="meh",
                agent_id="a1",
            )

    def test_corrected_sql_defaults_to_none(self) -> None:
        fb = QueryFeedback(
            question="q",
            generated_sql="SELECT 1",
            rating="up",
            agent_id="a1",
        )
        assert fb.corrected_sql is None

    def test_timestamp_defaults_to_utc_now(self) -> None:
        before = datetime.now(UTC)
        fb = QueryFeedback(
            question="q",
            generated_sql="SELECT 1",
            rating="up",
            agent_id="a1",
        )
        after = datetime.now(UTC)
        assert before <= fb.timestamp <= after


# ---------------------------------------------------------------------------
# record_feedback / load_feedback — round-trip
# ---------------------------------------------------------------------------


class TestFeedbackStore:
    def test_record_then_load_round_trip(self) -> None:
        fb = _make_fb()
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("nlqueries.feedback.store.config") as mock_cfg,
        ):
            mock_cfg.FEEDBACK_DIR = Path(tmpdir)
            record_feedback(fb)
            loaded = load_feedback("agent1")

        assert len(loaded) == 1
        result = loaded[0]
        assert result.question == fb.question
        assert result.generated_sql == fb.generated_sql
        assert result.rating == fb.rating
        assert result.agent_id == fb.agent_id
        assert result.corrected_sql is None
        assert result.timestamp == fb.timestamp

    def test_corrected_sql_round_trips(self) -> None:
        fb = _make_fb(corrected_sql="SELECT COUNT(*) FROM orders WHERE status='open'")
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("nlqueries.feedback.store.config") as mock_cfg,
        ):
            mock_cfg.FEEDBACK_DIR = Path(tmpdir)
            record_feedback(fb)
            loaded = load_feedback("agent1")

        assert loaded[0].corrected_sql == fb.corrected_sql

    def test_multiple_records_appended_in_order(self) -> None:
        records = [
            _make_fb(question=f"Question {i}", rating="up" if i % 2 == 0 else "down")
            for i in range(5)
        ]
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("nlqueries.feedback.store.config") as mock_cfg,
        ):
            mock_cfg.FEEDBACK_DIR = Path(tmpdir)
            for r in records:
                record_feedback(r)
            loaded = load_feedback("agent1")

        assert len(loaded) == 5
        for i, rec in enumerate(loaded):
            assert rec.question == f"Question {i}"

    def test_load_returns_empty_list_when_no_file(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("nlqueries.feedback.store.config") as mock_cfg,
        ):
            mock_cfg.FEEDBACK_DIR = Path(tmpdir)
            loaded = load_feedback("nonexistent_agent")
        assert loaded == []

    def test_separate_agents_stored_in_separate_files(self) -> None:
        fb_a = _make_fb(agent_id="agent_a", question="Q for A")
        fb_b = _make_fb(agent_id="agent_b", question="Q for B")
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("nlqueries.feedback.store.config") as mock_cfg,
        ):
            mock_cfg.FEEDBACK_DIR = Path(tmpdir)
            record_feedback(fb_a)
            record_feedback(fb_b)
            loaded_a = load_feedback("agent_a")
            loaded_b = load_feedback("agent_b")

        assert len(loaded_a) == 1
        assert loaded_a[0].question == "Q for A"
        assert len(loaded_b) == 1
        assert loaded_b[0].question == "Q for B"

    def test_agent_id_with_colons_sanitised_in_filename(self) -> None:
        fb = _make_fb(agent_id="postgres:localhost:mydb")
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("nlqueries.feedback.store.config") as mock_cfg,
        ):
            mock_cfg.FEEDBACK_DIR = Path(tmpdir)
            record_feedback(fb)
            files = list(Path(tmpdir).iterdir())

        assert len(files) == 1
        assert ":" not in files[0].name
        assert files[0].suffix == ".jsonl"

    def test_jsonl_file_has_one_record_per_line(self) -> None:
        records = [_make_fb(question=f"Q{i}") for i in range(3)]
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("nlqueries.feedback.store.config") as mock_cfg,
        ):
            mock_cfg.FEEDBACK_DIR = Path(tmpdir)
            for r in records:
                record_feedback(r)
            file_path = Path(tmpdir) / "agent1.jsonl"
            lines = [ln for ln in file_path.read_text().splitlines() if ln.strip()]

        assert len(lines) == 3
        for line in lines:
            parsed = json.loads(line)
            assert "question" in parsed
            assert "rating" in parsed

    def test_corrupt_line_skipped_silently(self) -> None:
        fb = _make_fb(question="Valid question")
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("nlqueries.feedback.store.config") as mock_cfg,
        ):
            mock_cfg.FEEDBACK_DIR = Path(tmpdir)
            record_feedback(fb)
            file_path = Path(tmpdir) / "agent1.jsonl"
            with file_path.open("a") as fh:
                fh.write("{corrupt json\n")
            loaded = load_feedback("agent1")

        assert len(loaded) == 1
        assert loaded[0].question == "Valid question"

    def test_down_rating_round_trips(self) -> None:
        fb = _make_fb(rating="down")
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("nlqueries.feedback.store.config") as mock_cfg,
        ):
            mock_cfg.FEEDBACK_DIR = Path(tmpdir)
            record_feedback(fb)
            loaded = load_feedback("agent1")

        assert loaded[0].rating == "down"

    def test_timestamp_preserved_as_iso_string(self) -> None:
        ts = datetime(2025, 3, 15, 9, 30, 0, tzinfo=UTC)
        fb = _make_fb()
        fb.timestamp = ts
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("nlqueries.feedback.store.config") as mock_cfg,
        ):
            mock_cfg.FEEDBACK_DIR = Path(tmpdir)
            record_feedback(fb)
            file_path = Path(tmpdir) / "agent1.jsonl"
            raw = json.loads(file_path.read_text().strip())

        assert raw["timestamp"] == ts.isoformat()


# ---------------------------------------------------------------------------
# CLI: feedback-stats
# ---------------------------------------------------------------------------


class TestFeedbackStatsCli:
    def test_feedback_stats_no_records(self) -> None:
        from nlqueries.cli.main import cli

        runner = CliRunner()
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("nlqueries.feedback.store.config") as mock_cfg,
        ):
            mock_cfg.FEEDBACK_DIR = Path(tmpdir)
            result = runner.invoke(cli, ["feedback-stats", "agent1"])

        assert result.exit_code == 0
        assert "No feedback recorded yet" in result.output

    def test_feedback_stats_shows_up_down_counts(self) -> None:
        from nlqueries.cli.main import cli

        runner = CliRunner()
        records = [
            _make_fb(rating="up"),
            _make_fb(rating="up"),
            _make_fb(rating="down"),
        ]
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("nlqueries.feedback.store.config") as mock_cfg,
        ):
            mock_cfg.FEEDBACK_DIR = Path(tmpdir)
            for r in records:
                record_feedback(r)
            result = runner.invoke(cli, ["feedback-stats", "agent1"])

        assert result.exit_code == 0
        assert "2" in result.output  # up count
        assert "1" in result.output  # down count

    def test_feedback_stats_shows_corrections(self) -> None:
        from nlqueries.cli.main import cli

        runner = CliRunner()
        fb = _make_fb(
            corrected_sql="SELECT COUNT(*) FROM orders WHERE status='open'",
            rating="down",
        )
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("nlqueries.feedback.store.config") as mock_cfg,
        ):
            mock_cfg.FEEDBACK_DIR = Path(tmpdir)
            record_feedback(fb)
            result = runner.invoke(cli, ["feedback-stats", "agent1"])

        assert result.exit_code == 0
        assert "Corrections" in result.output or "correction" in result.output.lower()
