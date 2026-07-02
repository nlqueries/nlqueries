"""
nlqueries benchmark harness
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Runs a golden question set through run_query() and emits a Markdown report.

Usage::

    python -m benchmarks.run_bench \
        --golden tests/golden/questions.yaml \
        --agent  sales-agent \
        --out    docs/benchmarks/

    # Replay mode — measure cache hit rate from a question log:
    python -m benchmarks.run_bench --replay questions.txt --agent sales-agent
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BenchRow:
    question: str
    agent_id: str
    latency_ms: int
    agent_type: str
    sql: str | None
    is_valid: bool  # True when sql present and no validation error in answer
    from_cache: bool
    tables_ok: bool  # expected_tables all present in sql
    contains_ok: bool  # expect_contains all present in answer
    hard: bool
    error: str | None = None


@dataclass
class BenchReport:
    rows: list[BenchRow] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    # Aggregate stats
    @property
    def total(self) -> int:
        return len(self.rows)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.rows if r.tables_ok and r.contains_ok and r.error is None)

    @property
    def cache_hits(self) -> int:
        return sum(1 for r in self.rows if r.from_cache)

    @property
    def p50_ms(self) -> int:
        if not self.rows:
            return 0
        latencies = sorted(r.latency_ms for r in self.rows)
        return latencies[len(latencies) // 2]

    @property
    def p95_ms(self) -> int:
        if not self.rows:
            return 0
        latencies = sorted(r.latency_ms for r in self.rows)
        return latencies[int(len(latencies) * 0.95)]

    @property
    def avg_ms(self) -> int:
        if not self.rows:
            return 0
        return int(sum(r.latency_ms for r in self.rows) / len(self.rows))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_tables(sql: str | None, expect_tables: list[str]) -> bool:
    if not expect_tables:
        return True
    if not sql:
        return False
    sql_lower = sql.lower()
    return all(t.lower() in sql_lower for t in expect_tables)


def _check_contains(answer: str, expect_contains: list[str]) -> bool:
    if not expect_contains:
        return True
    answer_lower = answer.lower()
    return all(c.lower() in answer_lower for c in expect_contains)


def _is_from_cache(answer: str) -> bool:
    # run_query joins all tokens; "from_cache" is in the last JSON chunk which
    # sync_runner strips. We detect it via a trailing marker in the joined text
    # if the orchestrator inserts one. For now we can't reliably detect it from
    # AgentQueryResult — this is a best-effort heuristic until TTFT tracking is
    # added in Phase 2.
    return False


async def _run_one(
    question: str,
    agent_id: str,
    expect_tables: list[str],
    expect_contains: list[str],
    hard: bool,
) -> BenchRow:
    from nlqueries.orchestrator.sync_runner import run_query

    start = time.monotonic()
    error: str | None = None
    sql: str | None = None
    agent_type = "unclear"
    answer = ""

    try:
        result = await run_query(question, agent_id)
        sql = result.sql
        agent_type = result.agent_type
        answer = result.answer or ""
        latency_ms = result.latency_ms
    except Exception as exc:  # noqa: BLE001
        latency_ms = int((time.monotonic() - start) * 1000)
        error = str(exc)

    tables_ok = _check_tables(sql, expect_tables)
    contains_ok = _check_contains(answer, expect_contains)
    is_valid = sql is not None and "validation_error" not in (answer or "").lower()

    return BenchRow(
        question=question,
        agent_id=agent_id,
        latency_ms=latency_ms,
        agent_type=agent_type,
        sql=sql,
        is_valid=is_valid,
        from_cache=_is_from_cache(answer),
        tables_ok=tables_ok,
        contains_ok=contains_ok,
        hard=hard,
        error=error,
    )


# ---------------------------------------------------------------------------
# Golden-set mode
# ---------------------------------------------------------------------------


async def run_golden(golden_path: Path, default_agent: str) -> BenchReport:
    items: list[dict[str, Any]] = yaml.safe_load(golden_path.read_text(encoding="utf-8")) or []
    report = BenchReport()

    for item in items:
        q = str(item.get("q", ""))
        agent = str(item.get("agent", default_agent))
        expect_tables: list[str] = item.get("expect_tables", [])
        expect_contains: list[str] = item.get("expect_contains", [])
        hard: bool = bool(item.get("hard", False))

        print(f"  [{agent}] {q[:70]}...", end="", flush=True)
        row = await _run_one(q, agent, expect_tables, expect_contains, hard)
        status = "OK" if (row.tables_ok and row.contains_ok and not row.error) else "FAIL"
        print(f" {status} ({row.latency_ms} ms)")
        report.rows.append(row)

    return report


# ---------------------------------------------------------------------------
# Replay mode (cache hit-rate measurement)
# ---------------------------------------------------------------------------


async def run_replay(replay_path: Path, agent_id: str) -> BenchReport:
    raw = replay_path.read_text(encoding="utf-8").splitlines()
    lines = [ln.strip() for ln in raw if ln.strip()]
    report = BenchReport()

    for question in lines:
        print(f"  {question[:70]}...", end="", flush=True)
        row = await _run_one(question, agent_id, [], [], False)
        print(f" {row.latency_ms} ms cache={'Y' if row.from_cache else 'N'}")
        report.rows.append(row)

    return report


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def _render_markdown(report: BenchReport, label: str) -> str:
    lines = [
        f"# NLQueries Benchmark — {label}",
        "",
        f"**Date:** {report.started_at}",
        f"**Total questions:** {report.total}",
        f"**Passed:** {report.passed} / {report.total}"
        f" ({100 * report.passed // max(report.total, 1)}%)",
        f"**Cache hits:** {report.cache_hits} / {report.total}",
        "",
        "## Latency",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| p50    | {report.p50_ms} ms |",
        f"| p95    | {report.p95_ms} ms |",
        f"| avg    | {report.avg_ms} ms |",
        "",
        "## Per-question results",
        "",
        "| # | Question | Agent | ms | Type | Tables | Contains | Cache | Hard | Error |",
        "|---|----------|-------|-----|------|--------|----------|-------|------|-------|",
    ]
    for i, row in enumerate(report.rows, 1):
        q_short = row.question[:50].replace("|", "\\|")
        lines.append(
            f"| {i} | {q_short} | {row.agent_id} | {row.latency_ms} "
            f"| {row.agent_type} | {'✓' if row.tables_ok else '✗'} "
            f"| {'✓' if row.contains_ok else '✗'} "
            f"| {'✓' if row.from_cache else '-'} "
            f"| {'H' if row.hard else '-'} "
            f"| {row.error or ''} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NLQueries benchmark harness")
    parser.add_argument("--golden", type=Path, help="Path to golden questions YAML")
    parser.add_argument("--replay", type=Path, help="Path to question log (one per line)")
    parser.add_argument("--agent", required=True, help="Default agent_id")
    parser.add_argument(
        "--out", type=Path, default=Path("docs/benchmarks"), help="Output directory for report"
    )
    parser.add_argument("--label", default="", help="Label for the report filename")
    return parser.parse_args()


async def _main() -> None:
    args = _parse_args()

    if args.golden:
        print(f"Running golden set: {args.golden}")
        report = await run_golden(args.golden, args.agent)
        mode = "golden"
    elif args.replay:
        print(f"Replay mode: {args.replay}")
        report = await run_replay(args.replay, args.agent)
        mode = "replay"
    else:
        print("ERROR: Provide --golden or --replay", file=sys.stderr)
        sys.exit(1)

    label = args.label or f"{mode}-{datetime.now(UTC).strftime('%Y%m%d-%H%M')}"
    md = _render_markdown(report, label)

    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / f"{label}.md"
    out_file.write_text(md, encoding="utf-8")

    print(f"\nReport: {out_file}")
    print(f"Passed {report.passed}/{report.total}  p50={report.p50_ms}ms  p95={report.p95_ms}ms")

    # Non-zero exit on failures so CI can gate on the golden set.
    if report.passed < report.total:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(_main())
