# nlqueries-core — OSS (BSL 1.1)
"""
nlqueries.sql_policy_report
~~~~~~~~~~~~~~~~~~~~~~~~~~~
What :mod:`nlqueries.sql_policy` would refuse, reported without refusing it.

The policy allows only what it lists: expression classes per dialect, in
:mod:`nlqueries.sql_constructs`, and functions sqlglot does not model, by name.
What a deployment's queries need beyond those lists depends on the deployment,
and on the installed sqlglot, whose classes the lists name.

This module runs the policy over statements a deployment has issued and reports
the outcome: how many would be refused, for what reason, which unrecognised
functions were called and which constructs not on the list were used, and how
often. Those are the candidates for the lists; the refusals are the cost of
enforcement.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from nlqueries.sql_policy import POLICY_VERSION, allowed_anonymous, evaluate


@dataclass(frozen=True)
class Refusal:
    """One statement the policy would refuse, and why."""

    sql: str
    reasons: tuple[str, ...]

    @property
    def excerpt(self) -> str:
        flat = " ".join(self.sql.split())
        return flat if len(flat) <= 120 else flat[:117] + "..."


@dataclass
class InventoryReport:
    """The result of running the policy over a deployment's statements."""

    dialect: str
    policy_version: str = POLICY_VERSION
    total: int = 0
    allowed: int = 0
    refusals: list[Refusal] = field(default_factory=list)
    #: Every unrecognised function seen, with the number of statements calling
    #: it. Includes allowlisted functions, so unused entries are also visible.
    anonymous_counts: Counter[str] = field(default_factory=Counter)
    #: Expression classes not on the dialect's list, with the number of
    #: statements using each.
    unlisted_counts: Counter[str] = field(default_factory=Counter)

    @property
    def refused(self) -> int:
        return len(self.refusals)

    @property
    def candidates(self) -> list[tuple[str, int]]:
        """Unrecognised functions not yet allowlisted, most-used first.

        These would have to be added for the refused statements to run.
        Ordered by usage.
        """
        permitted = allowed_anonymous(self.dialect)
        return [
            (name, n) for name, n in self.anonymous_counts.most_common() if name not in permitted
        ]

    def reason_counts(self) -> Counter[str]:
        """Refusals by rule, so a single dominant cause is visible."""
        counts: Counter[str] = Counter()
        for refusal in self.refusals:
            for reason in refusal.reasons:
                # Collapse the variable part: the rule matters, not the name.
                counts[reason.split(":")[0]] += 1
        return counts

    def render(self, examples: int = 5) -> str:
        """A plain-text report."""
        lines = [
            f"SQL policy inventory (dialect: {self.dialect}, policy version {self.policy_version})",
            "",
            f"  statements examined : {self.total}",
            f"  would be allowed    : {self.allowed}",
            f"  would be refused    : {self.refused}",
        ]
        if self.total:
            lines.append(f"  refusal rate        : {100 * self.refused / self.total:.1f}%")

        if self.refusals:
            lines += ["", "Refusals by rule:"]
            for reason, count in self.reason_counts().most_common():
                lines.append(f"  {count:5}  {reason}")

        if self.candidates:
            lines += ["", "Unrecognised functions not currently allowlisted:"]
            for name, count in self.candidates:
                lines.append(f"  {count:5}  {name}")
        elif self.anonymous_counts:
            lines += ["", "Every unrecognised function seen is already allowlisted."]

        if self.unlisted_counts:
            lines += ["", "Constructs not on the list (sqlglot expression classes):"]
            for name, count in self.unlisted_counts.most_common():
                lines.append(f"  {count:5}  {name}")

        if self.refusals:
            lines += ["", f"Examples (first {min(examples, len(self.refusals))}):"]
            for refusal in self.refusals[:examples]:
                lines.append(f"  - {refusal.excerpt}")
                lines.append(f"      {'; '.join(refusal.reasons)}")

        return "\n".join(lines)


def build_inventory(statements: list[str], dialect: str) -> InventoryReport:
    """Run the policy over *statements* and report, refusing nothing."""
    report = InventoryReport(dialect=dialect)
    for sql in statements:
        if not sql or not sql.strip():
            continue
        report.total += 1
        decision = evaluate(sql, dialect)
        report.anonymous_counts.update(decision.anonymous_functions)
        report.unlisted_counts.update(decision.unlisted_constructs)
        if decision.allowed:
            report.allowed += 1
        else:
            report.refusals.append(Refusal(sql=sql, reasons=decision.reasons))
    return report
