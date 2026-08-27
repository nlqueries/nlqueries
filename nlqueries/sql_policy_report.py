# nlqueries-core — OSS (BSL 1.1)
"""
nlqueries.sql_policy_report
~~~~~~~~~~~~~~~~~~~~~~~~~~~
What :mod:`nlqueries.sql_policy` would refuse, reported without refusing it.

The policy denies any function sqlglot does not model. That set is the vendor,
extension and user-defined one, and it contains both the payloads the policy
exists to stop and a minority of ordinary analytics functions. Which ordinary
functions appear depends on the deployment, so the allowlist cannot be settled
from a list written in advance.

This module runs the policy over statements a deployment has actually issued and
reports the outcome: how many would be refused, for what reason, and which
unrecognised functions were called and how often. The functions it lists are the
candidates for :data:`~nlqueries.sql_policy.ALLOWED_ANONYMOUS`; the refusals are
the cost of enabling it.

Enforcement is a separate change. Running this first is what makes the allowlist
a record of observed usage rather than a guess.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from nlqueries.sql_policy import ALLOWED_ANONYMOUS, POLICY_VERSION, evaluate


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
    #: it. Includes functions already allowlisted, so the list also shows which
    #: existing entries are still earning their place.
    anonymous_counts: Counter[str] = field(default_factory=Counter)

    @property
    def refused(self) -> int:
        return len(self.refusals)

    @property
    def candidates(self) -> list[tuple[str, int]]:
        """Unrecognised functions not yet allowlisted, most-used first.

        These are what an operator would have to add for the refused statements
        to run. Ordered by usage because the first entries buy the most back.
        """
        permitted = ALLOWED_ANONYMOUS.get(self.dialect.lower(), frozenset())
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
        if decision.allowed:
            report.allowed += 1
        else:
            report.refusals.append(Refusal(sql=sql, reasons=decision.reasons))
    return report
