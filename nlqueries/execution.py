"""
nlqueries.execution
~~~~~~~~~~~~~~~~~~~
Whether this request is allowed to touch a database, decided once at the edge
and carried the whole way down.

`--no-execute` was defined on the CLI and never reached the orchestrator, whose
signature is `run_query_sync(question, agent_id, **kwargs)`. The orchestrator
executed, populated the result, and only then did the CLI consult the flag — so
it suppressed a *second*, CLI-level execution and not the first one. Evaluation
had the same shape: it called the same function the same way and executed every
golden case against whatever connector was configured.

The lesson is not "add a check in the orchestrator too". It is that a boolean
reconstructed at each layer is a boolean that will eventually be reconstructed
wrongly, and no single reader can tell by looking. So permission becomes a value
the caller mints and everything downstream carries: immutable, explicit, and
impossible to widen — retries, sub-agents, cache replays and analysis all
inherit the policy they were given and none of them can hand themselves a better
one.

Deny is the default everywhere it is absent. A generation-only run that fails
closed produces SQL and no rows, which is inconvenient; an execute-by-default
run that fails open runs a language model's output against a customer's
database.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ExecutionMode(StrEnum):
    """What a caller is permitted to do with the SQL that comes back."""

    #: Produce SQL. Open no connector, make no connection, run nothing.
    GENERATE_ONLY = "generate_only"

    #: Run it, as a read. The read-only transaction and the database's own
    #: grants remain in force underneath — this permits execution, it does not
    #: promise the statement is safe.
    EXECUTE_READ_ONLY = "execute_read_only"


@dataclass(frozen=True)
class ExecutionPolicy:
    """An immutable permission to execute, or not.

    Frozen so it cannot be widened in place by code that receives it. There is
    deliberately no `escalate()` and no setter: a sub-agent, a retry or a cache
    replay that wants to execute must have been handed a policy that already
    said so.
    """

    mode: ExecutionMode

    @property
    def may_execute(self) -> bool:
        return self.mode is ExecutionMode.EXECUTE_READ_ONLY

    @classmethod
    def generate_only(cls) -> ExecutionPolicy:
        """Generate SQL and touch nothing. The default for evaluation."""
        return cls(mode=ExecutionMode.GENERATE_ONLY)

    @classmethod
    def execute_read_only(cls) -> ExecutionPolicy:
        """Permission to run the statement as a read.

        Minted at an entry point that has decided a human asked for data —
        never reconstructed downstream from a flag that happens to be in scope.
        """
        return cls(mode=ExecutionMode.EXECUTE_READ_ONLY)

    def __str__(self) -> str:  # pragma: no cover - for logs and telemetry
        return self.mode.value


class ExecutionNotPermitted(RuntimeError):
    """Raised when something tried to reach a database without permission.

    An exception rather than a silent skip: code that asks for rows and gets
    none has to be able to tell "the policy forbade this" from "the query
    returned nothing", and a caller that reaches here has a bug worth seeing.
    """

    def __init__(self, what: str, policy: ExecutionPolicy) -> None:
        super().__init__(
            f"{what} requires permission to execute, and this request carries "
            f"'{policy}'. Generation-only requests produce SQL without opening a "
            f"connector; if this one should run, it needs "
            f"ExecutionPolicy.execute_read_only() from the caller that started it."
        )
        self.policy = policy


#: What a caller gets when nobody says otherwise.
DEFAULT_POLICY = ExecutionPolicy.generate_only()
