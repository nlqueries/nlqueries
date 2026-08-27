"""
nlqueries.execution
~~~~~~~~~~~~~~~~~~~
Execution permission, determined once at the entry point and carried to the
connector.

Background. ``--no-execute`` was defined on the CLI but never reached the
orchestrator, whose signature is ``run_query_sync(question, agent_id,
**kwargs)``. The orchestrator executed the query and populated the result
before the CLI consulted the flag, so the flag suppressed only a second,
CLI-level execution. Evaluation had the same defect and executed every golden
case against the configured connector.

Permission is therefore modelled as a value the caller mints and every
downstream component carries, rather than a boolean re-derived at each layer.
The policy is immutable and cannot be widened, so retries, sub-agents, cache
replays and analysis inherit the permission they were given.

Where no policy is supplied, execution is denied. A generation-only run that
fails closed returns SQL and no rows; an execute-by-default run that fails open
executes model output against a customer's database.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ExecutionMode(StrEnum):
    """What a caller is permitted to do with the SQL that comes back."""

    #: Produce SQL only. No connector is opened and no statement is executed.
    GENERATE_ONLY = "generate_only"

    #: Execute as a read. The read-only transaction and the database's own
    #: grants remain in force. This permits execution; it does not assert that
    #: the statement is safe.
    EXECUTE_READ_ONLY = "execute_read_only"


@dataclass(frozen=True)
class ExecutionPolicy:
    """An immutable permission to execute.

    Frozen so that receiving code cannot widen it in place. No escalation
    method or setter is provided: a sub-agent, retry or cache replay that
    executes must have been passed a policy that already permitted it.
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
        """Permission to execute the statement as a read.

        Minted at an entry point that has established the request is a user
        asking for data. It is not re-derived downstream from a flag in scope.
        """
        return cls(mode=ExecutionMode.EXECUTE_READ_ONLY)

    def __str__(self) -> str:  # pragma: no cover - for logs and telemetry
        return self.mode.value


class ExecutionNotPermitted(RuntimeError):
    """Raised when a caller attempts to reach a database without permission.

    Raised rather than silently skipped, so that a caller which requests rows
    and receives none can distinguish a refusal by policy from an empty result
    set. Reaching this exception indicates a defect in the calling path.
    """

    def __init__(self, what: str, policy: ExecutionPolicy) -> None:
        super().__init__(
            f"{what} requires permission to execute, and this request carries "
            f"'{policy}'. Generation-only requests produce SQL without opening a "
            f"connector; if this one should run, it needs "
            f"ExecutionPolicy.execute_read_only() from the caller that started it."
        )
        self.policy = policy


#: The policy applied when a caller supplies none.
DEFAULT_POLICY = ExecutionPolicy.generate_only()
