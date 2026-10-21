# nlqueries-core — OSS (BSL 1.1)
"""
nlqueries.audit
~~~~~~~~~~~~~~~
Structured audit events: who was allowed to do what, and what ran against a
database.

Two kinds of event are recorded. An authorisation decision is made by the MCP
guard (:mod:`nlqueries.auth.enforcement`) and rendered here. A query execution
is recorded by :class:`~nlqueries.connectors.base.PermittedConnector`, the
per-request wrapper every statement passes through whichever surface asked for
it, so the CLI, the library, the MCP server and a cache replay are covered by
one call site rather than four.

Every event goes to the ``nlqueries.audit`` logger with its JSON in the message
and the same record in ``extra={"audit": ...}``, and, when ``NLQ_AUDIT_LOG``
names a file, to that file as one JSON line. The JSON is rendered with sorted
keys and ASCII escapes, so a carriage return or line feed inside a field is
written as ``\\r`` or ``\\n`` and cannot start a line of its own.

An execution event carries a SHA-256 digest of the statement and its length,
never the statement itself, the question that produced it, the rows it
returned or the credentials it ran under. The digest is the one
:func:`nlqueries.sql_policy.digest_of` computes, so it matches the
``sql_digest`` on a policy decision for the same bytes.

Who asked and through which surface are carried in a :class:`AuditContext`,
bound by the entry point: the CLI binds it once for the process, the MCP server
once per tool call. A caller that binds nothing is the library, and each of its
events gets a request id of its own.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import json
import logging
import os
import stat
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from nlqueries import config

if TYPE_CHECKING:
    from nlqueries.auth.principal import Principal
    from nlqueries.connectors.base import DatabaseConnector, QueryResult

logger = logging.getLogger("nlqueries.audit")

#: For problems with the file sink. A child of the audit logger, so a handler
#: on ``nlqueries.audit`` sees them, but named apart so they are not mistaken
#: for events.
_sink_log = logging.getLogger("nlqueries.audit.sink")

#: Incremented when a field is renamed, removed or changes meaning. Adding a
#: field does not change it; a consumer should ignore fields it does not know.
SCHEMA_VERSION = 1

#: Event names.
AUTHORIZATION = "authorization"
QUERY_EXECUTION = "query_execution"


class Surface(StrEnum):
    """Which entry point a request came through."""

    CLI = "cli"
    MCP = "mcp"
    #: Anything that called into the package without binding a context.
    LIBRARY = "library"


def new_request_id() -> str:
    """A fresh request id: 32 hex characters, random."""
    return uuid.uuid4().hex


@dataclass(frozen=True)
class AuditContext:
    """Who is asking, and through which surface, for the request in progress.

    Holds the principal's subject and source rather than the principal itself:
    a principal carries token claims, and nothing reached from here may log
    them.
    """

    surface: Surface
    request_id: str = field(default_factory=new_request_id)
    principal: str | None = None
    source: str | None = None

    @classmethod
    def for_principal(cls, surface: Surface, principal: Principal | None) -> AuditContext:
        if principal is None:
            return cls(surface=surface)
        return cls(surface=surface, principal=principal.subject, source=str(principal.source))


_context: ContextVar[AuditContext | None] = ContextVar("nlqueries_audit_context", default=None)

#: The context for a thread that inherited none. Set by an entry point that
#: owns the process (the CLI, the MCP server), because a thread started with
#: ``threading.Thread`` does not copy the starting thread's context variables,
#: and its events would otherwise be attributed to the library.
_process_context: AuditContext | None = None


def current() -> AuditContext:
    """The context bound for this request, or a library one made now."""
    return _context.get() or _process_context or AuditContext(surface=Surface.LIBRARY)


def bind_process(surface: Surface, principal: Principal | None = None) -> AuditContext:
    """Bind *surface* and *principal* for the rest of the process.

    For an entry point that serves one caller for its whole life, as a CLI
    command does. A context bound per request with :func:`bound` takes
    precedence where one is bound.
    """
    global _process_context  # noqa: PLW0603 - one process, one default
    _process_context = AuditContext.for_principal(surface, principal)
    return _process_context


@contextlib.contextmanager
def bound(surface: Surface, principal: Principal | None = None) -> Iterator[AuditContext]:
    """Bind a fresh request for the duration of the ``with`` block."""
    context = AuditContext.for_principal(surface, principal)
    token = _context.set(context)
    try:
        yield context
    finally:
        _context.reset(token)


def audited_tool(
    func: Callable[..., Any],
    surface: Surface,
    principal_for_call: Callable[[], Principal],
) -> Callable[..., Any]:
    """Wrap *func* so each call runs as one request, with its own id.

    The principal is resolved per call, as the guard resolves it, because it
    belongs to the request and the wrapper is built once. A resolver that fails
    leaves the principal unknown here; the guard inside makes the same call and
    refuses the request itself.

    The signature is kept (``functools.wraps``), because FastMCP publishes each
    tool's schema from it.
    """

    def principal() -> Principal | None:
        try:
            return principal_for_call()
        except Exception:  # noqa: BLE001 - the guard reports it
            return None

    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_audited(*args: Any, **kwargs: Any) -> Any:
            with bound(surface, principal()):
                return await func(*args, **kwargs)

        return async_audited

    @functools.wraps(func)
    def audited(*args: Any, **kwargs: Any) -> Any:
        with bound(surface, principal()):
            return func(*args, **kwargs)

    return audited


def render(record: Mapping[str, Any]) -> str:
    """*record* as one line of JSON.

    Sorted keys, so two renderings of one record are the same bytes. ASCII
    escapes, so a CR, an LF, or a Unicode line separator in any field is
    written as an escape and the record stays on one line.
    """
    return json.dumps(record, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str)


def publish(event: str, message: str, fields: Mapping[str, Any], level: int = logging.INFO) -> None:
    """Record *fields* as an *event*: to the audit logger, and to the file sink.

    The request id and surface come from the bound context, so an
    authorisation decision and the execution it let through share an id.
    """
    context = current()
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "event": event,
        "request_id": context.request_id,
        "surface": str(context.surface),
        **fields,
    }
    line = render(record)
    logger.log(level, "%s %s", message, line, extra={"audit": record})
    _write(line)


# -- query execution -----------------------------------------------------


def record_execution(
    connector: DatabaseConnector,
    sql: str,
    *,
    agent_id: str | None,
    started: float,
    result: QueryResult | None = None,
    error: BaseException | None = None,
) -> None:
    """Record one statement sent through *connector*.

    *started* is a :func:`time.monotonic` reading taken before the call. Pass
    the *result* the call returned or the *error* it raised.

    Never raises. Recording is not allowed to turn a query that ran into one
    that failed; a failure here is logged on ``nlqueries.audit.sink``.
    """
    duration_ms = round((time.monotonic() - started) * 1000, 3)
    try:
        if not (logger.isEnabledFor(logging.INFO) or config.AUDIT_LOG):
            # Nobody is listening. The policy evaluation below parses the
            # statement, which is not free, so it is skipped.
            return
        publish(
            QUERY_EXECUTION,
            "query executed",
            _execution_fields(connector, sql, agent_id, duration_ms, result, error),
        )
    except Exception:  # noqa: BLE001 - see the docstring
        _sink_log.warning("A query execution could not be recorded.", exc_info=True)


def _execution_fields(
    connector: DatabaseConnector,
    sql: str,
    agent_id: str | None,
    duration_ms: float,
    result: QueryResult | None,
    error: BaseException | None,
) -> dict[str, Any]:
    from nlqueries.sql_policy import POLICY_VERSION, digest_of, evaluate  # noqa: PLC0415

    context = current()
    connector_type, dialect = connector_identity(connector)

    decision: str | None = None
    if dialect is not None:
        decision = "allow" if evaluate(sql, dialect).allowed else "deny"

    if error is not None:
        outcome = "exception"
    elif result is not None and result.error:
        outcome = "error"
    else:
        outcome = "ok"

    return {
        "at": datetime.now(UTC).isoformat(),
        "principal": context.principal,
        "source": context.source,
        "agent_id": agent_id,
        "connector_type": connector_type,
        "dialect": dialect,
        "sql_sha256": digest_of(sql),
        "sql_bytes": len(sql.encode("utf-8", errors="surrogatepass")),
        "policy_version": POLICY_VERSION,
        "policy_decision": decision,
        "outcome": outcome,
        "row_count": result.row_count if result is not None else None,
        "truncated": result.truncated if result is not None else None,
        "duration_ms": duration_ms,
        # The class only. An exception's message can quote the statement, and
        # a database error's text often does.
        "error_class": type(error).__name__ if error is not None else None,
    }


def connector_identity(connector: DatabaseConnector) -> tuple[str, str | None]:
    """The registered type of *connector* and the SQL dialect it speaks.

    The type is the registry name of the nearest registered class in the
    connector's ancestry, so a subclass (an enterprise variant, say) reports
    the type it extends. A connector that is not registered reports its class
    name. The dialect is sqlglot's name for it; the generic SQLAlchemy
    connector's comes from its engine, since its type names no grammar. None
    when it cannot be told.
    """
    from nlqueries.connectors import CONNECTOR_REGISTRY  # noqa: PLC0415
    from nlqueries.sql_policy import _sqlglot_dialect  # noqa: PLC0415

    names = {cls: name for name, cls in CONNECTOR_REGISTRY.items()}
    connector_type = next(
        (names[cls] for cls in type(connector).__mro__ if cls in names),
        type(connector).__name__,
    )
    if connector_type == "sqlalchemy":
        engine = getattr(connector, "_engine", None)
        backend = getattr(getattr(engine, "dialect", None), "name", None)
        return connector_type, _sqlglot_dialect(backend) if isinstance(backend, str) else None
    if connector_type not in CONNECTOR_REGISTRY:
        return connector_type, None
    return connector_type, _sqlglot_dialect(connector_type)


# -- the file sink -------------------------------------------------------

_FILE_MODE = stat.S_IRUSR | stat.S_IWUSR

_write_lock = threading.Lock()

#: Paths a failure has already been reported for, so a full disk is one
#: warning rather than one per query.
_reported: set[str] = set()


def _write(line: str) -> None:
    """Append *line* to ``NLQ_AUDIT_LOG``, if one is set.

    Opened for each event, append-only, and created at 0600; an existing file
    with wider permissions is narrowed to 0600. Opening per event costs a
    system call or two, against a statement that has just been sent to a
    database, and means a rotated file is picked up without a restart.

    One ``write`` per line, on a descriptor opened with ``O_APPEND``, so lines
    from concurrent writers do not interleave.
    """
    path = config.AUDIT_LOG
    if not path:
        return
    data = (line + "\n").encode("ascii")
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    with _write_lock:
        try:
            fd = os.open(path, flags, _FILE_MODE)
            try:
                _narrow(fd)
                written = 0
                while written < len(data):
                    written += os.write(fd, data[written:])
            finally:
                os.close(fd)
        except OSError:
            if path not in _reported:
                _reported.add(path)
                _sink_log.warning(
                    "The audit log %s could not be written; events are still logged "
                    "to nlqueries.audit.",
                    path,
                    exc_info=True,
                )
            return
        _reported.discard(path)


def _narrow(fd: int) -> None:
    """Restrict the open file to its owner, if it is readable by anyone else."""
    fchmod = getattr(os, "fchmod", None)
    if fchmod is None:  # pragma: no cover - Windows, where the bits are not enforced
        return
    if os.fstat(fd).st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        with contextlib.suppress(OSError):
            fchmod(fd, _FILE_MODE)
