# nlqueries-core — OSS (BSL 1.1)
"""
nlqueries.connectors.login_identity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
What the login a connector uses is allowed to do, for engines other than
PostgreSQL (which has :mod:`nlqueries.connectors.postgres_identity`).

Every query runs in the most restrictive execution its engine offers, but that
constrains the transaction, not the login. A login that may write, change the
schema or administer the server keeps those rights whatever this application
does, and the database's own permissions are the boundary that does not depend
on this application being free of bugs. The per-engine modules read the login's
grants and turn them into :class:`LoginIdentity`, which ``nlqueries health``
reports. Nothing here refuses a connection.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sqlalchemy import event
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoginIdentity:
    """The login a connector uses, and what it holds beyond reading.

    The shape ``nlqueries health`` reads from a connector's ``identity``:
    :attr:`undetermined_reason`, :attr:`concerns` and :meth:`summary`.
    """

    engine: str
    user: str
    database: str
    #: One entry per write, DDL or administrative right the login holds.
    concerns: tuple[str, ...] = ()
    #: Set when the grants could not be read. Kept apart from :attr:`concerns`
    #: so that a check that did not complete is not mistaken for one that
    #: found nothing.
    undetermined_reason: str | None = None

    @property
    def is_least_privilege(self) -> bool:
        """True when the grants were read and nothing beyond reading was found."""
        return self.undetermined_reason is None and not self.concerns

    def summary(self) -> str:
        """A single line for logs and the health report."""
        if self.undetermined_reason is not None:
            return f"identity could not be determined: {self.undetermined_reason}"
        who = f"{self.user}@{self.database}" if self.database else self.user
        if not self.concerns:
            return f"{who}: least privilege"
        return f"{who}: {'; '.join(self.concerns)}"

    @classmethod
    def undetermined(cls, engine: str, reason: str) -> LoginIdentity:
        return cls(engine=engine, user="", database="", undetermined_reason=reason)


def log_identity(identity: LoginIdentity) -> None:
    """Warn when the login holds more than it needs, or could not be read."""
    if identity.undetermined_reason is not None:
        logger.warning("%s identity check: %s", identity.engine, identity.summary())
    elif identity.concerns:
        logger.warning(
            "The %s login is more privileged than this connector needs: %s. "
            "See docs/database-hardening.md for the login to create instead.",
            identity.engine,
            identity.summary(),
        )
    else:
        logger.debug("%s identity check: %s", identity.engine, identity.summary())


def watch_engine(
    engine: Engine,
    inspect: Callable[[Any], LoginIdentity],
    store: Callable[[LoginIdentity], None],
) -> None:
    """Read the login's grants once per physical connection the pool opens.

    On the pool's ``connect`` event rather than per query, as the PostgreSQL
    connector does: a new backend connection is when the answer can change.
    """

    @event.listens_for(engine, "connect")
    def _inspect(dbapi_connection: Any, _record: Any) -> None:
        identity = inspect(dbapi_connection)
        store(identity)
        log_identity(identity)
