"""
nlqueries.connectors.loader
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Open a live DatabaseConnector from the local connectors file.

Handles the sanitisation mismatch between agent IDs (underscores) and
connector IDs (colons), e.g. MCP tool receives ``postgres_localhost_imdb``
while the connector is keyed as ``postgres:localhost:imdb``.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import yaml

from nlqueries import config
from nlqueries.connectors import CONNECTOR_REGISTRY, DatabaseConnector
from nlqueries.connectors.base import PermittedConnector
from nlqueries.execution import DEFAULT_POLICY, ExecutionPolicy

logger = logging.getLogger(__name__)

_KEYRING_SERVICE = "nlqueries"

#: Keys in a connectors-file entry that describe the entry itself rather than
#: the connection, and so are not passed to ``DatabaseConnector.connect``.
#: Everything else in the entry is connector configuration and is passed
#: through -- see :func:`credentials_for`.
#:
#: ``url`` is NOT here: it is replaced rather than dropped, with the
#: password-resolved form, because ``SQLAlchemyConnector`` connects with it.
_ENTRY_ONLY_KEYS = frozenset({"db_type", "password_storage"})


# ---------------------------------------------------------------------------
# Connector reuse
#
# Every query used to build a new SQLAlchemy engine: a fresh TCP connection, TLS
# handshake and authentication against the customer's database, preceded by a
# YAML read and a keyring lookup — and the engines were never disposed, so they
# accumulated until garbage collection got round to them. At any real
# concurrency a customer's DBA sees that as connection churn, and it defeats
# SQLAlchemy's pooling entirely, since a pool that is discarded after one query
# has pooled nothing.
#
# Safe to cache because the connector carries no per-request state. Row filters
# and scope are applied ABOVE it: the enterprise layer wraps whatever this
# function returns, per call, from a ContextVar bound for that request. So a
# shared inner connector cannot leak one tenant's scope into another's query.
# That property is load-bearing — anything that later stores request state on a
# connector object breaks it.
# ---------------------------------------------------------------------------


@dataclass
class _Cached:
    connector: DatabaseConnector
    fingerprint: str
    created_at: float


_cache: OrderedDict[str, _Cached] = OrderedDict()
# Callers arrive from asyncio.to_thread, so this is genuinely concurrent.
_cache_lock = threading.Lock()


def _fingerprint(connector_id: str, cfg: dict[str, Any]) -> str:
    """Identify the configuration a cached connector was built from.

    Includes the resolved password, so a rotated credential produces a different
    fingerprint and the next query rebuilds — self-healing even with no explicit
    invalidation. Hashed rather than stored, so a cache entry never holds a
    readable secret.
    """
    payload = json.dumps(cfg, sort_keys=True, default=str)
    password = _load_password(connector_id, cfg) or ""
    return hashlib.sha256(f"{payload}|{password}".encode()).hexdigest()


def _dispose(cached: _Cached) -> None:
    with contextlib.suppress(Exception):
        cached.connector.close()


def invalidate_connector_cache(connector_id: str | None = None) -> None:
    """Drop cached connectors, disposing their engines.

    With no argument, drops everything. This is the hook the enterprise layer
    calls when a connector's configuration or credential changes, so a rotation
    takes effect immediately rather than at the next TTL expiry — a credential
    that keeps working for fifteen minutes after being rotated is a finding
    waiting to happen.
    """
    with _cache_lock:
        if connector_id is None:
            entries = list(_cache.values())
            _cache.clear()
        else:
            entry = _cache.pop(connector_id, None)
            entries = [entry] if entry is not None else []

    for entry in entries:
        _dispose(entry)
    if entries:
        logger.info("Invalidated %d cached connector(s)", len(entries))


def _cache_get(connector_id: str, fingerprint: str) -> DatabaseConnector | None:
    """Return a live cached connector, or None to build a fresh one."""
    now = time.monotonic()
    stale: _Cached | None = None

    with _cache_lock:
        entry = _cache.get(connector_id)
        if entry is None:
            return None
        expired = now - entry.created_at > config.CONNECTOR_CACHE_TTL_SECONDS
        if entry.fingerprint != fingerprint or expired:
            stale = _cache.pop(connector_id)
        else:
            _cache.move_to_end(connector_id)  # least-recently-used ordering
            return entry.connector

    if stale is not None:
        _dispose(stale)
    return None


def _cache_put(connector_id: str, connector: DatabaseConnector, fingerprint: str) -> None:
    evicted: list[_Cached] = []
    with _cache_lock:
        existing = _cache.pop(connector_id, None)
        if existing is not None:
            evicted.append(existing)
        _cache[connector_id] = _Cached(connector, fingerprint, time.monotonic())
        while len(_cache) > max(1, config.CONNECTOR_CACHE_MAX_ENTRIES):
            _, oldest = _cache.popitem(last=False)
            evicted.append(oldest)

    for entry in evicted:
        _dispose(entry)


def _load_connectors() -> dict[str, Any]:
    """The connectors file as a mapping, or ``{}`` when it does not exist."""
    if not config.CONNECTORS_FILE.exists():
        return {}
    loaded: dict[str, Any] = yaml.safe_load(config.CONNECTORS_FILE.read_text()) or {}
    return loaded


def _find_connector_id(agent_id: str, connectors: dict[str, Any] | None = None) -> str | None:
    """Return the connector_id in connectors.yaml that matches *agent_id*.

    Tries a direct lookup first (CLI usage where agent_id == connector_id),
    then a sanitized lookup (MCP usage where colons were replaced by ``_``).

    *connectors* lets a caller that has already read the file resolve against
    that same copy. ``open_connector_for_agent`` does, because reading it twice
    left a window in which the file could be rewritten between the two reads,
    and the branch guarding that window was indistinguishable from a genuinely
    missing entry.
    """
    if connectors is None:
        connectors = _load_connectors()
    if not connectors:
        return None
    if agent_id in connectors:
        return agent_id
    for key in connectors:
        if re.sub(r"[^\w.-]", "_", key) == agent_id:
            return key
    return None


def _load_password(connector_id: str, cfg: dict[str, Any]) -> str | None:
    if cfg.get("password_storage") == "keychain":
        try:
            import keyring  # noqa: PLC0415

            return keyring.get_password(_KEYRING_SERVICE, connector_id)
        except Exception:  # noqa: BLE001
            return None
    try:
        from sqlalchemy.engine import make_url  # noqa: PLC0415

        return make_url(cfg["url"]).password
    except Exception:  # noqa: BLE001
        return None


def _get_full_url(connector_id: str, cfg: dict[str, Any]) -> str:
    url: str = cfg["url"]
    if cfg.get("password_storage") == "keychain":
        pwd = _load_password(connector_id, cfg)
        if pwd is not None:
            try:
                from sqlalchemy.engine import make_url  # noqa: PLC0415

                url = make_url(url).set(password=pwd).render_as_string(hide_password=False)
            except Exception:  # noqa: BLE001
                pass
    return url


def credentials_for(connector_id: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Build the credentials dict to hand ``DatabaseConnector.connect``.

    *cfg* is one entry from the connectors file. Every key in it is connector
    configuration and is passed through, except the two that describe the entry
    rather than the connection (:data:`_ENTRY_ONLY_KEYS`).

    This used to be an allow-list of eight key names, written out at seven call
    sites -- here and six times in the CLI. It silently dropped every key it did
    not name, and what it dropped included the whole of a connector's TLS
    configuration: ``ssl_mode``, ``ssl_ca_cert``, ``ssl_client_cert`` and
    ``ssl_client_key``.

    That fails in both directions and neither is visible from the connector. A
    database with no TLS cannot be reached at all, because with no ``ssl_mode``
    the connector resolves the ``require`` default and libpq refuses the server.
    A connector configured to verify a private CA is opened *without* the
    certificate, so it resolves ``require`` rather than ``verify-full`` and comes
    up encrypted but authenticating nothing -- the posture the operator supplied
    a root certificate to avoid.

    A pass-through keeps callers out of the business of knowing which keys a
    connector understands. Every connector reads what it wants with ``.get()``,
    so a key it does not recognise costs nothing, whereas the allow-list had to
    be extended for each one in seven places and had fallen four keys behind.

    The URL is authoritative for the fields it carries, and is itself passed
    under ``url``: :func:`_get_full_url` resolves a keychain-stored password into
    it, so overlaying from the parsed URL is what makes that indirection work,
    and ``SQLAlchemyConnector`` connects with the URL directly.
    """
    from sqlalchemy.engine import make_url  # noqa: PLC0415

    url = _get_full_url(connector_id, cfg)
    parsed = make_url(url)
    credentials = {k: v for k, v in cfg.items() if k not in _ENTRY_ONLY_KEYS}
    credentials.update(
        {
            "url": url,
            "host": parsed.host or cfg.get("host", "localhost"),
            "port": parsed.port or cfg.get("port"),
            "database": parsed.database or cfg.get("database"),
            "user": parsed.username or cfg.get("user"),
            "password": parsed.password or _load_password(connector_id, cfg),
        }
    )
    return credentials


def open_connector_for_agent(
    agent_id: str, execution: ExecutionPolicy = DEFAULT_POLICY
) -> DatabaseConnector | None:
    """Return a connected DatabaseConnector for *agent_id*.

    Returns ``None`` when the connector cannot be found, the required driver is
    not installed, or the connection attempt fails. Every one of those is logged
    at warning with the cause, because they are five different faults with five
    different fixes and the caller sees one ``None``.

    They used to be silent. Downstream that becomes an answer with no rows and
    no error -- the SQL path treats a ``None`` connector as "nothing to execute"
    -- so a misconfigured agent looked exactly like a query that matched
    nothing, and the reason existed nowhere at all.

    The connector may be **shared with other in-flight requests**. It was
    previously built per query, and this docstring previously instructed callers
    not to cache it. That is no longer the case: callers must not close it or
    store per-request state on it, since reuse is required for pooling to have
    any effect.

    Set ``CONNECTOR_CACHE_ENABLED=false`` to restore the previous behaviour.

    *execution* is the calling request's permission. It is applied by wrapping
    the shared connector rather than by setting it on the connector, because the
    pooled object outlives the request and a policy stored there would be
    inherited by the next caller to receive it. Defaults to generate-only, so a
    caller that does not request execution is not granted it.
    """
    # Read once, and resolve everything from that copy. `_find_connector_id`
    # used to read the file itself, so this function read it twice and the two
    # reads could disagree.
    connectors = _load_connectors()
    if not connectors:
        logger.warning(
            "No connector could be opened for agent %s: %s is missing or empty, so "
            "nothing is configured for anything. The generated SQL will not run.",
            agent_id,
            config.CONNECTORS_FILE,
        )
        return None

    connector_id = _find_connector_id(agent_id, connectors)
    if connector_id is None:
        logger.warning(
            "No connector could be opened for agent %s: no entry in %s matches it, "
            "directly or after sanitising. The file holds %d: %s.",
            agent_id,
            config.CONNECTORS_FILE,
            len(connectors),
            ", ".join(sorted(connectors)),
        )
        return None

    cfg = connectors.get(connector_id)
    if not cfg:
        logger.warning(
            "No connector could be opened for agent %s: entry %s in %s is empty.",
            agent_id,
            connector_id,
            config.CONNECTORS_FILE,
        )
        return None

    fingerprint = ""
    if config.CONNECTOR_CACHE_ENABLED:
        fingerprint = _fingerprint(connector_id, cfg)
        cached = _cache_get(connector_id, fingerprint)
        if cached is not None:
            return PermittedConnector(cached, execution)

    db_type = cfg.get("db_type", "").lower()
    connector_cls = CONNECTOR_REGISTRY.get(db_type)
    if connector_cls is None:
        logger.warning(
            "No connector could be opened for agent %s: db_type %r is not registered. "
            "Its driver extra is probably not installed. Registered types: %s.",
            agent_id,
            db_type,
            ", ".join(sorted(CONNECTOR_REGISTRY)) or "(none)",
        )
        return None

    try:
        connector = connector_cls()
        connector.connect(credentials_for(connector_id, cfg))
        if config.CONNECTOR_CACHE_ENABLED:
            _cache_put(connector_id, connector, fingerprint)
        return PermittedConnector(connector, execution)
    except Exception:  # noqa: BLE001
        # With a traceback, as PostgresConnector.execute_query already does for
        # the same class of failure -- and that log line is the only reason a TLS
        # default silently refusing every query was ever diagnosed. The message
        # is what carries the diagnosis: "server does not support SSL, but SSL
        # was required" is the whole answer, and no summary of mine would have
        # said it.
        logger.warning(
            "No connector could be opened for agent %s (connector %s, db_type %r): "
            "the connection attempt failed.",
            agent_id,
            connector_id,
            db_type,
            exc_info=True,
        )
        return None
