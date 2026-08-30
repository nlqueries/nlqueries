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
#: through -- see the pass-through in :func:`open_connector_for_agent`.
_LOADER_ONLY_KEYS = frozenset({"db_type", "url", "password_storage"})


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


def _find_connector_id(agent_id: str) -> str | None:
    """Return the connector_id in connectors.yaml that matches *agent_id*.

    Tries a direct lookup first (CLI usage where agent_id == connector_id),
    then a sanitized lookup (MCP usage where colons were replaced by ``_``).
    """
    if not config.CONNECTORS_FILE.exists():
        return None
    connectors: dict[str, Any] = yaml.safe_load(config.CONNECTORS_FILE.read_text()) or {}
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


def open_connector_for_agent(
    agent_id: str, execution: ExecutionPolicy = DEFAULT_POLICY
) -> DatabaseConnector | None:
    """Return a connected DatabaseConnector for *agent_id*.

    Returns ``None`` when the connector cannot be found, the required driver is
    not installed, or the connection attempt fails.

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
    connector_id = _find_connector_id(agent_id)
    if connector_id is None:
        return None

    if not config.CONNECTORS_FILE.exists():
        return None

    connectors: dict[str, Any] = yaml.safe_load(config.CONNECTORS_FILE.read_text()) or {}
    cfg = connectors.get(connector_id)
    if not cfg:
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
        return None

    try:
        from sqlalchemy.engine import make_url  # noqa: PLC0415

        url = _get_full_url(connector_id, cfg)
        parsed = make_url(url)
        connector = connector_cls()
        # Start from the stored configuration rather than an allow-list of keys.
        # The allow-list silently dropped every key it did not name, and the
        # dropped keys included `ssl_mode`, `ssl_ca_cert`, `ssl_client_cert` and
        # `ssl_client_key` -- the whole of a connector's TLS configuration. So a
        # connector configured for a database with no TLS could not be opened
        # (the resolved mode fell back to `require`), and one configured to
        # verify a private CA was opened with `require` instead: encrypted, but
        # verifying nothing, which is the posture the operator had explicitly
        # configured against. Neither failure was visible from the connector.
        #
        # A pass-through keeps the loader out of the business of knowing which
        # keys a connector understands. Every connector reads what it wants with
        # `.get()`, so a key it does not recognise costs nothing -- whereas the
        # allow-list has to be extended for each one and had already fallen
        # behind by four.
        credentials = {k: v for k, v in cfg.items() if k not in _LOADER_ONLY_KEYS}
        # The URL stays authoritative for the fields it carries: `_get_full_url`
        # resolves a keychain-stored password into it, so reading these off the
        # parsed URL rather than off `cfg` is what makes that indirection work.
        credentials.update(
            {
                "host": parsed.host or cfg.get("host", "localhost"),
                "port": parsed.port or cfg.get("port"),
                "database": parsed.database or cfg.get("database"),
                "user": parsed.username or cfg.get("user"),
                "password": parsed.password or _load_password(connector_id, cfg),
            }
        )
        connector.connect(credentials)
        if config.CONNECTOR_CACHE_ENABLED:
            _cache_put(connector_id, connector, fingerprint)
        return PermittedConnector(connector, execution)
    except Exception:  # noqa: BLE001
        return None
