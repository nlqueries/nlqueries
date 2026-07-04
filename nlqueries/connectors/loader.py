"""
nlqueries.connectors.loader
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Open a live DatabaseConnector from the local connectors file.

Handles the sanitisation mismatch between agent IDs (underscores) and
connector IDs (colons), e.g. MCP tool receives ``postgres_localhost_imdb``
while the connector is keyed as ``postgres:localhost:imdb``.
"""

from __future__ import annotations

import re
from typing import Any

import yaml

from nlqueries import config
from nlqueries.connectors import CONNECTOR_REGISTRY, DatabaseConnector

_KEYRING_SERVICE = "nlqueries"


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


def open_connector_for_agent(agent_id: str) -> DatabaseConnector | None:
    """Open and return a connected DatabaseConnector for *agent_id*.

    Returns ``None`` when the connector cannot be found, the required driver
    is not installed, or the connection attempt fails.  Callers should treat
    the returned connector as short-lived and not cache it across requests.
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

    db_type = cfg.get("db_type", "").lower()
    connector_cls = CONNECTOR_REGISTRY.get(db_type)
    if connector_cls is None:
        return None

    try:
        from sqlalchemy.engine import make_url  # noqa: PLC0415

        url = _get_full_url(connector_id, cfg)
        parsed = make_url(url)
        connector = connector_cls()
        connector.connect(
            {
                "host": parsed.host or cfg.get("host", "localhost"),
                "port": parsed.port or cfg.get("port"),
                "database": parsed.database or cfg.get("database"),
                "user": parsed.username or cfg.get("user"),
                "password": parsed.password or _load_password(connector_id, cfg),
                "account": cfg.get("account"),
                "warehouse": cfg.get("warehouse"),
                "schema": cfg.get("schema"),
            }
        )
        return connector
    except Exception:  # noqa: BLE001
        return None
