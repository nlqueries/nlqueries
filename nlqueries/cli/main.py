"""
nlqueries.cli.main
~~~~~~~~~~~~~~~~~~
Entry point for the `nlqueries` command-line tool.

Commands
--------
  connect          Test a DB connection and register it as a named connector.
  extract-schema   Inspect a registered connector and print schema statistics.
  process-history  Run the Query Capsule pipeline over recent query history.
  export-kb        Generate and save the YAML knowledge base for a connector.
  ask              Ask an agent a natural-language question and stream the response.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import click
import yaml
from rich.console import Console
from rich.table import Table

# Suppress the HuggingFace Hub unauthenticated-request advisory that fires on
# every embedding call even when the model is fully cached locally.  Setting
# the env var before any HF import is the official suppression mechanism;
# setdefault means a user-supplied value is never overridden.
os.environ.setdefault("HF_HUB_VERBOSITY", "error")

from nlqueries.config import CONNECTORS_FILE, KB_PATH, QDRANT_URL
from nlqueries.connectors import CONNECTOR_REGISTRY

console = Console()
err_console = Console(stderr=True)

# ---------------------------------------------------------------------------
# Supported DB types -> SQLAlchemy driver schemes
# ---------------------------------------------------------------------------
_DB_SCHEMES: dict[str, str] = {
    "postgres": "postgresql+psycopg2",
    "postgresql": "postgresql+psycopg2",
    "mysql": "mysql+pymysql",
    "bigquery": "bigquery",
    "snowflake": "snowflake",
    "redshift": "redshift+redshift_connector",
    "mssql": "mssql+pymssql",
    "duckdb": "duckdb",
}

_DEFAULT_PORTS: dict[str, int] = {
    "postgres": 5432,
    "postgresql": 5432,
    "mysql": 3306,
    "bigquery": 443,
    "snowflake": 443,
    "redshift": 5439,
    "mssql": 1433,
}


# ---------------------------------------------------------------------------
# Connector registry helpers
# ---------------------------------------------------------------------------


def _load_connectors() -> dict[str, dict[str, Any]]:
    if CONNECTORS_FILE.exists():
        return yaml.safe_load(CONNECTORS_FILE.read_text()) or {}
    return {}


def _save_connector(connector_id: str, config: dict[str, Any]) -> None:
    CONNECTORS_FILE.parent.mkdir(parents=True, exist_ok=True)
    connectors = _load_connectors()
    connectors[connector_id] = config
    CONNECTORS_FILE.write_text(yaml.dump(connectors, default_flow_style=False, sort_keys=False))
    CONNECTORS_FILE.chmod(0o600)


_KEYRING_SERVICE = "nlqueries"


def _save_password(connector_id: str, password: str) -> bool:
    """Store *password* in the OS keychain for *connector_id*. Returns True on success."""
    try:
        import keyring as _kr  # noqa: PLC0415

        _kr.set_password(_KEYRING_SERVICE, connector_id, password)
        return True
    except Exception:  # noqa: BLE001
        return False


def _load_password(connector_id: str, cfg: dict[str, Any]) -> str | None:
    """Return the connector password from the OS keychain or the stored URL (legacy)."""
    if cfg.get("password_storage") == "keychain":
        try:
            import keyring as _kr  # noqa: PLC0415

            return _kr.get_password(_KEYRING_SERVICE, connector_id)
        except Exception:  # noqa: BLE001
            return None
    # Legacy path: password embedded in the SQLAlchemy URL.
    try:
        from sqlalchemy.engine import make_url as _mu  # noqa: PLC0415

        return _mu(cfg["url"]).password
    except Exception:  # noqa: BLE001
        return None


def _get_full_url(connector_id: str, cfg: dict[str, Any]) -> str:
    """Return the connection URL with the password injected from the keychain when needed."""
    url: str = cfg["url"]
    if cfg.get("password_storage") == "keychain":
        pwd = _load_password(connector_id, cfg)
        if pwd is not None:
            try:
                from sqlalchemy.engine import make_url as _mu  # noqa: PLC0415

                url = _mu(url).set(password=pwd).render_as_string(hide_password=False)
            except Exception:  # noqa: BLE001
                pass
    return url


def _session_path(agent_id: str) -> Path:
    safe_id = re.sub(r"[^\w.-]", "_", agent_id)
    path = Path.home() / ".nlqueries" / "sessions" / f"{safe_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _load_session(agent_id: str) -> list[Any]:
    """Return stored ConversationTurn objects for *agent_id*, newest last."""
    from datetime import datetime as _dt

    from nlqueries.orchestrator.conversation import ConversationTurn

    path = _session_path(agent_id)
    if not path.exists():
        return []
    turns: list[ConversationTurn] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
            turns.append(
                ConversationTurn(
                    role=raw["role"],
                    content=raw["content"],
                    agent_type=raw.get("agent_type"),
                    sql=raw.get("sql"),
                    timestamp=_dt.fromisoformat(raw["timestamp"]),
                )
            )
        except (KeyError, ValueError):
            continue
    return turns


def _save_turn(agent_id: str, role: str, content: str, **kwargs: Any) -> None:
    """Append a single ConversationTurn to the session JSONL file."""
    record = {
        "role": role,
        "content": content,
        "agent_type": kwargs.get("agent_type"),
        "sql": kwargs.get("sql"),
        "timestamp": datetime.now(UTC).isoformat(),
    }
    with _session_path(agent_id).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def _resolve_alias(value: str) -> str:
    """Resolve a connector alias to its full connector ID.

    If ``value`` is already a registered connector ID it is returned as-is.
    If it matches an ``alias`` field on any connector entry the real ID is
    returned instead.  Falls through unchanged when neither matches so the
    caller receives a clear "not found" error from ``_require_connector``.
    """
    connectors = _load_connectors()
    if value in connectors:
        return value
    for cid, cfg in connectors.items():
        if cfg.get("alias") == value:
            return cid
    return value


def _require_connector(connector_id: str) -> dict[str, Any]:
    connectors = _load_connectors()
    if connector_id not in connectors:
        raise click.ClickException(
            f"Connector '{connector_id}' not found.\n"
            f"  Register it first:  nlqueries connect <db-type> "
            f"--database <db> --user <u> --password <p>\n"
            f"  List connectors:    nlqueries connectors"
        )
    return connectors[connector_id]


def _build_url(
    db_type: str,
    host: str,
    port: int,
    database: str,
    user: str,
    password: str,
    account: str | None = None,  # Snowflake
    project: str | None = None,  # BigQuery
) -> str:
    db_type_l = db_type.lower()
    scheme = _DB_SCHEMES.get(db_type_l)
    if scheme is None:
        raise click.ClickException(
            f"Unsupported db-type '{db_type}'. Supported: {', '.join(sorted(set(_DB_SCHEMES)))}"
        )

    if db_type_l == "bigquery":
        proj = project or database
        return f"bigquery://{proj}"

    if db_type_l == "snowflake":
        acct = account or host
        return f"snowflake://{quote_plus(user)}:{quote_plus(password)}@{acct}/{database}"

    if db_type_l == "duckdb":
        # DuckDB uses a file path or :memory: — no host/port/user/password in the URL.
        db_path = database or ":memory:"
        return f"duckdb:///{db_path}" if db_path != ":memory:" else "duckdb:///:memory:"

    return f"{scheme}://{quote_plus(user)}:{quote_plus(password)}@{host}:{port}/{database}"


# ---------------------------------------------------------------------------
# Health-check helpers (#34)
# ---------------------------------------------------------------------------


class _CheckResult:
    """Result of a single service health check."""

    def __init__(self, service: str, status: str, detail: str, error: str = "") -> None:
        self.service = service
        self.status = status  # "ok" | "fail" | "skip" | "warn"
        self.detail = detail
        self.error = error


def _check_qdrant(qdrant_url: str) -> _CheckResult:
    import json as _json  # noqa: PLC0415
    import urllib.error  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415

    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(f"{qdrant_url}/", timeout=3) as resp:
            data = _json.loads(resp.read())
            ms = int((time.monotonic() - t0) * 1000)
            version = data.get("version", "?")
            host = qdrant_url.removeprefix("http://").removeprefix("https://")
            return _CheckResult("Qdrant", "ok", f"reachable at {host} (version {version}, {ms} ms)")
    except urllib.error.URLError as exc:
        return _CheckResult(
            "Qdrant",
            "fail",
            f"connection refused — is Qdrant running at {qdrant_url}?",
            str(exc),
        )
    except Exception as exc:  # noqa: BLE001
        return _CheckResult("Qdrant", "fail", f"unexpected error: {exc}", str(exc))


def _check_connectors(connector_filter: str | None) -> list[_CheckResult]:
    connectors = _load_connectors()
    if not connectors:
        return [
            _CheckResult(
                "Database", "skip", "no connectors registered — run 'nlqueries connect' first"
            )
        ]

    if connector_filter:
        resolved = _resolve_alias(connector_filter)
        if resolved not in connectors:
            return [_CheckResult("Database", "fail", f"connector '{connector_filter}' not found")]
        connectors = {resolved: connectors[resolved]}

    results: list[_CheckResult] = []
    for cid, cfg in connectors.items():
        db_type = cfg.get("db_type", "")
        alias = cfg.get("alias", "")
        label_name = alias if alias else cid
        service = f"Database ({label_name})"

        connector_cls = CONNECTOR_REGISTRY.get(db_type.lower())
        if connector_cls is None:
            results.append(
                _CheckResult(service, "skip", f"no connector class for db_type '{db_type}'")
            )
            continue

        try:
            from sqlalchemy.engine import make_url  # noqa: PLC0415

            parsed = make_url(cfg["url"])
            connector = connector_cls()
            connector.connect(
                {
                    "host": parsed.host or cfg.get("host", "localhost"),
                    "port": parsed.port or cfg.get("port"),
                    "database": parsed.database or cfg.get("database"),
                    "user": parsed.username or cfg.get("user"),
                    "password": _load_password(cid, cfg),
                    "account": cfg.get("account"),
                    "warehouse": cfg.get("warehouse"),
                    "schema": cfg.get("schema"),
                }
            )
            t0 = time.monotonic()
            qr = connector.execute_query("SELECT 1")
            ms = int((time.monotonic() - t0) * 1000)
            if qr.error:
                results.append(_CheckResult(service, "fail", f"SELECT 1 failed: {qr.error}"))
            else:
                db_name = cfg.get("database", "?")
                results.append(_CheckResult(service, "ok", f"{db_name} connected ({ms} ms)"))
        except Exception as exc:  # noqa: BLE001
            results.append(
                _CheckResult(
                    service, "fail", "connection failed — is the database running?", str(exc)
                )
            )

    return results


def _check_llm() -> _CheckResult:
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"))
    if not has_key:
        return _CheckResult(
            "LLM provider", "fail", "no API key set (ANTHROPIC_API_KEY or OPENAI_API_KEY)"
        )

    from nlqueries.config import LLM_MODEL, LLM_PROVIDER  # noqa: PLC0415

    try:
        from nlqueries.llm import get_llm_client  # noqa: PLC0415

        llm = get_llm_client()
        t0 = time.monotonic()
        llm.complete("You are a health check.", "Reply OK.", max_tokens=5)
        ms = int((time.monotonic() - t0) * 1000)
        return _CheckResult(
            "LLM provider", "ok", f"{LLM_PROVIDER} — {LLM_MODEL} responds ({ms} ms)"
        )
    except Exception as exc:  # noqa: BLE001
        return _CheckResult(
            "LLM provider", "fail", f"{LLM_PROVIDER} — {LLM_MODEL}: {exc}", str(exc)
        )


def _check_embedding() -> _CheckResult:
    from nlqueries.embeddings.embed_server import _DEFAULT_PORT  # noqa: PLC0415
    from nlqueries.embeddings.embedder import _try_daemon_single  # noqa: PLC0415

    vec = _try_daemon_single("health")
    if vec is not None:
        return _CheckResult(
            "Embedding model", "ok", f"daemon running on port {_DEFAULT_PORT} (dim {len(vec)})"
        )
    return _CheckResult(
        "Embedding model",
        "warn",
        "daemon not running — embeddings load ~9 s per invocation; "
        "start with 'nlqueries embed-server start'",
    )


def _check_config() -> _CheckResult:
    issues: list[str] = []
    notes: list[str] = []

    if KB_PATH.exists():
        notes.append("KB_PATH exists")
    else:
        issues.append(f"KB_PATH {KB_PATH} not found (run export-kb to create)")

    if os.environ.get("QDRANT_URL"):
        notes.append("QDRANT_URL set")
    else:
        notes.append("QDRANT_URL using default (localhost:6333)")

    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        issues.append("no LLM API key set")

    if issues:
        return _CheckResult("Config", "warn", "; ".join(issues))
    return _CheckResult("Config", "ok", ", ".join(notes))


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(package_name="nlqueries-core")
def cli() -> None:
    """NLQueries — natural-language query engine.

    \b
    Translate plain-English questions into SQL, build a self-updating YAML
    knowledge base from your schema, and expose everything via MCP or CLI.

    \b
    Typical workflow:
      1. nlqueries connect postgres --database mydb --user me --password s3cr3t
      2. nlqueries extract-schema postgres:localhost:mydb
      3. nlqueries process-history postgres:localhost:mydb --days 90
      4. nlqueries export-kb postgres:localhost:mydb --output kb.yaml
    """


# ---------------------------------------------------------------------------
# connect
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("db_type")
@click.option("--host", default="localhost", show_default=True, help="Database host.")
@click.option("--port", default=None, type=int, help="Database port (default varies by db-type).")
@click.option("--database", default=None, help="Database / catalog / project name.")
@click.option("--user", default=None, help="Database user.")
@click.option(
    "--password",
    default=None,
    hide_input=True,
    help=(
        "Database password. Omit to be prompted interactively "
        "(the value never appears in shell history)."
    ),
)
@click.option(
    "--password-env",
    "password_env",
    default=None,
    metavar="VAR",
    help="Read the password from environment variable VAR instead of the command line.",
)
@click.option("--account", default=None, help="Snowflake account identifier.")
@click.option("--warehouse", default=None, help="Snowflake warehouse to use.")
@click.option("--schema", "db_schema", default=None, help="Snowflake schema (optional).")
@click.option("--project-id", "project_id", default=None, help="BigQuery / GCP project ID.")
@click.option("--dataset-id", "dataset_id", default=None, help="BigQuery dataset ID (optional).")
@click.option(
    "--service-account-json",
    "service_account_json",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a BigQuery service-account JSON key file (omit to use ADC).",
)
@click.option(
    "--connector-id",
    "connector_id",
    default=None,
    help="Name to register this connector under (auto-generated if omitted).",
)
@click.option(
    "--alias",
    "alias",
    default=None,
    metavar="NAME",
    help=(
        "Short alias for this connector (e.g. 'prod'). "
        "Can also be set later with `nlqueries alias`."
    ),
)
def connect(
    db_type: str,
    host: str,
    port: int | None,
    database: str | None,
    user: str | None,
    password: str | None,
    password_env: str | None,
    account: str | None,
    warehouse: str | None,
    db_schema: str | None,
    project_id: str | None,
    dataset_id: str | None,
    service_account_json: str | None,
    connector_id: str | None,
    alias: str | None,
) -> None:
    """Test a database connection and register it as a named connector.

    \b
    DB_TYPE  one of: postgres, mysql, bigquery, snowflake, redshift, mssql, duckdb

    \b
    Examples:
      nlqueries connect postgres --database mydb --user alice --password secret
      nlqueries connect redshift --host cluster.abc.us-east-1.redshift.amazonaws.com \\
          --database dev --user awsuser --password secret
      nlqueries connect mssql --host myserver.database.windows.net \\
          --database mydb --user alice --password secret
      nlqueries connect duckdb --database /data/warehouse.db
      nlqueries connect snowflake --account acme-prod --database PROD --user bob \\
          --password s3cr3t --warehouse COMPUTE_WH --schema PUBLIC
      nlqueries connect bigquery --project-id acme-prod --dataset-id analytics \\
          --service-account-json /path/to/key.json
    """
    db_type_l = db_type.lower()

    # ------------------------------------------------------------------
    # Resolve the password without letting it hit shell history.
    # Priority: --password-env VAR > --password <value> > interactive prompt.
    # BigQuery uses service-account auth and never needs a password.
    # ------------------------------------------------------------------
    # DuckDB has no user/password — skip credential prompting entirely.
    _no_auth_types = {"bigquery", "duckdb"}
    if db_type_l not in _no_auth_types:
        if password_env is not None:
            password = os.environ.get(password_env) or ""
            if not password:
                raise click.ClickException(
                    f"Environment variable '{password_env}' is not set or is empty."
                )
        elif password is None:
            password = click.prompt("Database password", hide_input=True, default="")

    # Each db-type has a different minimal set of required credentials —
    # validate them up front with a clear, actionable message rather than
    # letting the connector fail with a confusing driver-level error.
    if db_type_l == "bigquery":
        project_id = project_id or database
        if not project_id:
            raise click.ClickException(
                "bigquery requires --project-id (or --database).\n"
                "  Example: nlqueries connect bigquery --project-id acme-prod "
                "--dataset-id analytics"
            )
    elif db_type_l == "snowflake":
        missing = [
            name
            for name, value in (
                ("--account", account),
                ("--warehouse", warehouse),
                ("--database", database),
                ("--user", user),
                ("--password", password),
            )
            if not value
        ]
        if missing:
            raise click.ClickException(
                f"snowflake requires {', '.join(missing)}.\n"
                f"  Example: nlqueries connect snowflake --account acme-prod "
                f"--database PROD --user bob --password s3cr3t --warehouse COMPUTE_WH"
            )
    elif db_type_l == "duckdb":
        # DuckDB only needs a database path (or :memory: which is the default).
        pass
    else:
        missing = [
            name
            for name, value in (
                ("--database", database),
                ("--user", user),
                ("--password", password),
            )
            if not value
        ]
        if missing:
            raise click.ClickException(
                f"{db_type} requires {', '.join(missing)}.\n"
                f"  Example: nlqueries connect {db_type_l} --database mydb "
                f"--user alice --password secret"
            )

    # Resolve port
    resolved_port: int = port or _DEFAULT_PORTS.get(db_type_l, 5432)

    # Build connection URL
    try:
        url = _build_url(
            db_type,
            host,
            resolved_port,
            database or "",
            user or "",
            password or "",
            account=account,
            project=project_id,
        )
    except click.ClickException:
        raise

    cid = connector_id or f"{db_type_l}:{host}:{database or project_id}"

    if db_type_l == "bigquery":
        console.print(f"[bold]Connecting[/bold] to BigQuery project [cyan]{project_id}[/cyan] …")
    elif db_type_l == "duckdb":
        db_label = database or ":memory:"
        cid = connector_id or f"duckdb:{db_label}"
        console.print(f"[bold]Connecting[/bold] to DuckDB at [cyan]{db_label}[/cyan] …")
    else:
        console.print(
            f"[bold]Connecting[/bold] to {db_type} at "
            f"[cyan]{host}:{resolved_port}/{database}[/cyan] …"
        )

    connector_cls = CONNECTOR_REGISTRY.get(db_type_l)

    try:
        if connector_cls is not None:
            # Use the registered DatabaseConnector implementation (e.g. PostgresConnector,
            # SnowflakeConnector). Pass through every credential field the connector might
            # need — extra keys are simply ignored by connectors that don't use them.
            connector = connector_cls()
            connector.connect(
                {
                    "host": host,
                    "port": resolved_port,
                    "database": database,
                    "user": user,
                    "password": password,
                    "account": account,
                    "warehouse": warehouse,
                    "schema": db_schema,
                    "project_id": project_id,
                    "dataset_id": dataset_id,
                    "service_account_json": service_account_json,
                }
            )
            if not connector.test_connection():
                raise RuntimeError("test_connection() returned False")
        else:
            # No dedicated connector registered for this db-type yet — fall back
            # to a raw SQLAlchemy connectivity check.
            from sqlalchemy import create_engine, text

            connect_args: dict[str, Any] = {}
            if db_type.lower() in ("postgres", "postgresql"):
                connect_args["connect_timeout"] = 10

            engine = create_engine(url, connect_args=connect_args)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))

    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[bold red]✗ Connection failed:[/bold red] {exc}")
        sys.exit(1)

    console.print("[bold green]✓ Connection successful.[/bold green]")

    # ------------------------------------------------------------------
    # Persist connector config.  Try to store the password in the OS keychain
    # so it never lives in plain text on disk.  Fall back to the old URL-
    # embedded format if keyring is unavailable (e.g. headless CI).
    # ------------------------------------------------------------------
    stored_in_keychain = False
    if password:
        stored_in_keychain = _save_password(cid, password)

    # Build the URL to store — strip the password when keychain is available.
    if stored_in_keychain:
        from sqlalchemy.engine import make_url as _mu  # noqa: PLC0415

        stored_url = str(_mu(url).set(password=None))
    else:
        stored_url = url

    config: dict[str, Any] = {
        "db_type": db_type_l,
        "host": host,
        "port": resolved_port,
        "database": database,
        "user": user,
        "url": stored_url,
        "registered": datetime.now(UTC).isoformat(),
    }
    if stored_in_keychain:
        config["password_storage"] = "keychain"
    if alias:
        config["alias"] = alias
    if db_type_l == "snowflake":
        config["account"] = account
        config["warehouse"] = warehouse
        if db_schema:
            config["schema"] = db_schema
    if db_type_l == "bigquery":
        config["project_id"] = project_id
        if dataset_id:
            config["dataset_id"] = dataset_id
        if service_account_json:
            config["service_account_json"] = service_account_json
    if db_type_l == "duckdb":
        # DuckDB has no host/port/user — store the database path directly.
        config["host"] = ""
        config["port"] = 0
        config["user"] = ""
    _save_connector(cid, config)

    console.print(f"  Connector registered as [bold]{cid!r}[/bold]")
    if alias:
        console.print(f"  Alias               : [bold]{alias}[/bold]")
    console.print(f"  Config saved to [dim]{CONNECTORS_FILE}[/dim]")
    if stored_in_keychain:
        console.print(
            "  [green]✓ Password stored in OS keychain[/green] (not written to the config file)."
        )
    else:
        console.print(
            "  [yellow]Note:[/yellow] keyring unavailable — password stored in config file. "
            "Ensure it is not world-readable."
        )


# ---------------------------------------------------------------------------
# alias
# ---------------------------------------------------------------------------


@cli.command("alias")
@click.argument("connector_id")
@click.argument("alias_name")
def set_alias(connector_id: str, alias_name: str) -> None:
    """Set a short alias for a connector ID.

    \b
    CONNECTOR_ID  the full connector identifier (e.g. postgres:localhost:mydb)
    ALIAS_NAME    the short name to use instead (e.g. mydb)

    \b
    Once set, every command that accepts a connector or agent ID will also
    accept the alias transparently.

    \b
    Example:
      nlqueries alias postgres:localhost:dvdrental dvdrental
      nlqueries ask dvdrental "How many customers?"
    """
    connector_id = _resolve_alias(connector_id)
    cfg = _require_connector(connector_id)
    cfg["alias"] = alias_name
    _save_connector(connector_id, cfg)
    console.print(
        f"  [bold green]✓[/bold green] Alias [bold]{alias_name!r}[/bold] "
        f"→ [dim]{connector_id}[/dim]"
    )


# ---------------------------------------------------------------------------
# update-password
# ---------------------------------------------------------------------------


@cli.command("update-password")
@click.argument("connector_id")
@click.option(
    "--password-env",
    "password_env",
    default=None,
    metavar="VAR",
    help="Read new password from environment variable VAR instead of prompting.",
)
def update_password(connector_id: str, password_env: str | None) -> None:
    """Rotate the password for a registered connector.

    \b
    CONNECTOR_ID  the connector to update (alias accepted)

    \b
    Prompts for the new password interactively (hidden input) unless
    --password-env VAR is given, in which case the value is read from
    that environment variable.

    \b
    The connector's connection is not re-tested — run 'nlqueries extract-schema
    <CONNECTOR_ID>' afterwards to verify the new credential works.

    \b
    Example:
      nlqueries update-password postgres:localhost:mydb
      nlqueries update-password mydb --password-env DB_PASSWORD
    """
    connector_id = _resolve_alias(connector_id)
    cfg = _require_connector(connector_id)

    if password_env is not None:
        new_password = os.environ.get(password_env) or ""
        if not new_password:
            raise click.ClickException(
                f"Environment variable '{password_env}' is not set or is empty."
            )
    else:
        new_password = click.prompt(
            f"New password for '{connector_id}'", hide_input=True, confirmation_prompt=True
        )

    stored = _save_password(connector_id, new_password)
    if stored:
        # Ensure the config marks this connector as using keychain storage and
        # the stored URL does not contain the old password.
        if cfg.get("password_storage") != "keychain":
            cfg["password_storage"] = "keychain"
            try:
                from sqlalchemy.engine import make_url as _mu  # noqa: PLC0415

                cfg["url"] = str(_mu(cfg["url"]).set(password=None))
            except Exception:  # noqa: BLE001
                pass
            _save_connector(connector_id, cfg)
        console.print(
            f"  [bold green]✓[/bold green] Password updated in OS keychain "
            f"for [cyan]{connector_id}[/cyan]."
        )
    else:
        console.print(
            "  [yellow]⚠ keyring unavailable — cannot store password securely.[/yellow]\n"
            "  Set the password manually in the connector URL inside "
            f"[dim]{CONNECTORS_FILE}[/dim]."
        )


# ---------------------------------------------------------------------------
# feedback
# ---------------------------------------------------------------------------


@cli.command("feedback")
@click.argument("agent_id")
@click.option("--question", required=True, help="The natural-language question that was asked.")
@click.option(
    "--generated-sql",
    "generated_sql",
    default="",
    help="The SQL the agent produced (optional — leave blank if unknown).",
)
@click.option(
    "--corrected-sql",
    "corrected_sql",
    default=None,
    help="The correct SQL (supply for thumbs-down corrections).",
)
@click.option("--thumbs-up", "rating", flag_value="up", help="Mark the answer as correct.")
@click.option("--thumbs-down", "rating", flag_value="down", help="Mark the answer as incorrect.")
def submit_feedback(
    agent_id: str,
    question: str,
    generated_sql: str,
    corrected_sql: str | None,
    rating: str | None,
) -> None:
    """Record thumbs-up or thumbs-down feedback for a query answer.

    \b
    AGENT_ID  the agent whose answer you are rating (alias accepted)

    \b
    Examples:
      nlqueries feedback dvdrental --question "Top customers?" --thumbs-up
      nlqueries feedback dvdrental \\
        --question "How many films have more than one actor?" \\
        --thumbs-down \\
        --corrected-sql "SELECT COUNT(*) FROM (SELECT film_id FROM film_actor
          GROUP BY film_id HAVING COUNT(actor_id) > 1) sub"
    """
    if not rating:
        raise click.UsageError("Provide either --thumbs-up or --thumbs-down.")

    agent_id = _resolve_alias(agent_id)

    from nlqueries.feedback.models import QueryFeedback
    from nlqueries.feedback.store import record_feedback

    fb = QueryFeedback(
        question=question,
        generated_sql=generated_sql,
        corrected_sql=corrected_sql,
        rating=rating,
        agent_id=agent_id,
    )
    record_feedback(fb)

    symbol = "[bold green]✓[/bold green]" if rating == "up" else "[bold red]✗[/bold red]"
    label = "thumbs-up" if rating == "up" else "thumbs-down"
    console.print(f"  {symbol} Feedback recorded ({label}) for [cyan]{agent_id}[/cyan]")
    if corrected_sql:
        console.print(
            f"  [dim]Correction saved. Re-run 'nlqueries export-kb {agent_id}' to apply it.[/dim]"
        )


# ---------------------------------------------------------------------------
# promote-feedback
# ---------------------------------------------------------------------------


@cli.command("promote-feedback")
@click.argument("agent_id")
def promote_feedback_cmd(agent_id: str) -> None:
    """Promote positively-rated feedback into the verified Qdrant collection.

    \b
    AGENT_ID  the agent whose feedback file to promote (alias accepted)

    Loads all ``thumbs-up`` feedback for AGENT_ID, validates each SQL against
    the current knowledge base schema, and upserts qualifying (question, SQL)
    pairs into the ``agent_{id}_verified`` Qdrant collection so future
    ``ask`` commands can blend them into the prompt as verified examples.

    \b
    This command is also called automatically at the end of 'export-kb'.

    \b
    Example:
      nlqueries promote-feedback postgres:localhost:mydb
    """
    agent_id = _resolve_alias(agent_id)

    from nlqueries.feedback.promoter import promote_feedback

    console.print(f"[bold]Promoting feedback[/bold] for [cyan]{agent_id}[/cyan] …")
    try:
        count = promote_feedback(agent_id)
    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[bold red]✗ Promotion failed:[/bold red] {exc}")
        sys.exit(1)

    if count:
        console.print(
            f"  [bold green]✓[/bold green] [bold]{count}[/bold] verified "
            f"pair(s) upserted to Qdrant collection "
            f"[cyan]agent_{agent_id}_verified[/cyan]."
        )
    else:
        console.print(
            "  [dim]No qualifying feedback found (need thumbs-up ratings with valid SQL).[/dim]"
        )


# ---------------------------------------------------------------------------
# connectors
# ---------------------------------------------------------------------------


@cli.command("connectors")
def list_connectors() -> None:
    """List all registered connectors and their aliases."""
    from rich.table import Table as _Table

    connectors = _load_connectors()
    if not connectors:
        console.print("[dim]No connectors registered. Run 'nlqueries connect' first.[/dim]")
        console.print(f"[dim]Config file: {CONNECTORS_FILE}[/dim]")
        return

    tbl = _Table(show_header=True, header_style="bold cyan")
    tbl.add_column("Connector ID")
    tbl.add_column("Type")
    tbl.add_column("Alias")

    for cid, cfg in connectors.items():
        alias = cfg.get("alias", "")
        db_type = cfg.get("db_type", "")
        tbl.add_row(cid, db_type, f"[bold]{alias}[/bold]" if alias else "[dim]—[/dim]")

    console.print(tbl)
    console.print(f"[dim]Config: {CONNECTORS_FILE}[/dim]")


# ---------------------------------------------------------------------------
# extract-schema
# ---------------------------------------------------------------------------


@cli.command("extract-schema")
@click.argument("connector_id")
def extract_schema(connector_id: str) -> None:
    """Inspect a connector's schema and print a summary.

    \b
    CONNECTOR_ID  the name used when you ran 'nlqueries connect'
                  (e.g. postgres:localhost:mydb)

    \b
    Prints:
      - Number of tables / views discovered
      - Total column count
      - Per-table row counts (via COUNT(*), sampled up to 50 tables)
    """
    connector_id = _resolve_alias(connector_id)
    cfg = _require_connector(connector_id)

    console.print(f"[bold]Extracting schema[/bold] for connector [cyan]{connector_id}[/cyan] …")

    connector_cls = CONNECTOR_REGISTRY.get(cfg.get("db_type", "").lower())

    if connector_cls is not None:
        # Use the registered DatabaseConnector implementation (e.g. PostgresConnector).
        try:
            from sqlalchemy.engine import make_url

            parsed = make_url(cfg["url"])
            connector = connector_cls()
            connector.connect(
                {
                    "host": parsed.host or cfg.get("host", "localhost"),
                    "port": parsed.port or cfg.get("port"),
                    "database": parsed.database or cfg.get("database"),
                    "user": parsed.username or cfg.get("user"),
                    "password": _load_password(connector_id, cfg),
                    # Snowflake-specific fields — absent from the URL, read from
                    # the persisted connector config (see `connect`).
                    "account": cfg.get("account"),
                    "warehouse": cfg.get("warehouse"),
                    "schema": cfg.get("schema"),
                }
            )
            schema = connector.extract_schema()
        except Exception as exc:  # noqa: BLE001
            err_console.print(f"[bold red]✗ Schema extraction failed:[/bold red] {exc}")
            sys.exit(1)

        total_columns = sum(len(t.columns) for t in schema.tables)

        console.print()
        console.print("[bold green]✓ Schema extraction complete[/bold green]")
        console.print(f"  Database: [bold]{schema.database}[/bold]")
        console.print(f"  Tables  : [bold]{len(schema.tables)}[/bold]")
        console.print(f"  Columns : [bold]{total_columns}[/bold] total across all tables")
        console.print(f"  Extracted at: [dim]{schema.extracted_at}[/dim]")

        if schema.tables:
            console.print()
            tbl = Table(
                "Schema",
                "Table",
                "Columns",
                "Rows",
                show_header=True,
                header_style="bold cyan",
            )
            for table_spec in schema.tables[:20]:
                rows_str = str(table_spec.row_count) if table_spec.row_count is not None else "—"
                tbl.add_row(
                    table_spec.schema,
                    table_spec.name,
                    str(len(table_spec.columns)),
                    rows_str,
                )
            if len(schema.tables) > 20:
                tbl.add_row(f"… and {len(schema.tables) - 20} more", "", "", "")
            console.print(tbl)
        return

    # --- Fallback: no dedicated connector registered for this db-type ---
    try:
        from sqlalchemy import create_engine, func, select, table
        from sqlalchemy import inspect as sa_inspect

        engine = create_engine(_get_full_url(connector_id, cfg))

        with engine.connect() as conn:
            inspector = sa_inspect(engine)
            table_names: list[str] = inspector.get_table_names()
            view_names: list[str] = inspector.get_view_names()

            total_columns = 0
            table_stats: list[dict[str, Any]] = []

            for tbl_name in table_names:
                cols = inspector.get_columns(tbl_name)
                col_count = len(cols)
                total_columns += col_count

                # Row count (best-effort; skip on error)
                try:
                    stmt = select(func.count()).select_from(table(tbl_name))
                    row = conn.execute(stmt).scalar()
                    row_count: int | None = int(row) if row is not None else None
                except Exception:  # noqa: BLE001
                    row_count = None

                table_stats.append(
                    {
                        "table": tbl_name,
                        "columns": col_count,
                        "rows": row_count,
                    }
                )

    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[bold red]✗ Schema extraction failed:[/bold red] {exc}")
        sys.exit(1)

    # --- Print summary ---
    console.print()
    console.print("[bold green]✓ Schema extraction complete[/bold green]")
    console.print(f"  Tables : [bold]{len(table_names)}[/bold]")
    console.print(f"  Views  : [bold]{len(view_names)}[/bold]")
    console.print(f"  Columns: [bold]{total_columns}[/bold] total across all tables")

    if table_stats:
        console.print()
        tbl = Table("Table", "Columns", "Rows", show_header=True, header_style="bold cyan")
        # Show up to 20 tables; summarise the rest
        for stat in table_stats[:20]:
            rows_str = str(stat["rows"]) if stat["rows"] is not None else "—"
            tbl.add_row(stat["table"], str(stat["columns"]), rows_str)
        if len(table_stats) > 20:
            tbl.add_row(f"… and {len(table_stats) - 20} more", "", "")
        console.print(tbl)


# ---------------------------------------------------------------------------
# doc-ingest
# ---------------------------------------------------------------------------


@cli.command("doc-ingest")
@click.argument("source_id")
@click.argument("file_path", type=click.Path(exists=True, dir_okay=False, readable=True))
def doc_ingest(source_id: str, file_path: str) -> None:
    """Ingest a document file and print the number of chunks produced.

    \b
    SOURCE_ID  opaque identifier for this document (e.g. a UUID or slug).
               Used to generate deterministic chunk IDs and for Qdrant filtering.
    FILE_PATH  path to the document file to ingest (must exist)

    \b
    Supported formats (requires the [docs] extra):
      .pdf — PDF documents via pdfplumber

    \b
    Example:
      nlqueries doc-ingest my-policy-doc-v1 /path/to/policy.pdf
    """
    from nlqueries.document_connectors import DOCUMENT_CONNECTOR_REGISTRY

    src = Path(file_path)
    suffix = src.suffix.lower()

    connector_key = next(
        (key for key, cls in DOCUMENT_CONNECTOR_REGISTRY.items() if cls().supports(src)),
        None,
    )
    if connector_key is None:
        raise click.ClickException(
            f"No document connector registered for '{suffix}'. "
            f"Supported extensions: "
            + ", ".join(
                f".{key}" if not key.startswith(".") else key for key in DOCUMENT_CONNECTOR_REGISTRY
            )
        )

    connector = DOCUMENT_CONNECTOR_REGISTRY[connector_key]()
    console.print(
        f"[bold]Ingesting[/bold] [cyan]{src.name}[/cyan] "
        f"(source_id=[bold]{source_id}[/bold], connector=[bold]{connector_key}[/bold]) …"
    )

    try:
        chunks = connector.ingest(src, source_id)
    except ImportError as exc:
        err_console.print(f"[bold red]✗ Missing dependency:[/bold red] {exc}")
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[bold red]✗ Ingestion failed:[/bold red] {exc}")
        sys.exit(1)

    console.print("[bold green]✓ Ingestion complete.[/bold green]")
    console.print(f"  Chunks produced : [bold]{len(chunks)}[/bold]")
    if chunks:
        pages = sorted({c.page_number for c in chunks if c.page_number is not None})
        console.print(f"  Pages covered   : [bold]{len(pages)}[/bold]")


# ---------------------------------------------------------------------------
# process-history
# ---------------------------------------------------------------------------


@cli.command("process-history")
@click.argument("connector_id")
@click.option(
    "--days",
    default=90,
    show_default=True,
    type=int,
    help="Number of days of query history to process.",
)
@click.option(
    "--min-executions",
    default=3,
    show_default=True,
    type=int,
    help="Minimum execution count for a query to be included.",
)
@click.option(
    "--max-queries",
    default=None,
    show_default=False,
    type=int,
    help=(
        "Maximum number of useful queries to return after filtering "
        f"(default: {500}, override with QUERY_HISTORY_LIMIT env var)."
    ),
)
@click.option(
    "--annotate/--no-annotate",
    default=True,
    show_default=True,
    help="Annotate capsules with LLM-generated intent descriptions.",
)
@click.option(
    "--embed/--no-embed",
    default=False,
    show_default=True,
    help="Upsert capsules into the Qdrant vector store after processing (requires Qdrant).",
)
@click.option(
    "--verbose/--no-verbose",
    default=False,
    show_default=True,
    help="Print sqlglot parse warnings and LLM provider log lines. Off by default.",
)
def process_history(
    connector_id: str,
    days: int,
    min_executions: int,
    max_queries: int | None,
    annotate: bool,
    embed: bool,
    verbose: bool,
) -> None:
    """Run the Query Capsule pipeline over recent query history.

    \b
    CONNECTOR_ID  the name used when you ran 'nlqueries connect'

    Reads query history from the information schema (or pg_stat_statements
    for PostgreSQL), de-duplicates and parameterises queries, clusters
    them by intent, and emits Query Capsules — normalised, annotated query
    templates ready for embedding and LLM context injection.

    \b
    Example:
      nlqueries process-history postgres:localhost:mydb --days 30
    """
    if not verbose:
        import logging as _logging

        for _noisy in ("sqlglot", "LiteLLM", "litellm", "httpx"):
            _logging.getLogger(_noisy).setLevel(_logging.ERROR)

    connector_id = _resolve_alias(connector_id)
    cfg = _require_connector(connector_id)

    # Preflight: LLM API key required when --annotate is on (the default).
    has_llm_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if annotate and not has_llm_key:
        err_console.print(
            "[bold red]✗ --annotate requires an LLM API key.[/bold red]\n"
            "  Set [bold]ANTHROPIC_API_KEY[/bold] or [bold]OPENAI_API_KEY[/bold] "
            "before running this command.\n"
            "  To skip annotation, run with [bold]--no-annotate[/bold]."
        )
        sys.exit(1)

    # Preflight: Qdrant must be reachable when --embed is on.
    if embed:
        import httpx as _httpx

        try:
            _httpx.get(f"{QDRANT_URL}/healthz", timeout=3.0).raise_for_status()
        except Exception:  # noqa: BLE001
            err_console.print(
                f"[bold red]✗ --embed requires Qdrant to be running at "
                f"{QDRANT_URL}[/bold red]\n"
                "  Start it with: [bold]docker run -d --name qdrant "
                "-p 6333:6333 qdrant/qdrant[/bold]\n"
                "  Or skip embedding with [bold]--no-embed[/bold] and run it later.\n"
                "  See Appendix B in the guide for full setup instructions."
            )
            sys.exit(1)

    console.print(
        f"[bold]Processing query history[/bold] for [cyan]{connector_id}[/cyan] "
        f"(last [bold]{days}[/bold] days) …"
    )

    connector_cls = CONNECTOR_REGISTRY.get(cfg.get("db_type", "").lower())
    if connector_cls is None:
        err_console.print(
            f"[bold red]✗ No connector registered for db_type '{cfg.get('db_type')}'.[/bold red]"
        )
        sys.exit(1)

    try:
        from sqlalchemy.engine import make_url

        from nlqueries.processing.pipeline import save_capsules

        parsed = make_url(cfg["url"])
        connector = connector_cls()
        connector.connect(
            {
                "host": parsed.host or cfg.get("host", "localhost"),
                "port": parsed.port or cfg.get("port"),
                "database": parsed.database or cfg.get("database"),
                "user": parsed.username or cfg.get("user"),
                "password": _load_password(connector_id, cfg),
                "account": cfg.get("account"),
                "warehouse": cfg.get("warehouse"),
                "schema": cfg.get("schema"),
                "project_id": cfg.get("project_id"),
                "dataset_id": cfg.get("dataset_id"),
                "service_account_json": cfg.get("service_account_json"),
            }
        )

        # Schema extraction is best-effort; the pipeline works without it
        # but uses SchemaSpec column types to sharpen placeholder typing.
        schema = None
        try:
            schema = connector.extract_schema()
        except Exception:  # noqa: BLE001
            console.print(
                "  [yellow]⚠ Schema extraction failed — "
                "placeholder types may default to VARCHAR.[/yellow]"
            )

        from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

        from nlqueries.config import QUERY_HISTORY_LIMIT
        from nlqueries.processing.parameterizer import parameterize_clusters
        from nlqueries.processing.query_clusterer import cluster_queries
        from nlqueries.processing.query_filter import filter_and_deduplicate

        effective_limit = max_queries if max_queries is not None else QUERY_HISTORY_LIMIT
        raw_limit = min(effective_limit * 3, 10_000)

        # Stage 1 — Extract
        with console.status(f"  [1] Extracting query history (last {days} days) …"):
            records = connector.extract_query_history(days=days, limit=raw_limit)
        console.print(f"  [1] Extracted [bold]{len(records)}[/bold] raw records")

        # Stage 2 — Filter + deduplicate
        with console.status("  [2] Filtering and deduplicating …"):
            filter_stats: dict[str, int] = {}
            normalized = filter_and_deduplicate(
                records, min_executions=min_executions, _stats=filter_stats
            )
            filter_stats["useful"] = len(normalized)
            normalized = normalized[:effective_limit]
        console.print(f"  [2] [bold]{len(normalized)}[/bold] useful queries kept")

        # Stage 3 — Cluster + parameterize
        with console.status("  [3] Clustering and parameterizing …"):
            clusters = cluster_queries(normalized)
            capsules = parameterize_clusters(clusters, schema=schema)
        console.print(f"  [3] [bold]{len(capsules)}[/bold] query capsules produced")

        # Stage 4 — Annotate (optional); per-capsule progress bar driven by
        # the on_capsule_done callback so it updates as concurrent threads finish.
        if annotate and capsules:
            from nlqueries.llm import get_llm_client
            from nlqueries.processing.intent_annotator import annotate_capsules

            _llm = get_llm_client()
            with Progress(
                SpinnerColumn(),
                TextColumn("  [4] Annotating"),
                BarColumn(),
                MofNCompleteColumn(),
                console=console,
            ) as _prog:
                _task = _prog.add_task("", total=len(capsules))
                annotate_capsules(capsules, _llm, on_capsule_done=lambda: _prog.advance(_task))

        # Stage 5 — Embed (optional)
        if embed and capsules:
            from nlqueries.config import QDRANT_COLLECTION
            from nlqueries.embeddings.qdrant_store import ensure_collection, upsert_capsules

            with console.status("  [5] Embedding capsules into Qdrant …"):
                ensure_collection(QDRANT_COLLECTION)
                upsert_capsules(QDRANT_COLLECTION, capsules)
            console.print(f"  [5] [bold]{len(capsules)}[/bold] capsules embedded into Qdrant")

        out_path = save_capsules(capsules, connector_id)

    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[bold red]✗ Pipeline failed:[/bold red] {exc}")
        sys.exit(1)

    annotated = sum(1 for c in capsules if c.intent)
    sys_dropped = filter_stats.get("system_command", 0) + filter_stats.get("system_schema", 0)

    console.print("[bold green]✓ Pipeline complete.[/bold green]")

    # Filter breakdown — helps diagnose why capsule count is low.
    if filter_stats:
        considered = filter_stats.get("considered", 0)
        useful = filter_stats.get("useful", 0)
        cap_reached = useful > effective_limit
        cap_note = (
            f"  [yellow](cap reached — {useful} useful found; "
            f"raise with --max-queries {effective_limit * 2})[/yellow]"
            if cap_reached
            else ""
        )
        console.print(f"  Queries scanned     : [bold]{considered}[/bold]{cap_note}")
        if filter_stats.get("too_few_executions"):
            console.print(
                f"  Dropped (low freq)  : [bold]{filter_stats['too_few_executions']}[/bold]"
                f"  [dim](lower with --min-executions 1)[/dim]"
            )
        if sys_dropped:
            console.print(f"  Dropped (system)    : [bold]{sys_dropped}[/bold]")
        if filter_stats.get("not_select"):
            console.print(f"  Dropped (non-SELECT): [bold]{filter_stats['not_select']}[/bold]")
        if filter_stats.get("normalize_failed"):
            console.print(
                f"  Dropped (parse err) : [bold]{filter_stats['normalize_failed']}[/bold]"
            )
        if filter_stats.get("duplicate"):
            console.print(f"  Duplicates merged   : [bold]{filter_stats['duplicate']}[/bold]")

    console.print(f"  Capsules produced   : [bold]{len(capsules)}[/bold]")
    if annotate:
        console.print(f"  Annotated           : [bold]{annotated}[/bold] / {len(capsules)}")
    if embed:
        console.print(f"  Embedded            : [bold]{len(capsules)}[/bold] capsules into Qdrant")
    console.print(f"  Saved to            : [dim]{out_path}[/dim]")

    # Warn when all queries appear to be internal driver queries.
    if len(capsules) == 0 and sys_dropped > 0:
        console.print(
            "\n  [yellow]⚠ No capsules produced — all queries appear to be internal "
            "PostgreSQL driver queries.[/yellow]\n"
            "  Run real business queries against your database first, then re-run "
            "process-history.\n"
            "  See 'Fresh or lightly-used databases' in the guide for examples."
        )


# ---------------------------------------------------------------------------
# export-kb
# ---------------------------------------------------------------------------


@cli.command("export-kb")
@click.argument("connector_id")
@click.option(
    "--output",
    "-o",
    default=None,
    show_default=False,
    type=click.Path(dir_okay=False, writable=True),
    help=(
        "Path to write the YAML knowledge base. "
        "Defaults to ~/.nlqueries/knowledge_base/<connector_id>.yaml "
        "(the same location that 'ask' and 'query' read from)."
    ),
)
@click.option(
    "--include-samples/--no-include-samples",
    default=True,
    show_default=True,
    help="Include sample rows for each table.",
)
@click.option(
    "--sample-rows",
    default=3,
    show_default=True,
    type=int,
    help="Number of sample rows to include per table.",
)
@click.option(
    "--describe-columns/--no-describe-columns",
    default=False,
    show_default=True,
    help=(
        "Use the LLM to auto-populate column descriptions from sample data. "
        "Skips surrogate-key columns (PKs, FKs, *_id/*_uuid/*_key). "
        "Manually written descriptions in an existing KB always win. "
        "Requires LLM_API_KEY to be set."
    ),
)
def export_kb(
    connector_id: str,
    output: str | None,
    include_samples: bool,
    sample_rows: int,
    describe_columns: bool,
) -> None:
    """Generate and save the YAML knowledge base for a connector.

    \b
    CONNECTOR_ID  the name used when you ran 'nlqueries connect'

    The knowledge base is a structured YAML file that describes your schema
    — tables, columns, types, foreign keys, and sample rows — in a format
    optimised for LLM context injection.

    \b
    Example:
      nlqueries export-kb postgres:localhost:mydb
      nlqueries export-kb postgres:localhost:mydb --output kb.yaml --sample-rows 5
    """
    connector_id = _resolve_alias(connector_id)
    cfg = _require_connector(connector_id)

    # Derive the canonical path used by `ask` and `query` when no --output given.
    if output is None:
        safe_id = re.sub(r"[^\w.-]", "_", connector_id)
        out_path = KB_PATH / f"{safe_id}.yaml"
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path = Path(output)

    console.print(f"[bold]Generating knowledge base[/bold] for [cyan]{connector_id}[/cyan] …")

    connector_cls = CONNECTOR_REGISTRY.get(cfg.get("db_type", "").lower())

    if connector_cls is not None:
        # Registered connector path — use SchemaSpec + kb_generator
        try:
            from sqlalchemy.engine import make_url

            from nlqueries.knowledge.kb_generator import (
                describe_columns as _describe_columns,
            )
            from nlqueries.knowledge.kb_generator import (
                generate_knowledge_base,
                save_knowledge_base,
            )
            from nlqueries.processing.pipeline import load_capsules

            parsed = make_url(cfg["url"])
            connector = connector_cls()
            connector.connect(
                {
                    "host": parsed.host or cfg.get("host", "localhost"),
                    "port": parsed.port or cfg.get("port"),
                    "database": parsed.database or cfg.get("database"),
                    "user": parsed.username or cfg.get("user"),
                    "password": _load_password(connector_id, cfg),
                    "account": cfg.get("account"),
                    "warehouse": cfg.get("warehouse"),
                    "schema": cfg.get("schema"),
                    "project_id": cfg.get("project_id"),
                    "dataset_id": cfg.get("dataset_id"),
                    "service_account_json": cfg.get("service_account_json"),
                }
            )
            schema = connector.extract_schema()

            # Preserve manual annotations from a previously generated KB
            existing_kb: dict[str, Any] | None = None
            if out_path.exists():
                existing_kb = yaml.safe_load(out_path.read_text(encoding="utf-8"))

            # Load capsules produced by process-history (best-effort)
            capsules = []
            try:
                capsules = load_capsules(connector_id)
            except FileNotFoundError:
                console.print(
                    "  [yellow]⚠ No capsules found — run process-history first "
                    "to include query_capsules in the KB.[/yellow]"
                )

            # Inject thumbs-down corrections as high-priority capsules so the
            # LLM sees the correct pattern before any auto-generated ones.
            try:
                from nlqueries.feedback.store import load_feedback
                from nlqueries.processing.parameterizer import QueryCapsule

                feedback_records = load_feedback(connector_id)
                correction_count = 0
                for fb in feedback_records:
                    if fb.rating == "down" and fb.corrected_sql:
                        capsules.insert(
                            0,
                            QueryCapsule(
                                template_sql=fb.corrected_sql,
                                placeholders=[],
                                tables=[],
                                columns=[],
                                frequency=999,
                                auto_description=f"User correction for: {fb.question}",
                                intent=f"Correction: {fb.question}",
                            ),
                        )
                        correction_count += 1
                if correction_count:
                    console.print(
                        f"  [dim]Incorporated {correction_count} feedback "
                        f"correction(s) as high-priority capsules.[/dim]"
                    )
            except Exception:  # noqa: BLE001
                pass

            # Optional: LLM-generated column descriptions from sample data
            llm_column_descriptions: dict[str, dict[str, str]] | None = None
            if describe_columns:
                has_llm_key = bool(
                    os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("LLM_API_KEY")
                )
                if not has_llm_key:
                    console.print(
                        "  [yellow]⚠ --describe-columns skipped: "
                        "LLM_API_KEY / ANTHROPIC_API_KEY not set.[/yellow]"
                    )
                else:
                    from nlqueries.llm import get_llm_client

                    llm = get_llm_client()
                    llm_column_descriptions = {}
                    for tbl in schema.tables:
                        result = connector.execute_query(
                            f"SELECT * FROM {tbl.name} LIMIT {sample_rows}"
                        )
                        if result.error:
                            continue
                        descs = _describe_columns(tbl, result.rows, result.columns, llm)
                        if descs:
                            llm_column_descriptions[tbl.name] = descs
                    described = sum(len(v) for v in llm_column_descriptions.values())
                    console.print(
                        f"  [dim]LLM described {described} column(s) "
                        f"across {len(llm_column_descriptions)} table(s).[/dim]"
                    )

            kb: dict[str, Any] = generate_knowledge_base(
                schema,
                capsules,
                agent_name=connector_id,
                existing_kb=existing_kb,
                llm_column_descriptions=llm_column_descriptions,
            )
            save_knowledge_base(kb, str(out_path))

        except Exception as exc:  # noqa: BLE001
            err_console.print(f"[bold red]✗ Knowledge base generation failed:[/bold red] {exc}")
            sys.exit(1)

        table_count = len(kb["schema"]["tables"])
        column_count = sum(len(t["columns"]) for t in kb["schema"]["tables"])
        capsule_count = len(kb["query_capsules"])

        console.print(
            f"[bold green]✓ Knowledge base written to[/bold green] [cyan]{out_path}[/cyan]"
        )
        console.print(f"  Tables   : [bold]{table_count}[/bold]")
        console.print(f"  Columns  : [bold]{column_count}[/bold]")
        console.print(f"  Capsules : [bold]{capsule_count}[/bold]")

        # Phase 5B: auto-promote positive feedback into the verified collection.
        try:
            from nlqueries.feedback.promoter import promote_feedback

            promoted = promote_feedback(connector_id)
            if promoted:
                console.print(
                    f"  [dim]Promoted {promoted} verified feedback pair(s) to Qdrant.[/dim]"
                )
        except Exception:  # noqa: BLE001
            pass  # best-effort; never fail export-kb because of this

        return

    # Fallback: no registered connector — raw SQLAlchemy introspection
    try:
        from sqlalchemy import create_engine, select, table, text
        from sqlalchemy import inspect as sa_inspect

        engine = create_engine(_get_full_url(connector_id, cfg))
        inspector = sa_inspect(engine)

        table_names = inspector.get_table_names()
        kb_fallback: dict[str, Any] = {
            "meta": {
                "connector_id": connector_id,
                "db_type": cfg["db_type"],
                "database": cfg["database"],
                "generated_at": datetime.now(UTC).isoformat(),
                "nlqueries_version": "0.1.0",
            },
            "tables": {},
        }

        with engine.connect() as conn:
            for tbl_name in table_names:
                columns = inspector.get_columns(tbl_name)
                pk_cols = inspector.get_pk_constraint(tbl_name).get("constrained_columns", [])
                fkeys = inspector.get_foreign_keys(tbl_name)
                indexes = inspector.get_indexes(tbl_name)

                col_defs = [
                    {
                        "name": c["name"],
                        "type": str(c["type"]),
                        "nullable": c.get("nullable", True),
                        "primary_key": c["name"] in pk_cols,
                        **({"default": str(c["default"])} if c.get("default") is not None else {}),
                    }
                    for c in columns
                ]

                fkey_defs = [
                    {
                        "columns": fk["constrained_columns"],
                        "references_table": fk["referred_table"],
                        "references_columns": fk["referred_columns"],
                    }
                    for fk in fkeys
                ]

                index_defs = [
                    {
                        "name": idx["name"],
                        "columns": idx["column_names"],
                        "unique": idx.get("unique", False),
                    }
                    for idx in indexes
                ]

                samples: list[dict[str, Any]] = []
                if include_samples and sample_rows > 0:
                    try:
                        stmt = select(text("*")).select_from(table(tbl_name)).limit(sample_rows)
                        rows = conn.execute(stmt)
                        col_names = list(rows.keys())
                        samples = [dict(zip(col_names, row, strict=False)) for row in rows]
                        samples = [
                            {
                                k: (
                                    str(v)
                                    if not isinstance(v, (str, int, float, bool, type(None)))
                                    else v
                                )
                                for k, v in row.items()
                            }
                            for row in samples
                        ]
                    except Exception:  # noqa: BLE001
                        samples = []

                kb_fallback["tables"][tbl_name] = {
                    "columns": col_defs,
                    "foreign_keys": fkey_defs,
                    "indexes": index_defs,
                    **({"sample_rows": samples} if include_samples else {}),
                }

    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[bold red]✗ Knowledge base generation failed:[/bold red] {exc}")
        sys.exit(1)

    # Write YAML
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        yaml.dump(kb_fallback, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)

    table_count = len(kb_fallback["tables"])
    column_count = sum(len(t["columns"]) for t in kb_fallback["tables"].values())

    console.print(f"[bold green]✓ Knowledge base written to[/bold green] [cyan]{out_path}[/cyan]")
    console.print(f"  Tables : [bold]{table_count}[/bold]")
    console.print(f"  Columns: [bold]{column_count}[/bold]")
    if include_samples:
        console.print(f"  Sample rows per table: up to {sample_rows}")
    console.print()
    console.print(
        "  [dim]Next step:[/dim] embed this knowledge base into Qdrant:\n"
        "  [dim]  python -m nlqueries.embeddings --kb-file {out_path}[/dim]"
    )


# ---------------------------------------------------------------------------
# annotate
# ---------------------------------------------------------------------------


@cli.command("annotate")
@click.argument("connector_id")
def annotate(connector_id: str) -> None:
    """Annotate saved Query Capsules with LLM-generated intent descriptions.

    \b
    CONNECTOR_ID  the name used when you ran 'nlqueries connect'

    Loads capsules saved by 'process-history' and calls the configured LLM to
    fill in the intent field with a concise business-question description.

    \b
    Example:
      nlqueries annotate postgres:localhost:mydb
    """
    connector_id = _resolve_alias(connector_id)
    _require_connector(connector_id)

    console.print(f"[bold]Annotating capsules[/bold] for [cyan]{connector_id}[/cyan] …")

    try:
        from nlqueries.llm import get_llm_client
        from nlqueries.processing.intent_annotator import annotate_capsules
        from nlqueries.processing.pipeline import load_capsules, save_capsules

        capsules = load_capsules(connector_id)

    except FileNotFoundError as exc:
        err_console.print(f"[bold red]✗ {exc}[/bold red]")
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[bold red]✗ Failed to load capsules:[/bold red] {exc}")
        sys.exit(1)

    if not capsules:
        console.print("[yellow]⚠ No capsules to annotate. Run process-history first.[/yellow]")
        return

    console.print(f"  Found [bold]{len(capsules)}[/bold] capsules to annotate.")

    try:
        llm = get_llm_client()
        annotate_capsules(capsules, llm)
        out_path = save_capsules(capsules, connector_id)

    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[bold red]✗ Annotation failed:[/bold red] {exc}")
        sys.exit(1)

    annotated = sum(1 for c in capsules if c.intent)
    console.print("[bold green]✓ Annotation complete.[/bold green]")
    console.print(f"  Annotated : [bold]{annotated}[/bold] / {len(capsules)} capsules")
    console.print(f"  Saved to  : [dim]{out_path}[/dim]")


# ---------------------------------------------------------------------------
# verify-oidc-token
# ---------------------------------------------------------------------------


@cli.command("verify-oidc-token")
@click.argument("discovery_url")
@click.argument("client_id")
@click.argument("id_token")
def verify_oidc_token(discovery_url: str, client_id: str, id_token: str) -> None:
    """Verify an OIDC ID token and print the decoded claims as JSON.

    \b
    DISCOVERY_URL  OIDC provider well-known endpoint
                   (e.g. https://accounts.google.com/.well-known/openid-configuration)
    CLIENT_ID      OAuth2 client ID the token was issued for
    ID_TOKEN       The raw JWT ID token string to verify

    \b
    Useful for debugging OIDC setups — confirms the token signature, expiry,
    audience, and issuer are all valid before wiring up the enterprise SSO flow.

    \b
    Example:
      nlqueries verify-oidc-token \\
        https://accounts.google.com/.well-known/openid-configuration \\
        my-client-id \\
        eyJhbGci...
    """
    import json as _json

    from nlqueries.auth.oidc_token import OidcTokenVerifier, OidcVerificationError

    try:
        verifier = OidcTokenVerifier(discovery_url)
        claims = verifier.verify(id_token, client_id)
    except OidcVerificationError as exc:
        err_console.print(f"[bold red]✗ OIDC verification failed:[/bold red] {exc}")
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[bold red]✗ Unexpected error:[/bold red] {exc}")
        sys.exit(1)

    output = {
        "sub": claims.sub,
        "email": claims.email,
        "name": claims.name,
        "given_name": claims.given_name,
        "family_name": claims.family_name,
        "picture": claims.picture,
        "email_verified": claims.email_verified,
        "raw": claims.raw,
    }
    console.print_json(_json.dumps(output, default=str))
    console.print("[bold green]✓ Token verified successfully.[/bold green]")


# ---------------------------------------------------------------------------
# ask
# ---------------------------------------------------------------------------


@cli.command("ask")
@click.argument("agent_id")
@click.argument("question")
@click.option(
    "--dialect",
    default="postgres",
    show_default=True,
    type=click.Choice(["postgres", "snowflake", "bigquery"]),
    help="SQL dialect used for generation and AST validation.",
)
def ask(agent_id: str, question: str, dialect: str) -> None:
    """Ask an agent a natural-language question and stream the response.

    \b
    AGENT_ID  the agent identifier whose knowledge base to query
              (must have been generated with 'nlqueries export-kb')
    QUESTION  the natural-language question, in quotes

    \b
    In interactive (TTY) mode: prints the natural-language reasoning
    response, then prints the validated SQL as a formatted line.
    The raw structured JSON chunk is suppressed.

    When stdout is redirected (pipe/file): emits all tokens including
    the final JSON chunk so scripts can parse it (e.g. | tail -1 | jq .sql).

    \b
    Example:
      nlqueries ask postgres:localhost:mydb "How many orders last month?"
      nlqueries ask my_agent "Top customers by revenue" --dialect snowflake
    """
    agent_id = _resolve_alias(agent_id)
    import json as _json

    from nlqueries.orchestrator import Orchestrator

    orchestrator = Orchestrator()

    async def _stream() -> None:
        try:
            if sys.stdout.isatty():
                # TTY: buffer tokens, strip the final JSON chunk, print cleanly.
                tokens: list[str] = []
                async for token in orchestrator.handle_question(
                    question, agent_id, dialect=dialect
                ):
                    tokens.append(token)
                sql: str | None = None
                text_tokens = tokens
                if tokens:
                    try:
                        last = _json.loads(tokens[-1])
                        if isinstance(last, dict) and last.get("type") == "sql":
                            text_tokens = tokens[:-1]
                            sql = last.get("sql") or None
                    except (ValueError, TypeError):
                        pass
                click.echo("".join(text_tokens))
                if sql:
                    console.print(f"\n[bold]SQL:[/bold] [dim]{sql}[/dim]")
            else:
                # Pipe / redirected: emit raw tokens including the JSON chunk.
                async for token in orchestrator.handle_question(
                    question, agent_id, dialect=dialect
                ):
                    click.echo(token, nl=False)
                click.echo()
        except FileNotFoundError as exc:
            err_console.print(f"[bold red]✗ {exc}[/bold red]")
            sys.exit(1)
        except Exception as exc:  # noqa: BLE001
            err_console.print(f"[bold red]✗ {exc}[/bold red]")
            sys.exit(1)

    asyncio.run(_stream())


# ---------------------------------------------------------------------------
# doc-ask
# ---------------------------------------------------------------------------


@cli.command("doc-ask")
@click.argument("collection")
@click.argument("question")
@click.option(
    "--source-id",
    "source_id",
    default=None,
    help="Restrict retrieval to a specific document source ID.",
)
@click.option(
    "--top-k",
    "top_k",
    default=5,
    show_default=True,
    type=int,
    help="Number of document chunks to retrieve.",
)
def doc_ask(collection: str, question: str, source_id: str | None, top_k: int) -> None:
    """Ask a question against an ingested document collection.

    \b
    COLLECTION  Qdrant collection name (convention: doc_{source_id}_chunks)
    QUESTION    The natural-language question, in quotes

    \b
    Streams the LLM answer to stdout, then prints a formatted citations block.

    \b
    Example:
      nlqueries doc-ask doc_my-policy-uuid_chunks "What is the refund policy?"
      nlqueries doc-ask doc_my-policy-uuid_chunks "Refund window?" --source-id my-policy-uuid
    """
    import json as _json

    from nlqueries.orchestrator.document_orchestrator import DocumentOrchestrator

    orchestrator = DocumentOrchestrator()

    async def _stream() -> None:
        citations: list[dict[str, Any]] = []
        try:
            async for token in orchestrator.handle_question(
                question,
                collection,
                source_id=source_id,
                top_k=top_k,
            ):
                try:
                    parsed = _json.loads(token)
                    if isinstance(parsed, dict) and parsed.get("type") == "citations":
                        citations.extend(parsed.get("citations", []))
                        continue
                except (ValueError, TypeError):
                    pass
                click.echo(token, nl=False)
            click.echo()  # final newline after streamed answer

            if citations:
                console.print("\n[bold cyan]Citations:[/bold cyan]")
                for i, cite in enumerate(citations, 1):
                    source = cite.get("source_name", "")
                    page = cite.get("page_number")
                    excerpt = cite.get("excerpt", "")
                    location = f"page {page}" if page is not None else "no page"
                    console.print(f"  [bold]{i}.[/bold] {source} ({location})")
                    if excerpt:
                        short = excerpt[:120] + "…" if len(excerpt) > 120 else excerpt
                        console.print(f"     [dim]{short}[/dim]")

        except Exception as exc:  # noqa: BLE001
            err_console.print(f"[bold red]✗ {exc}[/bold red]")
            sys.exit(1)

    asyncio.run(_stream())


# ---------------------------------------------------------------------------
# doc-sync-notion
# ---------------------------------------------------------------------------


@cli.command("doc-sync-notion")
@click.argument("source_id")
@click.argument("page_id")
@click.option(
    "--since",
    "since_ts",
    default=None,
    help=(
        "Only produce chunks if the page was modified after this ISO 8601 timestamp "
        "(e.g. 2024-01-15T00:00:00+00:00). Omit for a full sync."
    ),
)
def doc_sync_notion(source_id: str, page_id: str, since_ts: str | None) -> None:
    """Sync a Notion page and print the number of chunks produced.

    \b
    SOURCE_ID  Opaque identifier for this document source (e.g. a UUID or slug).
               Used to generate deterministic chunk IDs and for Qdrant filtering.
    PAGE_ID    Notion page ID or database ID to sync.

    \b
    The Notion API token is read from the NOTION_API_TOKEN environment variable.

    \b
    Examples:
      NOTION_API_TOKEN=secret_... nlqueries doc-sync-notion my-wiki-src abc123def456
      nlqueries doc-sync-notion my-wiki-src abc123 --since 2024-01-15T00:00:00+00:00
    """
    import os

    from nlqueries.document_connectors.notion import NotionConnector

    api_token = os.environ.get("NOTION_API_TOKEN", "")
    if not api_token:
        raise click.ClickException(
            "NOTION_API_TOKEN environment variable is not set.\n"
            "  Set it first:  export NOTION_API_TOKEN=<your-integration-token>"
        )

    since: datetime | None = None
    if since_ts:
        try:
            since = datetime.fromisoformat(since_ts)
        except ValueError as exc:
            raise click.ClickException(
                f"Invalid --since value: {since_ts!r}. "
                "Expected ISO 8601 format, e.g. 2024-01-15T00:00:00+00:00."
            ) from exc

    connector = NotionConnector(api_token=api_token)
    console.print(
        f"[bold]Syncing Notion page[/bold] [cyan]{page_id}[/cyan] "
        f"(source_id=[bold]{source_id}[/bold]) …"
    )

    try:
        chunks = connector.ingest(page_id, source_id, since=since)
    except ImportError as exc:
        err_console.print(f"[bold red]✗ Missing dependency:[/bold red] {exc}")
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[bold red]✗ Sync failed:[/bold red] {exc}")
        sys.exit(1)

    console.print("[bold green]✓ Sync complete.[/bold green]")
    console.print(f"  Chunks produced : [bold]{len(chunks)}[/bold]")
    if not chunks and since is not None:
        console.print(
            "  [yellow]⚠ No chunks — page may not have been modified "
            "after the given timestamp.[/yellow]"
        )


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------


@cli.command("query")
@click.argument("agent_id")
@click.argument("question")
@click.option(
    "--dialect",
    default="postgres",
    show_default=True,
    type=click.Choice(["postgres", "snowflake", "bigquery"]),
    help="SQL dialect used for generation.",
)
@click.option(
    "--execute/--no-execute",
    "execute_sql",
    default=True,
    show_default=True,
    help="Execute the generated SQL against the database and display rows.",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    default=False,
    help="Output the full result as raw JSON.",
)
@click.option(
    "--session/--no-session",
    default=True,
    show_default=True,
    help="Carry conversation context across queries to support follow-up questions.",
)
@click.option(
    "--new-session",
    "new_session",
    is_flag=True,
    default=False,
    help="Start a fresh conversation, discarding prior context.",
)
def query(
    agent_id: str,
    question: str,
    dialect: str,
    execute_sql: bool,
    output_json: bool,
    session: bool,
    new_session: bool,
) -> None:
    """Run a synchronous agent query and print the result.

    \b
    AGENT_ID  the agent identifier whose knowledge base to query
    QUESTION  the natural-language question, in quotes

    \b
    Generates SQL via the MultiAgentOrchestrator, then executes it against
    the registered connector and displays the result rows as a table.
    Use --no-execute to skip execution and see only the generated SQL.

    \b
    Examples:
      nlqueries query my_agent "How many orders last month?"
      nlqueries query my_agent "Top customers by revenue" --no-execute
      nlqueries query my_agent "Top customers by revenue" --json
    """
    agent_id = _resolve_alias(agent_id)

    from rich.table import Table

    from nlqueries.connectors.base import QueryResult
    from nlqueries.orchestrator.sync_runner import AgentQueryResult, run_query_sync

    # Session / conversational context
    history: list[Any] = []
    if session:
        if new_session:
            _session_path(agent_id).unlink(missing_ok=True)
        else:
            history = _load_session(agent_id)

    try:
        result: AgentQueryResult = run_query_sync(
            question, agent_id, dialect=dialect, history=history
        )
    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[bold red]✗ Query failed:[/bold red] {exc}")
        sys.exit(1)

    # Persist this exchange to the session file so follow-up questions work.
    if session:
        _save_turn(agent_id, "user", question)
        _save_turn(
            agent_id,
            "assistant",
            result.answer or result.merged_answer or "",
            agent_type=result.agent_type,
            sql=result.sql,
        )

    # Execute the generated SQL against the registered connector.
    sql_result: QueryResult | None = result.sql_result
    if execute_sql and result.sql and result.agent_type == "sql" and sql_result is None:
        try:
            from sqlalchemy.engine import make_url

            cfg = _require_connector(agent_id)
            connector_cls = CONNECTOR_REGISTRY.get(cfg.get("db_type", "").lower())
            if connector_cls is None:
                err_console.print(
                    f"[yellow]⚠ No connector registered for db_type "
                    f"'{cfg.get('db_type')}' — skipping execution.[/yellow]"
                )
            else:
                parsed = make_url(cfg["url"])
                connector = connector_cls()
                connector.connect(
                    {
                        "host": parsed.host or cfg.get("host", "localhost"),
                        "port": parsed.port or cfg.get("port"),
                        "database": parsed.database or cfg.get("database"),
                        "user": parsed.username or cfg.get("user"),
                        "password": _load_password(agent_id, cfg),
                        "account": cfg.get("account"),
                        "project": cfg.get("project"),
                    }
                )
                sql_result = connector.execute_query(result.sql)
        except Exception as exc:  # noqa: BLE001
            err_console.print(f"[bold red]✗ SQL execution failed:[/bold red] {exc}")

    if output_json:
        output = {
            "question": result.question,
            "resolved_question": result.resolved_question,
            "agent_type": result.agent_type,
            "answer": result.answer,
            "sql": result.sql,
            "sql_result": (
                {
                    "columns": sql_result.columns,
                    "rows": sql_result.rows,
                    "row_count": sql_result.row_count,
                    "execution_time_ms": sql_result.execution_time_ms,
                    "error": sql_result.error,
                }
                if sql_result
                else None
            ),
            "citations": [
                {
                    "source_name": c.source_name,
                    "page_number": c.page_number,
                    "excerpt": c.excerpt,
                }
                for c in result.citations
            ],
            "merged_answer": result.merged_answer,
            "latency_ms": result.latency_ms,
            "session_id": result.session_id,
        }
        console.print_json(json.dumps(output, default=str))
    else:
        if result.resolved_question and result.resolved_question != question:
            console.print(f"[dim]Resolved   : {result.resolved_question}[/dim]")
        console.print(f"[bold]Agent type :[/bold] {result.agent_type}")
        if result.sql:
            console.print(f"[bold]SQL        :[/bold] [dim]{result.sql}[/dim]")
        if sql_result:
            if sql_result.error:
                err_console.print(f"[bold red]✗ Execution error:[/bold red] {sql_result.error}")
            else:
                tbl = Table(show_header=True, header_style="bold cyan")
                for col in sql_result.columns:
                    tbl.add_column(col)
                for row in sql_result.rows:
                    tbl.add_row(*[str(v) if v is not None else "" for v in row])
                console.print(tbl)
                console.print(
                    f"[dim]{sql_result.row_count} row(s) "
                    f"in {sql_result.execution_time_ms:.1f} ms[/dim]"
                )
        elif result.agent_type != "sql":
            console.print(f"[bold]Answer     :[/bold] {result.answer}")
        if result.citations:
            console.print(f"[bold]Citations  :[/bold] {len(result.citations)} source(s)")
        console.print(f"[bold]Latency    :[/bold] {result.latency_ms} ms")


# ---------------------------------------------------------------------------
# doc-sync-confluence
# ---------------------------------------------------------------------------


@cli.command("doc-sync-confluence")
@click.argument("source_id")
@click.argument("space_key")
@click.option(
    "--base-url",
    "base_url",
    required=True,
    help="Confluence base URL (e.g. https://example.atlassian.net).",
)
@click.option("--username", "username", required=True, help="Confluence username (email).")
@click.option(
    "--since",
    "since_ts",
    default=None,
    help=(
        "Only fetch pages modified after this ISO 8601 timestamp "
        "(e.g. 2024-01-15T00:00:00+00:00). Omit for a full sync."
    ),
)
def doc_sync_confluence(
    source_id: str,
    space_key: str,
    base_url: str,
    username: str,
    since_ts: str | None,
) -> None:
    """Sync a Confluence space and print the number of chunks produced.

    \b
    SOURCE_ID  Opaque identifier for this document source (e.g. a UUID or slug).
               Used to generate deterministic chunk IDs and for Qdrant filtering.
    SPACE_KEY  Confluence space key to sync (e.g. ENG).

    \b
    The Confluence API token is read from the CONFLUENCE_API_TOKEN environment variable.

    \b
    Examples:
      CONFLUENCE_API_TOKEN=... nlqueries doc-sync-confluence my-src ENG \\
          --base-url https://acme.atlassian.net --username alice@acme.com
      nlqueries doc-sync-confluence my-src ENG --base-url https://acme.atlassian.net \\
          --username alice@acme.com --since 2024-01-15T00:00:00+00:00
    """
    import os

    from nlqueries.document_connectors.confluence import ConfluenceConnector

    api_token = os.environ.get("CONFLUENCE_API_TOKEN", "")
    if not api_token:
        raise click.ClickException(
            "CONFLUENCE_API_TOKEN environment variable is not set.\n"
            "  Set it first:  export CONFLUENCE_API_TOKEN=<your-api-token>"
        )

    since: datetime | None = None
    if since_ts:
        try:
            since = datetime.fromisoformat(since_ts)
        except ValueError as exc:
            raise click.ClickException(
                f"Invalid --since value: {since_ts!r}. "
                "Expected ISO 8601 format, e.g. 2024-01-15T00:00:00+00:00."
            ) from exc

    connector = ConfluenceConnector(base_url=base_url, username=username, api_token=api_token)
    console.print(
        f"[bold]Syncing Confluence space[/bold] [cyan]{space_key}[/cyan] "
        f"(source_id=[bold]{source_id}[/bold]) …"
    )

    try:
        chunks = connector.ingest(space_key, source_id, since=since)
    except ImportError as exc:
        err_console.print(f"[bold red]✗ Missing dependency:[/bold red] {exc}")
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[bold red]✗ Sync failed:[/bold red] {exc}")
        sys.exit(1)

    console.print("[bold green]✓ Sync complete.[/bold green]")
    console.print(f"  Chunks produced : [bold]{len(chunks)}[/bold]")
    if not chunks and since is not None:
        console.print(
            "  [yellow]⚠ No chunks — space may have no pages modified "
            "after the given timestamp.[/yellow]"
        )


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------


@cli.group("cache")
def cache_group() -> None:
    """Manage the semantic query cache for an agent.

    \b
    The semantic cache stores recent answers in Qdrant and serves them for
    semantically similar future questions (cosine similarity >= 0.97).

    \b
    Commands:
      nlqueries cache list  <agent-id>  — list cached questions and their SQL
      nlqueries cache stats <agent-id>  — show cache statistics
      nlqueries cache clear <agent-id>  — invalidate (delete) all cached entries
    """


@cache_group.command("list")
@click.argument("agent_id")
@click.option(
    "--limit",
    default=50,
    show_default=True,
    type=int,
    help="Maximum number of entries to display.",
)
def cache_list(agent_id: str, limit: int) -> None:
    """List cached questions and their generated SQL for AGENT_ID.

    \b
    AGENT_ID  the agent identifier whose cache to inspect

    \b
    Example:
      nlqueries cache list dvdrental
      nlqueries cache list dvdrental --limit 20
    """
    from nlqueries.cache.semantic_cache import SemanticCache

    agent_id = _resolve_alias(agent_id)
    cache = SemanticCache(agent_id)
    entries = cache.list_entries(limit=limit)

    if not entries:
        console.print(f"  No cached entries for [bold]{agent_id}[/bold]. Run a few queries first.")
        return

    tbl = Table(
        title=f"Cached entries — {agent_id}",
        show_header=True,
        header_style="bold",
        show_lines=True,
    )
    tbl.add_column("#", style="dim", width=4, no_wrap=True)
    tbl.add_column("Question", min_width=28)
    tbl.add_column("Type", width=10, no_wrap=True)
    tbl.add_column("Hits", width=5, no_wrap=True)
    tbl.add_column("SQL (preview)", min_width=36)
    tbl.add_column("Cached at", width=17, no_wrap=True)

    for i, entry in enumerate(entries, 1):
        sql_text = (entry.sql or "").replace("\n", " ").strip()
        sql_preview = sql_text[:70] + ("…" if len(sql_text) > 70 else "")
        tbl.add_row(
            str(i),
            entry.question,
            entry.agent_type,
            str(entry.hit_count),
            sql_preview or "[dim]—[/dim]",
            entry.created_at.strftime("%Y-%m-%d %H:%M"),
        )

    console.print(tbl)
    console.print(
        f"  Showing [bold]{len(entries)}[/bold] of "
        f"[bold]{cache.stats()['total_entries']}[/bold] total entries."
    )


@cache_group.command("stats")
@click.argument("agent_id")
def cache_stats(agent_id: str) -> None:
    """Show semantic cache statistics for AGENT_ID.

    \b
    AGENT_ID  the agent identifier whose cache to inspect

    \b
    Example:
      nlqueries cache stats postgres:localhost:mydb
    """
    from nlqueries.cache.semantic_cache import SemanticCache

    agent_id = _resolve_alias(agent_id)
    cache = SemanticCache(agent_id)
    info = cache.stats()
    console.print(f"[bold]Cache stats[/bold] for agent [cyan]{agent_id}[/cyan]")
    console.print(f"  Collection   : [bold]{info['collection']}[/bold]")
    console.print(f"  Total entries: [bold]{info['total_entries']}[/bold]")


@cache_group.command("clear")
@click.argument("agent_id")
def cache_clear(agent_id: str) -> None:
    """Invalidate (delete) all cached entries for AGENT_ID.

    \b
    AGENT_ID  the agent identifier whose cache to clear

    \b
    Example:
      nlqueries cache clear postgres:localhost:mydb
    """
    from nlqueries.cache.semantic_cache import SemanticCache

    agent_id = _resolve_alias(agent_id)
    cache = SemanticCache(agent_id)
    cache.invalidate(agent_id)
    console.print(f"[bold green]✓ Cache cleared[/bold green] for agent [cyan]{agent_id}[/cyan]")


# ---------------------------------------------------------------------------
# feedback-stats
# ---------------------------------------------------------------------------


@cli.command("feedback-stats")
@click.argument("agent_id")
def feedback_stats(agent_id: str) -> None:
    """Show feedback statistics for an agent from the local JSONL store.

    \b
    AGENT_ID  the agent identifier whose feedback to summarise

    Reads from ~/.nlqueries/feedback/<agent-id>.jsonl and prints a summary
    of thumbs-up / thumbs-down counts plus the most recent corrections.

    \b
    Example:
      nlqueries feedback-stats postgres:localhost:mydb
    """
    from nlqueries.feedback.store import load_feedback

    records = load_feedback(agent_id)

    if not records:
        console.print(
            f"[yellow]No feedback recorded yet for [bold]{agent_id!r}[/bold].[/yellow]\n"
            f"  Submit some with 'nlqueries feedback {agent_id} --question \"...\" --thumbs-up'\n"
            '  (or --thumbs-down --corrected-sql "..."),\n'
            "  or via the enterprise chat UI / API if you're on that edition."
        )
        return

    up_count = sum(1 for r in records if r.rating == "up")
    down_count = sum(1 for r in records if r.rating == "down")
    corrections = [r for r in records if r.corrected_sql]

    console.print(f"[bold]Feedback stats[/bold] for agent [cyan]{agent_id}[/cyan]")
    console.print(f"  Total entries : [bold]{len(records)}[/bold]")
    console.print(f"  Thumbs up     : [bold green]{up_count}[/bold green]")
    console.print(f"  Thumbs down   : [bold red]{down_count}[/bold red]")

    if corrections:
        console.print(f"\n  Corrections   : [bold]{len(corrections)}[/bold]")
        tbl = Table(
            "Question",
            "Generated SQL",
            "Corrected SQL",
            show_header=True,
            header_style="bold cyan",
        )
        for rec in corrections[-5:]:  # show up to 5 most recent
            q = rec.question[:60] + "…" if len(rec.question) > 60 else rec.question
            gen = rec.generated_sql[:50] + "…" if len(rec.generated_sql) > 50 else rec.generated_sql
            cor = (
                rec.corrected_sql[:50] + "…"
                if rec.corrected_sql and len(rec.corrected_sql) > 50
                else (rec.corrected_sql or "")
            )
            tbl.add_row(q, gen, cor)
        console.print(tbl)


# ---------------------------------------------------------------------------
# health (#34)
# ---------------------------------------------------------------------------


@cli.command("health")
@click.option(
    "--connector",
    "connector_filter",
    default=None,
    metavar="ALIAS",
    help="Check only the named connector instead of all.",
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Show full error details for failures.",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    default=False,
    help="Emit results as JSON for scripting / CI health checks.",
)
def health(connector_filter: str | None, verbose: bool, output_json: bool) -> None:
    """Check the status of all services NLQueries depends on.

    \b
    Checks Qdrant, registered database connectors, LLM provider, the
    embedding model daemon, and local configuration.

    \b
    Exits 0 when all checks pass, 1 when any check fails.

    \b
    Examples:
      nlqueries health
      nlqueries health --connector dvdrental
      nlqueries health --json
      nlqueries health --verbose
    """
    import json as _json  # noqa: PLC0415

    checks: list[_CheckResult] = []
    checks.append(_check_qdrant(QDRANT_URL))
    checks.extend(_check_connectors(connector_filter))
    checks.append(_check_llm())
    checks.append(_check_embedding())
    checks.append(_check_config())

    any_fail = any(c.status == "fail" for c in checks)

    if output_json:
        payload: dict[str, object] = {
            "checks": [
                {
                    "service": c.service,
                    "status": c.status,
                    "detail": c.detail,
                    **({"error": c.error} if verbose and c.error else {}),
                }
                for c in checks
            ],
            "healthy": not any_fail,
        }
        console.print_json(_json.dumps(payload))
    else:
        _STATUS_ICON = {
            "ok": "[bold green][OK]  [/bold green]",
            "fail": "[bold red][FAIL][/bold red]",
            "skip": "[dim][SKIP][/dim] ",
            "warn": "[yellow][WARN][/yellow]",
        }
        console.print("[bold]NLQueries health check[/bold]")
        console.print("─" * 46)
        for c in checks:
            icon = _STATUS_ICON.get(c.status, c.status)
            svc = c.service.ljust(20)
            console.print(f"  {icon} {svc} {c.detail}")
            if verbose and c.error:
                for line in c.error.splitlines():
                    console.print(f"           [dim]{line}[/dim]")
        console.print()
        if any_fail:
            fail_count = sum(1 for c in checks if c.status == "fail")
            console.print(
                f"  [bold red]{fail_count} service(s) unhealthy.[/bold red] "
                "Run [bold]nlqueries health --verbose[/bold] for details."
            )
        else:
            console.print("  [bold green]All services healthy.[/bold green]")

    if any_fail:
        sys.exit(1)


# ---------------------------------------------------------------------------
# kb-stats (#35)
# ---------------------------------------------------------------------------


@cli.command("kb-stats")
@click.argument("agent_id")
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Show per-table breakdown (row count, column coverage %).",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    default=False,
    help="Emit the full report as JSON for CI or scripting.",
)
def kb_stats(agent_id: str, verbose: bool, output_json: bool) -> None:
    """Print a coverage and quality report for a knowledge base.

    \b
    AGENT_ID  the connector ID used when you ran 'nlqueries export-kb'

    \b
    Checks schema coverage (tables and columns with descriptions),
    query-capsule coverage, join coverage (when connected), and quality
    signals (feedback, cache).

    \b
    Exits 0 when the knowledge base file exists, 1 when it is missing.

    \b
    Examples:
      nlqueries kb-stats postgres:localhost:mydb
      nlqueries kb-stats postgres:localhost:mydb --verbose
      nlqueries kb-stats postgres:localhost:mydb --json
    """
    import json as _json  # noqa: PLC0415
    from datetime import datetime  # noqa: PLC0415

    from nlqueries.knowledge.kb_stats import compute_kb_stats  # noqa: PLC0415

    agent_id = _resolve_alias(agent_id)
    safe_id = re.sub(r"[^\w.-]", "_", agent_id)
    kb_path = KB_PATH / f"{safe_id}.yaml"

    # Try to connect to the live DB — best-effort; skip gracefully on failure.
    connector: Any = None
    cfg = _load_connectors().get(agent_id)
    if cfg is not None:
        connector_cls = CONNECTOR_REGISTRY.get((cfg.get("db_type") or "").lower())
        if connector_cls is not None:
            try:
                from sqlalchemy.engine import make_url  # noqa: PLC0415

                parsed = make_url(cfg["url"])
                connector = connector_cls()
                connector.connect(
                    {
                        "host": parsed.host or cfg.get("host", "localhost"),
                        "port": parsed.port or cfg.get("port"),
                        "database": parsed.database or cfg.get("database"),
                        "user": parsed.username or cfg.get("user"),
                        "password": _load_password(agent_id, cfg),
                        "account": cfg.get("account"),
                        "warehouse": cfg.get("warehouse"),
                        "schema": cfg.get("schema"),
                    }
                )
            except Exception:  # noqa: BLE001
                connector = None

    stats = compute_kb_stats(agent_id, kb_path, connector)

    if not kb_path.exists():
        err_console.print(
            f"[bold red]Error:[/bold red] no knowledge base found at {kb_path}. "
            "Run [bold]nlqueries export-kb[/bold] first."
        )
        sys.exit(1)

    def _pct(num: int, den: int) -> str:
        return f"{num / den * 100:5.0f}%" if den else "  N/A"

    def _na(val: int | None) -> str:
        return str(val) if val is not None else "N/A"

    if output_json:
        payload: dict[str, Any] = {
            "agent_id": agent_id,
            "kb_path": str(kb_path),
            "kb_age_seconds": (
                int(datetime.now(UTC).timestamp() - stats.kb_mtime)
                if stats.kb_mtime is not None
                else None
            ),
            "schema_coverage": {
                "kb_tables": stats.kb_tables,
                "kb_tables_with_desc": stats.kb_tables_with_desc,
                "kb_columns": stats.kb_columns,
                "kb_columns_with_desc": stats.kb_columns_with_desc,
                "kb_tables_with_samples": stats.kb_tables_with_samples,
                "db_tables": stats.db_tables,
                "db_columns": stats.db_columns,
                "ambiguous_columns": stats.ambiguous_columns,
            },
            "query_coverage": {
                "capsule_count": stats.capsule_count,
                "capsule_with_intent": stats.capsule_with_intent,
                "joins_in_capsules": stats.joins_in_capsules,
            },
            "join_coverage": {
                "fk_joins": stats.fk_joins,
                "fk_joins_seen": stats.fk_joins_seen,
            },
            "quality_signals": {
                "feedback_total": stats.feedback_total,
                "feedback_thumbs_up": stats.feedback_thumbs_up,
                "feedback_thumbs_down": stats.feedback_thumbs_down,
                "feedback_corrections": stats.feedback_corrections,
                "cache_entries": stats.cache_entries,
            },
        }
        if verbose:
            payload["table_details"] = [
                {
                    "name": td.name,
                    "row_count": td.row_count,
                    "column_count": td.column_count,
                    "columns_with_desc": td.columns_with_desc,
                    "has_table_desc": td.has_table_desc,
                }
                for td in stats.table_details
            ]
        console.print_json(_json.dumps(payload))
        return

    # Human-readable output
    age_str = ""
    if stats.kb_mtime is not None:
        age_s = int(datetime.now(UTC).timestamp() - stats.kb_mtime)
        if age_s < 3600:
            age_str = f" ({age_s // 60} min ago)"
        elif age_s < 86400:
            age_str = f" ({age_s // 3600} h ago)"
        else:
            age_str = f" ({age_s // 86400} day(s) ago)"
        mtime_str = datetime.fromtimestamp(stats.kb_mtime).strftime("%Y-%m-%d %H:%M")
    else:
        mtime_str = "unknown"

    console.print(f"[bold]Knowledge Base — {agent_id}[/bold]")
    console.print(f"Last exported : {mtime_str}{age_str}")
    console.print("─" * 56)

    console.print()
    console.print("[bold]Schema coverage[/bold]")
    if stats.db_tables is not None:
        missing_tables = stats.db_tables - stats.kb_tables
        console.print(f"  Tables in database        : {stats.db_tables:>4}")
        console.print(
            f"  Tables in KB              : {stats.kb_tables:>4}"
            f"   ({_pct(stats.kb_tables, stats.db_tables)})"
            + (f"   ← {missing_tables} missing" if missing_tables else "")
        )
    else:
        console.print(f"  Tables in KB              : {stats.kb_tables:>4}")
    missing_desc = stats.kb_tables - stats.kb_tables_with_desc
    console.print(
        f"  Tables with a description : {stats.kb_tables_with_desc:>4}"
        f"   ({_pct(stats.kb_tables_with_desc, stats.kb_tables)})"
        + (f"   ← {missing_desc} missing" if missing_desc else "")
    )
    if stats.db_columns is not None:
        missing_cols = stats.db_columns - stats.kb_columns
        console.print(f"  Columns in database       : {stats.db_columns:>4}")
        console.print(
            f"  Columns in KB             : {stats.kb_columns:>4}"
            f"   ({_pct(stats.kb_columns, stats.db_columns)})"
            + (f"   ← {missing_cols} missing" if missing_cols else "")
        )
    else:
        console.print(f"  Columns in KB             : {stats.kb_columns:>4}")
    missing_col_desc = stats.kb_columns - stats.kb_columns_with_desc
    console.print(
        f"  Columns with a description: {stats.kb_columns_with_desc:>4}"
        f"   ({_pct(stats.kb_columns_with_desc, stats.kb_columns)})"
        + (f"   ← {missing_col_desc} blank" if missing_col_desc else "")
    )
    if stats.ambiguous_columns:
        console.print(
            f"  [yellow]Ambiguous columns (no desc): {stats.ambiguous_columns:>4}"
            "   ← status/type/code/value/…[/yellow]"
        )
    console.print(
        f"  Tables with sample rows   : {stats.kb_tables_with_samples:>4}"
        f"   ({_pct(stats.kb_tables_with_samples, stats.kb_tables)})"
    )

    console.print()
    console.print("[bold]Query coverage[/bold]")
    console.print(f"  Query capsules            : {stats.capsule_count:>4}")
    missing_intent = stats.capsule_count - stats.capsule_with_intent
    console.print(
        f"  Capsules with intent      : {stats.capsule_with_intent:>4}"
        f"   ({_pct(stats.capsule_with_intent, stats.capsule_count)})"
        + (f"   ← {missing_intent} unannotated" if missing_intent else "")
    )
    console.print(f"  JOIN keywords in capsules : {stats.joins_in_capsules:>4}")

    if stats.fk_joins is not None:
        console.print()
        console.print("[bold]Join coverage[/bold]")
        console.print(f"  FK-declared joins         : {stats.fk_joins:>4}")
        fk_unseen = stats.fk_joins - (stats.fk_joins_seen or 0)
        console.print(
            f"  FK joins seen in capsules : {_na(stats.fk_joins_seen):>4}"
            f"   ({_pct(stats.fk_joins_seen or 0, stats.fk_joins)})"
            + (f"   ← {fk_unseen} FK joins never used" if fk_unseen else "")
        )

    console.print()
    console.print("[bold]Quality signals[/bold]")
    cache_str = _na(stats.cache_entries) if stats.cache_entries != 0 else "0"
    console.print(f"  Cache entries             : {cache_str:>4}")
    fb_detail = (
        f"   ({stats.feedback_thumbs_up} 👍  {stats.feedback_thumbs_down} 👎)"
        if stats.feedback_total
        else ""
    )
    console.print(f"  Feedback recorded         : {stats.feedback_total:>4}{fb_detail}")
    console.print(f"  Corrections applied to KB : {stats.feedback_corrections:>4}")

    if verbose and stats.table_details:
        console.print()
        console.print("[bold]Per-table breakdown[/bold]")
        from rich.table import Table  # noqa: PLC0415

        tbl = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
        tbl.add_column("Table", style="cyan", no_wrap=True)
        tbl.add_column("Rows", justify="right")
        tbl.add_column("Cols", justify="right")
        tbl.add_column("Desc%", justify="right")
        tbl.add_column("Table desc?", justify="center")
        for td in sorted(stats.table_details, key=lambda x: x.name):
            rows_str = str(td.row_count) if td.row_count is not None else "?"
            desc_pct = _pct(td.columns_with_desc, td.column_count)
            has_desc = "✓" if td.has_table_desc else "✗"
            tbl.add_row(td.name, rows_str, str(td.column_count), desc_pct, has_desc)
        console.print(tbl)

    console.print()
    if verbose:
        console.print(
            f"  [dim]Run [bold]nlqueries kb-stats {agent_id}[/bold]"
            " without --verbose for the summary view.[/dim]"
        )
    else:
        console.print(
            f"  [dim]Run [bold]nlqueries kb-stats {agent_id} --verbose[/bold]"
            " for per-table and per-column breakdowns.[/dim]"
        )


# ---------------------------------------------------------------------------
# embed-server command group (#32 — persistent embedding daemon)
# ---------------------------------------------------------------------------


@cli.group("embed-server")
def embed_server_group() -> None:
    """Manage the persistent embedding daemon.

    \b
    The daemon loads all-MiniLM-L6-v2 once and keeps it in RAM, reducing
    per-invocation embedding latency from ~9 s to ~10 ms.

    \b
    Commands:
      nlqueries embed-server start   — launch daemon in background
      nlqueries embed-server stop    — stop the daemon
      nlqueries embed-server status  — show whether daemon is running
    """


@embed_server_group.command("start")
@click.option("--port", default=8765, show_default=True, help="Port to listen on.")
@click.option(
    "--foreground",
    is_flag=True,
    default=False,
    help="Run in foreground (blocks; useful for debugging).",
)
def embed_server_start(port: int, foreground: bool) -> None:
    """Start the embedding daemon in the background.

    \b
    Example:
      nlqueries embed-server start
      nlqueries embed-server start --port 9000
      nlqueries embed-server start --foreground
    """
    import subprocess
    import sys

    from nlqueries.embeddings.embed_server import _PID_FILE

    if _PID_FILE.exists():
        pid = int(_PID_FILE.read_text().strip())
        console.print(
            f"  Daemon already running (PID {pid}). Use [bold]embed-server stop[/bold] first."
        )
        return

    if foreground:
        from nlqueries.embeddings.embed_server import serve

        serve(port=port)
        return

    subprocess.Popen(
        [sys.executable, "-m", "nlqueries.embeddings.embed_server", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    console.print(
        f"  [green]✓[/green] Embedding daemon started on port {port}. "
        "Future queries will skip model load (~10 ms embedding instead of ~9 s)."
    )


@embed_server_group.command("stop")
def embed_server_stop() -> None:
    """Stop the embedding daemon.

    \b
    Example:
      nlqueries embed-server stop
    """
    import os as _os
    import signal as _sig

    from nlqueries.embeddings.embed_server import _PID_FILE

    if not _PID_FILE.exists():
        console.print("  Daemon is not running.")
        return

    pid = int(_PID_FILE.read_text().strip())
    try:
        _os.kill(pid, _sig.SIGTERM)
        console.print(f"  [green]✓[/green] Daemon stopped (PID {pid})")
    except ProcessLookupError:
        console.print(f"  Process {pid} not found — removing stale PID file.")
    finally:
        _PID_FILE.unlink(missing_ok=True)


@embed_server_group.command("status")
def embed_server_status() -> None:
    """Show whether the embedding daemon is running.

    \b
    Example:
      nlqueries embed-server status
    """
    import urllib.error
    import urllib.request

    from nlqueries.embeddings.embed_server import _DEFAULT_PORT, _PID_FILE

    if not _PID_FILE.exists():
        console.print(
            "  Daemon [red]not running[/red]. "
            "Start with [bold]nlqueries embed-server start[/bold]. "
            "Queries will load the model per-invocation (~9 s)."
        )
        return

    pid = int(_PID_FILE.read_text().strip())

    # Verify the OS process is actually alive before checking HTTP.
    from nlqueries.embeddings.embed_server import is_pid_alive

    if not is_pid_alive(pid):
        console.print(f"  Process {pid} not found — removing stale PID file.")
        _PID_FILE.unlink(missing_ok=True)
        return

    # Process alive — check whether HTTP server is ready yet.
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{_DEFAULT_PORT}/embed", timeout=1)
    except urllib.error.HTTPError:
        pass  # 405 on GET means the server is accepting connections
    except Exception:  # noqa: BLE001
        # Process running but HTTP not up yet — still loading the model.
        console.print(
            f"  Daemon [yellow]starting[/yellow] (PID {pid}) — "
            "model is still loading (~9 s). "
            "Run [bold]embed-server status[/bold] again in a moment."
        )
        return

    console.print(
        f"  Daemon [green]running[/green] (PID {pid}, port {_DEFAULT_PORT}). "
        "Embedding calls will use the daemon (~10 ms each)."
    )


# ---------------------------------------------------------------------------
# mcp-server
# ---------------------------------------------------------------------------


@cli.group("mcp-server")
def mcp_server_group() -> None:
    """Run the NLQueries MCP server.

    \b
    Exposes MCP tools to Claude Desktop and other MCP-compatible clients:
      list_agents       — discover available agents
      get_agent_schema  — inspect tables, columns, and FKs for an agent
      query             — ask a natural-language question to an agent
      submit_feedback   — record thumbs-up/down feedback for a result
      health            — check LLM, Qdrant, embed daemon, and config status
      invalidate_cache  — drop the semantic cache for an agent
      list_connectors   — list registered database connectors
      get_query_history — return recent queries and ratings for an agent
      get_cache_stats   — return cache size and collection info for an agent

    \b
    Commands:
      nlqueries mcp-server start            — stdio transport (Claude Desktop)
      nlqueries mcp-server start --sse      — SSE transport (network clients)
    """


@mcp_server_group.command("start")
@click.option(
    "--sse",
    is_flag=True,
    default=False,
    help="Use SSE transport instead of stdio.",
)
@click.option(
    "--host",
    default="0.0.0.0",
    show_default=True,
    help="Bind host for SSE transport.",
)
@click.option(
    "--port",
    default=8000,
    show_default=True,
    type=int,
    help="Port for SSE transport.",
)
def mcp_server_start(sse: bool, host: str, port: int) -> None:
    """Start the MCP server.

    \b
    Stdio mode (default) — for Claude Desktop.  Add this to claude_desktop_config.json:
      {
        "mcpServers": {
          "nlqueries": {
            "command": "nlqueries",
            "args": ["mcp-server", "start"]
          }
        }
      }

    \b
    SSE mode — for network/browser clients:
      nlqueries mcp-server start --sse --port 8000
    """
    from nlqueries.mcp_server.server import main  # noqa: PLC0415

    if sse:
        console.print(
            f"  [green]✓[/green] NLQueries MCP server (SSE) listening on http://{host}:{port}/sse"
        )
        main(transport="sse", host=host, port=port)
    else:
        main(transport="stdio")
