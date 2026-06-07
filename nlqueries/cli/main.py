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
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote_plus

import click
import yaml
from nlqueries.config import CONNECTORS_FILE
from nlqueries.connectors import CONNECTOR_REGISTRY
from rich.console import Console
from rich.table import Table

console = Console()
err_console = Console(stderr=True)

# ---------------------------------------------------------------------------
# Supported DB types -> SQLAlchemy driver schemes
# ---------------------------------------------------------------------------
_DB_SCHEMES: dict[str, str] = {
    "postgres":   "postgresql+psycopg2",
    "postgresql": "postgresql+psycopg2",
    "mysql":      "mysql+pymysql",
    "bigquery":   "bigquery",
    "snowflake":  "snowflake",
}

_DEFAULT_PORTS: dict[str, int] = {
    "postgres":   5432,
    "postgresql": 5432,
    "mysql":      3306,
    "bigquery":   443,
    "snowflake":  443,
}


# ---------------------------------------------------------------------------
# Connector registry helpers
# ---------------------------------------------------------------------------

def _load_connectors() -> dict:
    if CONNECTORS_FILE.exists():
        return yaml.safe_load(CONNECTORS_FILE.read_text()) or {}
    return {}


def _save_connector(connector_id: str, config: dict) -> None:
    CONNECTORS_FILE.parent.mkdir(parents=True, exist_ok=True)
    connectors = _load_connectors()
    connectors[connector_id] = config
    CONNECTORS_FILE.write_text(yaml.dump(connectors, default_flow_style=False, sort_keys=False))


def _require_connector(connector_id: str) -> dict:
    connectors = _load_connectors()
    if connector_id not in connectors:
        raise click.ClickException(
            f"Connector '{connector_id}' not found.\n"
            f"  Register it first:  nlqueries connect <db-type> "
            f"--database <db> --user <u> --password <p>\n"
            f"  List connectors:    cat {CONNECTORS_FILE}"
        )
    return connectors[connector_id]


def _build_url(
    db_type: str,
    host: str,
    port: int,
    database: str,
    user: str,
    password: str,
    account: str | None = None,    # Snowflake
    project: str | None = None,    # BigQuery
) -> str:
    db_type_l = db_type.lower()
    scheme = _DB_SCHEMES.get(db_type_l)
    if scheme is None:
        raise click.ClickException(
            f"Unsupported db-type '{db_type}'. "
            f"Supported: {', '.join(sorted(set(_DB_SCHEMES)))}"
        )

    if db_type_l == "bigquery":
        proj = project or database
        return f"bigquery://{proj}"

    if db_type_l == "snowflake":
        acct = account or host
        return (
            f"snowflake://{quote_plus(user)}:{quote_plus(password)}"
            f"@{acct}/{database}"
        )

    return (
        f"{scheme}://{quote_plus(user)}:{quote_plus(password)}"
        f"@{host}:{port}/{database}"
    )


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
@click.option("--password", default=None, hide_input=True, help="Database password.")
@click.option("--account", default=None, help="Snowflake account identifier.")
@click.option("--warehouse", default=None, help="Snowflake warehouse to use.")
@click.option("--schema", "db_schema", default=None, help="Snowflake schema (optional).")
@click.option("--project-id", "project_id", default=None, help="BigQuery / GCP project ID.")
@click.option(
    "--dataset-id", "dataset_id", default=None, help="BigQuery dataset ID (optional)."
)
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
def connect(
    db_type: str,
    host: str,
    port: int | None,
    database: str | None,
    user: str | None,
    password: str | None,
    account: str | None,
    warehouse: str | None,
    db_schema: str | None,
    project_id: str | None,
    dataset_id: str | None,
    service_account_json: str | None,
    connector_id: str | None,
) -> None:
    """Test a database connection and register it as a named connector.

    \b
    DB_TYPE  one of: postgres, mysql, bigquery, snowflake

    \b
    Examples:
      nlqueries connect postgres --database mydb --user alice --password secret
      nlqueries connect snowflake --account acme-prod --database PROD --user bob \\
          --password s3cr3t --warehouse COMPUTE_WH --schema PUBLIC
      nlqueries connect bigquery --project-id acme-prod --dataset-id analytics \\
          --service-account-json /path/to/key.json
    """
    db_type_l = db_type.lower()

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
            db_type, host, resolved_port, database or "", user or "", password or "",
            account=account, project=project_id,
        )
    except click.ClickException:
        raise

    cid = connector_id or f"{db_type_l}:{host}:{database or project_id}"

    if db_type_l == "bigquery":
        console.print(f"[bold]Connecting[/bold] to BigQuery project [cyan]{project_id}[/cyan] …")
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
            connector.connect({
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
            })
            if not connector.test_connection():
                raise RuntimeError("test_connection() returned False")
        else:
            # No dedicated connector registered for this db-type yet — fall back
            # to a raw SQLAlchemy connectivity check.
            from sqlalchemy import create_engine, text

            connect_args: dict = {}
            if db_type.lower() in ("postgres", "postgresql"):
                connect_args["connect_timeout"] = 10

            engine = create_engine(url, connect_args=connect_args)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))

    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[bold red]✗ Connection failed:[/bold red] {exc}")
        sys.exit(1)

    console.print("[bold green]✓ Connection successful.[/bold green]")

    # Persist connector config (store URL — password included; remind user)
    config = {
        "db_type":    db_type_l,
        "host":       host,
        "port":       resolved_port,
        "database":   database,
        "user":       user,
        "url":        url,          # ⚠ includes password — keep this file private
        "registered": datetime.now(UTC).isoformat(),
    }
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
    _save_connector(cid, config)

    console.print(f"  Connector registered as [bold]{cid!r}[/bold]")
    console.print(f"  Config saved to [dim]{CONNECTORS_FILE}[/dim]")
    console.print("  [yellow]Note:[/yellow] The config file contains the database password. "
                  "Ensure it is not world-readable.")


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
    cfg = _require_connector(connector_id)

    console.print(f"[bold]Extracting schema[/bold] for connector [cyan]{connector_id}[/cyan] …")

    connector_cls = CONNECTOR_REGISTRY.get(cfg.get("db_type", "").lower())

    if connector_cls is not None:
        # Use the registered DatabaseConnector implementation (e.g. PostgresConnector).
        try:
            from sqlalchemy.engine import make_url

            parsed = make_url(cfg["url"])
            connector = connector_cls()
            connector.connect({
                "host": parsed.host or cfg.get("host", "localhost"),
                "port": parsed.port or cfg.get("port"),
                "database": parsed.database or cfg.get("database"),
                "user": parsed.username or cfg.get("user"),
                "password": parsed.password,
                # Snowflake-specific fields — absent from the URL, read from
                # the persisted connector config (see `connect`).
                "account": cfg.get("account"),
                "warehouse": cfg.get("warehouse"),
                "schema": cfg.get("schema"),
            })
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
                "Schema", "Table", "Columns", "Rows",
                show_header=True, header_style="bold cyan",
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

        engine = create_engine(cfg["url"])

        with engine.connect() as conn:
            inspector = sa_inspect(engine)
            table_names: list[str] = inspector.get_table_names()
            view_names:  list[str] = inspector.get_view_names()

            total_columns = 0
            table_stats: list[dict] = []

            for tbl in table_names:
                cols = inspector.get_columns(tbl)
                col_count = len(cols)
                total_columns += col_count

                # Row count (best-effort; skip on error)
                try:
                    stmt = select(func.count()).select_from(table(tbl))
                    row = conn.execute(stmt).scalar()
                    row_count: int | None = int(row) if row is not None else None
                except Exception:  # noqa: BLE001
                    row_count = None

                table_stats.append({
                    "table":   tbl,
                    "columns": col_count,
                    "rows":    row_count,
                })

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
# process-history
# ---------------------------------------------------------------------------

@cli.command("process-history")
@click.argument("connector_id")
@click.option(
    "--days", default=90, show_default=True, type=int,
    help="Number of days of query history to process.",
)
@click.option(
    "--min-executions", default=3, show_default=True, type=int,
    help="Minimum execution count for a query to be included.",
)
def process_history(connector_id: str, days: int, min_executions: int) -> None:
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
    cfg = _require_connector(connector_id)

    console.print(
        f"[bold]Processing query history[/bold] for [cyan]{connector_id}[/cyan] "
        f"(last [bold]{days}[/bold] days) …"
    )

    try:
        # Pipeline stages — implemented in nlqueries.processing.*
        # Each stage is imported lazily so the CLI loads fast.
        from nlqueries.processing import (  # type: ignore[import]
            cluster_queries,
            fetch_history,
            filter_queries,
            parameterize,
        )

        raw       = fetch_history(cfg["url"], days=days)
        filtered  = filter_queries(raw, min_executions=min_executions)
        clusters  = cluster_queries(filtered)
        capsules  = parameterize(clusters)

    except ImportError:
        # Processing pipeline not yet implemented — emit a clear placeholder
        console.print()
        console.print("[yellow]⚠ Processing pipeline stubs not yet implemented.[/yellow]")
        console.print("  The following stages will run once nlqueries.processing is built:")
        console.print("    1. [dim]fetch_history[/dim]   — read pg_stat_statements / query logs")
        console.print("    2. [dim]filter_queries[/dim]  — drop DDL, noise, low-frequency queries")
        console.print(
            "    3. [dim]cluster_queries[/dim] — group by structural similarity (sqlglot AST)"
        )
        console.print("    4. [dim]parameterize[/dim]    — extract literals → typed placeholders")
        console.print()
        console.print(
            f"  Config: connector=[bold]{connector_id}[/bold]  "
            f"days={days}  min_executions={min_executions}"
        )
        return

    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[bold red]✗ Pipeline failed:[/bold red] {exc}")
        sys.exit(1)

    console.print("[bold green]✓ Pipeline complete.[/bold green]")
    console.print(f"  Raw queries fetched : {len(raw)}")
    console.print(f"  After filtering     : {len(filtered)}")
    console.print(f"  Clusters identified : {len(clusters)}")
    console.print(f"  Capsules produced   : [bold]{len(capsules)}[/bold]")


# ---------------------------------------------------------------------------
# export-kb
# ---------------------------------------------------------------------------

@cli.command("export-kb")
@click.argument("connector_id")
@click.option(
    "--output", "-o",
    default="knowledge_base.yaml",
    show_default=True,
    type=click.Path(dir_okay=False, writable=True),
    help="Path to write the YAML knowledge base.",
)
@click.option(
    "--include-samples/--no-include-samples",
    default=True, show_default=True,
    help="Include sample rows for each table.",
)
@click.option(
    "--sample-rows", default=3, show_default=True, type=int,
    help="Number of sample rows to include per table.",
)
def export_kb(
    connector_id: str,
    output: str,
    include_samples: bool,
    sample_rows: int,
) -> None:
    """Generate and save the YAML knowledge base for a connector.

    \b
    CONNECTOR_ID  the name used when you ran 'nlqueries connect'

    The knowledge base is a structured YAML file that describes your schema
    — tables, columns, types, foreign keys, and sample rows — in a format
    optimised for LLM context injection.

    \b
    Example:
      nlqueries export-kb postgres:localhost:mydb --output kb.yaml
      nlqueries export-kb postgres:localhost:mydb --output kb.yaml --sample-rows 5
    """
    cfg = _require_connector(connector_id)
    out_path = Path(output)

    console.print(
        f"[bold]Generating knowledge base[/bold] for [cyan]{connector_id}[/cyan] …"
    )

    try:
        from sqlalchemy import create_engine, select, table, text
        from sqlalchemy import inspect as sa_inspect

        engine = create_engine(cfg["url"])
        inspector = sa_inspect(engine)

        table_names = inspector.get_table_names()
        kb: dict = {
            "meta": {
                "connector_id": connector_id,
                "db_type":      cfg["db_type"],
                "database":     cfg["database"],
                "generated_at": datetime.now(UTC).isoformat(),
                "nlqueries_version": "0.1.0",
            },
            "tables": {},
        }

        with engine.connect() as conn:
            for tbl_name in table_names:
                columns = inspector.get_columns(tbl_name)
                pk_cols = inspector.get_pk_constraint(tbl_name).get("constrained_columns", [])
                fkeys   = inspector.get_foreign_keys(tbl_name)
                indexes = inspector.get_indexes(tbl_name)

                col_defs = [
                    {
                        "name":     c["name"],
                        "type":     str(c["type"]),
                        "nullable": c.get("nullable", True),
                        "primary_key": c["name"] in pk_cols,
                        **({"default": str(c["default"])} if c.get("default") is not None else {}),
                    }
                    for c in columns
                ]

                fkey_defs = [
                    {
                        "columns":            fk["constrained_columns"],
                        "references_table":   fk["referred_table"],
                        "references_columns": fk["referred_columns"],
                    }
                    for fk in fkeys
                ]

                index_defs = [
                    {
                        "name":    idx["name"],
                        "columns": idx["column_names"],
                        "unique":  idx.get("unique", False),
                    }
                    for idx in indexes
                ]

                samples: list[dict] = []
                if include_samples and sample_rows > 0:
                    try:
                        stmt = select(text("*")).select_from(table(tbl_name)).limit(sample_rows)
                        rows = conn.execute(stmt)
                        col_names = list(rows.keys())
                        samples = [dict(zip(col_names, row, strict=False)) for row in rows]
                        # Coerce non-serialisable types to strings
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

                kb["tables"][tbl_name] = {
                    "columns":      col_defs,
                    "foreign_keys": fkey_defs,
                    "indexes":      index_defs,
                    **({"sample_rows": samples} if include_samples else {}),
                }

    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[bold red]✗ Knowledge base generation failed:[/bold red] {exc}")
        sys.exit(1)

    # Write YAML
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        yaml.dump(kb, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)

    table_count  = len(kb["tables"])
    column_count = sum(len(t["columns"]) for t in kb["tables"].values())

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
