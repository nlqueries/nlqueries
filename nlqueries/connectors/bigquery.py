"""
nlqueries.connectors.bigquery
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
BigQuery implementation of :class:`~nlqueries.connectors.base.DatabaseConnector`.

Built on the official ``google-cloud-bigquery`` client. This module is part
of the public OSS API and has no dependency on the enterprise layer.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from google.cloud import bigquery
from google.oauth2 import service_account

from nlqueries import config
from nlqueries.connectors._budget import collect
from nlqueries.connectors.base import (
    ColumnSpec,
    DatabaseConnector,
    QueryRecord,
    QueryResult,
    SchemaSpec,
    TableSpec,
)

logger = logging.getLogger(__name__)


# Region used to qualify INFORMATION_SCHEMA.JOBS_BY_PROJECT when the
# connected dataset's location can't be determined (e.g. no dataset_id given).
_DEFAULT_REGION_QUALIFIER = "region-us"


class BigQueryConnector(DatabaseConnector):
    """Connector for Google BigQuery.

    Usage::

        connector = BigQueryConnector()
        connector.connect({
            "project_id": "acme-prod",
            "dataset_id": "analytics",          # optional
            "service_account_json": "/path/to/key.json",  # or a dict, or omit for ADC
        })
        if connector.test_connection():
            schema = connector.extract_schema()
    """

    def __init__(self) -> None:
        self._client: Any = None
        self._project_id: str | None = None
        self._dataset_id: str | None = None
        self._region_qualifier: str = _DEFAULT_REGION_QUALIFIER

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self, credentials: dict[str, Any]) -> None:
        """Build a BigQuery client from ``credentials``.

        Required keys: ``project_id``. Optional keys: ``dataset_id`` and
        ``service_account_json``.

        ``service_account_json`` may be a ``dict`` (already-parsed service
        account key), a path (``str`` or ``Path``) to a service account JSON
        key file, or omitted entirely — in which case Application Default
        Credentials (ADC) are used (e.g. ``gcloud auth application-default
        login``, a workload-identity binding, or ``GOOGLE_APPLICATION_
        CREDENTIALS``).
        """
        self._project_id = credentials["project_id"]
        self._dataset_id = credentials.get("dataset_id")

        creds = self._build_credentials(credentials.get("service_account_json"))

        if creds is not None:
            self._client = bigquery.Client(project=self._project_id, credentials=creds)
        else:
            # No explicit key provided — fall back to Application Default Credentials.
            self._client = bigquery.Client(project=self._project_id)

        self._region_qualifier = self._resolve_region_qualifier()

    @staticmethod
    def _build_credentials(service_account_json: Any) -> Any | None:
        """Build a ``Credentials`` object from ``service_account_json``, or ``None`` for ADC.

        Accepts a ``dict`` (parsed key), a path (``str``/``Path``) to a key
        file, or ``None``/empty (meaning: use Application Default Credentials).
        """
        if not service_account_json:
            return None

        if isinstance(service_account_json, dict):
            return service_account.Credentials.from_service_account_info(  # type: ignore[no-untyped-call]
                service_account_json
            )

        # Treat anything else (str, Path, ...) as a path to a key file.
        return service_account.Credentials.from_service_account_file(  # type: ignore[no-untyped-call]
            str(service_account_json)
        )

    def _resolve_region_qualifier(self) -> str:
        """Best-effort resolution of the ``region-<location>`` qualifier for JOBS_BY_PROJECT.

        ``INFORMATION_SCHEMA.JOBS_BY_PROJECT`` must be queried with a region
        qualifier matching the location of the data being queried (e.g.
        ``region-us``, ``region-us-central1``, ``region-europe-west1``).
        When a ``dataset_id`` was supplied at connect time, we look up that
        dataset's location and derive the qualifier from it; otherwise we
        fall back to :data:`_DEFAULT_REGION_QUALIFIER`. Callers in non-US
        regions without a configured ``dataset_id`` may need to adjust
        ``connector._region_qualifier`` directly.
        """
        if self._dataset_id and self._client is not None:
            try:
                dataset = self._client.get_dataset(self._dataset_id)
                location = dataset.location
                if location:
                    return f"region-{location.lower()}"
            except Exception:  # noqa: BLE001 — fall through to the default
                logger.debug(
                    "Could not resolve dataset location for region qualification; defaulting to %s",
                    _DEFAULT_REGION_QUALIFIER,
                )
        return _DEFAULT_REGION_QUALIFIER

    def _require_client(self) -> Any:
        if self._client is None:
            raise RuntimeError("BigQueryConnector.connect() must be called before use.")
        return self._client

    # ------------------------------------------------------------------
    # test_connection
    # ------------------------------------------------------------------

    def test_connection(self) -> bool:
        """Return True if ``SELECT 1`` succeeds as a BigQuery query job."""
        try:
            client = self._require_client()
            client.query("SELECT 1").result()
            return True
        except Exception:  # noqa: BLE001 — any failure means "not connected"
            logger.exception("BigQueryConnector.test_connection failed")
            return False

    # ------------------------------------------------------------------
    # extract_schema
    # ------------------------------------------------------------------

    def extract_schema(self) -> SchemaSpec:
        """Introspect the schema via the client's metadata APIs.

        Builds a full :class:`SchemaSpec` describing every table in the
        connected dataset (or, when no ``dataset_id`` was supplied at
        connect time, every table across every dataset in the project):
        its columns, row count, and description.

        Note: BigQuery has no enforced primary/foreign-key constraint
        system in the general case (``PRIMARY KEY`` / ``FOREIGN KEY``
        declarations exist only as unenforced metadata on certain table
        types and are not exposed via ``list_tables``/``get_table``), so
        ``ColumnSpec.is_primary_key``/``is_foreign_key`` are always
        ``False`` and ``references`` is always ``None`` here.

        Field ``mode`` is mapped to ``nullable`` as follows: ``REQUIRED``
        columns are non-nullable; ``NULLABLE`` and ``REPEATED`` columns are
        considered nullable (a ``REPEATED`` field's *array* may be empty,
        and BigQuery does not distinguish "empty array" from "no value"
        the way a scalar ``NULL`` would be distinguished).
        """
        client = self._require_client()
        database = self._project_id or ""

        tables: list[TableSpec] = []
        for dataset_ref in self._iter_dataset_refs(client):
            for table_ref in client.list_tables(dataset_ref):
                table = client.get_table(table_ref)
                tables.append(self._table_to_spec(table))

        return SchemaSpec(
            database=database,
            tables=tables,
            extracted_at=_utc_now_iso(),
        )

    def _iter_dataset_refs(self, client: Any) -> list[Any]:
        """Return the dataset reference(s) to introspect.

        If a ``dataset_id`` was supplied at connect time, only that dataset
        is introspected; otherwise every dataset in the project is.
        """
        if self._dataset_id:
            return [client.dataset(self._dataset_id)]
        return [dataset.reference for dataset in client.list_datasets()]

    @classmethod
    def _table_to_spec(cls, table: Any) -> TableSpec:
        """Convert a ``google.cloud.bigquery.Table`` into a :class:`TableSpec`."""
        columns = [cls._field_to_column_spec(field) for field in table.schema]
        return TableSpec(
            name=table.table_id,
            schema=table.dataset_id,
            row_count=int(table.num_rows) if table.num_rows is not None else None,
            columns=columns,
            description=table.description,
        )

    @staticmethod
    def _field_to_column_spec(field: Any) -> ColumnSpec:
        """Convert a ``SchemaField`` into a :class:`ColumnSpec`.

        ``REQUIRED`` -> not nullable; ``NULLABLE``/``REPEATED`` -> nullable.
        """
        return ColumnSpec(
            name=field.name,
            type=field.field_type,
            nullable=field.mode != "REQUIRED",
            is_primary_key=False,
            is_foreign_key=False,
            references=None,
            description=field.description,
        )

    # ------------------------------------------------------------------
    # extract_query_history
    # ------------------------------------------------------------------

    def extract_query_history(self, days: int = 30, limit: int = 500) -> list[QueryRecord]:
        """Return the top SELECT queries (by job count) from the last ``days`` days.

        Queries ``INFORMATION_SCHEMA.JOBS_BY_PROJECT`` (qualified by a
        ``region-<location>`` prefix — see :meth:`_resolve_region_qualifier`),
        filtered to completed ``SELECT`` query jobs, grouped by query text,
        and ordered by job count descending. Returns up to ``limit`` records.

        Returns an empty list (with a logged warning) if the view is not
        accessible — e.g. the caller's IAM role lacks
        ``bigquery.jobs.listAll``/``bigquery.resourceViewer`` at the project
        level required to read project-wide job history.
        """
        client = self._require_client()

        try:
            rows = self._query(
                client,
                f"""
                SELECT
                    query,
                    COUNT(*) AS job_count,
                    AVG(TIMESTAMP_DIFF(end_time, start_time, MILLISECOND)) AS avg_duration_ms,
                    MAX(end_time) AS last_executed
                FROM `{self._region_qualifier}`.INFORMATION_SCHEMA.JOBS_BY_PROJECT
                WHERE creation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {int(days)} DAY)
                  AND job_type = 'QUERY'
                  AND statement_type = 'SELECT'
                  AND state = 'DONE'
                  AND error_result IS NULL
                  AND query IS NOT NULL
                GROUP BY query
                ORDER BY job_count DESC
                LIMIT {limit}
                """,
            )
        except Exception:
            logger.warning(
                "extract_query_history: `%s`.INFORMATION_SCHEMA.JOBS_BY_PROJECT is not "
                "accessible — returning an empty query history. This typically requires "
                "the caller to have project-level job-history visibility "
                "(e.g. the `roles/bigquery.resourceViewer` or `bigquery.jobs.listAll` "
                "permission).",
                self._region_qualifier,
                exc_info=True,
            )
            return []

        return [
            QueryRecord(
                sql=row["query"],
                execution_count=int(row["job_count"]),
                avg_duration_ms=(
                    float(row["avg_duration_ms"]) if row["avg_duration_ms"] is not None else None
                ),
                last_executed=(
                    str(row["last_executed"]) if row["last_executed"] is not None else None
                ),
            )
            for row in rows
        ]

    @staticmethod
    def _query(client: Any, sql: str) -> list[dict[str, Any]]:
        """Run ``sql`` as a query job and return rows as ``{column_name: value}`` dicts."""
        result = client.query(sql).result()
        return [dict(row.items()) for row in result]

    # ------------------------------------------------------------------
    # execute_query
    # ------------------------------------------------------------------

    def _execute_query(
        self,
        sql: str,
        timeout_seconds: float | None = None,
        max_rows: int | None = None,
    ) -> QueryResult:
        """Run ``sql`` as a BigQuery query job and return a :class:`QueryResult`.

        The job is bounded by *timeout_seconds* when given, else the
        ``CONNECTOR_STATEMENT_TIMEOUT_SECONDS`` default — set as the job's
        ``job_timeout_ms`` so BigQuery cancels the job server-side rather than
        letting it run indefinitely. A budget of 0 disables it.

        Any exception raised during execution is caught and surfaced via
        ``QueryResult.error`` rather than propagating, so callers can treat
        query execution as always returning a result object.

        ``QueryResult`` has no dedicated field for job metadata, so —
        per spec — the job's ``total_bytes_processed`` is surfaced as a
        logged metadata note (at INFO level) rather than bolted onto the
        dataclass, keeping the shared interface unchanged for every connector.
        """
        effective_timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else config.CONNECTOR_STATEMENT_TIMEOUT_SECONDS
        )
        start = time.perf_counter()
        _truncated, _reason = False, None
        try:
            client = self._require_client()
            # BigQuery has no transaction to roll back for a single job, so
            # unlike every other connector here there is nothing to undo a write
            # with. `create_session=False` at least keeps a job from opening a
            # session that later statements could join, and `use_legacy_sql=False`
            # keeps the dialect the one every validator in front of this parsed.
            #
            # The real control is IAM, and it is the operator's: a service
            # account holding `roles/bigquery.dataViewer` and
            # `roles/bigquery.jobUser` cannot write whatever SQL asks for. See
            # docs/database-hardening.md.
            job_config = bigquery.QueryJobConfig(
                use_legacy_sql=False,
                create_session=False,
            )
            # Set only when a timeout actually applies, rather than passing None
            # to mean "no timeout". On the installed 3.41 the setter runs the
            # value through `_int_or_none` and clears the property, so both forms
            # work -- but the declared floor is `google-cloud-bigquery>=3.0` and
            # this avoids depending on that detail holding across the whole range
            # we say we support. Behaviour is identical either way.
            if effective_timeout is not None and effective_timeout > 0:
                job_config.job_timeout_ms = int(effective_timeout * 1000)
            query_job = client.query(sql, job_config=job_config)
            result = query_job.result()
            elapsed_ms = (time.perf_counter() - start) * 1000

            # An audit signal, deliberately not a control: the job has already
            # run by the time this is readable, so refusing here would refuse
            # only the *results* of a write that already happened. It is logged
            # at warning so that a statement type other than SELECT reaching a
            # database is visible to whoever reads the logs, rather than being
            # inferred later from the data.
            statement_type = getattr(query_job, "statement_type", None)
            if statement_type is not None and statement_type != "SELECT":
                logger.warning(
                    "BigQueryConnector executed a %s statement (job_id=%s). The SQL policy "
                    "and the caller's IAM role are what prevent this; the job has already "
                    "run by the time this is known.",
                    statement_type,
                    query_job.job_id,
                )

            schema = result.schema or []
            columns = [field.name for field in schema]
            rows, _truncated, _reason = (
                collect((list(row.values()) for row in result), max_rows)
                if columns
                else ([], False, None)
            )

            logger.info(
                "BigQueryConnector.execute_query metadata: total_bytes_processed=%s (job_id=%s)",
                query_job.total_bytes_processed,
                query_job.job_id,
            )

            return QueryResult(
                columns=columns,
                rows=rows,
                row_count=len(rows),
                truncated=_truncated,
                truncation_reason=_reason,
                execution_time_ms=elapsed_ms,
                error=None,
            )
        except Exception as exc:  # noqa: BLE001 — surfaced via QueryResult.error, not raised
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.exception("BigQueryConnector.execute_query failed")
            return QueryResult(
                columns=[],
                rows=[],
                row_count=0,
                execution_time_ms=elapsed_ms,
                error=str(exc),
            )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017 -- keep py3.10-compatible
