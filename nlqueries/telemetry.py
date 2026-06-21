"""
nlqueries.telemetry
~~~~~~~~~~~~~~~~~~~~
OpenTelemetry tracer and meter setup for nlqueries-core.

When no OTel SDK exporter is configured (the default), all instrumentation
calls are no-ops — existing tests and CLI usage are unaffected.

Usage
-----
    from nlqueries.telemetry import get_tracer, get_meter
    from nlqueries.telemetry import query_counter, query_latency, chunk_search_latency

    tracer = get_tracer()
    with tracer.start_as_current_span("my.operation") as span:
        span.set_attribute("key", "value")
        ...

    query_counter.add(1, {"dialect": "postgres"})
    query_latency.record(elapsed_ms, {"dialect": "postgres"})
"""

from __future__ import annotations

from opentelemetry import metrics, trace

_INSTRUMENTATION_NAME = "nlqueries.core"


def get_tracer() -> trace.Tracer:
    """Return a :class:`opentelemetry.trace.Tracer` for nlqueries-core.

    When no global TracerProvider is configured the OTel SDK returns a
    no-op tracer so instrumented code is never broken by a missing exporter.
    """
    return trace.get_tracer(_INSTRUMENTATION_NAME)


def get_meter() -> metrics.Meter:
    """Return a :class:`opentelemetry.metrics.Meter` for nlqueries-core.

    Same no-op guarantee as :func:`get_tracer` — safe to call without a
    configured MeterProvider.
    """
    return metrics.get_meter(_INSTRUMENTATION_NAME)


# ---------------------------------------------------------------------------
# Module-level meter used to define pre-built instruments.
# These are initialised once at import time; callers can import them directly.
# ---------------------------------------------------------------------------

_meter: metrics.Meter = get_meter()

# Total number of queries processed (SQL or document).
# Recommended attributes: {"dialect": "postgres"|"snowflake"|..., "agent_type": "sql"|"document"}
query_counter: metrics.Counter = _meter.create_counter(
    "nlqueries.queries.total",
    description="Total number of NLQueries queries processed.",
)

# End-to-end query latency in milliseconds.
# Recommended attributes: {"dialect": ..., "agent_type": ...}
query_latency: metrics.Histogram = _meter.create_histogram(
    "nlqueries.queries.latency_ms",
    description="End-to-end query latency in milliseconds.",
    unit="ms",
)

# Qdrant chunk search latency in milliseconds.
# Recommended attributes: {"collection": ...}
chunk_search_latency: metrics.Histogram = _meter.create_histogram(
    "nlqueries.chunks.search_latency_ms",
    description="Qdrant document chunk search latency in milliseconds.",
    unit="ms",
)
