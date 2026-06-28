# nlqueries-core — OSS (BSL 1.1)
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from nlqueries.llm.client import LLMClient
from nlqueries.processing.parameterizer import QueryCapsule

_log = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an expert data analyst. Given a parameterized SQL template, write a concise "
    "natural language description phrased as a business question a non-technical user might ask. "
    "Output only the description, no explanation."
)
_MAX_INTENT_LENGTH = 200
_BATCH_SIZE = 10
_BATCH_DELAY = 1.0  # kept for backward-compat imports; no longer used
_MAX_CONCURRENT = 5


def annotate_capsule(capsule: QueryCapsule, llm: LLMClient) -> QueryCapsule:
    """Fill capsule.intent with a single LLM call. Returns the same capsule."""
    tables_str = ", ".join(capsule.tables) if capsule.tables else "unknown"
    user_msg = f"SQL Template: {capsule.template_sql}\nTables involved: {tables_str}"
    response = llm.complete(system=_SYSTEM_PROMPT, user=user_msg)
    capsule.intent = response[:_MAX_INTENT_LENGTH]
    return capsule


def annotate_capsules(
    capsules: list[QueryCapsule],
    llm: LLMClient,
    batch_size: int = _BATCH_SIZE,  # noqa: ARG001 — kept for API compat
) -> list[QueryCapsule]:
    """Annotate all capsules concurrently using up to 5 threads.

    Each capsule's ``intent`` field is updated in-place via LLM completion.
    Returns the same list. Raises the first per-capsule exception encountered.
    """
    if not capsules:
        return capsules

    with ThreadPoolExecutor(max_workers=_MAX_CONCURRENT) as pool:
        futures = {pool.submit(annotate_capsule, cap, llm): cap for cap in capsules}
        for i, future in enumerate(as_completed(futures), 1):
            future.result()  # re-raises any exception from the thread
            _log.info("Annotated %d/%d capsules", i, len(capsules))
    return capsules
