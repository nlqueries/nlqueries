# nlqueries-core — OSS (BSL 1.1)
from __future__ import annotations

import logging
import time

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
_BATCH_DELAY = 1.0  # seconds between batches to avoid rate limits


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
    batch_size: int = _BATCH_SIZE,
) -> list[QueryCapsule]:
    """Annotate all capsules in batches, sleeping 1 s between batches.

    Each capsule's ``intent`` field is updated in-place via LLM completion.
    Returns the same list.
    """
    total = len(capsules)
    for batch_start in range(0, total, batch_size):
        batch = capsules[batch_start : batch_start + batch_size]
        for capsule in batch:
            annotate_capsule(capsule, llm)
        batch_end = min(batch_start + batch_size, total)
        _log.info("Annotated %d/%d capsules", batch_end, total)
        if batch_end < total:
            time.sleep(_BATCH_DELAY)
    return capsules
