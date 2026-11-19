"""
nlqueries.orchestrator.prompt_values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Stored values a value index offers the prompt (F20, part 2d).

A layer above core can give an agent's :class:`~nlqueries.connectors.loader.
LookupSource` a ``prompt_values`` provider. Where the request's ``value_index``
setting is on, prompt assembly asks it once per question and shows:

- each short column's every value with the schema, as ``values: [...]`` -- the
  same for every question, so the provider's prompt cache keeps it;
- for a longer column, the values matching words of the question, in the
  per-question part.

Core supplies no provider, so on its own nothing changes. The provider is
called on a daemon thread in the request's context -- a wrapper may read
request-bound state, such as the row filters an index must have been built
under -- and given up on after :data:`TIMEOUT_S`: the prompt is then built
without, as it would be with the setting off. A failing provider is logged at
debug and never fails the request.
"""

from __future__ import annotations

import concurrent.futures
import contextvars
import logging
import threading
from typing import Any

from nlqueries.connectors.loader import PromptValues
from nlqueries.query_settings import current as current_settings

_log = logging.getLogger(__name__)

#: How long the prompt waits for the provider. It runs before every question's
#: model call, so it is kept short: a slow index costs this much at most.
TIMEOUT_S = 1.0

#: Safety caps on what is rendered, whatever a provider returns.
MAX_LIST = 100
MAX_MATCHING = 20

EMPTY = PromptValues()


def prompt_values_for(agent_id: str | None, question: str) -> PromptValues:
    """What *agent_id*'s value index offers the prompt for *question*, or
    :data:`EMPTY` when it offers nothing, the setting is off, or it failed or
    ran out of time."""
    if not agent_id or not current_settings().value_index:
        return EMPTY
    from nlqueries.connectors import loader  # noqa: PLC0415 -- looked up when called

    try:
        source = loader.lookup_source(agent_id)
    except Exception:  # noqa: BLE001 -- the prompt is built without
        _log.debug("No lookup source for the prompt's stored values.", exc_info=True)
        return EMPTY
    provider = getattr(source, "prompt_values", None)
    if provider is None:
        return EMPTY
    found = _call(provider, question)
    return found if isinstance(found, PromptValues) else EMPTY


def _call(provider: Any, question: str) -> Any:
    future: concurrent.futures.Future[Any] = concurrent.futures.Future()
    context = contextvars.copy_context()

    def run() -> None:
        try:
            future.set_result(context.run(provider, question))
        except BaseException as exc:  # noqa: BLE001 - handed to the waiter
            future.set_exception(exc)

    # A daemon thread, not a pool: a provider stuck in its database must not
    # hold the process on its way out.
    threading.Thread(target=run, name="nlq-prompt-values", daemon=True).start()
    try:
        return future.result(timeout=TIMEOUT_S)
    except concurrent.futures.TimeoutError:
        _log.debug("The prompt's stored values were given up on after %.1f s.", TIMEOUT_S)
    except Exception:  # noqa: BLE001 -- the prompt is built without
        _log.debug("The prompt's stored values could not be read.", exc_info=True)
    return None


def column_key(schema: Any, table: Any, column: Any) -> tuple[str, str, str]:
    """The key :class:`PromptValues` uses: each part lower-cased, the schema
    ``""`` where none is named."""
    return (str(schema or "").lower(), str(table or "").lower(), str(column or "").lower())


__all__ = ["EMPTY", "MAX_LIST", "MAX_MATCHING", "TIMEOUT_S", "column_key", "prompt_values_for"]
