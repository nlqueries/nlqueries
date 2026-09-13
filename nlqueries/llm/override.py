# nlqueries-core — OSS (BSL 1.1)
"""
nlqueries.llm.override
~~~~~~~~~~~~~~~~~~~~~~~
The per-request LLM configuration, and the token budget resolved from it.

Its own module so the clients can read the budget. `output_budget` lives beside
the override it reads, and both live below `nlqueries.llm`, which imports the
clients -- a client importing from the package root would be a cycle. Everything
here is re-exported from `nlqueries.llm`, so the public import path is unchanged.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from nlqueries import config


@dataclass(frozen=True)
class LLMOverride:
    """Per-invocation LLM configuration that supersedes the process env/config.

    Every field is optional; a ``None`` field falls back to the env-derived
    ``nlqueries.config`` default, so a partial override (e.g. only ``api_key``)
    is fine. The host application (e.g. the enterprise layer resolving a
    per-tenant key from its settings store) sets one for the duration of a
    request via :func:`use_llm_override`; :func:`get_llm_client` reads it.

    ``extra`` carries provider-specific keyword arguments that this package has
    no opinion about — AWS region and credentials for a Bedrock deployment, for
    instance. They are passed verbatim to the underlying completion call, which
    keeps cloud-provider vocabulary out of core: the host application knows what
    its provider needs, and core only has to carry it. Note that supplying it
    makes the instance unhashable, as a dict field does to any frozen dataclass;
    nothing hashes an override today.
    """

    provider: str | None = None
    model: str | None = None
    fast_model: str | None = None
    api_key: str | None = None
    api_base: str | None = None
    extra: dict[str, Any] | None = None
    #: Tokens an answer may generate, overriding ``config.LLM_MAX_OUTPUT_TOKENS``.
    #:
    #: A first-class field rather than a key in ``extra`` because ``extra`` may
    #: not carry it: ``LiteLLMClient`` rejects ``max_tokens`` there, since the
    #: two code paths disagree about which value wins and one of them loses
    #: silently. That refusal is right, and it left a host with no way at all to
    #: set a budget -- which is what this field is for.
    max_tokens: int | None = None


# Task-local so concurrent requests on one event loop never see each other's
# override: contextvars are copied per asyncio Task, and the value set here
# propagates through every ``await`` in the same task (where get_llm_client is
# called) without leaking across tasks.
_override: ContextVar[LLMOverride | None] = ContextVar("nlqueries_llm_override", default=None)


@contextlib.contextmanager
def use_llm_override(override: LLMOverride | None) -> Iterator[None]:
    """Bind *override* for the duration of the ``with`` block (task-local).

    Passing ``None`` is a no-op binding (the process default keeps applying),
    which lets callers wrap a block unconditionally.
    """
    token = _override.set(override)
    try:
        yield
    finally:
        _override.reset(token)


def current_llm_override() -> LLMOverride | None:
    """Return the override bound in the current context, if any."""
    return _override.get()


#: Bad override budgets already warned about, so a per-request code path does
#: not emit the same line on every question.
_WARNED_BUDGETS: set[int] = set()

#: What each kind of call gets, as a share of the answer budget and a floor.
#:
#: One configurable number, several call sites that need less than an answer.
#: The shares take over as an operator raises the budget; the floors are what
#: hold at and below the default. Without the shares, raising the budget for a
#: reasoning model would fix the answer and leave classification starved at 200
#: -- and that is the tier that returns an EMPTY string rather than a short one,
#: because the reasoning is billed first.
#:
#: At the default of 1024 an operator gets 1024 / 512 / 200 -- precisely the
#: numbers these call sites hard-coded before this was configurable -- but the
#: two tiers get there differently, and the difference matters.
#:
#: `correction` is exactly its share: half of 1024 is 512, equal to its floor.
#: `classification` is NOT: an eighth of 1024 is 128, so the FLOOR is what
#: yields 200, and the share does not take over until the total passes 1600.
#:
#: So an operator who raises the budget modestly for a reasoning model -- 1024 to
#: 1536, say -- moves the answer and correction tiers and leaves classification
#: at exactly 200, which is the tier this whole change exists to relieve. It is
#: the tier that returns an EMPTY string rather than a short one when reasoning
#: eats the allowance. Raising past 1600 is what moves it; 4096 gives it 512.
#: The floors are also the lower bound on a nonsense budget. ``config`` clamps
#: what it reads from the environment, but an ``LLMOverride`` is the other
#: channel and arrives unclamped -- from a settings store, where a person can
#: type 0 into a form. Both converge here, so this is the one place that has to
#: hold.
_BUDGET_TIERS: dict[str, tuple[float, int]] = {
    # The answer itself: the whole budget, and never less than a token.
    "answer": (1.0, 1),
    # A retry that rewrites SQL, or a candidate set. Structured, bounded output.
    "correction": (0.5, 512),
    # A label or a short JSON object: intent, and follow-up resolution.
    "classification": (0.125, 200),
}


def output_budget(tier: str = "answer") -> int:
    """Tokens this kind of call may generate.

    Resolved from the bound :class:`LLMOverride` when it names a budget, else
    from ``config.LLM_MAX_OUTPUT_TOKENS``.

    Callers pass a tier rather than a number so the policy lives here. The
    numbers they used to pass -- 1024, 512, 200 -- were chosen against models
    that emit only the answer; a reasoning model spends the same allowance on
    reasoning first, and the smallest budgets are the ones that then return
    nothing at all.
    """
    if tier not in _BUDGET_TIERS:
        raise ValueError(f"Unknown budget tier: {tier!r}. Available: {sorted(_BUDGET_TIERS)}")
    override = _override.get()
    # A non-positive budget is not a budget, so it is IGNORED and the configured
    # value stands. One rule for 0 and for negatives, which the first version
    # did not have: it resolved with `or`, so 0 was falsy and fell back to the
    # configured value while a negative passed through to be clamped at the tier
    # floor -- two different behaviours, and a warning that described only the
    # second. 0 is the one a person actually types into a settings form.
    #
    # Ignoring beats clamping here. Clamping a bad value to the floor yields a
    # one-token answer, which is useless; falling back to the configured budget
    # is the behaviour the deployment already had.
    #
    # Warned once per value rather than per call: this runs on every LLM
    # request, and a warning that floods is a warning nobody reads.
    total = config.LLM_MAX_OUTPUT_TOKENS
    requested = override.max_tokens if override else None
    if requested is not None and requested >= 1:
        total = requested
    elif requested is not None and requested not in _WARNED_BUDGETS:
        _WARNED_BUDGETS.add(requested)
        logging.getLogger(__name__).warning(
            "LLMOverride.max_tokens=%d is not a usable token budget and is being "
            "ignored; using LLM_MAX_OUTPUT_TOKENS=%d. Set a positive value to "
            "override it.",
            requested,
            total,
        )
    share, floor = _BUDGET_TIERS[tier]
    return max(floor, int(total * share))
