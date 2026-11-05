"""
nlqueries.query_settings
~~~~~~~~~~~~~~~~~~~~~~~~
The answer-behaviour settings one request runs with.

How a value the question names is matched, and which of the checks and repairs
that shape an answer run, differ between users: one needs values matched
exactly as stored, another wants 'legal' to find 'Legal'. A single process
serves both in enterprise, where an admin sets these per tenant and per agent,
so they cannot be read from the process's configuration alone.

:class:`QuerySettings` holds them. Its defaults are the process's configuration
(the environment, ``.env`` and ``query_config.env``, see
:data:`nlqueries.config.QUERY_SETTING_VARS`), read when they are asked for, not
when this module is imported. A caller that knows better for one request binds
its own with :func:`use_query_settings`, and every step of that request reads
:func:`current`: the binding is a context variable, so it reaches the threads
and tasks the request runs on, and never another request.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
from typing import Any

from nlqueries import config

#: Each setting's environment variable, in the order they are documented.
ENV_VARS: dict[str, str] = {
    "match_mode": "NLQ_MATCH_MODE",
    "result_checks": "NLQ_RESULT_CHECKS",
    "grain_check": "NLQ_GRAIN_CHECK",
    "plan_check": "NLQ_PLAN_CHECK",
    "plan_check_min_row_reads": "NLQ_PLAN_CHECK_MIN_ROW_READS",
    "literal_grounding": "NLQ_LITERAL_GROUNDING",
    "literal_grounding_repair": "NLQ_LITERAL_GROUNDING_REPAIR",
}


@dataclass(frozen=True)
class QuerySettings:
    """The answer-behaviour settings for one request. See ``docs/configuration.md``."""

    match_mode: str = "loose"
    result_checks: bool = True
    grain_check: bool = True
    plan_check: bool = True
    plan_check_min_row_reads: int = 1_000_000_000
    literal_grounding: bool = True
    literal_grounding_repair: bool = True

    def __post_init__(self) -> None:
        if self.match_mode not in config.MATCH_MODES:
            raise ValueError(
                f"match_mode must be one of {', '.join(config.MATCH_MODES)}, "
                f"not {self.match_mode!r}"
            )
        if self.plan_check_min_row_reads < 0:
            raise ValueError("plan_check_min_row_reads must not be negative")

    @property
    def grounds_literals(self) -> bool:
        """Whether a literal may be replaced with the stored value it matches
        loosely: never when values are matched exactly."""
        return self.literal_grounding and self.match_mode != "exact"

    @classmethod
    def from_config(cls) -> QuerySettings:
        """The process's configuration, as :mod:`nlqueries.config` holds it now."""
        return cls(
            match_mode=config.MATCH_MODE,
            result_checks=config.RESULT_CHECKS,
            grain_check=config.GRAIN_CHECK,
            plan_check=config.PLAN_CHECK,
            plan_check_min_row_reads=config.PLAN_CHECK_MIN_ROW_READS,
            literal_grounding=config.LITERAL_GROUNDING,
            literal_grounding_repair=config.LITERAL_GROUNDING_REPAIR,
        )

    def overridden(self, values: Mapping[str, Any]) -> QuerySettings:
        """These settings with *values* in place, by field name; for a caller
        that keeps per-tenant or per-agent settings. A name that is not a
        setting, or a value of the wrong kind, raises ``ValueError``: a typo in
        an admin's settings should be seen, not silently run as the default."""
        known = {f.name: f.type for f in fields(self)}
        unknown = sorted(set(values) - set(known))
        if unknown:
            raise ValueError(f"not query settings: {', '.join(unknown)}")
        coerced: dict[str, Any] = {}
        for name, value in values.items():
            current_value = getattr(self, name)
            if isinstance(current_value, bool):
                if not isinstance(value, bool):
                    raise ValueError(f"{name} must be true or false, not {value!r}")
            elif isinstance(current_value, int):
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(f"{name} must be a whole number, not {value!r}")
            elif not isinstance(value, str):
                raise ValueError(f"{name} must be text, not {value!r}")
            coerced[name] = value.strip().lower() if isinstance(value, str) else value
        return replace(self, **coerced)

    def fingerprint(self) -> str:
        """A short digest of every setting, for keying what was worked out under
        them: the semantic cache binds an entry to it, so an answer cached under
        loose matching is not served to a request that matches exactly."""
        canonical = json.dumps(self.as_env(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def as_env(self) -> dict[str, str]:
        """``{environment variable: value}``, as ``query_config.env`` writes them."""
        out: dict[str, str] = {}
        for name, var in ENV_VARS.items():
            value = getattr(self, name)
            out[var] = ("true" if value else "false") if isinstance(value, bool) else str(value)
        return out


#: What each setting does, one line, for the file `template` writes.
_DESCRIPTIONS: dict[str, str] = {
    "match_mode": (
        "How a value the question names is matched to what a column stores: "
        "loose (ignore case and surrounding spaces), exact (as stored), or contains "
        "(as loose, then a stored value the phrase is part of)."
    ),
    "result_checks": "Check each result for signs it missed the question, and correct it once.",
    "grain_check": "Check the statement against the model's plan line of what one row is.",
    "plan_check": "Rewrite a subquery the database would re-run for every row, before it runs.",
    "plan_check_min_row_reads": "Row reads at or above which the plan check asks for a rewrite.",
    "literal_grounding": "Replace a literal with the stored value it matches loosely.",
    "literal_grounding_repair": "Ask the model once to correct a literal no stored value matches.",
}


def template() -> str:
    """A ``query_config.env`` with every setting at its default, commented
    out, and what each does."""
    lines = [
        "# NLQueries query settings: how answers are worked out.",
        "# Uncomment a line to change it. The environment, and .env, take",
        "# precedence over this file. Run 'nlqueries query-config' to see the",
        "# values in effect and where each one came from.",
    ]
    for name, (var, value) in zip(ENV_VARS, QuerySettings().as_env().items(), strict=True):
        lines += ["", f"# {_DESCRIPTIONS[name]}", f"# {var}={value}"]
    return "\n".join(lines) + "\n"


def source_of(var: str) -> str:
    """Where the process's value of *var* came from: ``query_config.env``,
    ``environment`` (the shell or ``.env``) or ``default``."""
    import os  # noqa: PLC0415

    if var in config.QUERY_CONFIG_APPLIED:
        return "query_config.env"
    return "environment" if var in os.environ else "default"


_bound: contextvars.ContextVar[QuerySettings | None] = contextvars.ContextVar(
    "nlqueries_query_settings", default=None
)


def current() -> QuerySettings:
    """The settings this request runs with: those bound by
    :func:`use_query_settings`, else the process's configuration."""
    bound = _bound.get()
    return bound if bound is not None else QuerySettings.from_config()


@contextmanager
def use_query_settings(settings: QuerySettings) -> Iterator[QuerySettings]:
    """Run the enclosed request with *settings*."""
    token = _bound.set(settings)
    try:
        yield settings
    finally:
        _bound.reset(token)
