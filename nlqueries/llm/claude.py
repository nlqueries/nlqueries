# nlqueries-core — OSS (BSL 1.1)
"""What Claude 5 and later need from a request that earlier models did not.

Two rules, both keyed on the model's generation:

- **No sampling parameters.** Claude Sonnet 5.5 and Haiku 5.5 answer a
  ``temperature`` with a 400 ("`temperature` is deprecated for this model"), and
  Anthropic's documentation says the same of ``top_p`` and ``top_k``. Core sends
  only ``temperature``, and only from the self-consistency candidates.
- **An effort.** They think adaptively, and ``output_config.effort`` caps how
  much. :data:`nlqueries.config.LLM_EFFORT` says which level, or none.

Judged from the id, because nothing else is available at the point of the
call: LiteLLM's model map does not list either 5.5 model, and its own
``supports_sampling_params`` check answers True for an id it does not know.
"""

from __future__ import annotations

import re

from nlqueries import config

# Family first, then the major version: `claude-sonnet-5-5`,
# `us.anthropic.claude-haiku-5-5`, `anthropic/claude-sonnet-4.6`. The pre-4
# names put the version first (`claude-3-5-sonnet-20241022`) and do not match,
# which is right: neither rule applies to them.
_FAMILY_GENERATION = re.compile(r"claude-(?:opus|sonnet|haiku)-(\d+)")


def claude_generation(model: str) -> int | None:
    """The major version of a Claude id, whatever prefix routes it; else ``None``."""
    match = _FAMILY_GENERATION.search(model.lower())
    return int(match.group(1)) if match else None


def _from_claude_5(model: str) -> bool:
    generation = claude_generation(model)
    return generation is not None and generation >= 5


def accepts_temperature(model: str) -> bool:
    """False for Claude 5 and later, which reject it; True for every other model."""
    return not _from_claude_5(model)


def effort_for(model: str) -> str | None:
    """The ``output_config.effort`` to send *model*, or ``None`` to send none.

    The caller decides whether its route reaches the Anthropic API at all; this
    decides only whether the model takes an effort, and which one.
    """
    return config.LLM_EFFORT if _from_claude_5(model) else None
