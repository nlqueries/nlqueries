# nlqueries-core — OSS (BSL 1.1)
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from typing import Any

from nlqueries.llm.override import output_budget

#: Finish reasons meaning "stopped because it ran out of room", across providers.
#: LiteLLM normalises to ``length``; the Anthropic SDK reports ``max_tokens``.
TRUNCATED = frozenset({"length", "max_tokens"})


class OutputBudgetExhausted(RuntimeError):
    """The model used its whole output allowance without writing an answer.

    Raised rather than returning ``""`` because an empty string is
    indistinguishable from a model that had nothing to say, and the two want
    opposite responses from the caller: one is a configuration problem with a
    named fix, the other is not.

    This is the failure a reasoning model produces. Reasoning is billed from the
    same allowance as the answer and spent first, so a budget sized for the
    answer alone returns a truncated answer -- or, when the allowance is small
    enough, nothing at all. Measured on `deepseek-v4-pro`: at 5 tokens, all five
    went to reasoning and the content came back empty with
    ``finish_reason=length``.

    Only raised when NOTHING was produced. A truncated answer is still an
    answer, and is left to the caller.
    """

    def __init__(self, model: str, budget: int) -> None:
        self.model = model
        self.budget = budget
        super().__init__(
            f"{model} used its entire {budget}-token output budget without "
            f"producing an answer. Raise the output-token limit -- a reasoning "
            f"model spends this allowance before it writes anything."
        )


class LLMTimeout(RuntimeError):
    """A single LLM call did not finish inside its deadline.

    ``phase`` names which deadline ran out, and the message changes with it
    because the remedy does. ``read``, ``write`` and ``pool`` are all set from
    ``LLM_TIMEOUT_SECONDS``; ``connect`` alone is held at a short fixed value
    so an unreachable endpoint fails fast, and is the only one raising that
    setting does not affect. ``pool`` gets its own message: it means every
    pooled connection was busy, which is neither a slow model nor an
    unreachable endpoint.

    Both clients raise this in place of their SDK's own timeout type, so a host
    has one thing to catch and one message to render. The alternative is asking
    every caller to know both ``litellm.Timeout`` and
    ``anthropic.APITimeoutError``, and to keep knowing them as providers are
    added.

    Defined here rather than beside either client because this module has no
    third-party imports, and importing it must stay cheap.
    """

    def __init__(self, model: str, seconds: object, phase: str = "read") -> None:
        self.model = model
        self.seconds = seconds
        self.phase = phase
        # `seconds` is whatever the deadline was configured as, and a host can
        # put something other than a number there: litellm's `timeout` accepts
        # `str` and `httpx.Timeout` as well as a float, and `extra` forwards it
        # untouched. `:g` raises on both -- `ValueError` for a string,
        # `TypeError` for an object -- so formatting it blindly would replace
        # the provider's timeout with a formatting error, in the constructor
        # that exists to report the timeout clearly.
        shown = f"{seconds:g}" if isinstance(seconds, (int, float)) else str(seconds)
        # The advice has to match the phase that expired, or it sends the
        # operator to the wrong setting.
        #
        # `connect` ONLY, not `connect`/`pool`. Connect is held at a few seconds
        # while the rest of the deadline is minutes, so a blocked egress or a
        # mistyped endpoint fails fast and raising LLM_TIMEOUT_SECONDS does not
        # touch it. `pool` is different on both counts: it IS set from that
        # setting, so saying otherwise sends the operator away from the one
        # control that would help, and it means every pooled connection was
        # busy rather than that the endpoint was unreachable.
        if phase == "connect":
            super().__init__(
                f"{model} could not be reached within {shown}s (connect timeout). "
                f"Check the endpoint and whether outbound access to the provider "
                f"is allowed. Raising LLM_TIMEOUT_SECONDS will not help: it sets "
                f"the response deadline, not this one."
            )
        elif phase == "pool":
            super().__init__(
                f"{model} waited {shown}s for a free connection and did not get "
                f"one (pool timeout). Every pooled connection was busy, so this "
                f"is concurrency rather than a slow or unreachable model. Raise "
                f"LLM_TIMEOUT_SECONDS to wait longer, or reduce how many "
                f"questions run at once."
            )
        else:
            super().__init__(
                f"{model} did not respond within {shown}s. Raise "
                f"LLM_TIMEOUT_SECONDS if this model is legitimately slow, or check "
                f"whether the provider is reachable."
            )


#: Exception class names every httpx-shaped transport uses for a deadline.
#:
#: Matched by name because the httpx an SDK raises from is not necessarily the
#: one imported alongside it: newer `anthropic` builds against `httpx2`, and CI
#: found the matching split on a constructor argument while the local run
#: passed, the two being the same object there. An `isinstance` check is
#: therefore true in one environment and false in another with nothing in this
#: repository having changed -- and the half that fails is the silent one, a
#: mid-stream stall escaping untranslated.
TIMEOUT_NAMES = frozenset(
    {
        "TimeoutException",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
    }
)


def looks_like_timeout(exc: BaseException) -> bool:
    """Whether *exc* is a transport deadline, judged by class name.

    The name-based half of each client's check. Each adds the SDK types it
    knows by identity; this covers the ones it cannot name because they come
    from a distribution it did not import.
    """
    return any(t.__name__ in TIMEOUT_NAMES for t in type(exc).__mro__)


def exhausted(finish_reason: object, content: str) -> bool:
    """Whether a reply is an exhausted budget rather than a short answer.

    ``finish_reason`` is read with ``getattr(..., None)`` at every call site, and
    ``None`` answers False here. LiteLLM normalises a hundred-odd providers and
    does not promise the field on all of them, so reading it directly would turn
    a missing attribute into an AttributeError on every call -- which is how the
    existing usage tests caught this, their fakes not carrying the field either.
    A reply whose finish reason is unknown is treated as an ordinary reply.
    """
    return str(finish_reason) in TRUNCATED and not content.strip()


# System parameter type: plain string OR a list of typed blocks (Anthropic format).
# List form is only meaningful for providers that support prompt caching
# (see `supports_prompt_caching` on each client class).
SystemParam = str | list[dict[str, Any]]


class LLMClient(ABC):
    # Set to True in provider subclasses that support Anthropic-style
    # cache_control blocks in the system parameter.
    supports_prompt_caching: bool = False

    @abstractmethod
    def complete(self, system: SystemParam, user: str, max_tokens: int | None = None) -> str:
        """Return the full response string for a single-turn completion."""

    @abstractmethod
    def stream(self, system: SystemParam, user: str) -> Iterator[str]:
        """Yield response tokens one at a time."""

    # ------------------------------------------------------------------
    # Async API — default implementations bridge to the sync methods via
    # asyncio.to_thread() so that existing subclasses keep working.
    # Provider-specific subclasses (AnthropicClient, LiteLLMClient) override
    # these with native async implementations for true concurrency.
    # ------------------------------------------------------------------

    async def acomplete(
        self,
        system: SystemParam,
        user: str,
        max_tokens: int | None = None,
        *,
        temperature: float | None = None,
    ) -> str:
        """Async completion.  Default: runs sync complete() in a thread.

        Args:
            system:      System prompt (string or list of typed blocks).
            user:        User message.
            max_tokens:  Maximum tokens to generate. ``None`` means the
                         configured answer budget, resolved per call -- not a
                         literal default, because a default is bound at import
                         and the budget is a runtime value from
                         ``LLM_MAX_OUTPUT_TOKENS`` or the bound ``LLMOverride``.
            temperature: Sampling temperature (0.0–1.0).  ``None`` uses the
                         provider default.  Passed through on provider clients
                         that override this method; ignored by the default
                         thread-bridge implementation.
        """
        # Resolved here, not forwarded as ``None``. This bridge used to hand a
        # subclass the literal 1024, and an out-of-tree ``LLMClient`` typed the
        # way the in-repo doubles are -- ``max_tokens: int = 1024`` -- would take
        # a ``None`` straight to its SDK, which rejects it. Widening the base
        # signature is our business; changing what a subclass is handed is not.
        return await asyncio.to_thread(
            self.complete, system, user, max_tokens or output_budget("answer")
        )

    async def astream(self, system: SystemParam, user: str) -> AsyncIterator[str]:
        """Async token stream.  Default: collects sync stream() in a thread, then yields.

        .. warning::
           This default **destroys time to first token**: it drains the entire
           response in a worker thread before yielding anything, so a caller
           streaming to a user sees nothing until the model has finished. It is
           a correctness shim, not a streaming implementation.

           Both shipped providers override it, so nothing pays this today. A new
           provider that does not override it will look fine in tests — the
           tokens all arrive — and feel broken in the product.
        """
        tokens: list[str] = await asyncio.to_thread(list, self.stream(system, user))
        for token in tokens:
            yield token
