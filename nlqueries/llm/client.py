# nlqueries-core — OSS (BSL 1.1)
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from typing import Any

from nlqueries.llm.override import output_budget

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
