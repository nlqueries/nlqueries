# nlqueries-core — OSS (BSL 1.1)
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from typing import Any

# System parameter type: plain string OR a list of typed blocks (Anthropic format).
# List form is only meaningful for providers that support prompt caching
# (see `supports_prompt_caching` on each client class).
SystemParam = str | list[dict[str, Any]]


class LLMClient(ABC):
    # Set to True in provider subclasses that support Anthropic-style
    # cache_control blocks in the system parameter.
    supports_prompt_caching: bool = False

    @abstractmethod
    def complete(self, system: SystemParam, user: str, max_tokens: int = 1024) -> str:
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
        max_tokens: int = 1024,
        *,
        temperature: float | None = None,
    ) -> str:
        """Async completion.  Default: runs sync complete() in a thread.

        Args:
            system:      System prompt (string or list of typed blocks).
            user:        User message.
            max_tokens:  Maximum tokens to generate.
            temperature: Sampling temperature (0.0–1.0).  ``None`` uses the
                         provider default.  Passed through on provider clients
                         that override this method; ignored by the default
                         thread-bridge implementation.
        """
        return await asyncio.to_thread(self.complete, system, user, max_tokens)

    async def astream(self, system: SystemParam, user: str) -> AsyncIterator[str]:
        """Async token stream.  Default: collects sync stream() in a thread, then yields."""
        tokens: list[str] = await asyncio.to_thread(list, self.stream(system, user))
        for token in tokens:
            yield token
