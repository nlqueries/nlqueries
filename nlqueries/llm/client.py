# nlqueries-core — OSS (BSL 1.1)
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator


class LLMClient(ABC):
    @abstractmethod
    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        """Return the full response string for a single-turn completion."""

    @abstractmethod
    def stream(self, system: str, user: str) -> Iterator[str]:
        """Yield response tokens one at a time."""
