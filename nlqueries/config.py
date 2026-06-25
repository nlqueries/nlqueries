"""
nlqueries.config
~~~~~~~~~~~~~~~~
Loads runtime configuration from environment variables (and an optional
.env file in the working directory).

All other modules should import settings from here rather than reading
os.environ directly, so the source of truth is a single place.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the current working directory (or any parent).
# This is a no-op when the variables are already set in the environment,
# so it is safe to call unconditionally.
load_dotenv(override=False)

# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------
QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
"""URL of the Qdrant vector store. Defaults to a local instance."""

QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "nlqueries")
"""Name of the Qdrant collection used for query embeddings."""

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
"""Anthropic API key for Claude-based query generation and summarisation."""


def _detect_provider() -> str:
    """Resolve the LLM provider from the environment.

    Priority: explicit LLM_PROVIDER env var > key-based auto-detection.
    If OPENAI_API_KEY is set (and LLM_PROVIDER is not), routes through
    LiteLLM so the existing litellm client handles OpenAI calls.
    """
    explicit = os.getenv("LLM_PROVIDER", "").strip()
    if explicit:
        return explicit
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("OPENAI_API_KEY"):
        return "litellm"
    return "anthropic"


def _detect_model(provider: str) -> str:
    """Resolve the LLM model from the environment.

    Priority: explicit LLM_MODEL env var > provider default.
    When auto-detected as litellm (OpenAI key present), defaults to
    'openai/gpt-4o' so LiteLLM routes to OpenAI without extra config.
    """
    explicit = os.getenv("LLM_MODEL", "").strip()
    if explicit:
        return explicit
    if provider == "litellm" and os.getenv("OPENAI_API_KEY"):
        return "openai/gpt-4o"
    return "claude-sonnet-4-5"


LLM_PROVIDER: str = _detect_provider()
"""LLM provider to use. Auto-detected from available API keys if not set explicitly."""

LLM_MODEL: str = _detect_model(LLM_PROVIDER)
"""LLM model identifier. Defaults based on detected provider if not set explicitly."""

# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------
KB_PATH: Path = Path(os.getenv("KB_PATH", str(Path.home() / ".nlqueries" / "knowledge_base")))
"""Directory where generated YAML knowledge-base files are stored."""

KB_REFRESH_INTERVAL: int = int(os.getenv("KB_REFRESH_INTERVAL", "3600"))
"""How often (seconds) the knowledge base is refreshed from the source DB."""

# ---------------------------------------------------------------------------
# Connectors registry
# ---------------------------------------------------------------------------
CONNECTORS_FILE: Path = Path(
    os.getenv("CONNECTORS_FILE", str(Path.home() / ".nlqueries" / "connectors.yaml"))
)
"""YAML file that stores registered connector configurations."""

# ---------------------------------------------------------------------------
# Query Capsules
# ---------------------------------------------------------------------------
CAPSULES_DIR: Path = Path(os.getenv("CAPSULES_DIR", str(Path.home() / ".nlqueries" / "capsules")))

# ---------------------------------------------------------------------------
# Query history
# ---------------------------------------------------------------------------
QUERY_HISTORY_LIMIT: int = int(os.getenv("QUERY_HISTORY_LIMIT", "500"))
"""Maximum number of queries to fetch from the database's query history view."""
"""Directory where serialised QueryCapsule JSON files are stored."""

# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------
FEEDBACK_DIR: Path = Path(os.getenv("FEEDBACK_DIR", str(Path.home() / ".nlqueries" / "feedback")))
"""Directory where per-agent feedback JSONL files are stored."""

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
