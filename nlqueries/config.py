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

LLM_MODEL: str = os.getenv("LLM_MODEL", "claude-sonnet-4-5")
"""Default Anthropic model used for query generation."""

LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "anthropic")
"""LLM provider to use. Resolved by nlqueries.llm.get_llm_client()."""

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
"""Directory where serialised QueryCapsule JSON files are stored."""

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
