"""
Shared pytest fixtures for the nlqueries-core test suite.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _no_semantic_cache():
    """Prevent tests from hitting real Qdrant via SemanticCache.

    Any test that exercises MultiAgentOrchestrator.handle_question() would
    otherwise embed the question and query/write real Qdrant, making test
    results non-deterministic (cached runs return word-split tokens instead
    of the mock agent tokens).  Patching SemanticCache at the module level
    keeps all orchestrator tests hermetic.
    """
    mock_cache = MagicMock()
    mock_cache.get.return_value = None  # always a cache miss
    with patch(
        "nlqueries.orchestrator.multi_agent_orchestrator.SemanticCache",
        return_value=mock_cache,
    ):
        yield
