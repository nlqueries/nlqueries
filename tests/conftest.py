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


def granted(connector):
    """Give *connector* permission to execute, explicitly.

    Connectors deny execution until something grants it (SEC-07): in production
    the loader binds this request's policy to a per-request wrapper. A unit test
    that builds a connector directly has to say the same thing out loud.

    Deliberately a helper called at each site rather than an autouse fixture.
    Blanket-granting in conftest would mean no test could ever observe a refusal,
    and the first thing to notice would be a security control that had quietly
    stopped working.
    """
    from nlqueries.execution import ExecutionPolicy

    connector.bind_execution_policy(ExecutionPolicy.execute_read_only())
    return connector
