"""
Shared pytest fixtures for the nlqueries-core test suite.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _stamp(path: Path) -> tuple[object, ...]:
    """Enough of *path* to notice any change to it, including a write-and-restore."""
    try:
        raw = path.read_bytes()
    except OSError:
        return (False,)
    return (True, len(raw), hashlib.sha256(raw).hexdigest())


@pytest.fixture(scope="session", autouse=True)
def _the_operators_connectors_file_is_left_alone() -> object:
    """Fail the session if the suite writes to the real ``CONNECTORS_FILE``.

    Not hypothetical. Two tests driving `nlqueries connect` destroyed the
    developer's own `~/.nlqueries/connectors.yaml`, replacing three registered
    connectors with `{}`, and passed while doing it. The cause is easy to
    reproduce by accident: `nlqueries/cli/main.py` binds `CONNECTORS_FILE` at
    import, so a fixture patching `config.CONNECTORS_FILE` redirects every
    reader and none of the CLI's writes -- and the writer's *read* goes through
    `load_connectors_for_update`, which is dynamic, so it sees the empty
    temporary file and writes `{}` over the real one.

    Redirecting neither name fails loudly. Redirecting exactly one is silent,
    which is why this is a guard rather than a note in a docstring: the fixture
    that gets it wrong is the fixture that looks right.

    Session-scoped, so it reads the path before any test can patch it. It names
    no test -- a bisect does that -- but it turns a silent loss into a failure
    that cannot be missed.
    """
    from nlqueries import config

    real: Path = config.CONNECTORS_FILE
    before = _stamp(real)
    yield
    after = _stamp(real)
    if before != after:
        pytest.fail(
            f"The suite modified {real}, which is the operator's real connector "
            f"registry and must never be written by a test. A fixture is very "
            f"likely patching `config.CONNECTORS_FILE` without also rebinding "
            f"`nlqueries.cli.main.CONNECTORS_FILE`, which the CLI's writer uses.",
            pytrace=False,
        )


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
