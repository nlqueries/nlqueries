"""
Shared pytest fixtures for the nlqueries-core test suite.
"""

from __future__ import annotations

import atexit
import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# The suite writes under a temporary state directory, never the operator's.
#
# `config.STATE_DIR` is the root of everything NLQueries keeps between runs, and
# five values derive from it -- `KB_PATH`, `CONNECTORS_FILE`, `CAPSULES_DIR`,
# `FEEDBACK_DIR`, and the embed server's pid file -- plus the session log and the
# cache signing key, which are built from it at use.
#
# Guarding one of those was what we had, and it was not a mechanism. Three
# separate escapes into `~/.nlqueries` were found in one afternoon -- the
# connectors file, the KB under `KB_PATH`, and the session log under `STATE_DIR`
# -- and each was found by a reviewer pointing at it or by walking the directory
# by hand. Moving the root redirects a test that writes somewhere *new*, which is
# the case neither of those finds.
#
# **This must run before anything imports `nlqueries`,** because `STATE_DIR` is
# read at import and `cli/main.py` and `embed_server.py` bind values derived from
# it at import too. Setting it afterwards would redirect the readers and none of
# the writers, which is precisely the silent half-fix the guard below exists to
# catch. Measured rather than assumed: `sys.modules` holds no `nlqueries` module
# when this file executes, so a module-level assignment here is early enough.
# `tests/test_state_dir_redirect.py` asserts it at run time, so a future plugin
# that imports core first fails loudly instead of quietly writing home.
#
# An outer value still wins -- CI or a developer pinning it for a reproduction
# should not be silently overridden -- but only a *usable* one. An empty
# `NLQ_STATE_DIR` survives `setdefault`, and `Path("")` is the working
# directory, so the suite would scatter `connectors.yaml`, `knowledge_base/` and
# the rest through the checkout. Both assertions in
# `test_state_dir_redirect.py` pass in that state, because the cwd is not
# `~/.nlqueries`: the guard reports the redirect intact while it is not. Treated
# as a value somebody meant to disable, which is how `config` reads a blanked
# variable elsewhere.
_TEST_STATE_DIR = tempfile.mkdtemp(prefix="nlq-test-state-")
if not os.environ.get("NLQ_STATE_DIR"):
    os.environ["NLQ_STATE_DIR"] = _TEST_STATE_DIR
atexit.register(shutil.rmtree, _TEST_STATE_DIR, ignore_errors=True)

# And the per-path variables are removed, not merely left to fall back.
#
# `config` reads `KB_PATH`, `CONNECTORS_FILE`, `CAPSULES_DIR` and `FEEDBACK_DIR`
# from their own variables FIRST and only then from `STATE_DIR`, and
# `load_dotenv(override=False)` runs inside `config` at import -- so a `.env` in
# the working directory, which is the local setup `docs/configuration.md`
# describes, reattaches any one of those branches to the operator's real files
# despite the root having moved.
#
# The asymmetry with the root is deliberate. Honouring an outer `NLQ_STATE_DIR`
# means honouring somebody who said "put the whole tree here"; honouring an
# outer `KB_PATH` during a test run means letting one branch quietly point home
# while everything else is redirected, which is the half-redirect this file
# exists to prevent. `test_everything_derived_from_it_moved_too` does catch it,
# but only once it runs -- and in collection order `test_cli.py`,
# `test_connector_resolver_seam.py`, `test_feedback.py` and `test_kb_stats.py`
# have each had their chance to write by then.
for _per_path in ("KB_PATH", "CONNECTORS_FILE", "CAPSULES_DIR", "FEEDBACK_DIR"):
    os.environ.pop(_per_path, None)


def _stamp(path: Path) -> tuple[object, ...]:
    """Enough of *path* to notice any write to it, including one that changes nothing.

    Modification time as well as content, because content alone is not enough:
    rewriting a file with the bytes it already held leaves the hash identical, and
    so does writing something and putting the original back. Either means a test
    wrote to the operator's file and got away with it -- against a file holding
    real connectors, the same write is the destructive one. An earlier version of
    this compared content only, and missed exactly that.
    """
    try:
        raw = path.read_bytes()
        mtime = path.stat().st_mtime_ns
    except OSError:
        return (False,)
    return (True, mtime, len(raw), hashlib.sha256(raw).hexdigest())


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

    Session-scoped, so it reads the path before any test can patch it. Two limits
    worth knowing rather than discovering: it names no test, so a bisect is what
    identifies the writer, and it stamps at the first test's setup, which is after
    collection -- a write during module import happens before it is watching.

    Kept after the `NLQ_STATE_DIR` redirect at the top of this file, which should
    make reaching the real file impossible. That is exactly why it stays: it is
    now the check that the redirect is holding, and it costs one stat per
    session. A guard that only ever fires when a mechanism has failed is worth
    more than one that fires often.
    """
    from nlqueries import config

    real: Path = config.CONNECTORS_FILE
    before = _stamp(real)
    yield
    after = _stamp(real)
    if before == after:
        return

    # Two different faults, and one diagnosis for both would send half the readers
    # after the wrong thing. Changed content is a test that wrote something else
    # over the file. Identical content with a newer timestamp is a test that wrote
    # the same bytes back -- harmless here, and the same call against a file with
    # real connectors in it is the destructive one, so it is reported just as
    # loudly and with the fact that made it survivable stated rather than implied.
    same_content = before[2:] == after[2:]
    what = (
        "rewrote it with the bytes it already held (the contents are unchanged, "
        "so nothing was lost this time)"
        if same_content
        else "changed its contents"
    )
    pytest.fail(
        f"The suite {what}: {real}. That is the operator's real connector registry "
        f"and must never be written by a test. The usual cause is a fixture "
        f"patching `config.CONNECTORS_FILE` without also rebinding "
        f"`nlqueries.cli.main.CONNECTORS_FILE`, which the CLI's writer uses -- and "
        f"the same applies to `KB_PATH`. Note that this fixture starts watching at "
        f"the first test's setup, so a write during collection happens before it "
        f"is looking.",
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
