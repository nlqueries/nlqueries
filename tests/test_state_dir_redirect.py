"""
tests.test_state_dir_redirect
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The suite writes under a temporary state directory, and does so *in time*.

``tests/conftest.py`` sets ``NLQ_STATE_DIR`` at module level, before anything
imports ``nlqueries``. That ordering is the whole mechanism, and it is invisible
when it fails -- so it is asserted here rather than trusted.
"""

from __future__ import annotations

from pathlib import Path


def test_the_state_directory_is_not_the_operators() -> None:
    """The redirect took effect at all.

    Without it the suite writes under `~/.nlqueries`, which holds the operator's
    real state: their connector registry, their knowledge bases, their session
    logs. Two tests driving `nlqueries connect` once replaced three registered
    connectors with `{}` and passed while doing it.
    """
    import os

    from nlqueries import config

    state = config.STATE_DIR.resolve()

    assert state != (Path.home() / ".nlqueries").resolve(), "NLQ_STATE_DIR did not take effect"
    # And it is the directory conftest made, rather than some other override
    # that happens not to be the default.
    assert state == Path(os.environ["NLQ_STATE_DIR"]).resolve()

    # Deliberately *not* asserting the state directory sits outside the home
    # directory. It was the obvious phrasing and it is wrong on Windows, where
    # the system temp directory is `C:/Users/<user>/AppData/Local/Temp` -- so the
    # assertion failed on a correct redirect. What matters is that it is not the
    # operator's `~/.nlqueries`, not where the OS keeps its scratch space.


def test_the_bound_copies_cannot_disagree_with_the_config() -> None:
    """The property an environment variable buys that monkeypatching did not.

    `cli/main.py` binds `CONNECTORS_FILE`, `KB_PATH` and `STATE_DIR` at import.
    Patching `config.CONNECTORS_FILE` at run time redirects the readers and none
    of those bound copies -- "Redirecting neither name fails loudly. Redirecting
    exactly one is silent", as `conftest`'s guard puts it -- and that silent half
    is what replaced a developer's connector registry with `{}`.

    Setting the environment variable removes the failure mode rather than
    guarding it: `STATE_DIR` is read once, at import, so every bound copy is
    derived from the same value and they cannot drift apart. This asserts that,
    and is therefore a *property* test rather than a trap.

    Worth stating plainly, because I first wrote this claiming to catch a
    redirect that landed too late. It does not, and cannot: if something imports
    `nlqueries` before `conftest` runs, `config` itself reads the home directory
    and every derived value follows it there -- so that failure surfaces in
    `test_the_state_directory_is_not_the_operators` above, which is where
    mutating the ordering actually fails. There is no state in which the config
    is redirected and the bound copies are not.
    """
    from nlqueries import config
    from nlqueries.cli import main as cli_main

    assert cli_main.CONNECTORS_FILE == config.CONNECTORS_FILE, (
        "cli.main bound CONNECTORS_FILE before the redirect landed; its readers "
        "and writers now disagree, which stays silent until something writes"
    )
    assert cli_main.KB_PATH == config.KB_PATH
    assert cli_main.STATE_DIR == config.STATE_DIR


def test_everything_derived_from_it_moved_too() -> None:
    """Moving the root is only worth anything if the branches followed.

    These are what `config` derives from `STATE_DIR`, plus the embed server's pid
    file, which builds its path at import. The point of redirecting the root
    rather than guarding one file is that a test writing to any of them --
    including one nobody has thought of yet -- lands in the temporary tree.
    """
    from nlqueries import config

    state = config.STATE_DIR.resolve()
    for name in ("KB_PATH", "CONNECTORS_FILE", "CAPSULES_DIR", "FEEDBACK_DIR"):
        path: Path = getattr(config, name).resolve()
        assert path.is_relative_to(state), f"config.{name} is outside the test state dir: {path}"

    from nlqueries.embeddings import embed_server

    assert embed_server._PID_FILE.resolve().is_relative_to(state)
