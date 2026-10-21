"""Only ``./.env``, in the working directory, is read.

``load_dotenv()`` with no path searches upward from the file that calls it,
which for an installed package is ``site-packages/nlqueries``, not the
directory the user runs ``nlqueries`` from; it looks at the working directory
only in an interactive session or ``python -c``. So the README's
``cp .env.example .env`` did nothing for a pip install. The check has to run
as a script file in a fresh interpreter, the way the ``nlqueries`` launcher
does, or it passes on the broken code.

Nor is a ``.env`` in a parent directory read: it belongs to whatever project
that directory is, not to this one.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_PROBE = """\
import nlqueries.config as c
print("MODEL=" + c._detect_model("anthropic"))
print("FROM=" + c.__file__)
"""

#: The package under test, so the child interpreter imports this checkout's
#: code rather than whatever the venv's editable install points at.
_REPO = Path(__file__).resolve().parents[1]


def _run_probe(tmp_path: Path, env_file: str | None, parent_env_file: str | None = None) -> str:
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)
    if env_file is not None:
        (project / ".env").write_text(env_file, encoding="utf-8")
    if parent_env_file is not None:
        (tmp_path / ".env").write_text(parent_env_file, encoding="utf-8")
    script = tmp_path / "probe.py"
    script.write_text(_PROBE, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k not in ("LLM_MODEL", "PYTHON_DOTENV_DISABLED")}
    env["NLQ_STATE_DIR"] = str(tmp_path / "state")
    env["PYTHONPATH"] = str(_REPO)
    done = subprocess.run(
        [sys.executable, str(script)],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    lines = done.stdout.splitlines()
    source = next(line for line in lines if line.startswith("FROM="))
    assert Path(source[5:]).resolve().is_relative_to(_REPO), source
    return next(line for line in lines if line.startswith("MODEL="))


def test_a_dotenv_in_the_working_directory_is_read(tmp_path: Path) -> None:
    line = _run_probe(tmp_path, "LLM_MODEL=model-from-the-working-directory\n")

    assert line == "MODEL=model-from-the-working-directory"


def test_a_dotenv_in_a_parent_directory_is_not_read(tmp_path: Path) -> None:
    line = _run_probe(tmp_path, None, parent_env_file="LLM_MODEL=model-from-the-parent\n")

    control = tmp_path / "control"
    control.mkdir()
    assert line != "MODEL=model-from-the-parent"
    assert line == _run_probe(control, None)


def test_the_working_directory_dotenv_is_read_beside_a_parents(tmp_path: Path) -> None:
    line = _run_probe(
        tmp_path,
        "LLM_MODEL=model-from-the-working-directory\n",
        parent_env_file="LLM_MODEL=model-from-the-parent\n",
    )

    assert line == "MODEL=model-from-the-working-directory"


def test_the_environment_still_wins_over_the_dotenv(tmp_path: Path) -> None:
    project_env = "LLM_MODEL=model-from-the-working-directory\n"
    project = tmp_path / "project"
    project.mkdir()
    (project / ".env").write_text(project_env, encoding="utf-8")
    script = tmp_path / "probe.py"
    script.write_text(_PROBE, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k != "PYTHON_DOTENV_DISABLED"}
    env["LLM_MODEL"] = "model-from-the-environment"
    env["NLQ_STATE_DIR"] = str(tmp_path / "state")
    env["PYTHONPATH"] = str(_REPO)
    done = subprocess.run(
        [sys.executable, str(script)],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )

    assert "MODEL=model-from-the-environment" in done.stdout
