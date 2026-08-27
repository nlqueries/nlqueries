"""
nlqueries.feedback.store
~~~~~~~~~~~~~~~~~~~~~~~~
Local JSONL persistence for :class:`~nlqueries.feedback.models.QueryFeedback`.

Each agent's feedback is written to a separate file::

    ~/.nlqueries/feedback/<agent-id>.jsonl

One JSON object per line; appended on every ``record_feedback()`` call so the
file is readable with a simple ``readlines()`` even while new records are being
written.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from nlqueries import config
from nlqueries.feedback.models import QueryFeedback
from nlqueries.state_files import private_dir, restrict


def _agent_path(agent_id: str) -> Path:
    """Return the JSONL path for *agent_id*, creating parent dirs if needed."""
    safe_id = re.sub(r"[^\w.-]", "_", agent_id)
    return private_dir(config.FEEDBACK_DIR) / f"{safe_id}.jsonl"


def record_feedback(fb: QueryFeedback) -> None:
    """Append *fb* as a JSON line to the agent's local feedback file."""
    path = _agent_path(fb.agent_id)
    record = {
        "question": fb.question,
        "generated_sql": fb.generated_sql,
        "corrected_sql": fb.corrected_sql,
        "rating": fb.rating,
        "agent_id": fb.agent_id,
        "timestamp": fb.timestamp.isoformat(),
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    # Applied after writing: a file created by `open` takes the process umask.
    restrict(path)


def load_feedback(agent_id: str) -> list[QueryFeedback]:
    """Load all feedback records for *agent_id* from the local JSONL file.

    Returns an empty list if no feedback file exists yet.
    Lines that cannot be parsed are silently skipped so a corrupt entry does
    not prevent loading the rest.
    """
    path = _agent_path(agent_id)
    if not path.exists():
        return []

    results: list[QueryFeedback] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
            results.append(
                QueryFeedback(
                    question=raw["question"],
                    generated_sql=raw["generated_sql"],
                    corrected_sql=raw.get("corrected_sql"),
                    rating=raw["rating"],
                    agent_id=raw["agent_id"],
                    timestamp=datetime.fromisoformat(raw["timestamp"]),
                )
            )
        except (KeyError, ValueError):
            continue

    return results
