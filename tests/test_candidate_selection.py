"""Execution-verified candidate selection for the first answer.

Candidates come from a fake LLM, keyed by prompt variant; they run on a real
SQLite file through per-call connectors, as the orchestrator's would.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml
from nlqueries.connectors.base import QueryResult
from nlqueries.connectors.loader import LookupSource
from nlqueries.connectors.sqlite import SQLiteConnector
from nlqueries.execution import ExecutionPolicy
from nlqueries.orchestrator import candidate_selection as csel
from nlqueries.orchestrator.candidate_selection import (
    VARIANTS,
    Candidate,
    adopt,
    choose_first_answer,
)
from nlqueries.orchestrator.match_notes import (
    Choice,
    add_choice,
    add_match_note,
    collect_choices,
    collect_match_notes,
)
from nlqueries.orchestrator.previous_attempt import PreviousAttempt, use_previous_attempt
from nlqueries.orchestrator.prompt_assembly import AssembledPrompt, _build_static_system
from nlqueries.orchestrator.provenance import Provenance, use_provenance
from nlqueries.orchestrator.sql_generation import SQLGenerationResult
from nlqueries.query_settings import QuerySettings, use_query_settings

KB: dict[str, Any] = {
    "db_name": "shop",
    "schema": {
        "tables": [
            {
                "name": "t",
                "columns": [
                    {"name": "id", "type": "INTEGER", "is_primary_key": True},
                    {"name": "name", "type": "TEXT", "samples": ["a", "b"]},
                    {"name": "score", "type": "INTEGER"},
                ],
            }
        ]
    },
}
ROWS = [(1, "a", 10), (2, "b", 60), (3, "c", 90)]
QUESTION = "Which names score well?"
TABLES_FIRST = VARIANTS[2].user_suffix
# A run-time error that static validation cannot see.
FAILS_WHEN_RUN = "SELECT abs(-9223372036854775808) FROM t"


class _Recording(SQLiteConnector):
    def __init__(self) -> None:
        super().__init__()
        self.closed = False

    def close(self) -> None:
        self.closed = True
        super().close()


class _Source:
    """A lookup source on a SQLite file, recording the connectors it opens."""

    def __init__(self, db: Path, fail: bool = False) -> None:
        self.db = db
        self.fail = fail
        self.opened: list[_Recording] = []
        self.source = LookupSource(key=str(db), open=self._open)
        #: The lookup_source the orchestrator asked for this source, when driven.
        self.finder: MagicMock | None = None

    def _open(self) -> _Recording:
        if self.fail:
            raise ConnectionError("database unreachable")
        connector = _Recording()
        connector.connect({"database": str(self.db)})
        connector.bind_execution_policy(ExecutionPolicy.execute_read_only())
        self.opened.append(connector)
        return connector


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "shop.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT, score INTEGER)")
        conn.executemany("INSERT INTO t VALUES (?, ?, ?)", ROWS)
    conn.close()
    return path


class _LLM:
    """Answers by prompt variant; counts every call."""

    supports_prompt_caching = True

    def __init__(self, by_variant: dict[str, str]) -> None:
        self.by_variant = by_variant
        self.calls: list[tuple[Any, str]] = []
        self.streams = 0

    @staticmethod
    def variant_of(system: Any, user: str) -> str:
        if user.endswith(TABLES_FIRST):
            return "tables_first"
        static = system[0]["text"] if isinstance(system, list) else str(system)
        return "prompt" if "samples:" in static else "no_samples"

    async def acomplete(
        self, system: Any, user: str, max_tokens: int | None = None, **_: Any
    ) -> str:
        self.calls.append((system, user))
        reply = self.by_variant[self.variant_of(system, user)]
        if isinstance(reply, BaseException):
            raise reply
        return f"Reasoning for {self.variant_of(system, user)}.\n<sql>{reply}</sql>"

    async def astream(self, system: Any, user: str) -> Any:
        self.streams += 1
        yield f"Reasoning.\n<sql>{self.by_variant['prompt']}</sql>"


def _choose(
    llm: _LLM,
    src: _Source,
    lookups: LookupSource | None = None,
    *,
    timeout_seconds: float | None = None,
    deadline: float | None = None,
) -> tuple[Candidate, dict[str, Any]]:
    prompt = AssembledPrompt(
        static_system=_build_static_system(KB), dynamic_context="", user_question=QUESTION
    )
    collected = Provenance()

    async def run() -> Candidate:
        with use_provenance(collected):
            return await choose_first_answer(
                prompt=prompt,
                knowledge_base=KB,
                llm=llm,  # type: ignore[arg-type]
                dialect="sqlite",
                question=QUESTION,
                source=src.source,
                lookups=lookups,
                cache=True,
                timeout_seconds=timeout_seconds,
                deadline=deadline,
            )

    chosen = asyncio.run(run())
    assert collected.candidate_selection is not None
    return chosen, collected.candidate_selection


# --- Choosing ------------------------------------------------------------------------


def test_the_largest_group_of_identical_results_wins(db: Path) -> None:
    """Not the first candidate: the two that agree."""
    llm = _LLM(
        {
            "prompt": "SELECT name FROM t WHERE score > 5",
            "no_samples": "SELECT name FROM t WHERE score > 50",
            "tables_first": "SELECT name FROM t WHERE score >= 51",
        }
    )

    chosen, record = _choose(llm, _Source(db))

    assert chosen.index == 1  # of the agreeing two, the shorter SQL
    assert record["groups"] == [2, 1] and record["executed"] == 3 and record["chosen"] == 1
    assert record["reason"].startswith("2 of 3 executed candidates")


def test_on_a_tie_a_non_empty_result_beats_an_empty_one(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The empty one is the shorter SQL, so only the non-empty rule picks the other."""
    monkeypatch.setattr("nlqueries.config.CANDIDATE_COUNT", 2)
    llm = _LLM(
        {
            "prompt": "SELECT name FROM t WHERE id > 9",
            "no_samples": "SELECT name FROM t WHERE score > 50",
        }
    )

    chosen, record = _choose(llm, _Source(db))

    assert chosen.index == 1
    assert "non-empty" in record["reason"]


def test_on_a_tie_a_candidate_valid_without_repair_wins(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The repaired one (a string LIMIT, fixed mechanically) is the shorter SQL."""
    monkeypatch.setattr("nlqueries.config.CANDIDATE_COUNT", 2)
    llm = _LLM(
        {
            "prompt": "SELECT name FROM t LIMIT '1'",
            "no_samples": "SELECT name FROM t WHERE score > 50 ORDER BY name",
        }
    )

    chosen, record = _choose(llm, _Source(db))

    assert chosen.index == 1
    assert "valid without repair" in record["reason"]


def test_a_candidate_whose_run_fails_never_wins(db: Path) -> None:
    """Two failures with the same error are not a group of two."""
    llm = _LLM(
        {
            "prompt": FAILS_WHEN_RUN,
            "no_samples": FAILS_WHEN_RUN,
            "tables_first": "SELECT name FROM t WHERE score > 50",
        }
    )

    chosen, record = _choose(llm, _Source(db))

    assert chosen.index == 2
    assert record["executed"] == 1 and record["groups"] == [1]
    statuses = [c["status"] for c in record["candidates"]]
    assert statuses == ["error", "error", "ok"]
    assert "integer overflow" in record["candidates"][0]["error"]


def test_a_candidate_that_runs_past_its_deadline_never_wins(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("nlqueries.config.CANDIDATE_COUNT", 2)
    monkeypatch.setattr("nlqueries.config.CANDIDATE_EXEC_TIMEOUT_S", 1.0)
    release = threading.Event()
    real = _Source(db)

    class _SlowFirst:
        def __init__(self) -> None:
            self.inner = real._open()

        def execute_query(self, sql: str, *args: Any, **kwargs: Any) -> QueryResult:
            if "id > 0" in sql:
                release.wait(10)
            return self.inner.execute_query(sql, *args, **kwargs)

        def close(self) -> None:
            self.inner.close()

    src = _Source(db)
    src.source = LookupSource(key="slow", open=_SlowFirst)
    llm = _LLM(
        {
            "prompt": "SELECT name FROM t WHERE id > 0",
            "no_samples": "SELECT name FROM t WHERE score > 50",
        }
    )
    try:
        chosen, record = _choose(llm, src)
    finally:
        release.set()

    assert chosen.index == 1
    assert [c["status"] for c in record["candidates"]] == ["timeout", "ok"]


def test_with_nothing_run_the_first_candidate_is_the_answer(db: Path) -> None:
    llm = _LLM(
        {
            "prompt": "SELECT name FROM t WHERE score > 5",
            "no_samples": "SELECT name FROM t WHERE score > 50",
            "tables_first": "SELECT name FROM t WHERE score > 50",
        }
    )

    chosen, record = _choose(llm, _Source(db, fail=True))

    assert chosen.index == 0
    assert chosen.result is not None and chosen.result.sql == "SELECT name FROM t WHERE score > 5"
    assert record["executed"] == 0 and record["groups"] == []
    assert record["reason"].startswith("no candidate executed")


# --- How candidates are made and run --------------------------------------------------


def test_the_variants_differ_in_their_prompts_and_every_call_is_counted(db: Path) -> None:
    llm = _LLM({v.name: "SELECT name FROM t" for v in VARIANTS})

    _choose(llm, _Source(db))

    assert len(llm.calls) == 3
    seen = sorted(_LLM.variant_of(system, user) for system, user in llm.calls)
    assert seen == ["no_samples", "prompt", "tables_first"]
    # Each system keeps its own cache marker.
    assert all(system[0].get("cache_control") for system, _ in llm.calls)


def test_each_candidate_runs_on_its_own_connector_and_closes_it(db: Path) -> None:
    src = _Source(db)
    llm = _LLM({v.name: "SELECT name FROM t WHERE score > 50" for v in VARIANTS})

    _choose(llm, src)

    assert len(src.opened) == 3
    assert len({id(c) for c in src.opened}) == 3
    assert all(c.closed for c in src.opened)


def test_the_answers_provenance_is_the_chosen_candidates_alone(db: Path) -> None:
    """Every candidate grounds its literal; one substitution is recorded, not three."""
    src = _Source(db)
    sql = "SELECT id FROM t WHERE name = 'A'"
    llm = _LLM({v.name: sql for v in VARIANTS})
    prompt = AssembledPrompt(
        static_system=_build_static_system(KB), dynamic_context="", user_question=QUESTION
    )
    collected = Provenance()

    async def run() -> tuple[Candidate, list[Any]]:
        with use_provenance(collected):
            chosen = await choose_first_answer(
                prompt=prompt,
                knowledge_base=KB,
                llm=llm,  # type: ignore[arg-type]
                dialect="sqlite",
                question=QUESTION,
                source=src.source,
                lookups=src.source,
                cache=True,
            )
            before = list(collected.literals_grounded)
            adopt(chosen)
            return chosen, before

    chosen, before = asyncio.run(run())

    assert before == [], "no candidate recorded into the request's provenance"
    assert chosen.result is not None and chosen.result.sql == "SELECT id FROM t WHERE name = 'a'"
    assert collected.literals_grounded == [
        {"table": "t", "column": "name", "before": "A", "after": "a"}
    ]
    record = collected.to_dict()["candidate_selection"]
    assert set(record) == {"count", "executed", "groups", "chosen", "reason", "candidates"}
    assert set(record["candidates"][0]) == {"variant", "sql", "status", "rows"}
    assert record["candidates"][0]["rows"] == 1


def test_variants_beyond_the_table_cycle_through_it(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("nlqueries.config.CANDIDATE_COUNT", 4)
    llm = _LLM({v.name: "SELECT name FROM t" for v in VARIANTS})

    _, record = _choose(llm, _Source(db))

    assert [c["variant"] for c in record["candidates"]] == [
        "prompt",
        "no_samples",
        "tables_first",
        "prompt",
    ]


def test_the_calls_are_made_concurrently(db: Path) -> None:
    """Each call waits until all three are in flight: made one after another,
    the first would wait for the others for ever, and time out."""
    in_flight = 0
    all_in = asyncio.Event()
    inner = _LLM({v.name: "SELECT name FROM t" for v in VARIANTS})

    class _Gathering(_LLM):
        async def acomplete(self, system: Any, user: str, *args: Any, **kwargs: Any) -> str:
            nonlocal in_flight
            in_flight += 1
            if in_flight == len(VARIANTS):
                all_in.set()
            await asyncio.wait_for(all_in.wait(), 5)
            return await inner.acomplete(system, user)

    chosen, record = _choose(_Gathering(inner.by_variant), _Source(db))

    assert record["executed"] == 3


def test_rows_are_compared_after_sorting_then_capping(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same rows in a different order agree, cap or no cap."""
    monkeypatch.setattr("nlqueries.config.CANDIDATE_COUNT", 2)
    monkeypatch.setattr("nlqueries.config.CANDIDATE_MAX_ROWS", 2)
    llm = _LLM(
        {
            "prompt": "SELECT name FROM t ORDER BY name",
            "no_samples": "SELECT name FROM t ORDER BY name DESC",
        }
    )

    _, record = _choose(llm, _Source(db))

    assert record["groups"] == [2]
    assert [c["rows"] for c in record["candidates"]] == [2, 2]


def test_a_cut_off_reply_is_validated_as_cut_off_and_a_whole_one_is_not(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Judged per candidate, from its own reply, as the streamed answer is."""
    monkeypatch.setattr("nlqueries.config.CANDIDATE_COUNT", 2)
    seen: dict[str, bool] = {}
    real = csel.validate_and_repair

    async def validate(sql: str, *args: Any, cut_off: bool = False, **kwargs: Any) -> Any:
        seen[sql] = cut_off
        return await real(sql, *args, cut_off=cut_off, **kwargs)

    class _CutFirst(_LLM):
        async def acomplete(self, system: Any, user: str, *a: Any, **kw: Any) -> str:
            if self.variant_of(system, user) == "prompt":
                return "Reasoning.\n<sql>SELECT name FROM t WHERE"
            return await super().acomplete(system, user)

    monkeypatch.setattr(csel, "validate_and_repair", validate)
    llm = _CutFirst({"prompt": "", "no_samples": "SELECT name FROM t"})

    _choose(llm, _Source(db))

    assert seen == {"SELECT name FROM t WHERE": True, "SELECT name FROM t": False}


def test_when_every_call_fails_the_first_error_is_raised(db: Path) -> None:
    llm = _LLM({v.name: RuntimeError(f"{v.name} is down") for v in VARIANTS})  # type: ignore[misc]

    with pytest.raises(RuntimeError, match="prompt is down"):
        _choose(llm, _Source(db))


def test_one_failed_call_leaves_the_others_to_choose_from(db: Path) -> None:
    llm = _LLM(
        {
            "prompt": RuntimeError("busy"),  # type: ignore[dict-item]
            "no_samples": "SELECT name FROM t WHERE score > 50",
            "tables_first": "SELECT name FROM t WHERE score >= 51",
        }
    )

    chosen, record = _choose(llm, _Source(db))

    assert chosen.index == 1
    assert record["candidates"][0] == {
        "variant": "prompt",
        "sql": "",
        "status": "no_reply",
        "error": "busy",
    }


# --- Each candidate's notes and choices are its own -----------------------------------


_NOTE_VALUES = {"prompt": ("A",), "no_samples": ("B",), "tables_first": ("C",)}


def _noting(variant_of_sql: dict[str, str], *, choose: set[str]) -> Any:
    """A validate_and_repair that leaves a note, and for *choose* a choice,
    naming the candidate it validated."""

    async def validate(sql: str, *args: Any, **kwargs: Any) -> SQLGenerationResult:
        variant = variant_of_sql[sql]
        add_match_note(f"note from {variant}", _NOTE_VALUES[variant])
        if variant in choose:
            add_choice(
                Choice(key=f"t.name:{variant}", column="t.name", asked="x", options=("x1", "x2"))
            )
        return SQLGenerationResult(
            sql=sql, is_valid=True, validation_error=None, dialect="sqlite", attempt_count=1
        )

    return validate


def test_only_the_chosen_candidates_notes_and_choices_reach_the_request(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sqls = {
        "prompt": "SELECT name FROM t WHERE score > 5",
        "no_samples": "SELECT name FROM t WHERE score > 50",
        "tables_first": "SELECT name FROM t WHERE score >= 51",
    }
    monkeypatch.setattr(
        csel, "validate_and_repair", _noting({v: k for k, v in sqls.items()}, choose=set())
    )
    llm = _LLM(sqls)
    prompt = AssembledPrompt(
        static_system=_build_static_system(KB), dynamic_context="", user_question=QUESTION
    )

    async def run() -> tuple[Candidate, list[Any], list[Any], list[Any]]:
        with collect_match_notes() as notes, collect_choices() as choices:
            chosen = await choose_first_answer(
                prompt=prompt,
                knowledge_base=KB,
                llm=llm,  # type: ignore[arg-type]
                dialect="sqlite",
                question=QUESTION,
                source=_Source(db).source,
                lookups=None,
                cache=True,
            )
            before = list(notes)
            adopt(chosen)
            return chosen, before, list(notes), list(choices)

    chosen, before, notes, choices = asyncio.run(run())

    assert before == [], "no candidate wrote into the request's collection"
    assert chosen.variant == "no_samples"
    assert notes == [("note from no_samples", ("B",))]
    assert choices == []


def test_adopt_records_the_chosen_ones_grounding_choices_and_repair() -> None:
    chosen = Candidate(index=0, variant="prompt")
    chosen.provenance.validator.append("repaired after: not allowed")
    chosen.provenance.literals_grounded.append(
        {"table": "t", "column": "name", "before": "A", "after": "a"}
    )
    chosen.provenance.literal_repair = {
        "attempted": True,
        "changed": True,
        "notes": ["n"],
        "reason": None,
    }
    choice = Choice(key="t.name:x", column="t.name", asked="x", options=("x1", "x2"))
    chosen.notes = [("a note", ("a",))]
    chosen.choices = [choice]
    collected = Provenance()

    with use_provenance(collected), collect_match_notes() as notes, collect_choices() as choices:
        adopt(chosen)

    assert collected.validator == ["repaired after: not allowed"]
    assert collected.literals_grounded == chosen.provenance.literals_grounded
    assert collected.literal_repair is not None
    assert (collected.literal_repair["changed"], collected.literal_repair["notes"]) == (True, ["n"])
    assert notes == [("a note", ("a",))] and choices == [choice]


def test_a_candidate_that_leaves_a_choice_for_the_user_is_not_run(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing runs until the user has chosen; the other candidates still do."""
    sqls = {
        "prompt": "SELECT name FROM t WHERE name = 'x'",
        "no_samples": "SELECT name FROM t WHERE score > 50",
        "tables_first": "SELECT name FROM t WHERE score >= 51",
    }
    monkeypatch.setattr(
        csel, "validate_and_repair", _noting({v: k for k, v in sqls.items()}, choose={"prompt"})
    )
    src = _Source(db)

    chosen, record = _choose(_LLM(sqls), src)

    assert [c["status"] for c in record["candidates"]] == ["awaiting_choice", "ok", "ok"]
    assert len(src.opened) == 2
    assert chosen.variant == "no_samples"


def test_with_every_candidate_awaiting_a_choice_the_first_is_the_answer_and_asks(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sql = "SELECT name FROM t WHERE name = 'x'"
    monkeypatch.setattr("nlqueries.config.CANDIDATE_COUNT", 1)
    monkeypatch.setattr(csel, "validate_and_repair", _noting({sql: "prompt"}, choose={"prompt"}))
    src = _Source(db)

    chosen, record = _choose(_LLM({"prompt": sql}), src)

    assert chosen.index == 0 and chosen.choices and src.opened == []
    assert record["reason"].startswith("no candidate executed")


# --- Through the orchestrator ---------------------------------------------------------


_ANSWER_ROWS = QueryResult(
    columns=["name"], rows=[["b"], ["c"]], row_count=2, execution_time_ms=1.0, error=None
)


def _drive(
    on: bool,
    llm: _LLM,
    src: _Source | None,
    execution: ExecutionPolicy,
    *,
    previous: PreviousAttempt | None = None,
    deadline: float | None = None,
    timeout_seconds: float | None = None,
) -> tuple[dict[str, Any], Provenance, MagicMock, str]:
    from nlqueries.orchestrator.orchestrator import Orchestrator

    finder = MagicMock(return_value=src.source if src is not None else None)
    if src is not None:
        src.finder = finder
    # The connector the answer itself runs on, for the user, when the policy
    # lets it: not one of the candidates'.
    answer_connector = MagicMock()
    answer_connector.execute_query.return_value = _ANSWER_ROWS
    answer_connector.explain_plan.return_value = None
    frames: list[dict[str, Any]] = []
    text: list[str] = []
    collected = Provenance()
    settings = QuerySettings(
        literal_grounding=False,
        result_checks=False,
        plan_check=False,
        empty_result_explain=False,
        reading_line=False,
        candidate_selection=on,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "shop.yaml").write_text(yaml.dump(KB), encoding="utf-8")
        with (
            patch("nlqueries.orchestrator.orchestrator.config") as cfg,
            patch("nlqueries.orchestrator.orchestrator.get_llm_client", return_value=llm),
            patch("nlqueries.connectors.loader.lookup_source", new=finder),
            patch(
                "nlqueries.connectors.loader.open_connector_for_agent",
                return_value=answer_connector,
            ),
            patch("nlqueries.embeddings.qdrant_store.search_schema", return_value=[]),
            patch("nlqueries.embeddings.qdrant_store.search", return_value=[]),
            patch("nlqueries.embeddings.embedder.embed_text", return_value=[0.0] * 384),
            patch("nlqueries.orchestrator.prompt_assembly._search_verified", return_value=[]),
            use_query_settings(settings),
            use_previous_attempt(previous),
        ):
            cfg.KB_PATH = Path(tmpdir)
            cfg.CONNECTOR_STATEMENT_TIMEOUT_SECONDS = 30.0

            async def collect() -> None:
                with use_provenance(collected):
                    async for token in Orchestrator().handle_question(
                        QUESTION,
                        "shop",
                        dialect="sqlite",
                        execution=execution,
                        deadline=deadline,
                        timeout_seconds=timeout_seconds,
                    ):
                        if token.startswith("{") and '"type": "sql"' in token:
                            frames.append(json.loads(token))
                        elif not token.startswith("{"):
                            text.append(token)

            asyncio.run(collect())
    return frames[-1], collected, answer_connector, "".join(text)


_DIFFERING = {
    "prompt": "SELECT name FROM t WHERE score > 5",
    "no_samples": "SELECT name FROM t WHERE score > 50",
    "tables_first": "SELECT name FROM t WHERE score >= 51",
}


@pytest.mark.parametrize(
    "execution", [ExecutionPolicy.generate_only(), ExecutionPolicy.execute_read_only()]
)
def test_off_makes_exactly_one_call_and_runs_no_candidate(
    db: Path, execution: ExecutionPolicy
) -> None:
    src = _Source(db)
    llm = _LLM({v.name: "SELECT name FROM t WHERE score > 50" for v in VARIANTS})

    frame, collected, _, _ = _drive(False, llm, src, execution)

    assert llm.streams == 1 and llm.calls == []
    assert src.opened == []
    assert collected.candidate_selection is None
    assert frame["sql"] == "SELECT name FROM t WHERE score > 50"


def test_on_answers_with_the_chosen_candidate_and_runs_it_for_the_user(db: Path) -> None:
    src = _Source(db)
    llm = _LLM(_DIFFERING)

    frame, collected, answer_connector, _ = _drive(
        True, llm, src, ExecutionPolicy.execute_read_only()
    )

    assert llm.streams == 0 and len(llm.calls) == 3
    assert len(src.opened) == 3 and all(c.closed for c in src.opened)
    assert frame["sql"] == "SELECT name FROM t WHERE score > 50"
    assert frame["sql_table"]["rows"] == [["b"], ["c"]]
    (run,) = [c.args[0] for c in answer_connector.execute_query.call_args_list]
    assert run == "SELECT name FROM t WHERE score > 50"
    assert collected.candidate_selection is not None
    assert collected.candidate_selection["chosen"] == 1


def test_the_chosen_candidates_reasoning_is_what_the_caller_reads(db: Path) -> None:
    _, _, _, text = _drive(True, _LLM(_DIFFERING), _Source(db), ExecutionPolicy.execute_read_only())

    assert "Reasoning for no_samples." in text
    assert "Reasoning for prompt." not in text and "<sql>" not in text


def test_on_under_generate_only_takes_the_single_call(db: Path) -> None:
    """Checking a candidate means running the model's statement, which this
    policy forbids."""
    src = _Source(db)
    llm = _LLM(_DIFFERING)

    frame, collected, _, _ = _drive(True, llm, src, ExecutionPolicy.generate_only())

    assert llm.streams == 1 and llm.calls == []
    assert src.opened == []
    assert collected.candidate_selection is None
    assert frame["sql"] == _DIFFERING["prompt"]


def test_on_with_no_registered_connector_takes_the_single_call(db: Path) -> None:
    llm = _LLM({v.name: "SELECT name FROM t WHERE score > 50" for v in VARIANTS})

    _, collected, _, _ = _drive(True, llm, None, ExecutionPolicy.execute_read_only())

    assert llm.streams == 1 and llm.calls == []
    assert collected.candidate_selection is None


def test_on_a_correction_of_a_previous_attempt_takes_no_candidates(db: Path) -> None:
    src = _Source(db)

    class _Correcting(_LLM):
        async def acomplete(self, system: Any, user: str, *a: Any, **kw: Any) -> str:
            self.calls.append((system, user))
            return "<sql>SELECT name FROM t WHERE score > 50</sql>"

    llm = _Correcting(_DIFFERING)
    previous = PreviousAttempt(sql="SELECT nme FROM t", reason="no such column: nme")

    frame, collected, _, _ = _drive(
        True, llm, src, ExecutionPolicy.execute_read_only(), previous=previous
    )

    assert len(llm.calls) == 1 and llm.streams == 0
    assert src.opened == []
    # Not even asked where candidates would run (grounding is off here).
    assert src.finder is not None and src.finder.call_count == 0
    assert collected.candidate_selection is None
    assert frame["sql"] == "SELECT name FROM t WHERE score > 50"


def test_the_setting_is_a_query_setting_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NLQ_CANDIDATE_SELECTION", raising=False)
    assert QuerySettings().candidate_selection is False
    assert QuerySettings().overridden({"candidate_selection": True}).candidate_selection is True
    assert QuerySettings(candidate_selection=True).as_env()["NLQ_CANDIDATE_SELECTION"] == "true"
    with pytest.raises(ValueError, match="true or false"):
        QuerySettings().overridden({"candidate_selection": "on"})


def test_the_chosen_candidates_match_note_is_said_with_the_answer(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Through the orchestrator: the chosen one's note, once; no other's."""
    sqls = {
        "prompt": "SELECT name FROM t WHERE name = 'A' OR score > 5",
        "no_samples": "SELECT name FROM t WHERE name = 'B' OR score > 50",
        "tables_first": "SELECT name FROM t WHERE name = 'C' OR score >= 51",
    }
    monkeypatch.setattr(
        csel, "validate_and_repair", _noting({v: k for k, v in sqls.items()}, choose=set())
    )

    frame, _, _, _ = _drive(True, _LLM(sqls), _Source(db), ExecutionPolicy.execute_read_only())

    assert frame["sql"] == sqls["no_samples"]
    assert frame["result_notes"] == ["note from no_samples"]


# --- Bounded by the request -------------------------------------------------------------


class _Timed(SQLiteConnector):
    """Records the statement timeout each query is given."""

    seen: list[float] = []

    def execute_query(self, sql: str, timeout_seconds: float = 0, **kw: Any) -> QueryResult:
        _Timed.seen.append(timeout_seconds)
        return super().execute_query(sql, timeout_seconds, **kw)


def _timed_source(db: Path) -> _Source:
    src = _Source(db)

    def _open() -> _Timed:
        connector = _Timed()
        connector.connect({"database": str(db)})
        connector.bind_execution_policy(ExecutionPolicy.execute_read_only())
        return connector

    src.source = LookupSource(key="timed", open=_open)
    return src


def test_each_run_is_bounded_by_the_requests_statement_timeout(db: Path) -> None:
    _Timed.seen = []
    llm = _LLM({v.name: "SELECT name FROM t" for v in VARIANTS})

    _choose(llm, _timed_source(db), timeout_seconds=3.0)

    assert _Timed.seen == [3.0, 3.0, 3.0]


def test_a_request_with_no_statement_timeout_keeps_the_setting(db: Path) -> None:
    """Zero is the connectors' "no timeout", not a budget of nothing."""
    _Timed.seen = []
    llm = _LLM({v.name: "SELECT name FROM t" for v in VARIANTS})

    _choose(llm, _timed_source(db), timeout_seconds=0)

    assert _Timed.seen == [10.0, 10.0, 10.0]


def test_the_runs_take_at_most_half_the_time_left_before_the_deadline(db: Path) -> None:
    _Timed.seen = []
    llm = _LLM({v.name: "SELECT name FROM t" for v in VARIANTS})

    _choose(llm, _timed_source(db), deadline=time.monotonic() + 8.0)

    assert len(_Timed.seen) == 3 and all(3.0 < t <= 4.0 for t in _Timed.seen), _Timed.seen


def test_with_too_little_time_left_no_candidate_runs(db: Path) -> None:
    src = _Source(db)
    llm = _LLM(_DIFFERING)

    chosen, record = _choose(llm, src, deadline=time.monotonic() + 1.5)

    assert src.opened == []
    assert chosen.index == 0
    assert [c["status"] for c in record["candidates"]] == ["not_run"] * 3
    assert record["reason"].startswith("no candidate executed")


def test_the_orchestrator_hands_its_deadline_to_the_runs(db: Path) -> None:
    src = _Source(db)
    llm = _LLM(_DIFFERING)

    frame, collected, _, _ = _drive(
        True, llm, src, ExecutionPolicy.execute_read_only(), deadline=time.monotonic() + 1.5
    )

    assert src.opened == []
    assert collected.candidate_selection is not None
    assert {c["status"] for c in collected.candidate_selection["candidates"]} == {"not_run"}
    assert frame["sql"] == _DIFFERING["prompt"]


def test_a_result_the_connector_cut_short_is_not_compared(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cut at the fetch limit in each statement's own order, two candidates
    with the same rows hold different prefixes."""
    monkeypatch.setattr("nlqueries.config.CONNECTOR_MAX_FETCH_ROWS", 2)
    llm = _LLM(
        {
            "prompt": "SELECT name FROM t ORDER BY name",
            "no_samples": "SELECT name FROM t ORDER BY name DESC",
            "tables_first": "SELECT name FROM t WHERE score > 50",
        }
    )

    chosen, record = _choose(llm, _Source(db))

    assert [c["status"] for c in record["candidates"]] == ["truncated", "truncated", "ok"]
    assert "more rows than the connector fetches" in record["candidates"][0]["error"]
    assert chosen.index == 2


def test_the_orchestrator_hands_its_statement_timeout_to_the_runs(db: Path) -> None:
    _Timed.seen = []
    llm = _LLM({v.name: "SELECT name FROM t" for v in VARIANTS})

    _drive(True, llm, _timed_source(db), ExecutionPolicy.execute_read_only(), timeout_seconds=2.0)

    assert _Timed.seen == [2.0, 2.0, 2.0]
