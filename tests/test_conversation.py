"""Tests for nlqueries.orchestrator.conversation and assemble_prompt_with_history."""

from __future__ import annotations

from nlqueries.orchestrator.conversation import ConversationSession, create_session
from nlqueries.orchestrator.prompt_assembly import assemble_prompt_with_history

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session_with_turns(num_turns: int) -> ConversationSession:
    """Create a session with alternating user/assistant turns."""
    session = create_session("test-agent")
    for i in range(num_turns):
        if i % 2 == 0:
            session.add_turn("user", f"Question {i}")
        else:
            session.add_turn(
                "assistant",
                f"Answer {i}",
                agent_type="sql",
                sql=f"SELECT {i} FROM orders",
            )
    return session


def _minimal_kb() -> dict[str, object]:
    return {
        "schema": {"tables": []},
        "business_context": {"glossary": [], "rules": []},
        "query_capsules": [],
    }


# ---------------------------------------------------------------------------
# test_get_context_window_limits_to_max_turns
# ---------------------------------------------------------------------------


def test_get_context_window_limits_to_max_turns() -> None:
    # 10 turns total (indices 0–9); last 6 are indices 4–9
    session = _make_session_with_turns(10)
    window = session.get_context_window(max_turns=6)
    assert len(window) == 6
    # Turn at index 4 is a user turn ("Question 4")
    assert window[0].content == "Question 4"
    assert window[0].role == "user"


# ---------------------------------------------------------------------------
# test_to_prompt_messages_excludes_sql
# ---------------------------------------------------------------------------


def test_to_prompt_messages_excludes_sql() -> None:
    session = create_session("agent")
    session.add_turn("user", "How many orders?")
    session.add_turn(
        "assistant",
        "There were 100 orders.",
        agent_type="sql",
        sql="SELECT COUNT(*) FROM orders",
    )
    messages = session.to_prompt_messages()

    assert len(messages) == 2
    # Each message dict must have exactly "role" and "content" keys
    for msg in messages:
        assert set(msg.keys()) == {"role", "content"}

    # The SQL must not appear in any message content
    for msg in messages:
        assert "SELECT" not in msg["content"]

    assert messages[1]["content"] == "There were 100 orders."
    assert messages[1]["role"] == "assistant"


# ---------------------------------------------------------------------------
# test_add_turn_updates_timestamp
# ---------------------------------------------------------------------------


def test_add_turn_updates_timestamp() -> None:
    session = create_session("agent")
    before = session.updated_at

    session.add_turn("user", "Hello")

    assert session.updated_at >= before
    assert len(session.turns) == 1
    assert session.turns[0].role == "user"
    assert session.turns[0].content == "Hello"
    assert session.turns[0].agent_type is None
    assert session.turns[0].sql is None


# ---------------------------------------------------------------------------
# test_assemble_prompt_with_history_includes_prior_messages
# ---------------------------------------------------------------------------


def test_assemble_prompt_with_history_includes_prior_messages() -> None:
    kb = _minimal_kb()
    history: list[dict[str, str]] = [
        {"role": "user", "content": "How many customers?"},
        {"role": "assistant", "content": "There are 500 customers."},
    ]

    system_prompt, user_prompt, prior = assemble_prompt_with_history(
        "Show me those customers",
        kb,
        history,
    )

    # system_prompt is now a list of typed blocks (AssembledPrompt.system_blocks())
    assert isinstance(system_prompt, list)
    assert len(system_prompt) > 0
    # Every block must have a non-empty "text" field
    assert all(isinstance(b.get("text"), str) and b["text"].strip() for b in system_prompt)
    assert user_prompt == "Show me those customers"
    assert prior is history  # same list reference — not a copy
    assert len(prior) == 2
    assert prior[0]["role"] == "user"
    assert prior[1]["role"] == "assistant"
    assert prior[1]["content"] == "There are 500 customers."
