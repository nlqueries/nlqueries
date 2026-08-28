"""
How much a caller may ask for (SEC-14).

Authorisation says a principal may run `query` on an agent. It says nothing
about how often, and each call is an LLM charge, a query against the customer's
database, and a slot in a process with no cap on how many it will run at once.

The concurrency limit is the one that matters for a slow tool: `query` is
allowed forty-five seconds, so a caller ignoring the rate limit's refusals could
still hold open as many slow calls as they can start.
"""

from __future__ import annotations

import threading

import pytest
from nlqueries.auth.admission import (
    DEFAULT_CONCURRENCY,
    DEFAULT_RATE_LIMIT,
    AdmissionControl,
    TooManyRequests,
    from_config,
)


class TestTheRateLimit:
    def test_calls_up_to_the_limit_are_admitted(self) -> None:
        control = AdmissionControl(rate_limit=3, max_concurrent=0)

        for _ in range(3):
            control.acquire("alice")

    def test_the_next_call_is_refused(self) -> None:
        control = AdmissionControl(rate_limit=3, max_concurrent=0)
        for _ in range(3):
            control.acquire("alice")

        with pytest.raises(TooManyRequests, match="per minute"):
            control.acquire("alice")

    def test_one_caller_does_not_spend_another_s_allowance(self) -> None:
        """Per principal. A shared counter would let one caller deny everyone
        else, which is a denial of service wearing the clothes of a control."""
        control = AdmissionControl(rate_limit=2, max_concurrent=0)
        control.acquire("alice")
        control.acquire("alice")

        control.acquire("bob")

    def test_the_window_resets(self, monkeypatch) -> None:
        control = AdmissionControl(rate_limit=2, max_concurrent=0)
        clock = [1000.0]
        monkeypatch.setattr(control, "_now", lambda: clock[0])

        control.acquire("alice")
        control.acquire("alice")
        clock[0] += 61.0

        control.acquire("alice")

    def test_zero_disables_it(self) -> None:
        control = AdmissionControl(rate_limit=0, max_concurrent=0)

        for _ in range(100):
            control.acquire("alice")


class TestTheConcurrencyLimit:
    def test_calls_in_flight_up_to_the_limit_are_admitted(self) -> None:
        control = AdmissionControl(rate_limit=0, max_concurrent=2)

        control.acquire("alice")
        control.acquire("alice")

        assert control.in_flight("alice") == 2

    def test_the_next_one_is_refused(self) -> None:
        control = AdmissionControl(rate_limit=0, max_concurrent=2)
        control.acquire("alice")
        control.acquire("alice")

        with pytest.raises(TooManyRequests, match="in flight"):
            control.acquire("alice")

    def test_releasing_frees_a_slot(self) -> None:
        control = AdmissionControl(rate_limit=0, max_concurrent=1)
        control.acquire("alice")
        control.release("alice")

        control.acquire("alice")

    def test_releasing_a_slot_nobody_took_is_harmless(self) -> None:
        """The guard releases in `finally`, which can run for a call that was
        refused admission."""
        control = AdmissionControl(rate_limit=0, max_concurrent=1)

        control.release("alice")

        assert control.in_flight("alice") == 0

    def test_a_release_does_not_lend_a_slot_to_someone_else(self) -> None:
        control = AdmissionControl(rate_limit=0, max_concurrent=1)
        control.acquire("alice")

        control.release("bob")

        with pytest.raises(TooManyRequests):
            control.acquire("alice")


class TestTheTwoTogether:
    def test_a_refused_concurrency_slot_still_spends_the_rate_allowance(self) -> None:
        """Deliberate. A caller hammering a full pool is making requests, and not
        counting them would let them retry without limit."""
        control = AdmissionControl(rate_limit=5, max_concurrent=1)
        control.acquire("alice")

        for _ in range(4):
            with pytest.raises(TooManyRequests, match="in flight"):
                control.acquire("alice")

        with pytest.raises(TooManyRequests, match="per minute"):
            control.acquire("alice")

    def test_concurrent_callers_cannot_both_take_the_last_slot(self) -> None:
        """Both limits are checked and the slot taken under one lock."""
        control = AdmissionControl(rate_limit=0, max_concurrent=1)
        admitted: list[bool] = []
        barrier = threading.Barrier(8)

        def attempt() -> None:
            barrier.wait()
            try:
                control.acquire("alice")
                admitted.append(True)
            except TooManyRequests:
                admitted.append(False)

        threads = [threading.Thread(target=attempt) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert sum(admitted) == 1


class TestConfiguration:
    def test_the_defaults_apply_when_nothing_is_set(self, monkeypatch) -> None:
        monkeypatch.delenv("NLQ_MCP_RATE_LIMIT_PER_MINUTE", raising=False)
        monkeypatch.delenv("NLQ_MCP_MAX_CONCURRENT", raising=False)

        control = from_config()

        assert control.rate_limit == DEFAULT_RATE_LIMIT
        assert control.max_concurrent == DEFAULT_CONCURRENCY

    def test_the_limits_can_be_set(self, monkeypatch) -> None:
        monkeypatch.setenv("NLQ_MCP_RATE_LIMIT_PER_MINUTE", "10")
        monkeypatch.setenv("NLQ_MCP_MAX_CONCURRENT", "2")

        control = from_config()

        assert (control.rate_limit, control.max_concurrent) == (10, 2)

    @pytest.mark.parametrize("value", ["not-a-number", "-5"])
    def test_an_unusable_value_falls_back_to_the_default(self, monkeypatch, value: str) -> None:
        """Loudly, and to the default rather than to no limit: a typo in a
        number should not switch a control off."""
        monkeypatch.setenv("NLQ_MCP_RATE_LIMIT_PER_MINUTE", value)

        assert from_config().rate_limit == DEFAULT_RATE_LIMIT


class TestWindowsDoNotAccumulate:
    """`_windows` only ever gained keys: one per distinct subject for the life
    of the process. Small each, but a server in front of an identity provider
    accumulates one per user who has ever called."""

    def test_expired_windows_are_dropped_when_one_rolls_over(self, monkeypatch) -> None:
        control = AdmissionControl(rate_limit=5, max_concurrent=0)
        clock = [1000.0]
        monkeypatch.setattr(control, "_now", lambda: clock[0])

        for i in range(50):
            control.acquire(f"user-{i}")
        assert control.tracked_subjects() == 50

        clock[0] += 61.0
        control.acquire("user-0")

        assert control.tracked_subjects() == 1

    def test_a_window_still_inside_its_minute_is_kept(self, monkeypatch) -> None:
        """The sweep must not discard an allowance someone is still spending."""
        control = AdmissionControl(rate_limit=2, max_concurrent=0)
        clock = [1000.0]
        monkeypatch.setattr(control, "_now", lambda: clock[0])

        control.acquire("alice")
        control.acquire("alice")
        clock[0] += 30.0
        control.acquire("bob")

        with pytest.raises(TooManyRequests):
            control.acquire("alice")

    def test_the_sweep_does_not_reset_the_caller_who_triggered_it(self, monkeypatch) -> None:
        """Their new window starts at one call, not zero."""
        control = AdmissionControl(rate_limit=1, max_concurrent=0)
        clock = [1000.0]
        monkeypatch.setattr(control, "_now", lambda: clock[0])

        control.acquire("alice")
        clock[0] += 61.0
        control.acquire("alice")

        with pytest.raises(TooManyRequests):
            control.acquire("alice")
