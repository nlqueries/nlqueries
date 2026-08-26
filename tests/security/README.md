# The security corpus

Reproductions of the findings from the 2026-07-02 and 2026-08-25 reviews, run
against disposable containers. Both reviews built a lab like this and deleted it
afterwards; this one lives here so a finding's status is something you can run
rather than something you look up.

```bash
pytest -m security            # just these
pytest tests/security -q      # same thing, by path
```

Docker is required. Without it the whole directory skips, so a machine that
cannot run the lab reports that rather than reporting success.

## What the tests assert

**The absence of an effect, not the presence of an error.** The lab database has
a `lab.marker` table and a `lab.mark()` function that writes to it through a
plain `SELECT`. Tests read the marker back and compare the sequence before and
after. A control that returns a tidy refusal while the row still lands is not a
control, and the error string alone cannot tell you which happened.

`test_the_lab_can_actually_record_an_effect` checks the instrument itself. A
corpus whose marker silently did nothing would pass every assertion for the
wrong reason.

## The xfail contract

A finding that is still open is `xfail(strict=True)`, with a reason naming the
register row and the wave that closes it:

```python
@pytest.mark.xfail(strict=True, reason="SEC-16 — needs the SQL policy (W4)")
```

Not a skip. `strict=True` means the suite **fails when the test starts passing**,
which is what forces somebody to delete the marker and update
`internal-docs/security-findings-register.md`. An open finding cannot quietly
become closed, and a closed one cannot quietly reopen.

So when your change makes one of these pass:

1. delete its `xfail` marker (or its entry in `STILL_OPEN`),
2. update the register row to *Verified fixed* with the PR number,
3. and leave the test.

**This has already earned its keep.** `SELECT ... FOR UPDATE` was listed as open
on the first run and came back `XPASS`: PostgreSQL refuses a row lock in a
read-only transaction, because it needs a transaction id it will not assign.
That had been assumed rather than measured, and strict xfail turned a wrong line
in a register into a build failure.

## Adding a payload

Put it in `payloads.py`, not in a test. Payloads are data so the same corpus can
be driven through every route that reaches a database — fresh generation, cache
replay, retry, promotion, the direct connector API — as those routes gain gates,
instead of being rewritten once per route.

Each carries what it does (`effect`), the register row (`finding`), and the
layers expected to stop it (`stopped_by`). More than one layer is the point: a
payload with a single line of defence is a payload waiting for that line to have
a bug in it.

## Adding a safe query

`SAFE_POSTGRES` is the other half, and the more important one to keep growing.
Every control here can be made to pass by refusing more, and the failure mode of
a policy that refuses a `GROUP BY` is that somebody turns the policy off. If you
add a rule, add the analytics it must not break.

## Everything is synthetic

Loopback-only containers, made-up data, canary files written by the tests
themselves. Nothing here reads a real database, a real credential, or a file the
test did not create — so a failure means the engine did something it should not,
not that the machine happened to have something interesting on it.
