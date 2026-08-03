"""Tests for the machine-wide JVM concurrency budget.

The load-bearing test here is ``test_sigkilled_holder_releases_its_slot``: the
whole reason this uses ``flock`` rather than a PID file or a sentinel is that the
kernel releases the lock when the holder dies *however* it dies. This project
kills JVMs routinely (SIGALRM guards, slice timeouts, hang-recovery pkill), so a
budget that could go stale would wedge every later run.

No JVM and no OPSIN is involved -- these are pure lock-semantics tests.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import time

import pytest

from orthonym import jvm_budget as jb


@pytest.fixture
def slots(tmp_path, monkeypatch):
    """Isolate the budget in a temp dir so tests never touch the real one."""
    monkeypatch.setenv("ORTHONYM_JVM_SLOT_DIR", str(tmp_path / "slots"))
    monkeypatch.delenv("ORTHONYM_JVM_BUDGET", raising=False)
    monkeypatch.setenv("ORTHONYM_JVM_SLOTS", "4")
    return tmp_path / "slots"


# --------------------------------------------------------------- sizing

def test_total_slots_reads_env(monkeypatch):
    monkeypatch.setenv("ORTHONYM_JVM_SLOTS", "7")
    assert jb.total_slots() == 7


def test_total_slots_default_reserves_cpus(monkeypatch):
    monkeypatch.delenv("ORTHONYM_JVM_SLOTS", raising=False)
    expected = max(1, (os.cpu_count() or 4) - 4)
    assert jb.total_slots() == expected


@pytest.mark.parametrize("bad", ["0", "-3", "twelve", ""])
def test_total_slots_ignores_invalid_env(monkeypatch, bad):
    """An unparseable override must fall back, never crash a naming run."""
    monkeypatch.setenv("ORTHONYM_JVM_SLOTS", bad)
    assert jb.total_slots() == max(1, (os.cpu_count() or 4) - 4)


# --------------------------------------------------------------- acquisition

def test_acquires_and_reports_held(slots):
    with jb.jvm_slots(2, purpose="unit-a") as h:
        assert h.enforced is True
        st = jb.status()
        assert st["total"] == 4
        assert st["free"] == 2
        assert len(st["held"]) == 2
        assert all(x["purpose"] == "unit-a" for x in st["held"])
        assert all(x["pid"] == os.getpid() for x in st["held"])
    # released on exit
    assert jb.status()["free"] == 4


def test_slots_released_on_exception(slots):
    with pytest.raises(ValueError):
        with jb.jvm_slots(3, purpose="unit-boom"):
            raise ValueError("boom")
    assert jb.status()["free"] == 4


def test_over_budget_request_is_capped_not_refused(slots):
    """Asking for more than the machine has must still run the work."""
    with jb.jvm_slots(99, purpose="unit-greedy") as h:
        assert h.enforced is True
        assert jb.status()["free"] == 0


def test_timeout_proceeds_by_default(slots):
    """A caller must never LOSE work to the budget -- default is fail-open."""
    with jb.jvm_slots(4, purpose="unit-hog"):
        t0 = time.monotonic()
        with jb.jvm_slots(1, purpose="unit-late", timeout=0.5) as h2:
            assert h2.enforced is False  # proceeded without a reservation
        assert time.monotonic() - t0 >= 0.5


def test_timeout_can_raise_when_caller_prefers_refusing(slots):
    with jb.jvm_slots(4, purpose="unit-hog"):
        with pytest.raises(jb.BudgetTimeout):
            with jb.jvm_slots(1, purpose="unit-strict", timeout=0.3,
                              on_timeout="raise"):
                pass


def test_disabled_by_env_is_a_noop(slots, monkeypatch):
    monkeypatch.setenv("ORTHONYM_JVM_BUDGET", "off")
    assert jb.is_enabled() is False
    with jb.jvm_slots(4, purpose="unit-off") as h:
        assert h.enforced is False
        # nothing was taken, so a second full reservation also succeeds
        with jb.jvm_slots(4, purpose="unit-off-2") as h2:
            assert h2.enforced is False


def test_unusable_slot_dir_fails_open(monkeypatch, tmp_path):
    """The budget must never be able to fail a run. Point it at a file."""
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    monkeypatch.setenv("ORTHONYM_JVM_SLOT_DIR", str(blocker))
    monkeypatch.delenv("ORTHONYM_JVM_BUDGET", raising=False)
    with jb.jvm_slots(2, purpose="unit-nodir") as h:
        assert h.enforced is False  # proceeded anyway


# ------------------------------------------------- the load-bearing property

_HOLDER = textwrap.dedent(
    """
    import os, sys, time
    sys.path.insert(0, {src!r})
    os.environ["ORTHONYM_JVM_SLOT_DIR"] = {d!r}
    os.environ["ORTHONYM_JVM_SLOTS"] = "4"
    from orthonym.jvm_budget import jvm_slots
    with jvm_slots(4, purpose="unit-victim"):
        print("HELD", flush=True)
        time.sleep(300)
    """
)


def test_sigkilled_holder_releases_its_slot(slots, tmp_path):
    """SIGKILL a process holding the whole budget; the slots must free up.

    This is the property a PID file or a sentinel file does NOT have, and it is
    why `pgrep`/PID-based guards were the wrong primitive.
    """
    src = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(jb.__file__)))))
    code = _HOLDER.format(src=src, d=str(slots))
    proc = subprocess.Popen([sys.executable, "-c", code],
                            stdout=subprocess.PIPE, text=True)
    try:
        # Wait for it to actually hold the budget -- never assert on a race.
        line = proc.stdout.readline().strip()
        assert line == "HELD", f"holder did not start: {line!r}"
        assert jb.status()["free"] == 0, "holder should own every slot"

        proc.kill()          # SIGKILL -- no chance to clean up
        proc.wait(timeout=10)

        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if jb.status()["free"] == 4:
                break
            time.sleep(0.1)
        assert jb.status()["free"] == 4, (
            "SIGKILLed holder leaked its slots -- flock semantics violated")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)


def test_two_processes_serialise_on_the_budget(slots):
    """A peer needing the whole budget waits until the first releases it."""
    src = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(jb.__file__)))))
    code = _HOLDER.format(src=src, d=str(slots))
    proc = subprocess.Popen([sys.executable, "-c", code],
                            stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout.readline().strip() == "HELD"
        # We cannot get even one slot while it holds all four.
        with jb.jvm_slots(1, purpose="unit-peer", timeout=0.4) as h:
            assert h.enforced is False
        proc.kill()
        proc.wait(timeout=10)
        # Now we can.
        with jb.jvm_slots(1, purpose="unit-peer2", timeout=10) as h:
            assert h.enforced is True
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)


# ------------------------------------------------- contention with N > 2

_STRESS_WORKER = textwrap.dedent(
    """
    import json, os, sys, time
    sys.path.insert(0, {src!r})
    os.environ["ORTHONYM_JVM_SLOT_DIR"] = {d!r}
    os.environ["ORTHONYM_JVM_SLOTS"] = "3"
    os.environ["ORTHONYM_JVM_BUDGET"] = {budget!r}
    from orthonym.jvm_budget import jvm_slots
    want = int(sys.argv[1])
    with jvm_slots(want, purpose="stress", timeout=60) as h:
        t0 = time.monotonic(); time.sleep(0.15); t1 = time.monotonic()
        print(json.dumps({{"want": want, "start": t0, "end": t1}}))
    """
)


def _peak_overlap(slots_dir, budget_env: str) -> tuple[int, int]:
    """Run 12 contending workers over a 3-slot budget; return (peak_held, n_records).

    ``time.monotonic()`` is CLOCK_MONOTONIC on Linux, which is system-wide, so
    timestamps from separate processes are directly comparable.
    """
    src = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(jb.__file__))))
    code = _STRESS_WORKER.format(src=src, d=str(slots_dir), budget=budget_env)
    widths = [1] * 8 + [2] * 3 + [3]
    procs = [subprocess.Popen([sys.executable, "-c", code, str(w)],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True) for w in widths]
    recs = []
    for p in procs:
        out, _ = p.communicate(timeout=120)
        line = out.strip().split("\n")[-1] if out.strip() else ""
        if line:
            recs.append(json.loads(line))
    events = []
    for r in recs:
        events.append((r["start"], r["want"]))
        events.append((r["end"], -r["want"]))
    events.sort()
    cur = peak = 0
    for _, d in events:
        cur += d
        peak = max(peak, cur)
    return peak, len(recs)


def test_many_processes_never_oversell_the_budget(slots):
    """12 contending workers over 3 slots: all acquire, and never more than 3 at once.

    Covers what 2 processes cannot: that the acquire loop's release-on-partial
    does not livelock, and that a worker wanting the FULL budget is not starved.
    """
    peak, n = _peak_overlap(slots, "on")
    assert n == 12, f"only {n}/12 workers produced a record -- livelock or crash"
    assert peak <= 3, f"OVERSOLD: {peak} slots held at once, budget is 3"


def test_the_oversell_check_can_actually_fail(slots):
    """Negative control: with the budget disabled the same probe MUST oversell.

    Without this, `peak <= 3` above could pass because the overlap arithmetic is
    broken rather than because the semaphore works. Measured: 33 with the budget
    off vs exactly 6 with it on, at a 6-slot budget.
    """
    peak, n = _peak_overlap(slots, "off")
    assert n == 12
    assert peak > 3, ("budget disabled yet peak <= 3 -- the overlap measurement "
                      "cannot detect oversell, so the positive test proves nothing")


def test_status_records_purpose_and_age(slots):
    with jb.jvm_slots(1, purpose="v22_gate"):
        h = jb.status()["held"]
        assert len(h) == 1
        assert h[0]["purpose"] == "v22_gate"
        assert h[0]["age_s"] >= 0
        raw = json.loads((slots / "slot-000").read_text())
        assert raw["purpose"] == "v22_gate"
        assert raw["pid"] == os.getpid()
