"""The measurement tools' wall-clock limit (``orthonym.wallclock``).

A tool's per-molecule alarm used to raise an ``Exception`` subclass. The naming
path holds hundreds of ``except Exception`` handlers, so an alarm that fired inside
one of them was caught there and naming carried on from a half-finished choice: a
36-ring-atom molecule that abstains in every unloaded run was named in a loaded
one, because ``VonBaeyerAnalyzer._choose_lowest_locant_numbering`` kept its
incumbent numbering when its candidate search was cut off. ``WallClockTimeout`` is a
``BaseException``: it leaves ``name_tiered`` and reaches the tool, which writes its
TIMEOUT row.
"""
from __future__ import annotations

import ast
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from orthonym.wallclock import (
    NAMING_DEPTHS,
    NAMING_REQUEST_CTX,
    NAMING_TIER_CTX,
    REFIRE_INTERVAL_S,
    WallClockTimeout,
    clear_leaked_naming_state,
    leaked_naming_state,
    wall_clock_limit,
)

REPO = Path(__file__).resolve().parents[2]

# A substituted tricyclic cage whose von Baeyer analysis runs the lowest-locant
# candidate search inside ``_choose_lowest_locant_numbering``'s ``try`` (traced: the
# search with ``return_candidates=True`` is called for it at the PIN tier, as for
# 'CC1C2CCC3C1CC3C2'; it is not called for the bicyclic cages, which take another
# path).
_VB_CAGE = "CC1CC2CC3CC1C3C2"


def _timer_disarmed() -> bool:
    return signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


# --------------------------------------------------------------------- the class

def test_timeout_is_a_base_exception_not_an_exception():
    assert issubclass(WallClockTimeout, BaseException)
    assert not issubclass(WallClockTimeout, Exception)


# ------------------------------------------------------------- the context manager

def test_limit_raises_on_time_and_disarms():
    before = signal.getsignal(signal.SIGALRM)
    t0 = time.monotonic()
    with pytest.raises(WallClockTimeout) as info:
        with wall_clock_limit(0.2):
            time.sleep(5)
    assert time.monotonic() - t0 < 2.0
    assert info.value.seconds == 0.2
    assert _timer_disarmed()
    assert signal.getsignal(signal.SIGALRM) is before


def test_limit_passes_through_except_exception():
    """The defect class: a broad ``except Exception`` between the alarm and the tool
    must not catch the timeout."""
    caught_by_engine_style_handler = []
    with pytest.raises(WallClockTimeout):
        with wall_clock_limit(0.1):
            try:
                time.sleep(5)
            except Exception:  # noqa: BLE001 - the engine's handler shape
                caught_by_engine_style_handler.append(True)
    assert caught_by_engine_style_handler == []
    assert _timer_disarmed()


def test_no_timeout_leaves_no_timer_and_restores_the_handler():
    before = signal.getsignal(signal.SIGALRM)
    with wall_clock_limit(5):
        x = sum(range(1000))
    assert x == 499500
    assert _timer_disarmed()
    assert signal.getsignal(signal.SIGALRM) is before


def test_other_exceptions_disarm_too():
    with pytest.raises(ZeroDivisionError):
        with wall_clock_limit(5):
            1 / 0  # noqa: B018
    assert _timer_disarmed()


def test_limits_nest_and_the_outer_deadline_holds():
    t0 = time.monotonic()
    with pytest.raises(WallClockTimeout) as info:
        with wall_clock_limit(0.4):
            with wall_clock_limit(10):
                pass
            time.sleep(5)
    assert info.value.seconds == 0.4
    assert time.monotonic() - t0 < 2.0
    assert _timer_disarmed()


def test_no_limit_values_run_unguarded():
    for seconds in (None, 0, -1):
        with wall_clock_limit(seconds):
            pass
    assert _timer_disarmed()


def test_a_dropped_timeout_still_ends_the_with_as_a_timeout():
    """The first timeout can be dropped on its way out (an alarm inside a garbage
    collector callback is printed as "Exception ignored"); a limit that fired is a
    timeout whatever the body did next."""
    with pytest.raises(WallClockTimeout):
        with wall_clock_limit(0.05):
            try:
                time.sleep(2)
            except BaseException:  # noqa: BLE001 - stands in for the dropped raise
                pass
    assert _timer_disarmed()


def test_a_dropped_timeout_is_raised_again_while_the_body_runs():
    t0 = time.monotonic()
    dropped = []
    with pytest.raises(WallClockTimeout):
        with wall_clock_limit(0.05):
            try:
                time.sleep(2)
            except BaseException:  # noqa: BLE001
                dropped.append(time.monotonic() - t0)
            time.sleep(10)
    assert len(dropped) == 1
    assert time.monotonic() - t0 < 0.05 + REFIRE_INTERVAL_S + 1.0
    assert _timer_disarmed()


def test_a_timeout_converted_to_another_exception_still_reads_as_a_timeout():
    with pytest.raises(WallClockTimeout) as info:
        with wall_clock_limit(0.05):
            try:
                time.sleep(2)
            except BaseException as exc:  # noqa: BLE001
                raise RuntimeError("converted") from exc
    assert isinstance(info.value.__cause__, RuntimeError)


def test_off_the_main_thread_the_body_runs_unguarded():
    out = []

    def body():
        with wall_clock_limit(0.05):
            time.sleep(0.2)
            out.append("done")

    t = threading.Thread(target=body)
    t.start()
    t.join(5)
    assert out == ["done"]


# ---------------------------------------------------- the engine does not absorb it

def _cut_off_candidate_search(monkeypatch, exc_type):
    """Make the von Baeyer lowest-locant candidate search raise ``exc_type``, the way
    an alarm landing in it would; return the list that records each firing."""
    from orthonym.rules import polycyclic

    original = polycyclic.VonBaeyerAnalyzer._find_main_ring
    fired = []

    def _cut_off(self, *args, **kwargs):
        if kwargs.get("return_candidates"):
            fired.append(exc_type.__name__)
            raise exc_type()
        return original(self, *args, **kwargs)

    monkeypatch.setattr(polycyclic.VonBaeyerAnalyzer, "_find_main_ring", _cut_off)
    return fired


def test_an_exception_timeout_is_absorbed_inside_the_engine(monkeypatch):
    """Control: the old alarm class, ``TimeoutError``, raised in the candidate search,
    is caught by the engine and naming completes -- the defect this module fixes."""
    from orthonym import Orthonym
    from orthonym.errors import is_failure_name

    fired = _cut_off_candidate_search(monkeypatch, TimeoutError)
    res = Orthonym(style="pin").name_tiered(_VB_CAGE)
    assert fired, "the candidate search was not reached; the control is vacuous"
    assert res.get("name") and not is_failure_name(res["name"])


def test_wall_clock_timeout_propagates_out_of_name_tiered(monkeypatch):
    from orthonym import Orthonym

    fired = _cut_off_candidate_search(monkeypatch, WallClockTimeout)
    with pytest.raises(WallClockTimeout):
        Orthonym(style="pin").name_tiered(_VB_CAGE)
    assert fired == ["WallClockTimeout"]
    assert leaked_naming_state() == {}


def test_wall_clock_timeout_propagates_at_the_best_effort_tier(monkeypatch):
    from orthonym import Orthonym
    from orthonym.cli import _emit_tier_flags

    fired = _cut_off_candidate_search(monkeypatch, WallClockTimeout)
    with pytest.raises(WallClockTimeout):
        Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(_VB_CAGE)
    assert fired
    assert leaked_naming_state() == {}


# ---------------------------------------------- the state after a real alarm unwinds

def _fresh_process_name(smiles: str) -> str:
    code = ("import sys, json\n"
            "from orthonym import Orthonym\n"
            "print(json.dumps(Orthonym(style='pin').name_tiered(sys.argv[1]).get('name')))\n")
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    out = subprocess.run([sys.executable, "-c", code, smiles], capture_output=True,
                         text=True, timeout=300, env=env, cwd=str(REPO))
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_real_alarms_unwind_clean_and_the_next_naming_is_the_fresh_one():
    """Real SIGALRMs landing at different points of a naming: after each, the known
    process-global naming state is at its value outside a naming call, and the same
    process then names the molecule as a fresh process does."""
    from orthonym import Orthonym

    fresh = _fresh_process_name(_VB_CAGE)
    # Cold, the early alarms land in the lazy imports and the JVM start; warm (a
    # naming of ~20 ms on an idle machine), they land in the naming itself (seen:
    # namer.name, multiplicative, amino_acids, partial_saturation, polycyclic,
    # the InChIKey step, and once name's own finally block, which left
    # session_depth at 1 before the limit put the state back).
    limits = (0.001, 0.01, 0.1, 0.5)
    limits += (None,)  # warm the process
    limits += (0.001, 0.002, 0.003, 0.005, 0.008, 0.012, 0.018, 0.03)
    timeouts = 0
    for limit in limits:
        try:
            with wall_clock_limit(limit):
                Orthonym(style="pin").name_tiered(_VB_CAGE)
        except WallClockTimeout:
            timeouts += 1
            assert leaked_naming_state() == {}, f"state left behind at {limit} s"
        assert _timer_disarmed()
    assert timeouts >= 1, "no alarm landed inside a naming; the test is vacuous"
    assert Orthonym(style="pin").name_tiered(_VB_CAGE).get("name") == fresh


def _plant_left_behind_state():
    """The state an alarm between a ``set`` and its ``try``, or inside a ``finally``,
    leaves behind."""
    import orthonym.namer  # noqa: F401 - loads the modules the check reads
    from orthonym.assembly import fragment_naming, memo
    from orthonym.metrics import provenance

    provenance.best_effort_ctx.set(True)
    provenance.best_effort_request_ctx.set(False)
    fragment_naming._fragment_guard.name_call_depth = 1
    fragment_naming._fragment_guard.session_depth = 1
    memo._cache_var.set({("k", 1): "v"})
    orthonym.namer._LP_INPUT.open = True


_PLANTED = {"best_effort_ctx", "best_effort_request_ctx", "name_call_depth",
            "session_depth", "_cache_var", "_LP_INPUT.open"}


def test_the_limit_puts_back_the_state_the_interrupted_call_changed():
    import orthonym.namer  # noqa: F401

    assert leaked_naming_state() == {}
    with pytest.raises(WallClockTimeout) as info:
        with wall_clock_limit(0.05):
            _plant_left_behind_state()
            time.sleep(2)
    assert leaked_naming_state() == {}
    assert {"orthonym.metrics.provenance.best_effort_ctx",
            "orthonym.metrics.provenance.best_effort_request_ctx",
            "orthonym.assembly.memo._cache_var",
            "orthonym.assembly.fragment_naming._fragment_guard",
            "orthonym.namer._LP_INPUT"} <= set(info.value.restored)


def test_clear_leaked_naming_state_resets_what_it_reports():
    """Known positive for the check the tools run after a timeout."""
    assert leaked_naming_state() == {}
    _plant_left_behind_state()
    leaked = clear_leaked_naming_state()
    assert set(leaked) == _PLANTED
    assert leaked_naming_state() == {}
    assert clear_leaked_naming_state() == {}


@pytest.mark.parametrize("tier", ["pin", "best-effort"])
def test_a_finished_naming_leaves_the_known_state_idle(tier):
    """No false report: every value the check reads is at its idle value after a
    naming that finished (a peptide, an isotope label, a salt, a cage, an ester)."""
    from orthonym import Orthonym
    from orthonym.cli import _emit_tier_flags

    namer = (Orthonym(style="pin") if tier == "pin"
             else Orthonym(style="pin", **_emit_tier_flags(tier)))
    for smiles in ("CC(N)C(=O)NCC(=O)O", "[2H]C([2H])([2H])O", "[Na+].CC(=O)[O-]",
                   _VB_CAGE, "CCOC(=O)c1ccccc1", "O=C1CCCN1"):
        namer.name_tiered(smiles)
        assert leaked_naming_state() == {}, smiles


def test_the_known_state_includes_the_test_suite_guard_set():
    """The test suite's isolation guard (tests/conftest.py) and the tools' check read
    the same ContextVars and counters."""
    tree = ast.parse((REPO / "tests" / "conftest.py").read_text())
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and \
                isinstance(node.targets[0], ast.Name) and \
                node.targets[0].id in ("_NAMING_TIER_CTX", "_NAMING_REQUEST_CTX",
                                       "_NAMING_DEPTHS"):
            found[node.targets[0].id] = ast.literal_eval(node.value)
    assert found == {"_NAMING_TIER_CTX": NAMING_TIER_CTX,
                     "_NAMING_REQUEST_CTX": NAMING_REQUEST_CTX,
                     "_NAMING_DEPTHS": NAMING_DEPTHS}


# ------------------------------------------- the persistent OPSIN pipe after an interrupt

class _InterruptFirstRead:
    """A reader queue whose first ``get`` is interrupted by a tool's alarm."""

    def __init__(self, q):
        self.q = q
        self.fired = False

    def get(self, timeout=None):
        if not self.fired:
            self.fired = True
            raise WallClockTimeout(0.1)
        return self.q.get(timeout=timeout)


def test_persistent_opsin_does_not_hand_a_late_answer_to_the_next_name(monkeypatch):
    """An interrupt between writing a name and reading its answer leaves the answer in
    the pipe; the next call must not read it as its own."""
    from orthonym.validation import opsin_server

    real_popen = subprocess.Popen

    def _echo_popen(argv, **kwargs):  # one line out per line in, like the OPSIN CLI
        return real_popen(["cat"], **kwargs)

    monkeypatch.setattr(opsin_server.subprocess, "Popen", _echo_popen)
    srv = opsin_server.PersistentOpsin("unused.jar", read_timeout=5.0)
    try:
        assert srv.invoke("first") == ("first", True)
        srv._q = _InterruptFirstRead(srv._q)
        with pytest.raises(WallClockTimeout):
            srv.invoke("second")
        assert srv.invoke("third") == ("third", True)
    finally:
        srv.close()
