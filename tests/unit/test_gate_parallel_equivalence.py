"""Permanent regression guard: the PARALLEL gate verdict == the SERIAL gate verdict.

The phase gate is the project's correctness oracle. Parallelizing it (``--jobs N``)
must be a DISTRIBUTION change only: fanning per-row naming (``run_conformance``) and
per-probe naming (``run_determinism``) out over an ``mp.Pool`` must NOT change the verdict.
A parallel run that "passes" but silently skips/reorders rows is exactly the false-PASS
failure mode this guard exists to catch (feedback_harness_that_reports_success), so it
asserts, for jobs=1 vs jobs=4 over a fixed fixture:

  * same matched / protect_regressions / target_passes / target_total, AND
  * the SAME SET of non-MATCH rows (def_id / smiles), AND
  * full per-row record equality and per-pack summary equality (the strongest form),

and the same for determinism (same new_nondeterministic set, identical records).

WHY PROCESSES, NOT THREADS: ``run_with_timeout`` uses SIGALRM, which only fires on a
process's MAIN thread; each ``mp.Pool`` worker is its own process (own main thread) so the
per-row hang-timeout keeps working. A ThreadPool would break it (the [99Tc]-hang lesson).

WHY PARALLEL RUNS FIRST HERE: an ``mp.Pool`` on Linux forks. Forking a process that
already holds a live JPype JVM corrupts OPSIN in the children — a CALLER invariant the real
gate honours by never instantiating a namer in the parent before its pool (verified
separately by a fresh-process serial-vs-parallel run; see the task report). This in-process
test would otherwise trip that invariant by warming the parent JVM in the serial pass and
then forking. Running the parallel config FIRST (while the parent is still JVM-free), then
the serial config (which forks nothing), keeps the fork clean and isolates what this test is
actually about: the distribute→map→regroup→aggregate logic. Run this file on its own (the
project's standard targeted-file pattern) so no earlier test has warmed the parent JVM.

The gate's naming config is pinned OFF via the environment so the parent and every worker
name identically regardless of the multiprocessing start method (equivalence holds whatever
the gate state is, as long as all processes agree on it).
"""
from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path

import pytest

# The pools use the process default start method. The REAL gate is a plain, single-threaded
# python process, so its default `fork` forks cleanly. The pytest host, however, is
# multi-threaded (plugins/fixtures), and forking a multi-threaded process can DEADLOCK
# (Python even warns about it). Force `forkserver` for THIS test process only: workers are
# forked from a clean single-threaded server, so the hang cannot occur. This does not change
# the verdict — naming is independent of the multiprocessing start method (proven: fresh-process
# serial vs parallel runs, which have different PYTHONHASHSEEDs, are byte-identical), and env
# vars (the pinned gate config below) are inherited by forkserver workers just as by fork.
try:  # pragma: no cover - environment-dependent
    mp.set_start_method("forkserver", force=True)
except (RuntimeError, ValueError):
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pin_conformance_eval import run_conformance  # noqa: E402
from determinism_eval import run_determinism  # noqa: E402

# A fixed fixture of simple molecules across TWO packs, mixing categories and a deliberately
# wrong expectation, so the run exercises MATCH, MISMATCH, protect and target branches and
# the multi-pack regroup — without depending on any gold file.
_FIXTURE_PACKS = [
    ("equiv_pack_a", [
        {"smiles": "CCO", "expected_pin": "ethanol", "def_id": "EQ-A1", "category": "target"},
        {"smiles": "CC(=O)O", "expected_pin": "acetic acid", "def_id": "EQ-A2", "category": "protect"},
        {"smiles": "CCCO", "expected_pin": "propan-1-ol", "def_id": "EQ-A3", "category": "target"},
    ]),
    ("equiv_pack_b", [
        {"smiles": "CO", "expected_pin": "methanol", "def_id": "EQ-B1", "category": "target"},
        {"smiles": "C1CCCCC1", "expected_pin": "cyclohexane", "def_id": "EQ-B2", "category": "target"},
        # Deliberately wrong expectation -> guarantees a non-MATCH row (exercises the
        # non-MATCH def_id-set path) whatever the exact emitted name is.
        {"smiles": "CCC", "expected_pin": "not-the-name-of-propane", "def_id": "EQ-B3", "category": "target"},
    ]),
]

_DET_PROBES = [
    {"smiles": s, "def_id": "", "source": "test"}
    for s in ["CCO", "CC(=O)O", "c1ccccc1", "CCN", "COC", "CC(C)O"]
]


@pytest.fixture(autouse=True)
def _pin_gate_off(monkeypatch):
    # Deterministic naming config in EVERY process (parent + pool workers), so a
    # serial/parallel difference can only be a distribution bug, never a config skew.
    monkeypatch.setenv("ORTHONYM_DISABLE_OPSIN_VALIDITY_GATE", "1")


def _conf_verdict(report: dict) -> dict:
    agg = report["aggregate"]
    return {
        "total": agg["total"],
        "matched": agg["matched"],
        "protect_regressions": agg["protect_regressions"],
        "target_passes": agg["target_passes"],
        "target_total": agg["target_total"],
        "nonmatch_rows": sorted(
            (r["pack"], r["smiles"], r["def_id"], r["verdict"])
            for r in report["records"] if r["verdict"] != "MATCH"
        ),
        "nonmatch_def_ids": sorted(
            {r["def_id"] for r in report["records"] if r["verdict"] != "MATCH"}
        ),
    }


def test_conformance_parallel_equals_serial():
    # Parallel FIRST (parent JVM-free at fork), then serial (forks nothing). See module docstring.
    parallel = run_conformance(_FIXTURE_PACKS, timeout=30.0, classify_rt=False, jobs=4)
    serial = run_conformance(_FIXTURE_PACKS, timeout=30.0, classify_rt=False, jobs=1)

    assert _conf_verdict(serial) == _conf_verdict(parallel), (
        "PARALLEL conformance verdict differs from SERIAL — a distribution bug. "
        "Do NOT ship the parallel gate until this is byte-identical."
    )
    # Strongest forms: full record list and per-pack summaries identical.
    assert serial["records"] == parallel["records"]
    assert serial["packs"] == parallel["packs"]
    # Sanity: the fixture actually exercised both branches (not all-abstain / all-error).
    v = _conf_verdict(serial)
    assert v["matched"] >= 1, "fixture produced no MATCH — naming did not run"
    assert len(v["nonmatch_def_ids"]) >= 1, "fixture produced no non-MATCH row"


def test_determinism_parallel_equals_serial():
    parallel = run_determinism(_DET_PROBES, n_rand=2, seed=0, timeout=30.0, quarantine=[], jobs=4)
    serial = run_determinism(_DET_PROBES, n_rand=2, seed=0, timeout=30.0, quarantine=[], jobs=1)

    assert serial["total_probes"] == parallel["total_probes"]
    assert serial["deterministic"] == parallel["deterministic"]
    assert serial["nondeterministic"] == parallel["nondeterministic"]
    # SAME new_nondeterministic set (the determinism gate's FAIL set).
    assert (
        {r["smiles"] for r in serial["new_nondeterministic"]}
        == {r["smiles"] for r in parallel["new_nondeterministic"]}
    )
    # Strongest form: identical per-probe records.
    assert serial["records"] == parallel["records"]
