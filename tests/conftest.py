"""
Pytest configuration and shared fixtures for Orthonym.
"""

import pytest
import os
import warnings
from pathlib import Path
from typing import Dict

from rdkit import Chem

# Shared jar helpers (the jars live outside the repo; see orthonym.jars).
# Re-exported here; tests import them from ``tests.support.jars``.
from tests.support.jars import REQUIRE_JARS, jar_or_none, jar_or_skip, jar_unavailable  # noqa: F401
# Fixtures every test can request: the natural-product and nucleoside producers on their own.
from tests.support.np_producer import np_producer_only  # noqa: F401
from tests.support.nucleoside_producer import nucleoside_producer_only  # noqa: F401

# Paths
PROJECT_ROOT = Path(__file__).parent.parent


# ============================================================================
# Local-only files (docs/, a temp dir/, untracked.planning/) -- see
# tests/support/local_only.py.
# ============================================================================
# A test that reads a file the published tree does not ship carries
# `@local_only("<repo-relative path>")`: it runs where the file exists and is
# skipped, naming the missing path, where it does not. This replaced a
# module-level hook that skipped whole modules when `.planning/` was absent -- a
# proxy that stopped firing once part of `.planning/` became tracked, so the
# readers raised FileNotFoundError in every clean checkout (TRIAGE g7 C02), and
# that skipped the tests of a module that needed no file at all.


# ============================================================================
# Molecule Fixtures
# ============================================================================

@pytest.fixture
def mol_from_smiles():
    """Factory fixture to create RDKit molecules from SMILES."""
    from tests.support.read_back import mol
    return mol


@pytest.fixture
def canonical():
    """Factory fixture to get canonical SMILES."""
    def _canonicalize(smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, f"Invalid SMILES: {smiles}"
        return Chem.MolToSmiles(mol, canonical=True)
    return _canonicalize


@pytest.fixture(scope="module")
def namer():
    """The default engine (``Orthonym``, the PIN tier), one per test module. A module that
    needs another tier or other flags defines its own ``namer``."""
    from orthonym import Orthonym
    return Orthonym()


# ============================================================================
# OPSIN Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def opsin_jar() -> str:
    """Path to the OPSIN CLI jar, or None when it cannot be found.

    Delegates to ``orthonym.validation.opsin_roundtrip._find_opsin_jar`` — the
    SAME resolver ``opsin_parse`` uses internally — so "the fixture found a jar"
    and "the parse helper can use a jar" cannot disagree.

    A hand-rolled ``glob`` over five patterns used to stand here. It could
    resolve a jar that ``_find_opsin_jar`` would NOT return, which would have
    made ``opsin_to_smiles`` skip-free but silently None-returning — the exact
    blind state this file exists to prevent.

    ``_find_opsin_jar`` now delegates to ``orthonym.jars.find_jar`` and RAISES
    ``JarUnavailable`` when the pinned jar is missing; that maps to None here,
    except under ``ORTHONYM_REQUIRE_JARS=1``, where it fails the test.
    """
    try:
        from orthonym.validation.opsin_roundtrip import _find_opsin_jar
        jar = _find_opsin_jar()
    except Exception as exc:
        if REQUIRE_JARS:
            pytest.fail(f"ORTHONYM_REQUIRE_JARS=1 but the OPSIN jar is unavailable: {exc}")
        return None
    if jar is None and REQUIRE_JARS:
        pytest.fail("ORTHONYM_REQUIRE_JARS=1 but the OPSIN jar is unavailable (reduced mode)")
    return jar


@pytest.fixture
def opsin_available(opsin_jar) -> bool:
    """Check if OPSIN is available for round-trip testing."""
    from tests.support.jars import java_on_path
    return opsin_jar is not None and java_on_path()


@pytest.fixture(scope="session")
def _opsin_parse_canary(opsin_jar) -> bool:
    """Prove the OPSIN parse path actually parses before any test trusts it.

    ⚠ HISTORY — this is why the canary exists. ``opsin_to_smiles`` used to shell
    out as ``java -jar <jar> -osmi <name>``. OPSIN's CLI reads a TRAILING
    argument as an INPUT FILE, so every single call died with::

        java.io.FileNotFoundException: butan-2-ol (No such file or directory)
        exit 1

    The fixture's ``if result.returncode == 0`` therefore never held and it
    returned ``None`` for EVERY name ever passed to it. Sixteen tests guarded on
    ``if parsed:`` and so passed vacuously for as long as the fixture existed;
    the seventeenth failed and was papered over with an xfail whose recorded
    reason named an unrelated cause.

    A None return is indistinguishable from "OPSIN legitimately could not
    interpret this name", so the breakage can only be caught by probing a name
    OPSIN is known to parse. That is this fixture. It ASSERTS rather than skips:
    a present-but-unusable jar must be loud.
    """
    if opsin_jar is None:
        return False

    from orthonym.validation.opsin_roundtrip import opsin_parse

    probe = opsin_parse("butan-2-ol")
    assert probe is not None, (
        "OPSIN canary failed: opsin_parse('butan-2-ol') returned None even "
        f"though a jar was found at {opsin_jar}. Every round-trip test "
        "depending on this fixture would silently validate nothing. Fix the "
        "OPSIN invocation before trusting any round-trip result."
    )
    assert Chem.CanonSmiles(probe) == Chem.CanonSmiles("CCC(C)O"), (
        "OPSIN canary parsed 'butan-2-ol' to the WRONG structure "
        f"({probe!r}); the parse path is returning something other than the "
        "SMILES for that name."
    )
    return True


@pytest.fixture
def opsin_to_smiles(opsin_available, _opsin_parse_canary):
    """Convert an IUPAC name to SMILES with OPSIN, or None if OPSIN cannot
    interpret it.

    Thin alias for ``orthonym.validation.opsin_roundtrip.opsin_parse`` — the
    maintained primitive. It feeds the name on **stdin** (the CLI's only correct
    input channel for a name) and prefers the in-process JPype JVM, so it is
    both correct and ~100x cheaper than a per-name process launch. Do not
    reintroduce a third hand-rolled subprocess call here.
    """
    if not opsin_available:
        pytest.skip("OPSIN not available")

    from orthonym.validation.opsin_roundtrip import opsin_parse

    return opsin_parse


# ============================================================================
# A test MODULE must not change the ORTHONYM_* environment of the run.
# ============================================================================
# Under xdist every worker imports every test module at collection, so a
# module-level `os.environ[...] =...` stays in that worker's environment and is
# inherited by EVERY child process the rest of the suite spawns there. The
# whole-suite run of 2026-09-26 lost 11 subprocess tests and 14 more this way
# (TRIAGE g8 C1, g7 C01): two modules set ORTHONYM_DISABLE_OPSIN_VALIDITY_GATE=1
# and ORTHONYM_SELF_CONSISTENCY_GATE=off at import, and the children then named
# with both gates off ('benzene' for a quinoxaline-quinoline, labelled
# pin_verified). Fixed at the source in 02310cf0e; this check keeps the class
# from coming back: after collection, any ORTHONYM_* variable a module changed is
# put back and reported as a warning. Set per test with monkeypatch.setenv, or in
# the child's env= only.
_ORTHONYM_ENV_BEFORE_COLLECTION: Dict[str, str] = {}


def _orthonym_env() -> Dict[str, str]:
    return {k: v for k, v in os.environ.items() if k.startswith("ORTHONYM_")}


def pytest_collection_finish(session):
    """Undo (and report) any ORTHONYM_* change made while importing test modules."""
    before = dict(_ORTHONYM_ENV_BEFORE_COLLECTION)
    now = _orthonym_env()
    changed = sorted(k for k in set(before) | set(now) if before.get(k) != now.get(k))
    if not changed:
        return
    for key in changed:
        if key in before:
            os.environ[key] = before[key]
        else:
            os.environ.pop(key, None)
    warnings.warn(pytest.PytestWarning(
        "a test module changed the environment at import and it was put back "
        f"({', '.join(f'{k}={now.get(k)!r}' for k in changed)}): every child process "
        "the suite spawns would have inherited it (TRIAGE g8 C1). Use "
        "monkeypatch.setenv in the test, or pass env= to the child."))


def _keep_hotspots_sigsegv_handler(config) -> None:
    """Take pytest's faulthandler plugin out of the session once it has configured itself.

    The plugin calls ``faulthandler.enable`` at configure, before the first test starts the
    in-process JVM (HotSpot then chains to faulthandler's handler), and ``faulthandler.disable``
    at unconfigure, while the JVM still runs. ``disable`` restores the disposition saved at
    ``enable``, SIG_DFL, over HotSpot's SIGSEGV handler. The JVM raises SIGSEGV in normal
    operation (implicit null checks, safepoint polls), so a fault during worker teardown killed
    the worker with signal 11, unseen by xdist because no test was running (audit 2026-10-13,
    internal notes section 1; pinned by
    tests/unit/diagnostics/test_pytest_faulthandler_jvm.py).

    Unregistering the plugin here, from the conftest every public runner loads (CI passes
    ``-o addopts=""``, which drops anything set in pyproject.toml), removes its unconfigure half
    and keeps the rest: faulthandler stays on for the whole session, so a native crash still
    leaves a Python traceback. The plugin's ``faulthandler_timeout`` option stops working with it;
    the suite's time limit is pytest-timeout's thread method. ``-p no:faulthandler`` leaves
    nothing to unregister.
    """
    config.pluginmanager.unregister(name="faulthandler")


def _worktree_import_guard(root=None):
    """Stop the run when this tree is a git worktree (``.git`` is a file) and ``orthonym`` imports
    from elsewhere. The shared ``.venv`` is an editable install of the MAIN tree, so a bare
    ``pytest`` in a worktree tests main's code (the contributor guide a project rule(c)); the repair is
    ``PYTHONPATH=<tree>/src``. An orthonym installed into site-packages also counts as elsewhere.
    Silent in the main tree, in CI and in a published copy (no ``.git`` file there)."""
    root = Path(root) if root is not None else Path(__file__).resolve().parent.parent
    if not (root / ".git").is_file():
        return
    import orthonym
    where = Path(orthonym.__file__).resolve()
    if not where.is_relative_to((root / "src").resolve()):
        pytest.exit(f"orthonym imports from {where.parent}, not from this worktree's "
                    f"{root / 'src'}: run with PYTHONPATH={root / 'src'}", returncode=4)


# trylast: after the faulthandler plugin's own pytest_configure, which this undoes in part.
@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    """Register the markers pyproject.toml does not; snapshot the ORTHONYM_* environment."""
    _worktree_import_guard()
    _keep_hotspots_sigsegv_handler(config)
    _ORTHONYM_ENV_BEFORE_COLLECTION.clear()
    _ORTHONYM_ENV_BEFORE_COLLECTION.update(_orthonym_env())
    config.addinivalue_line(
        "markers",
        "real_opsin_cli: run this test with the real `java -jar opsin` command line "
        "(the PR run answers the test code's OPSIN command-line calls in process, "
        "tests/support/opsin_cli_inproc.py); for a test that compares something with it"
    )
    config.addinivalue_line(
        "markers",
        "opsin_gate: run this test with the OPSIN validity gate ENABLED "
        "(it is disabled suite-wide by default). Skips if the OPSIN jar is "
        "absent, because the gate fails OPEN without it."
    )


# ============================================================================
# The OPSIN validity gate in tests — declarative, and never silently blind.
# ============================================================================
# The gate is default-ON in production: suppressing a name OPSIN cannot parse,
# or that OPSIN parses to a DIFFERENT molecule , is the whole point of
# it. The suite disables it by default because most tests assert RAW generator
# output — some deliberately malformed — and must not pay a per-name OPSIN call.
#
# That default is a TRAP, and it has been sprung. A test written *about* gate
# behaviour is green-but-blind unless it re-enables the gate, and nothing says
# so. Measured 2026-07-31 on the canary `CC(=O)N(CC1CO1)C(C)C`:
#
# gate OFF -> '(5-carbamoylpentyl)oxirane' <- a DIFFERENT molecule
# gate ON -> 'unknown organic compound' <- suppressed it
#
# so "assert this molecule abstains" passes for the wrong reason with the gate
# off, and would keep passing if the gate were deleted outright.
#
# There is now exactly ONE supported way to ask for the gate:
#
# pytestmark = pytest.mark.opsin_gate # module-wide, or
# @pytest.mark.opsin_gate # per-test, or
# def test_x(opsin_gate):... # fixture form
#
# and it is VERIFIED rather than merely requested. Two silent-failure modes are
# closed:
#
# 1. A rename of the flag. The old `raising=False` meant a rename would turn
# every re-enable in the suite into a no-op at once, silently. The setattr
# below raises.
# 2. A missing OPSIN jar. `_final_opsin_validity_gate` fails OPEN when the jar
# is absent , so "gate on, no jar" is the same blind state by another
# route — and a worktree checkout has no jar (the `opsin` gitlink has no
#.gitmodules to fetch from). `pytest_runtest_call` below skips those tests
# instead of passing them.
#
# (2) is enforced at the hook level rather than in the fixture, so it also covers
# the ~36 legacy files that still hand-roll
# `monkeypatch.setattr(namer, "_DISABLE_VALIDITY_GATE", False)`. Those keep
# working; the marker is the way to write new ones.
# ============================================================================

_GATE_FLAG = "_DISABLE_VALIDITY_GATE"


def _opsin_jar_present() -> bool:
    """True when the validity gate can actually reach OPSIN.

    Mirrors `namer._validity_gate_jar_present` rather than calling it, so this
    stays a test-side observation of the same fact and does not depend on the
    gate module's own import graph being healthy.
    """
    try:
        from orthonym.validation.opsin_roundtrip import _find_opsin_jar
        return _find_opsin_jar() is not None
    except Exception:  # incl. orthonym.jars.JarUnavailable
        return False


def _gate_is_on() -> bool:
    """Read the live flag. Absent module/attr counts as gate-off (not blind)."""
    try:
        import orthonym.namer as _namer
        return getattr(_namer, _GATE_FLAG) is False
    except Exception:
        return False


@pytest.fixture(autouse=True)
def _opsin_validity_gate_state(request, monkeypatch):
    """Set the gate per test: OFF by default, ON under the `opsin_gate` marker.

    Note `raising` is left at its default True — a rename of the flag must be a
    loud AttributeError here, not a suite-wide silent no-op.
    """
    import orthonym.namer as _namer

    wants_gate = (
        request.node.get_closest_marker("opsin_gate") is not None
        or "opsin_gate" in request.fixturenames
    )
    monkeypatch.setattr(_namer, _GATE_FLAG, not wants_gate)
    yield


@pytest.fixture
def opsin_gate(_opsin_validity_gate_state):
    """Fixture form of `@pytest.mark.opsin_gate`, for tests that prefer to
    request it. The autouse fixture above sees this in `request.fixturenames`
    and has already enabled the gate; this only asserts that it did."""
    import orthonym.namer as _namer
    assert getattr(_namer, _GATE_FLAG) is False, (
        "the opsin_gate fixture did not enable the gate — the autouse "
        "fixture's fixturenames check has broken"
    )
    return True


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    """Refuse to run a gate-ON test blind.

    Runs after every fixture has been set up, so it observes the gate state the
    test body will actually see — whether that came from the `opsin_gate`
    marker or from a legacy hand-rolled monkeypatch. With the gate on but no
    jar, `_final_opsin_validity_gate` returns the name unchanged (
    fail-OPEN) and the test asserts nothing about the gate.
    """
    if _gate_is_on() and not _opsin_jar_present():
        # jar_unavailable skips -- or FAILS under ORTHONYM_REQUIRE_JARS=1.
        jar_unavailable(
            "OPSIN jar absent: the validity gate fails OPEN without it (D-13), "
            "so this gate-enabled test would be green-but-blind. Fetch the "
            "OPSIN jar (`orthonym --fetch-jars`) to run it."
        )
    return (yield)


# ============================================================================
# Process-global naming state must not outlive the test that set it.
# ============================================================================
# `Orthonym.name` publishes four tier ContextVars for the length of a top-level
# call, and the fragment recursion keeps three thread-local counters. A test (or
# an engine exit path) that leaves any of them set changes every LATER naming on
# the same xdist worker: a default name_compound call then runs at the leaked
# tier, and a PIN-tier test sees best-effort names. The whole-suite run of
# 2026-09-26 had six such failures that passed alone (TRIAGE g8 C2): the
# N-hydroxy rows (engine leak in the isotope decorator's re-entry, fixed in
# 745d3ed6f), the hydrazinyl furan/thiophene scope guard, and
# test_runtime_cache (a test-side session_depth leak).
#
# The known state is the one ``orthonym.wallclock.leaked_naming_state`` reads (the
# tier ContextVars, the request tier, the fragment counters, the memo scope and the
# producers' re-entry guards). This guard names the leaker instead of letting a later, unrelated test fail:
# after each test it checks the state, resets what leaked, and ERRORS the test
# that left it behind. A leak from inside the engine is a defect (a name must
# never depend on what the process named before), so it is reported, not
# silently absorbed. Validated on two known positives: the pre-745d3ed6f engine
# (one best-effort naming of [2H]C([2H])([2H])O leaves three ContextVars True)
# and the pre-fix test_runtime_cache (session_depth 1).
@pytest.fixture
def reset_naming_state_after():
    """For a test that drives an engine internal (the fragment recursion) outside a ``name`` scope, which
    would put the naming state back itself: reset it after the test, before the guard below reads it."""
    yield
    from orthonym.wallclock import reset_naming_state
    reset_naming_state()


@pytest.fixture(autouse=True)
def _naming_state_isolation():
    """Error the test that leaves naming state behind; reset it for the next.

    A state already leaked before this test started (by a test outside the
    guard's reach) is reset first, so this test does not inherit it.
    """
    from orthonym.wallclock import leaked_naming_state, reset_naming_state
    if leaked_naming_state():
        reset_naming_state()
    yield
    leaked = leaked_naming_state()
    if leaked:
        reset_naming_state()
        pytest.fail(
            f"the test left process-global naming state behind: {leaked}. Every "
            "later naming on this worker would have run with it (TRIAGE g8 C2). "
            "Reset it in the test (the reset_naming_state_after fixture, for a test that drives an "
            "engine internal outside name()), or fix the engine "
            "exit that leaked it.", pytrace=False)


@pytest.fixture(scope="session", autouse=True)
def _opsin_cli_in_process():
    """PR runs: answer the test code's ``java -jar opsin -osmi`` calls in process.

    A JVM start per case made about 60 files the slowest of the suite. The served
    result is what the command line prints (``jvm_bridge._opsin_stdout_uncached``'s
    contract, compared on 40 names by ``test_opsin_cli_inproc.py``). The nightly run sets
    ORTHONYM_TEST_REAL_OPSIN_CLI=1 and runs every call against the real command line; the
    ``real_opsin_cli`` marker does the same for one test. Calls from ``src/orthonym`` are
    never served in process. See tests/support/opsin_cli_inproc.py for the exact scope.
    """
    from tests.support import opsin_cli_inproc
    installed = opsin_cli_inproc.install()
    yield installed
    if installed:
        opsin_cli_inproc.uninstall()


@pytest.fixture(autouse=True)
def _real_opsin_cli_marker(request):
    """A test marked ``real_opsin_cli`` runs against the real OPSIN command line."""
    if request.node.get_closest_marker("real_opsin_cli") is None:
        yield
        return
    from tests.support import opsin_cli_inproc
    opsin_cli_inproc.SUSPENDED = True
    try:
        yield
    finally:
        opsin_cli_inproc.SUSPENDED = False


@pytest.fixture(autouse=True)
def _phase146_clear_thread_locals():
    """Auto-clear the three thread-local stores after each test.

    Stores covered:
      - orthonym.assembly.coverage_scoring._confidence_store
      - orthonym.assembly.candidate_pool._pool_store
      - orthonym.rules.parent_correctness._pc_context.reference_name

    The clear_* helpers are imported lazily and wrapped in try/except so
    a missing import (during partial module reloads) never breaks an
    otherwise-passing test. Reference: internal notes
    """
    yield
    try:
        from orthonym.assembly.coverage_scoring import clear_confidence
        clear_confidence()
    except Exception:
        pass
    try:
        from orthonym.assembly.candidate_pool import clear_pool
        clear_pool()
    except Exception:
        pass
    try:
        from orthonym.rules.parent_correctness import clear_reference_name
        clear_reference_name()
    except Exception:
        pass
