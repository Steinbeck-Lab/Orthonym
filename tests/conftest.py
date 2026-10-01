"""
Pytest configuration and shared fixtures for Orthonym.
"""

import pytest
import csv
import os
import warnings
from pathlib import Path
from typing import List, Dict

from rdkit import Chem

# Shared jar helpers (the jars live outside the repo; see orthonym.jars).
# Re-exported here; tests import them from ``tests.support.jars``.
from tests.support.jars import REQUIRE_JARS, jar_or_none, jar_or_skip, jar_unavailable  # noqa: F401

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
FIXTURES_DIR = Path(__file__).parent / "fixtures"


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
    def _create(smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, f"Invalid SMILES: {smiles}"
        return mol
    return _create


@pytest.fixture
def canonical():
    """Factory fixture to get canonical SMILES."""
    def _canonicalize(smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, f"Invalid SMILES: {smiles}"
        return Chem.MolToSmiles(mol, canonical=True)
    return _canonicalize


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def test_data() -> List[Dict[str, str]]:
    """Load test dataset from CSV if available."""
    csv_path = PROJECT_ROOT / "test_data.csv"
    
    if not csv_path.exists():
        return []
    
    data = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Handle different possible column names
            smiles = row.get('smiles') or row.get('SMILES') or row.get('Smiles')
            name = row.get('iupac_name') or row.get('name') or row.get('Name') or row.get('IUPAC')
            if smiles and name:
                data.append({'smiles': smiles, 'name': name})
    
    return data


@pytest.fixture
def quick_sample(test_data) -> List[Dict[str, str]]:
    """First 500 compounds for quick testing."""
    return test_data[:500]


@pytest.fixture
def medium_sample(test_data) -> List[Dict[str, str]]:
    """First 5000 compounds for medium testing."""
    return test_data[:5000]


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
    import shutil

    if not shutil.which("java"):
        return False

    return opsin_jar is not None


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

    from rdkit import Chem
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
# Compound Class Test Data
# ============================================================================

@pytest.fixture
def simple_alkanes():
    """Simple alkane test cases."""
    return [
        ("C", "methane"),
        ("CC", "ethane"),
        ("CCC", "propane"),
        ("CCCC", "butane"),
        ("CCCCC", "pentane"),
        ("CCCCCC", "hexane"),
        ("CCCCCCC", "heptane"),
        ("CCCCCCCC", "octane"),
        ("CCCCCCCCC", "nonane"),
        ("CCCCCCCCCC", "decane"),
    ]


@pytest.fixture
def branched_alkanes():
    """Branched alkane test cases."""
    return [
        ("CC(C)C", "2-methylpropane"),
        ("CC(C)CC", "2-methylbutane"),
        ("CCC(C)C", "2-methylbutane"),
        ("CC(C)(C)C", "2,2-dimethylpropane"),
        ("CC(C)C(C)C", "2,3-dimethylbutane"),
        ("CC(C)(C)CC", "2,2-dimethylbutane"),
        ("CCC(CC)CC", "3-ethylpentane"),
    ]


@pytest.fixture
def simple_alcohols():
    """Simple alcohol test cases."""
    return [
        ("CO", "methanol"),
        ("CCO", "ethanol"),
        ("CCCO", "propan-1-ol"),
        ("CC(O)C", "propan-2-ol"),
        ("CCCCO", "butan-1-ol"),
        ("CCC(O)C", "butan-2-ol"),
        ("CC(C)O", "propan-2-ol"),
        ("CC(C)CO", "2-methylpropan-1-ol"),
    ]


@pytest.fixture
def simple_aldehydes():
    """Simple aldehyde test cases."""
    return [
        ("C=O", "formaldehyde"),
        ("CC=O", "acetaldehyde"),
        ("CCC=O", "propanal"),
        ("CCCC=O", "butanal"),
        ("CCCCC=O", "pentanal"),
    ]


@pytest.fixture
def simple_ketones():
    """Simple ketone test cases."""
    return [
        ("CC(C)=O", "propan-2-one"),
        ("CCC(C)=O", "butan-2-one"),
        ("CCCC(C)=O", "pentan-2-one"),
        ("CCC(CC)=O", "pentan-3-one"),
    ]


@pytest.fixture
def simple_acids():
    """Simple carboxylic acid test cases."""
    return [
        ("C(=O)O", "formic acid"),
        ("CC(=O)O", "acetic acid"),
        ("CCC(=O)O", "propanoic acid"),
        ("CCCC(=O)O", "butanoic acid"),
        ("CCCCC(=O)O", "pentanoic acid"),
    ]


@pytest.fixture
def simple_heterocycles():
    """Simple heterocycle test cases (retained names)."""
    return [
        ("c1ccoc1", "furan"),
        ("c1ccsc1", "thiophene"),
        ("c1cc[nH]c1", "pyrrole"),
        ("c1ccncc1", "pyridine"),
        ("C1CCOC1", "oxolane"),
        ("C1CCNC1", "pyrrolidine"),
        ("C1CCNCC1", "piperidine"),
        ("C1COCCN1", "morpholine"),
    ]


# ============================================================================
# Test Markers
# ============================================================================

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


def pytest_configure(config):
    """Configure custom markers; snapshot the ORTHONYM_* environment; record reloads."""
    _ORTHONYM_ENV_BEFORE_COLLECTION.clear()
    _ORTHONYM_ENV_BEFORE_COLLECTION.update(_orthonym_env())
    _install_reload_recorder()
    config.addinivalue_line(
        "markers", "unit: Fast unit tests (<1s each)"
    )
    config.addinivalue_line(
        "markers", "integration: Compound class integration tests"
    )
    config.addinivalue_line(
        "markers", "roundtrip: Round-trip validation with OPSIN"
    )
    config.addinivalue_line(
        "markers", "slow: Full 100k validation suite"
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
# This guard names the leaker instead of letting a later, unrelated test fail:
# after each test it checks the state, resets what leaked, and ERRORS the test
# that left it behind. A leak from inside the engine is a defect (a name must
# never depend on what the process named before), so it is reported, not
# silently absorbed. Validated on two known positives: the pre-745d3ed6f engine
# (one best-effort naming of [2H]C([2H])([2H])O leaves three ContextVars True)
# and the pre-fix test_runtime_cache (session_depth 1).
_NAMING_TIER_CTX = ("general_fallback_ctx", "best_effort_ctx",
                    "allow_aromatic_general_ctx", "full_coverage_ctx")
# The request tier (``provenance.best_effort_request_ctx``) is None outside a
# request; a value left behind, False included, would set the tier of every later
# call on the worker that reads it outside a request: the ring ceilings
# (``vonbaeyer_universal.cage_caps``) and ``name_with_confidence``, which sets it
# only when it is None.
_NAMING_REQUEST_CTX = ("best_effort_request_ctx",)
_NAMING_DEPTHS = ("session_depth", "name_call_depth")


def _leaked_naming_state() -> dict:
    """The non-default process-global naming state of the calling thread."""
    import sys
    leaked = {}
    pv = sys.modules.get("orthonym.metrics.provenance")
    if pv is not None:
        for attr in _NAMING_TIER_CTX:
            var = getattr(pv, attr, None)
            if var is not None and var.get() not in (False, None):
                leaked[attr] = var.get()
        for attr in _NAMING_REQUEST_CTX:
            var = getattr(pv, attr, None)
            if var is not None and var.get() is not None:
                leaked[attr] = var.get()
    fn = sys.modules.get("orthonym.assembly.fragment_naming")
    guard = getattr(fn, "_fragment_guard", None) if fn is not None else None
    if guard is not None:
        for attr in _NAMING_DEPTHS:
            if getattr(guard, attr, 0):
                leaked[attr] = getattr(guard, attr)
        if getattr(guard, "visited", None):
            leaked["visited"] = len(guard.visited)
    return leaked


def _reset_naming_state() -> None:
    import sys
    pv = sys.modules.get("orthonym.metrics.provenance")
    if pv is not None:
        for attr in _NAMING_TIER_CTX:
            var = getattr(pv, attr, None)
            if var is not None:
                var.set(False)
        for attr in _NAMING_REQUEST_CTX:
            var = getattr(pv, attr, None)
            if var is not None:
                var.set(None)
    fn = sys.modules.get("orthonym.assembly.fragment_naming")
    guard = getattr(fn, "_fragment_guard", None) if fn is not None else None
    if guard is not None:
        for attr in _NAMING_DEPTHS:
            setattr(guard, attr, 0)
        guard.visited = set()
        guard.cache = None


@pytest.fixture(autouse=True)
def _naming_state_isolation():
    """Error the test that leaves naming state behind; reset it for the next.

    A state already leaked before this test started (by a test outside the
    guard's reach) is reset first, so this test does not inherit it.
    """
    if _leaked_naming_state():
        _reset_naming_state()
    yield
    leaked = _leaked_naming_state()
    if leaked:
        _reset_naming_state()
        pytest.fail(
            f"the test left process-global naming state behind: {leaked}. Every "
            "later naming on this worker would have run with it (TRIAGE g8 C2). "
            "Reset it in the test (try/finally, or a fixture), or fix the engine "
            "exit that leaked it.", pytrace=False)


# ============================================================================
# A module a test reloads must be put back before the next test.
# ============================================================================
# `importlib.reload` re-executes a module inside the same module object, so its
# import-time reads of the environment and its classes are replaced for the rest
# of the process; monkeypatch restoring the environment does not undo that. Four
# tests reloaded coverage_scoring and candidate_pool under
# ORTHONYM_USE_V18_WEIGHTS=true and left them so: every later test on the worker
# named with the V18 weights (the canary oxime row
# '(3Z,6E)-2,4,4,7-tetramethylnona-6,8-dien-3-one oxime' became '...-3-oxime')
# and saw another CandidateName class (isinstance failed). TRIAGE 'Canary oxime --
# test-order flake'. Reload with `tests.support.module_reload.reloaded`, which
# puts the module back. This guard records the namespace of every orthonym module
# a test reloads (before its first reload in the test) and, after the test, puts
# back any module that was left changed and ERRORS the test that left it.
# Validated on the known positives: each of the four reload sites without
# `reloaded` errors here.
_RELOADED_MODULES: Dict[str, dict] = {}
_ORIGINAL_RELOAD = None


def _recording_reload(module):
    """`importlib.reload`, recording an orthonym module's namespace first."""
    name = getattr(module, "__name__", "") or ""
    if (name == "orthonym" or name.startswith("orthonym.")) and name not in _RELOADED_MODULES:
        _RELOADED_MODULES[name] = dict(vars(module))
    return _ORIGINAL_RELOAD(module)


def _install_reload_recorder() -> None:
    global _ORIGINAL_RELOAD
    import importlib
    if _ORIGINAL_RELOAD is None:
        _ORIGINAL_RELOAD = importlib.reload
        importlib.reload = _recording_reload


def pytest_unconfigure(config):
    """Put `importlib.reload` back."""
    global _ORIGINAL_RELOAD
    import importlib
    if _ORIGINAL_RELOAD is not None:
        importlib.reload = _ORIGINAL_RELOAD
        _ORIGINAL_RELOAD = None


@pytest.fixture(autouse=True)
def _module_reload_isolation():
    """Error the test that reloads an orthonym module and leaves it changed."""
    _RELOADED_MODULES.clear()
    yield
    import sys
    left = []
    for name, namespace in _RELOADED_MODULES.items():
        module = sys.modules.get(name)
        if module is None:
            continue
        live = vars(module)
        if live.keys() != namespace.keys() or any(live[k] is not v for k, v in namespace.items()):
            live.clear()
            live.update(namespace)
            left.append(name)
    _RELOADED_MODULES.clear()
    if left:
        pytest.fail(
            f"the test reloaded {', '.join(left)} and left the reloaded module in place "
            "(its import-time settings and classes); every later test on this worker "
            "would have run with it (TRIAGE 'Canary oxime -- test-order flake'). It was "
            "put back. Reload with tests.support.module_reload.reloaded.", pytrace=False)


# ============================================================================
# An OPSIN oracle built for a jar a test simulated must not outlive the test.
# ============================================================================
# The validity gate keeps ONE OpsinOracle for the process (`namer._VALIDITY_ORACLE`;
# the split gate keeps `group_splitting._DEFAULT_ORACLE` the same way). It is built
# at its first use from the jar `_find_opsin_jar` names at that moment and is never
# built again. A test that forces the jar absent (monkeypatch `_find_opsin_jar` ->
# None, the `_force_jar_absent` pattern) and is the first test on its worker to reach
# that oracle builds it with jar=None (an N-acyl ring amino acid such as captopril
# reaches it through `fragment_acid_name_verified`, which asks OPSIN without checking
# for the jar first). monkeypatch puts the resolver back, but the oracle keeps
# jar=None. For the rest of the worker the gate runs again, every OPSIN answer is
# 'unavailable', the gate fails closed on that, and simple molecules come out as
# 'unknown organic compound' (438 failures in one worker's order; TRIAGE 'Unit suite
# -- the unknown-organic-compound order leak'). The engine does the same outside the
# tests after `orthonym.jars.fetch_all` in a process that started in reduced mode;
# the engine fix (the oracle follows the jar resolution) is in namer.py and waits for
# the lane merge (strict xfail in tests/unit/test_simulated_jar_isolation.py). Until
# then this guard runs after each test's teardown (monkeypatch has been undone by
# then). An orthonym module global that holds an OpsinOracle which the test replaced
# with one built for another jar than the real one is put back: the object from
# before the test if its jar is the real one, otherwise None (the next use builds it
# again). The test is not failed: simulating a missing jar is what it is for, and the
# resolver it patched was put back. Validated on the known positives in that section.
_ORACLES_BEFORE_TEST: Dict[tuple, object] = {}


def _module_oracles() -> Dict[tuple, object]:
    """(module, attribute) -> value, for each orthonym module global that holds an OpsinOracle."""
    import sys
    found = {}
    for name, module in list(sys.modules.items()):
        if module is None or not (name == "orthonym" or name.startswith("orthonym.")):
            continue
        for attr, value in list(vars(module).items()):
            # By class name, so an instance made before a reload of its module still counts.
            if type(value).__name__ == "OpsinOracle" and hasattr(value, "_jar"):
                found[(name, attr)] = value
    return found


def _real_opsin_jar():
    """The jar this test process really has (None in reduced mode or without a jar)."""
    try:
        from orthonym.validation.opsin_roundtrip import _find_opsin_jar
        return _find_opsin_jar()
    except Exception:  # incl. orthonym.jars.JarUnavailable
        return None


def _put_back_simulated_jar_oracles() -> list:
    """Put back each module-global OpsinOracle the test replaced with one built for
    another jar than the real one; return the (module, attribute) pairs put back."""
    import sys
    changed = {key: value for key, value in _module_oracles().items()
               if _ORACLES_BEFORE_TEST.get(key) is not value}
    if not changed:
        return []
    real = _real_opsin_jar()
    put_back = []
    for (name, attr), oracle in changed.items():
        if oracle._jar == real:
            continue
        before = _ORACLES_BEFORE_TEST.get((name, attr))
        keep = before if before is not None and getattr(before, "_jar", None) == real else None
        setattr(sys.modules[name], attr, keep)
        put_back.append((name, attr))
    return put_back


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup(item):
    """Record the module-global OpsinOracles before any fixture of the test runs."""
    global _ORACLES_BEFORE_TEST
    _ORACLES_BEFORE_TEST = _module_oracles()
    return (yield)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_teardown(item, nextitem):
    """After every finalizer of the test (monkeypatch undone), put back the oracles
    built for a simulated jar."""
    try:
        return (yield)
    finally:
        _put_back_simulated_jar_oracles()


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
