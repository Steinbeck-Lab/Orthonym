"""
Pytest configuration and shared fixtures for Orthonym.
"""

import pytest
import csv
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
# Public-repo fixture guard — skip tests that read local-only.planning /
# a temp dir working directories.
# ============================================================================
# The published repository does NOT ship the local `.planning/` and
# `a temp dir/` working trees. A small set of tests reads real files from under
# them (dev baselines, audit docs, an offline measurement instrument) and would
# raise FileNotFoundError at run time when those files are absent.
#
# This single hook fixes that WITHOUT touching any test:
# * `.planning/` present (local dev) -> no-op, every test runs unchanged.
# * `.planning/` absent (public repo) -> every test in a fixture-dependent
# module is SKIPPED, not errored.
#
# `.planning/` presence is the proxy for "this is the local working tree": when
# it is gone, so is `a temp dir/`, so the single check also covers the
# a temp dir reader.
#
# `FIXTURE_DEPENDENT` is deliberately MINIMAL — only modules that actually READ
# a file under `.planning/` or `a temp dir/` at run time (and would therefore
# fail if it were absent) are listed. Modules that merely MENTION such a path in
# a comment, docstring or assert-message string read nothing and are excluded.
# All reads below are inside test functions/methods (never at import time), so a
# collection-time skip is sufficient — no module fails to import when the files
# are gone.
_PLANNING_DIR = PROJECT_ROOT / ".planning"

FIXTURE_DEPENDENT = {
    # exec's a temp dir/asm_robustness.py via importlib.spec_from_file_location
    # (unguarded); the other tests in this module do not need the fixture, but
    # the guard is applied at module granularity.
    "test_v28_composer1.py",
    # NOTE (2026-09-11): test_orgm/test_frn byte-identical canaries and
    # test_build_ledger now carry their OWN module-level skip/skipif (retired /
    # fixtures pruned), and test_m2_substituent_routing skips only its single
    # fixture-reading test — so they no longer need this dir-level guard (which
    # would over-skip m2's still-valid routing tests in the public repo).
}


def pytest_collection_modifyitems(config, items):
    """Skip fixture-dependent modules when the local `.planning/` tree is absent.

    No-op in local development (where `.planning/` exists and the fixtures are
    present). In the published repo the working directories are not shipped, so
    the run-time FileNotFoundError those modules would raise is turned into a
    clean skip. See the `FIXTURE_DEPENDENT` note above.
    """
    if _PLANNING_DIR.exists():
        return
    skip_marker = pytest.mark.skip(
        reason="requires local .planning/scratchpad fixtures (not published)"
    )
    for item in items:
        # nodeid is "<relpath>::<test>"; take the module file's basename.
        basename = Path(item.nodeid.split("::", 1)[0]).name
        if basename in FIXTURE_DEPENDENT:
            item.add_marker(skip_marker)


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

def pytest_configure(config):
    """Configure custom markers."""
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
