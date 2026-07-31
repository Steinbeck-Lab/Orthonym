"""
Pytest configuration and shared fixtures for Orthonym.
"""

import pytest
import csv
from pathlib import Path
from typing import List, Dict

from rdkit import Chem

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
FIXTURES_DIR = Path(__file__).parent / "fixtures"


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
    """Find OPSIN JAR path if available."""
    import glob
    
    # Search patterns for OPSIN JAR
    patterns = [
        str(PROJECT_ROOT / "opsin" / "opsin-cli" / "target" / "opsin-cli-*-jar-with-dependencies.jar"),
        str(PROJECT_ROOT / "opsin-cli-*.jar"),
        str(PROJECT_ROOT / "opsin.jar"),
        "opsin-cli-*-jar-with-dependencies.jar",
        "opsin.jar",
    ]
    
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    
    return None


@pytest.fixture
def opsin_available(opsin_jar) -> bool:
    """Check if OPSIN is available for round-trip testing."""
    import shutil
    
    if not shutil.which("java"):
        return False
    
    return opsin_jar is not None


@pytest.fixture
def opsin_to_smiles(opsin_available, opsin_jar):
    """Convert IUPAC name to SMILES using OPSIN."""
    if not opsin_available:
        pytest.skip("OPSIN not available")
    
    import subprocess
    
    def _convert(name: str) -> str:
        try:
            result = subprocess.run(
                ['java', '-jar', opsin_jar, '-osmi', name],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None
    
    return _convert


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
# or that OPSIN parses to a DIFFERENT molecule (SELF-01), is the whole point of
# it. The suite disables it by default because most tests assert RAW generator
# output — some deliberately malformed — and must not pay a per-name OPSIN call.
#
# That default is a TRAP, and it has been sprung. A test written *about* gate
# behaviour is green-but-blind unless it re-enables the gate, and nothing says
# so. Measured 2026-07-31 on the canary `CC(=O)N(CC1CO1)C(C)C`:
#
#     gate OFF -> '(5-carbamoylpentyl)oxirane'   <- a DIFFERENT molecule
#     gate ON  -> 'unknown organic compound'     <- SELF-01 suppressed it
#
# so "assert this molecule abstains" passes for the wrong reason with the gate
# off, and would keep passing if the gate were deleted outright.
#
# There is now exactly ONE supported way to ask for the gate:
#
#     pytestmark = pytest.mark.opsin_gate       # module-wide, or
#     @pytest.mark.opsin_gate                   # per-test, or
#     def test_x(opsin_gate): ...               # fixture form
#
# and it is VERIFIED rather than merely requested. Two silent-failure modes are
# closed:
#
#   1. A rename of the flag. The old `raising=False` meant a rename would turn
#      every re-enable in the suite into a no-op at once, silently. The setattr
#      below raises.
#   2. A missing OPSIN jar. `_final_opsin_validity_gate` fails OPEN when the jar
#      is absent (D-13), so "gate on, no jar" is the same blind state by another
#      route — and a worktree checkout has no jar (the `opsin` gitlink has no
#      .gitmodules to fetch from). `pytest_runtest_call` below skips those tests
#      instead of passing them.
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
    except Exception:
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
    jar, `_final_opsin_validity_gate` returns the name unchanged (D-13
    fail-OPEN) and the test asserts nothing about the gate.
    """
    if _gate_is_on() and not _opsin_jar_present():
        pytest.skip(
            "OPSIN jar absent: the validity gate fails OPEN without it (D-13), "
            "so this gate-enabled test would be green-but-blind. Build/fetch "
            "the OPSIN jar to run it."
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
    otherwise-passing test. Reference: 146-RESEARCH.md §8.4.
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
