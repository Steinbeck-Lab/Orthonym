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


# ============================================================================
# Phase 146 RESEARCH §8.4 — thread-local cleanup auto-fixture.
# Prevents state leakage when later plans introduce monkeypatch.setenv
# + importlib.reload patterns for V17/V18 parametrized testing
# (test_feature_flag.py, test_score_based_mode.py).
# ============================================================================
@pytest.fixture(autouse=True)
def _disable_opsin_validity_gate_for_tests(monkeypatch):
    """SUB-03/D-13: the OPSIN-parse validity gate is default-ON in production,
    but the test suite asserts RAW output (incl. some deliberately-malformed
    names) and must not pay a per-name OPSIN subprocess. Disable the gate by
    default for every test; the gate's own tests (test_opsin_validity_gate.py)
    re-enable it via monkeypatch. Best-effort — a missing attr never breaks a
    test (raising=False)."""
    try:
        import orthonym.namer as _namer
        monkeypatch.setattr(_namer, "_DISABLE_VALIDITY_GATE", True, raising=False)
    except Exception:
        pass
    yield


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
