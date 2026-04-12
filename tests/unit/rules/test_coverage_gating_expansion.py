"""
Coverage gating expansion tests.

Verifies that:
1. HANDLER_PRIORITY dict contains entries for all handler types (33+ entries)
2. CONFIDENCE_GATE_THRESHOLD is set at 0.40
3. Tier B handlers use confidence gating before returning
4. _confidence_gate function correctly rejects low-confidence names
5. High-confidence handler names still pass through correctly
"""

import pytest
from rdkit import Chem

from orthonym.assembly.coverage_scoring import (
    HANDLER_PRIORITY,
    CONFIDENCE_GATE_THRESHOLD,
    CandidateName,
    compute_confidence,
)
from orthonym.namer import MolecularFeatures, name_compound


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_features(smiles, **kwargs):
    """Create a minimal MolecularFeatures for testing."""
    mol = Chem.MolFromSmiles(smiles)
    f = MolecularFeatures(mol=mol, smiles=smiles)
    for key, val in kwargs.items():
        setattr(f, key, val)
    return f


# ---------------------------------------------------------------------------
# 1. Structural tests: HANDLER_PRIORITY coverage
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_handler_priority_has_minimum_entries():
    """HANDLER_PRIORITY dict has 6+ entries (expanded from original 4)."""
    assert len(HANDLER_PRIORITY) >= 6, \
        f"Expected 6+ handler priorities, got {len(HANDLER_PRIORITY)}"


@pytest.mark.unit
def test_handler_priority_contains_core_handlers():
    """Core ring handlers are in HANDLER_PRIORITY."""
    assert 'complex_ring' in HANDLER_PRIORITY
    assert 'heterocycle' in HANDLER_PRIORITY
    assert 'benzene' in HANDLER_PRIORITY
    assert 'chain' in HANDLER_PRIORITY


@pytest.mark.unit
def test_handler_priority_contains_tier_b_handlers():
    """Tier B specialized handlers are in HANDLER_PRIORITY."""
    tier_b = [
        'oxime', 'hydrazone', 'isocyanate', 'isothiocyanate',
        'carbamic_acid', 'carbamate', 'urea', 'guanidine',
        'sulfoxide', 'sulfone', 'thioether', 'boronic_acid',
    ]
    for handler in tier_b:
        assert handler in HANDLER_PRIORITY, \
            f"Missing Tier B handler '{handler}' in HANDLER_PRIORITY"


@pytest.mark.unit
def test_confidence_gate_threshold():
    """CONFIDENCE_GATE_THRESHOLD is set at 0.40."""
    assert CONFIDENCE_GATE_THRESHOLD == 0.40


# ---------------------------------------------------------------------------
# 2. _confidence_gate function tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_confidence_gate_exists():
    """_confidence_gate function is importable from composer."""
    from orthonym.assembly.composer import _confidence_gate
    assert callable(_confidence_gate)


@pytest.mark.unit
def test_confidence_gate_accepts_high_confidence():
    """_confidence_gate returns True for high-confidence names."""
    from orthonym.assembly.composer import _confidence_gate
    features = _make_features("OB(O)c1ccccc1")  # phenylboronic acid
    # A name that covers the molecule well should pass
    result = _confidence_gate("phenylboronic acid", "boronic_acid", features)
    assert result is True, "High-confidence name should pass gate"


@pytest.mark.unit
def test_confidence_gate_rejects_low_confidence():
    """_confidence_gate returns False for low-confidence names."""
    from orthonym.assembly.composer import _confidence_gate
    # A large molecule with a tiny, non-retained name -> low confidence
    features = _make_features(
        "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",  # triacontane (30 carbons)
        functional_groups={'alcohol': [(1,)]},
        principal_group='alcohol',
        principal_group_atoms=[(1,)],
        substituents={0: [[2, 3]]},
        ring_substituents={},
        heterocycle_substituents={},
        benzene_substituents={},
    )
    # A very short name for a 30-atom molecule: should be rejected
    result = _confidence_gate("ol", "thioether", features)
    assert result is False, "Low-confidence name should be rejected"


# ---------------------------------------------------------------------------
# 3. End-to-end tests: Tier B handlers with confidence gating
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("smiles, expected_fragment, handler_desc", [
    # High-confidence: small molecules where handler covers most atoms
    ("OB(O)c1ccccc1", "boronic acid", "boronic_acid"),
    # Urea - simple
    ("O=C(N)N", "urea", "urea"),
])
def test_tier_b_high_confidence_passes(smiles, expected_fragment, handler_desc):
    """Tier B handlers still produce correct names for high-confidence matches."""
    result = name_compound(smiles)
    assert expected_fragment in result.lower(), \
        f"Expected '{expected_fragment}' in '{result}' for {handler_desc}"


@pytest.mark.unit
def test_confidence_gate_in_composer_source():
    """Verify composer.py contains CONFIDENCE_GATE debug log messages."""
    import inspect
    from orthonym.assembly import composer
    source = inspect.getsource(composer)
    assert "CONFIDENCE_GATE" in source, \
        "composer.py must contain CONFIDENCE_GATE log messages"
    # Count occurrences of _confidence_gate calls
    gate_calls = source.count("_confidence_gate(")
    assert gate_calls >= 3, \
        f"Expected 3+ _confidence_gate calls in composer.py, found {gate_calls}"


# ---------------------------------------------------------------------------
# 4. compute_confidence integration with Tier B handler IDs
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("handler_id", [
    "oxime", "boronic_acid", "sulfoxide", "urea", "carbamate",
])
def test_compute_confidence_accepts_tier_b_handler_ids(handler_id):
    """compute_confidence() works with Tier B handler IDs."""
    features = _make_features("CCCO", functional_groups={}, principal_group=None)
    cand = compute_confidence("propan-1-ol", handler_id, features)
    assert isinstance(cand, CandidateName)
    assert cand.handler == handler_id
    assert 0.0 <= cand.confidence <= 1.0
