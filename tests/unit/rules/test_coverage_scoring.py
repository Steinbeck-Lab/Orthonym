"""Unit tests for coverage_scoring module.

Tests CandidateName dataclass, compute_confidence 4-factor scoring,
select_best_candidate selection logic, and thread-local confidence store.
"""

import logging
import pytest
from rdkit import Chem

from orthonym.assembly.coverage_scoring import (
    CandidateName,
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
    FACTOR_WEIGHTS,
    HANDLER_PRIORITY,
    compute_confidence,
    select_best_candidate,
    log_confidence,
    store_confidence,
    retrieve_confidence,
    clear_confidence,
    _compute_fg_recognition,
    _compute_substituent_completeness,
)
from orthonym.namer import MolecularFeatures


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_features(smiles, functional_groups=None, principal_group=None,
                   principal_group_atoms=None, substituents=None,
                   ring_substituents=None, heterocycle_substituents=None,
                   benzene_substituents=None):
    """Create a minimal MolecularFeatures for testing."""
    mol = Chem.MolFromSmiles(smiles)
    f = MolecularFeatures(mol=mol, smiles=smiles)
    if functional_groups is not None:
        f.functional_groups = functional_groups
    if principal_group is not None:
        f.principal_group = principal_group
    if principal_group_atoms is not None:
        f.principal_group_atoms = principal_group_atoms
    if substituents is not None:
        f.substituents = substituents
    if ring_substituents is not None:
        f.ring_substituents = ring_substituents
    if heterocycle_substituents is not None:
        f.heterocycle_substituents = heterocycle_substituents
    if benzene_substituents is not None:
        f.benzene_substituents = benzene_substituents
    return f


# ---------------------------------------------------------------------------
# 1. CandidateName creation
# ---------------------------------------------------------------------------

def test_candidate_name_creation():
    """CandidateName stores all required fields."""
    c = CandidateName(
        name="ethanol",
        handler="chain",
        confidence=0.85,
        factors={'ratio': 0.9, 'atom_coverage': 0.8,
                 'fg_recognition': 1.0, 'substituent_completeness': 0.7},
    )
    assert c.name == "ethanol"
    assert c.handler == "chain"
    assert c.confidence == 0.85
    assert len(c.factors) == 4
    assert c.factors['ratio'] == 0.9


# ---------------------------------------------------------------------------
# 2-5. compute_confidence tests
# ---------------------------------------------------------------------------

def test_compute_confidence_simple_alkane():
    """Ethane: no FGs to miss should give high fg_recognition."""
    features = _make_features("CC")
    cand = compute_confidence("ethane", "chain", features)
    assert 0.0 <= cand.confidence <= 1.0
    assert cand.factors['fg_recognition'] == 1.0
    assert cand.handler == "chain"
    assert cand.name == "ethane"


def test_compute_confidence_with_functional_groups():
    """Ethanol: has an alcohol FG, principal group should be recognised."""
    features = _make_features(
        "CCO",
        functional_groups={'primary_alcohol': [(1,)]},
        principal_group='primary_alcohol',
        principal_group_atoms=[(1,)],
    )
    cand = compute_confidence("ethanol", "chain", features)
    assert 0.0 <= cand.confidence <= 1.0
    assert cand.factors['fg_recognition'] == 1.0  # 1/1 recognised
    assert all(k in cand.factors for k in FACTOR_WEIGHTS)


def test_compute_confidence_no_heavy_atoms():
    """Edge case: molecule with no heavy atoms returns 1.0 factors."""
    # Hydrogen molecule has no heavy atoms in RDKit
    mol = Chem.MolFromSmiles("[H][H]")
    features = MolecularFeatures(mol=mol, smiles="[H][H]")
    cand = compute_confidence("hydrogen", "chain", features)
    # With 0 heavy atoms, ratio_raw = len("hydrogen") / 1 = 8, normalised to 1.0
    assert cand.factors['ratio'] == 1.0
    assert cand.confidence > 0.0


def test_compute_confidence_retained_name_boost():
    """Retained name (benzene) gets atom_coverage=1.0 regardless of size."""
    features = _make_features("c1ccccc1")
    cand = compute_confidence("benzene", "benzene", features)
    assert cand.factors['atom_coverage'] == 1.0


# ---------------------------------------------------------------------------
# 6-8. select_best_candidate tests
# ---------------------------------------------------------------------------

def test_select_best_candidate_highest_wins():
    """Candidate with clearly higher confidence wins."""
    low = CandidateName(name="a", handler="chain", confidence=0.3)
    high = CandidateName(name="b", handler="heterocycle", confidence=0.9)
    assert select_best_candidate([low, high]).name == "b"
    assert select_best_candidate([high, low]).name == "b"


def test_select_best_candidate_tiebreak_by_priority():
    """Within EPSILON, higher priority handler wins."""
    a = CandidateName(name="a", handler="chain", confidence=0.80)
    b = CandidateName(name="b", handler="complex_ring", confidence=0.805)
    # Difference is 0.005, within EPSILON=0.01 -> tiebreak by priority
    result = select_best_candidate([a, b])
    assert result.handler == "complex_ring"


def test_select_best_candidate_single():
    """Single candidate returns itself."""
    c = CandidateName(name="only", handler="chain", confidence=0.5)
    assert select_best_candidate([c]) is c


# ---------------------------------------------------------------------------
# 9-10. FG recognition tests
# ---------------------------------------------------------------------------

def test_fg_recognition_no_fgs():
    """Hydrocarbons with no FGs return 1.0."""
    features = _make_features("CCCC")  # butane
    assert _compute_fg_recognition(features) == 1.0


def test_fg_recognition_all_recognized():
    """All FGs have known prefix forms -> 1.0."""
    features = _make_features(
        "OCC(=O)O",  # glycolic acid
        functional_groups={
            'carboxylic_acid': [(2, 3, 4)],
            'primary_alcohol': [(0,)],
        },
        principal_group='carboxylic_acid',
        principal_group_atoms=[(2, 3, 4)],
    )
    result = _compute_fg_recognition(features)
    assert result == 1.0  # carboxylic acid (principal) + alcohol has "hydroxy" prefix


# ---------------------------------------------------------------------------
# 11-12. Substituent completeness tests
# ---------------------------------------------------------------------------

def test_substituent_completeness_no_subs():
    """Returns 1.0 when no substituents are expected."""
    features = _make_features("CC")
    assert _compute_substituent_completeness("ethane", features) == 1.0


def test_substituent_completeness_partial():
    """Correctly estimates partial coverage from locants in name."""
    features = _make_features(
        "CC(C)C",
        substituents={2: [[3]]},  # 1 substituent at locant 2
    )
    # Name "2-methylpropane" has "2-" as one locant group
    result = _compute_substituent_completeness("2-methylpropane", features)
    assert result == 1.0  # 1 named / 1 expected

    # Name without locant -> 0 named
    result_no_locant = _compute_substituent_completeness("propane", features)
    assert result_no_locant == 0.0


# ---------------------------------------------------------------------------
# 13-14. Thread-local store tests
# ---------------------------------------------------------------------------

def test_thread_local_store_clear():
    """clear_confidence resets the store."""
    cand = CandidateName(name="test", handler="chain", confidence=0.5)
    store_confidence(cand)
    clear_confidence()
    result = retrieve_confidence()
    assert result['name'] == ''
    # v29 C4: was `== 0.0`. An empty store measured NOTHING, so it now reports
    # confidence=None / verification='unverified' rather than 0.0 -- which was
    # a fabricated FAIL, just as the old name_with_confidence() 1.0 was a
    # fabricated PASS. The project invariant admits no verdict either way on
    # insufficient evidence. Contract locked by
    # tests/unit/test_coverage_contract.py.
    assert result['confidence'] is None
    assert result['verification'] == 'unverified'
    assert result['factors'] == {}
    # 'unknown' is preserved deliberately: the namer quality gates use
    # handler != 'unknown' as their "was anything actually scored?" guard.
    assert result['handler'] == 'unknown'


def test_thread_local_store_round_trip():
    """store then retrieve returns same data."""
    cand = CandidateName(
        name="ethanol",
        handler="chain",
        confidence=0.85,
        factors={'ratio': 0.9, 'atom_coverage': 0.8,
                 'fg_recognition': 1.0, 'substituent_completeness': 0.7},
    )
    store_confidence(cand)
    result = retrieve_confidence()
    assert result['name'] == "ethanol"
    assert result['confidence'] == 0.85
    assert result['handler'] == "chain"
    assert result['factors']['ratio'] == 0.9
    # Clean up
    clear_confidence()


# ---------------------------------------------------------------------------
# 15. Logging level tests
# ---------------------------------------------------------------------------

def test_log_confidence_levels(caplog):
    """Correct log levels for each confidence band."""
    high = CandidateName(name="a", handler="chain", confidence=0.80)
    med = CandidateName(name="b", handler="chain", confidence=0.50)
    low = CandidateName(name="c", handler="chain", confidence=0.30)

    with caplog.at_level(logging.DEBUG, logger="orthonym.assembly.coverage_scoring"):
        log_confidence(high)
    assert any("accepted" in r.message and r.levelno == logging.DEBUG
               for r in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="orthonym.assembly.coverage_scoring"):
        log_confidence(med)
    assert any("moderate" in r.message and r.levelno == logging.INFO
               for r in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="orthonym.assembly.coverage_scoring"):
        log_confidence(low)
    assert any("low-confidence" in r.message and r.levelno == logging.WARNING
               for r in caplog.records)


# ---------------------------------------------------------------------------
# Phase 145.1: parent_atom_indices field extension (SC-1)
# ---------------------------------------------------------------------------

def test_candidate_name_has_parent_atom_indices_field():
    """Phase 145.1: CandidateName dataclass exposes parent_atom_indices."""
    c = CandidateName(name="ethanol", handler="chain")
    assert hasattr(c, "parent_atom_indices")
    assert c.parent_atom_indices is None


def test_candidate_name_parent_atom_indices_accepts_set():
    """Phase 145.1: parent_atom_indices stores a set of atom indices."""
    c = CandidateName(
        name="ethanol", handler="chain",
        parent_atom_indices={0, 1, 2},
    )
    assert c.parent_atom_indices == {0, 1, 2}


def test_candidate_name_default_factory_unchanged():
    """Phase 145.1 regression: factors default_factory still works."""
    c = CandidateName(name="x", handler="h")
    assert c.factors == {}
    # Mutating one instance must not affect another
    c.factors['ratio'] = 0.5
    c2 = CandidateName(name="y", handler="h")
    assert c2.factors == {}


# ---------------------------------------------------------------------------
# Phase 145.1: FACTOR_WEIGHTS extension + byte-identical proof (SC-3, D-14)
# ---------------------------------------------------------------------------

def test_factor_weights_includes_parent_correctness():
    """SC-3: FACTOR_WEIGHTS contains parent_correctness key with value 0.0."""
    from orthonym.assembly.coverage_scoring import FACTOR_WEIGHTS
    assert 'parent_correctness' in FACTOR_WEIGHTS
    assert FACTOR_WEIGHTS['parent_correctness'] == 0.0


def test_factor_weights_parent_correctness_is_last_key():
    """Risk 2 (PATTERNS): parent_correctness MUST be inserted at LAST position
    to preserve dict iteration order in compute_confidence's sum-loop."""
    from orthonym.assembly.coverage_scoring import FACTOR_WEIGHTS
    keys = list(FACTOR_WEIGHTS.keys())
    assert keys[-1] == 'parent_correctness', (
        f"parent_correctness must be LAST key for byte-identical safety; "
        f"actual order: {keys}"
    )


def test_factor_weights_total_5_keys():
    """SC-3: FACTOR_WEIGHTS has exactly 5 keys after extension."""
    from orthonym.assembly.coverage_scoring import FACTOR_WEIGHTS
    assert len(FACTOR_WEIGHTS) == 5


def test_factor_weights_existing_keys_unchanged():
    """Risk 2: existing 4 keys preserved in their original order."""
    from orthonym.assembly.coverage_scoring import FACTOR_WEIGHTS
    keys = list(FACTOR_WEIGHTS.keys())
    assert keys[:4] == [
        'ratio', 'atom_coverage', 'fg_recognition', 'substituent_completeness'
    ], f"Existing key order changed; actual: {keys}"


def test_factor_weights_existing_values_unchanged():
    """Risk 2: non-demoted factor weights preserved at Phase 145.1 calibration.

    Phase 145.2 D-09-a.1 update: the 'ratio' factor is demoted to 0.0 because
    it has zero IUPAC Blue Book justification (name-length/HA is a heuristic,
    not a naming rule). Byte-identical preserved via position-based
    pool.best() selection. The other three weights remain at their Phase 81
    calibration values until Phase 146 SC-4 recalibrates.
    """
    from orthonym.assembly.coverage_scoring import FACTOR_WEIGHTS
    assert FACTOR_WEIGHTS['ratio'] == 0.0  # Phase 145.2 D-09-a.1 demotion
    assert FACTOR_WEIGHTS['atom_coverage'] == 0.20
    assert FACTOR_WEIGHTS['fg_recognition'] == 0.35
    assert FACTOR_WEIGHTS['substituent_completeness'] == 0.25


def test_byte_identical_confidence_with_zero_weight_factor():
    """D-14 BYTE-IDENTICAL PROOF: adding parent_correctness=0.0 does NOT
    change compute_confidence's output value.

    Phase 145.2 D-09-a.1 update: 'ratio' is ALSO demoted to 0.0 as a
    separate invariant (zero IUPAC justification). The 4-key "control"
    dict used in this test must mirror the live FACTOR_WEIGHTS values
    (sans the 5th parent_correctness key) so that the test remains a
    clean proof of D-14's "last-position 0.0 key insertion is a no-op"
    claim. If ratio's live value changes in a future phase (Phase 146
    recalibration), update the control dict to match.

    Strategy: compute confidence via the live FACTOR_WEIGHTS (5 keys);
    then patch FACTOR_WEIGHTS to remove the 5th key; recompute; assert
    equal to 4-decimal precision. IEEE 754 + dict-order guarantees this.
    """
    import orthonym.assembly.coverage_scoring as cs
    from orthonym.assembly.coverage_scoring import compute_confidence
    # Use a representative molecule (benzaldehyde -- non-trivial confidence)
    features = _make_features(
        "O=Cc1ccccc1",
        functional_groups={'aldehyde': [(1, 0)]},
        principal_group='aldehyde',
        principal_group_atoms=[(1, 0)],
    )
    # Compute with the live (5-key) FACTOR_WEIGHTS
    cand_5key = compute_confidence("benzaldehyde", "benzene", features)
    confidence_5key = cand_5key.confidence

    # Save and patch to a 4-key dict (omits parent_correctness).
    # Phase 145.2 D-09-a.1: 'ratio' mirrors its demoted value (0.0) so
    # this test isolates the D-14 "5th-key-as-no-op" invariant.
    original = cs.FACTOR_WEIGHTS
    cs.FACTOR_WEIGHTS = {
        'ratio': 0.0,   # Phase 145.2 D-09-a.1 demotion
        'atom_coverage': 0.20,
        'fg_recognition': 0.35,
        'substituent_completeness': 0.25,
    }
    try:
        cand_4key = compute_confidence("benzaldehyde", "benzene", features)
        confidence_4key = cand_4key.confidence
    finally:
        cs.FACTOR_WEIGHTS = original

    # IEEE 754: x + 0.0 = x (for finite x); rounded to 4 decimals must match
    assert confidence_5key == confidence_4key, (
        f"BYTE-IDENTICAL VIOLATION: 5-key confidence={confidence_5key:.6f} "
        f"!= 4-key confidence={confidence_4key:.6f}. D-14 invariant broken."
    )


def test_handler_priority_extended_with_iss002_entries():
    """ISS-002 REGRESSION: HANDLER_PRIORITY contains the entries Plan 01's
    HANDLER_POLICIES needs. Single source of truth -- no hardcoded literals
    in candidate_pool.py."""
    from orthonym.assembly.coverage_scoring import HANDLER_PRIORITY
    assert HANDLER_PRIORITY['n_oxide'] == 5, "n_oxide must be 5 (matches direct-return peers)"
    assert HANDLER_PRIORITY['amine'] == 5, "amine must be 5 (matches direct-return peers)"
    assert HANDLER_PRIORITY['simple_molecule'] == 1, "simple_molecule must be 1 (terminal fallback)"
    assert HANDLER_PRIORITY['polyfunctional'] == 4, "polyfunctional already present, must be 4"
