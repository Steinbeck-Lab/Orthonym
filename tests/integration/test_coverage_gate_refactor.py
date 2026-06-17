"""Integration tests for coverage gate refactor (Phase 81).

Verifies that the refactored candidate collection system:
1. Still returns names for compounds that previously triggered DROP-20
2. Non-gated handlers remain unchanged
3. Candidate collection picks the best handler
4. Canary compounds remain stable
"""

import pytest
from orthonym import name_compound


# ---------------------------------------------------------------------------
# 1. Former DROP-20 compounds still return names
# ---------------------------------------------------------------------------

class TestFormerDrop20Compounds:
    """Compounds from IMPROVED_COMPOUNDS that previously triggered DROP-20."""

    def test_large_indole_peptide(self):
        """35-atom molecule with indole should not just return '1H-indole'."""
        smi = "CC(=O)N[C@@H](CC(C)C)C(=O)N(C)[C@@H](Cc1ccccc1)" \
              "C(=O)N/C=C\\c1c[nH]c2ccccc12"
        result = name_compound(smi)
        assert result is not None
        assert result != "1H-indole", "Should not return bare scaffold name"
        assert len(result) > 10, "Name should be descriptive"

    def test_morphinan_derivative(self):
        """Complex polycyclic molecule should return a name."""
        smi = ("COC1=CC=C2[C@H]3Cc4ccc(OC)c5c4[C@@]2"
               "(C[C@@H](C2=C[C@@]4(O)[C@H]6Cc7ccc(O)c8c7"
               "[C@@]4(CCN6C)[C@@H](O8)C2=O)N3C)[C@H]1O5")
        result = name_compound(smi)
        assert result is not None
        assert result != "morphinan", "Should not return bare scaffold name"

    def test_large_ester_chain(self):
        """Ester with long chains should return a name."""
        smi = ("CCCCC/C=C\\C/C=C\\CCCCCCCCCCCC(=O)O"
               "[C@H](COC(=O)CCCCCCCCCCCCCCCCCC)"
               "COP(=O)(O)OC[C@H](N)C(=O)O")
        result = name_compound(smi)
        assert result is not None
        assert len(result) > 5


# ---------------------------------------------------------------------------
# 2. DROP-20 compounds have confidence metadata
# ---------------------------------------------------------------------------

class TestConfidenceMetadata:
    """Verify include_confidence=True returns metadata for gated compounds.

    Note: include_confidence parameter is added in Task 3 of this plan.
    These tests validate the full API after Task 3 completes.
    """

    def test_benzene_with_confidence(self):
        result = name_compound("c1ccccc1", include_confidence=True)
        assert isinstance(result, dict)
        assert 'name' in result
        assert 'confidence' in result
        assert 'factors' in result
        assert 'handler' in result
        assert result['confidence'] > 0

    def test_pyridine_with_confidence(self):
        result = name_compound("c1ccncc1", include_confidence=True)
        assert isinstance(result, dict)
        assert result['confidence'] > 0

    def test_caffeine_with_confidence(self):
        result = name_compound("Cn1c(=O)c2c(ncn2C)n(C)c1=O",
                               include_confidence=True)
        assert isinstance(result, dict)
        assert result['confidence'] > 0


# ---------------------------------------------------------------------------
# 3. Non-gated handlers unchanged
# ---------------------------------------------------------------------------

class TestNonGatedHandlers:
    """Verify handlers that return directly are not affected."""

    def test_naphthalene_polycyclic(self):
        """Polycyclic handler: direct return, not gated."""
        assert name_compound("c1ccc2ccccc2c1") == "naphthalene"

    def test_tetrahydronaphthalene_partial_sat(self):
        """Partial saturation handler: direct return, not gated."""
        result = name_compound("C1CCC2=CC=CC=C2C1")
        assert "tetrahydronaphthalene" in result.lower() or "tetralin" in result.lower()

    def test_cyclohexanecarbonitrile_ring_nitrile(self):
        """Ring nitrile handler: direct return, not gated."""
        result = name_compound("C1CCCCC1C#N")
        assert "nitrile" in result.lower() or "cyano" in result.lower() or "carbonitrile" in result.lower()

    def test_acetamide_amide(self):
        """Amide handler: direct return, not gated."""
        result = name_compound("CC(=O)N")
        assert "amide" in result.lower() or "acetamide" in result.lower()

    def test_n_methylamine(self):
        """Amine handler: direct return, not gated."""
        result = name_compound("CNC")
        assert result is not None
        assert "methyl" in result.lower()


# ---------------------------------------------------------------------------
# 4. Candidate collection picks best
# ---------------------------------------------------------------------------

class TestCandidateSelection:
    """Verify candidate collection selects the best handler."""

    def test_fused_heterocycle_beats_simple(self):
        """Fused heterocycle (complex_ring) should win over simple heterocycle."""
        # Quinoline with substituent -- complex_ring should win
        result = name_compound("CCCCCCCCCc1cc(=O)c2ccccc2n1C")
        assert "quinolin" in result.lower(), \
            f"Expected quinoline-based name, got: {result}"

    def test_adenine_retained_wins(self):
        """Adenine retained name should win (core retained name boost)."""
        result = name_compound("Nc1ncnc2nc[nH]c12")
        assert result == "adenine"


# ---------------------------------------------------------------------------
# 5. Canary stability spot check
# ---------------------------------------------------------------------------

CANARY_SPOT_CHECK = [
    ("CCO", "ethanol"),
    ("CC(=O)O", "acetic acid"),
    ("c1ccccc1", "benzene"),
    ("Cc1ccccc1", "toluene"),
    ("c1ccc2ccccc2c1", "naphthalene"),
    ("c1ccncc1", "pyridine"),
    ("c1cc[nH]c1", "pyrrole"),
    ("Cn1c(=O)c2c(ncn2C)n(C)c1=O",
     "1,3,7-trimethyl-3,7-dihydro-1H-purine-2,6-dione"),
    ("O=C(O)c1ccccc1", "benzoic acid"),
    # v22 Phase B (DD1 Fix 4): triethylamine is general-nomenclature; PIN is the
    # substitutive N,N-diethylethanamine (P-62.2.1.2).
    ("CCN(CC)CC", "N,N-diethylethanamine"),
]


@pytest.mark.parametrize(
    "smiles,expected",
    CANARY_SPOT_CHECK,
    ids=[name for _, name in CANARY_SPOT_CHECK],
)
def test_canary_stability_spot_check(smiles, expected):
    """Spot-check 10 representative canary compounds."""
    assert name_compound(smiles) == expected
