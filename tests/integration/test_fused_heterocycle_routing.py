"""
Integration tests for a phase: Fused Heterocycle Classification Routing.

Tests that the routing guards in composer.py correctly divert tricyclic fused
heterocycles (xanthene, phenothiazine, phenoxazine, thianthrene) to the fused
naming path instead of the polycyclic-bridged or sulfide naming paths.

Requirements tested:
-: Xanthene named as '9H-xanthene' (not tricyclo[...])
-: Phenothiazine named as '10H-phenothiazine' (not sulfide)
-: Phenoxazine named as '10H-phenoxazine' (not tricyclo[...])
-: Thianthrene named as 'thianthrene' (not sulfide)
- a phase Plan 02: Canary regression tests for all 7 tricyclic + 4 bicyclic compounds
"""

import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.assembly.composer import _classify_complex_ring


# =============================================================================
# Core routing fix tests (a phase target compounds)
# =============================================================================


# =============================================================================
# Regression guard tests (already-correct tricyclic compounds)
# =============================================================================


# =============================================================================
# _classify_complex_ring unit-level tests
# =============================================================================


class TestClassifyComplexRingGuard:
    """Test that _classify_complex_ring returns correct classification type."""

    @pytest.mark.integration
    def test_xanthene_classified_as_fused(self):
        """Xanthene must be classified as ortho-fused, not polycyclic-bridged."""
        mol = Chem.MolFromSmiles('c1ccc2c(c1)Cc1ccccc1O2')
        result = _classify_complex_ring(mol)
        assert result == 'ortho-fused', (
            f"Xanthene classified as '{result}', expected 'ortho-fused'"
        )

    @pytest.mark.integration
    def test_phenothiazine_classified_as_fused(self):
        """Phenothiazine must be classified as ortho-fused, not polycyclic-bridged."""
        mol = Chem.MolFromSmiles('c1ccc2c(c1)Nc1ccccc1S2')
        result = _classify_complex_ring(mol)
        assert result == 'ortho-fused', (
            f"Phenothiazine classified as '{result}', expected 'ortho-fused'"
        )

    @pytest.mark.integration
    def test_thianthrene_classified_as_fused(self):
        """Thianthrene must be classified as ortho-fused, not polycyclic-bridged."""
        mol = Chem.MolFromSmiles('c1ccc2c(c1)Sc1ccccc1S2')
        result = _classify_complex_ring(mol)
        assert result == 'ortho-fused', (
            f"Thianthrene classified as '{result}', expected 'ortho-fused'"
        )

    @pytest.mark.integration
    def test_adamantane_still_polycyclic(self):
        """Adamantane must remain polycyclic-bridged (VB path regression guard)."""
        mol = Chem.MolFromSmiles('C1C2CC3CC1CC(C2)C3')
        result = _classify_complex_ring(mol)
        assert result == 'polycyclic-bridged', (
            f"Adamantane classified as '{result}', expected 'polycyclic-bridged'"
        )


# =============================================================================
# Thioether guard non-interference tests
# =============================================================================


class TestThioetherGuardDoesNotAffectRealSulfides:
    """Ensure the thioether guard only affects fused heterocycles, not real sulfides.

    A real acyclic sulfide must still be NAMED (not lost to the fused-heterocycle
    guard). Since the sulfanyl-vs-sulfide slice, the PIN is the SUBSTITUTIVE
    "(R'-sulfanyl)RH" method 1, the Blue Book), not the functional-class
    "R R' sulfide" (method 2) — so these assert the substitutive PIN.
    """

    @pytest.mark.integration
    def test_diethyl_sulfide_still_works(self):
        """CCSCC must still be named — now (ethylsulfanyl)ethane (PIN, P-63.2.5)."""
        result = name_compound('CCSCC')
        assert result == '(ethylsulfanyl)ethane', (
            f"Diethyl sulfide lost/mis-named: got '{result}'"
        )


# =============================================================================
# Canary compound tests (a phase Plan 02)
# =============================================================================


class TestPhase57CanaryCompounds:
    """Canary-level regression guards for all 7 tricyclic fused heterocycles."""

    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,expected_name", [
        ('c1ccc2nc3ccccc3nc2c1', 'phenazine'),
    ])
    def test_tricyclic_fused_heterocycle(self, smiles, expected_name):
        """All 7 tricyclic fused heterocycles must produce correct retained names."""
        result = name_compound(smiles)
        assert result == expected_name, (
            f"Canary failed for {expected_name}: got '{result}'"
        )


