"""
Integration tests for Phase 57: Fused Heterocycle Classification Routing.

Tests that the routing guards in composer.py correctly divert tricyclic fused
heterocycles (xanthene, phenothiazine, phenoxazine, thianthrene) to the fused
naming path instead of the polycyclic-bridged or sulfide naming paths.

Requirements tested:
- PRNT-01: Xanthene named as '9H-xanthene' (not tricyclo[...])
- PRNT-02: Phenothiazine named as '10H-phenothiazine' (not sulfide)
- PRNT-03: Phenoxazine named as '10H-phenoxazine' (not tricyclo[...])
- PRNT-04: Thianthrene named as 'thianthrene' (not sulfide)
- Phase 57 Plan 02: Canary regression tests for all 7 tricyclic + 4 bicyclic compounds
"""

import pytest
from rdkit import Chem

from src.orthonym import name_compound
from src.orthonym.assembly.composer import _classify_complex_ring


# =============================================================================
# Core routing fix tests (Phase 57 target compounds)
# =============================================================================


class TestPhase57RoutingFixes:
    """Test that the 4 target compounds now produce correct retained names."""

    @pytest.mark.integration
    def test_xanthene_not_tricyclo(self):
        """PRNT-01: Xanthene must use retained name, not VB polycyclic."""
        result = name_compound('c1ccc2c(c1)Cc1ccccc1O2')
        assert result == '9H-xanthene', (
            f"Xanthene routing failed: got '{result}', expected '9H-xanthene'"
        )

    @pytest.mark.integration
    def test_phenothiazine_not_sulfide(self):
        """PRNT-02: Phenothiazine must use retained name, not sulfide."""
        result = name_compound('c1ccc2c(c1)Nc1ccccc1S2')
        assert result == '10H-phenothiazine', (
            f"Phenothiazine routing failed: got '{result}', expected '10H-phenothiazine'"
        )

    @pytest.mark.integration
    def test_phenoxazine_not_tricyclo(self):
        """PRNT-03: Phenoxazine must use retained name, not VB polycyclic."""
        result = name_compound('c1ccc2c(c1)Nc1ccccc1O2')
        assert result == '10H-phenoxazine', (
            f"Phenoxazine routing failed: got '{result}', expected '10H-phenoxazine'"
        )

    @pytest.mark.integration
    def test_thianthrene_not_sulfide(self):
        """PRNT-04: Thianthrene must use retained name, not sulfide."""
        result = name_compound('c1ccc2c(c1)Sc1ccccc1S2')
        assert result == 'thianthrene', (
            f"Thianthrene routing failed: got '{result}', expected 'thianthrene'"
        )


# =============================================================================
# Regression guard tests (already-correct tricyclic compounds)
# =============================================================================


class TestPhase57RegressionGuard:
    """Ensure 3 already-correct tricyclic fused heterocycles remain unchanged."""

    @pytest.mark.integration
    def test_carbazole_still_correct(self):
        """Carbazole must remain '9H-carbazole' after routing changes."""
        result = name_compound('c1ccc2c(c1)[nH]c1ccccc12')
        assert result == '9H-carbazole', (
            f"Carbazole regressed: got '{result}', expected '9H-carbazole'"
        )

    @pytest.mark.integration
    def test_acridine_still_correct(self):
        """Acridine must remain 'acridine' after routing changes."""
        result = name_compound('c1ccc2nc3ccccc3cc2c1')
        assert result == 'acridine', (
            f"Acridine regressed: got '{result}', expected 'acridine'"
        )

    @pytest.mark.integration
    def test_phenazine_still_correct(self):
        """Phenazine must remain 'phenazine' after routing changes."""
        result = name_compound('c1ccc2nc3ccccc3nc2c1')
        assert result == 'phenazine', (
            f"Phenazine regressed: got '{result}', expected 'phenazine'"
        )


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
    """Ensure the thioether guard only affects fused heterocycles, not real sulfides."""

    @pytest.mark.integration
    def test_dimethyl_sulfide_still_works(self):
        """Dimethyl sulfide must still be named as a sulfide."""
        result = name_compound('CSC')
        assert 'sulfide' in result, (
            f"Dimethyl sulfide lost sulfide naming: got '{result}'"
        )

    @pytest.mark.integration
    def test_diethyl_sulfide_still_works(self):
        """Diethyl sulfide must still be named as a sulfide."""
        result = name_compound('CCSCC')
        assert 'sulfide' in result, (
            f"Diethyl sulfide lost sulfide naming: got '{result}'"
        )


# =============================================================================
# Canary compound tests (Phase 57 Plan 02)
# =============================================================================


class TestPhase57CanaryCompounds:
    """Canary-level regression guards for all 7 tricyclic fused heterocycles."""

    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,expected_name", [
        ('c1ccc2c(c1)Cc1ccccc1O2', '9H-xanthene'),
        ('c1ccc2c(c1)Nc1ccccc1S2', '10H-phenothiazine'),
        ('c1ccc2c(c1)Nc1ccccc1O2', '10H-phenoxazine'),
        ('c1ccc2c(c1)Sc1ccccc1S2', 'thianthrene'),
        ('c1ccc2c(c1)[nH]c1ccccc12', '9H-carbazole'),
        ('c1ccc2nc3ccccc3cc2c1', 'acridine'),
        ('c1ccc2nc3ccccc3nc2c1', 'phenazine'),
    ])
    def test_tricyclic_fused_heterocycle(self, smiles, expected_name):
        """All 7 tricyclic fused heterocycles must produce correct retained names."""
        result = name_compound(smiles)
        assert result == expected_name, (
            f"Canary failed for {expected_name}: got '{result}'"
        )


class TestBicyclicFusedHeterocyclesUnchanged:
    """Verify bicyclic fused compounds still name correctly after Phase 57 changes."""

    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,expected_name", [
        ('c1ccc2[nH]ccc2c1', '1H-indole'),
        ('c1ccc2ncccc2c1', 'quinoline'),
        ('c1ccc2cnccc2c1', 'isoquinoline'),
        ('c1ccc2[nH]cnc2c1', '1H-benzimidazole'),
    ])
    def test_bicyclic_fused_heterocycle(self, smiles, expected_name):
        """Bicyclic fused heterocycles must not be affected by tricyclic routing guards."""
        result = name_compound(smiles)
        assert result == expected_name, (
            f"Bicyclic regression for {expected_name}: got '{result}'"
        )
