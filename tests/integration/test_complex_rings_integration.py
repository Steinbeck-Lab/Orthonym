"""
End-to-end integration tests for a phase: Complex Ring Systems.

Verifies complex ring naming requirements (COMPLEX-01 through COMPLEX-05):
- COMPLEX-01: Bicyclo compound naming (bicyclo[x.y.z] format)
- COMPLEX-02: Spiro compound naming (spiro[a.b] format)
- COMPLEX-03: Fused heterocycle naming (retained names with tautomer locants)
- COMPLEX-05: Complex ring integration in composer

Tests the complete naming pipeline from SMILES to final IUPAC name.

Reference: IUPAC 2013 Blue Book, Sections (Bridged Systems),
            (Spiro Systems), (Fused Systems)
"""

import pytest
from orthonym import name_compound


# =============================================================================
# Perception Layer Accessibility Tests
# =============================================================================

class TestPerceptionLayerAccessibility:
    """Verify perception functions are accessible through rules modules.

    CRITICAL: These tests verify that the complex ring modules can be imported
    and that their detection functions work correctly.
    """

    @pytest.mark.integration
    def test_bicyclo_functions_accessible(self):
        """Verify bicyclo detection functions are accessible."""
        from orthonym.rules.bicyclo import is_bicyclo_system, name_bicyclo_system
        from rdkit import Chem

        # Test with norbornane
        mol = Chem.MolFromSmiles('C1CC2CCC1C2')
        assert is_bicyclo_system(mol) == True
        assert name_bicyclo_system(mol) == 'bicyclo[2.2.1]heptane'  # R11 (2026-09-25, pre-existing-failures plan, Task 5)::9881, only adamantane/cubane are retained; "bicyclo[2.2.1]heptane (PIN)":2038

    @pytest.mark.integration
    def test_spiro_functions_accessible(self):
        """Verify spiro detection functions are accessible."""
        from orthonym.rules.spiro import is_spiro_system, name_spiro_system
        from rdkit import Chem

        # Test with spiro[4.5]decane
        mol = Chem.MolFromSmiles('C1CCC2(CC1)CCCC2')
        assert is_spiro_system(mol) == True
        result = name_spiro_system(mol)
        assert result is not None
        assert result[0] == 'spiro[4.5]decane'

    @pytest.mark.integration
    def test_fused_functions_accessible(self):
        """Verify fused ring detection functions are accessible."""
        from orthonym.rules.fused_rings import (
            classify_fused_system,
            name_fused_heterocycle,
        )
        from rdkit import Chem

        # Test with indole
        mol = Chem.MolFromSmiles('c1ccc2[nH]ccc2c1')
        assert classify_fused_system(mol) == 'ortho-fused'
        result = name_fused_heterocycle(mol)
        assert result is not None
        assert result[0] == '1H-indole'

    @pytest.mark.integration
    def test_composer_imports_work(self):
        """Verify composer can import all complex ring functions."""
        # This tests that the imports at the top of composer.py work
        from orthonym.assembly.composer import (
            _is_complex_ring_system,
            _classify_complex_ring,
            _assemble_complex_ring_name,
        )
        from rdkit import Chem

        mol = Chem.MolFromSmiles('C1CC2CCC1C2')
        assert _is_complex_ring_system(mol) == True
        assert _classify_complex_ring(mol) == 'bicyclo'


# =============================================================================
# COMPLEX-01: Bicyclo System Integration Tests
# =============================================================================

class TestCOMPLEX01:
    """COMPLEX-01: Bicyclo compound naming through main API."""

    @pytest.mark.integration
    def test_bicyclo_systematic_when_no_retained(self):
        """Test systematic naming when no retained name exists."""
        result = name_compound("C1CC2CCC1CC2")  # bicyclo[2.2.2]octane
        assert "bicyclo" in result
        assert "[2.2.2]" in result


# =============================================================================
# COMPLEX-02: Spiro System Integration Tests
# =============================================================================

class TestCOMPLEX02:
    """COMPLEX-02: Spiro compound naming through main API."""

    @pytest.mark.integration
    def test_spiro_descriptor_order(self):
        """Test that spiro descriptor uses correct order (smaller first)."""
        result = name_compound("C1CCC2(CC1)CCCC2")  # 5-ring fused to 6-ring
        # Should be spiro[4.5] not spiro[5.4]
        assert "spiro[4.5]" in result


# =============================================================================
# COMPLEX-03 & COMPLEX-05: Fused Heterocycle Integration Tests
# =============================================================================

class TestCOMPLEX03:
    """COMPLEX-03: Fused heterocycle naming with retained names."""

    @pytest.mark.integration
    def test_indole_has_tautomer_locant(self):
        """Test that indole includes 1H- tautomer locant (IUPAC 2013 PIN)."""
        result = name_compound("c1ccc2[nH]ccc2c1")
        assert result == "1H-indole"
        assert result.startswith("1H-")

    @pytest.mark.integration
    def test_quinoline_no_tautomer_locant(self):
        """Test that quinoline has no tautomer locant (no N-H)."""
        result = name_compound("c1ccc2ncccc2c1")
        assert result == "quinoline"
        assert "H-" not in result


# =============================================================================
# PAH Integration Tests (verify existing still work)
# =============================================================================


# =============================================================================
# Regression Tests (ensure simple rings still work)
# =============================================================================


# =============================================================================
# Complex Ring Routing Tests
# =============================================================================

class TestComplexRingRouting:
    """Test that composer correctly routes complex ring systems."""

    @pytest.mark.integration
    def test_bicyclo_routes_before_cycloalkane(self):
        """Test bicyclo detected before simple cycloalkane classification."""
        # Norbornane could be misclassified as cycloalkane
        result = name_compound("C1CC2CCC1C2")
        # R11 (2026-09-25, pre-existing-failures plan, Task 5)::9881, only adamantane/cubane are retained; "bicyclo[2.2.1]heptane (PIN)":2038
        assert result == "bicyclo[2.2.1]heptane"
        # If it were misclassified, it might be named as a substituted cyclopentane
        assert "cyclopentane" not in result

    @pytest.mark.integration
    def test_spiro_routes_before_cycloalkane(self):
        """Test spiro detected before simple cycloalkane classification."""
        result = name_compound("C1CCC2(CC1)CCCC2")
        assert "spiro" in result
        # If misclassified, might be named as substituted cyclohexane
        assert "cyclohexane" not in result

    @pytest.mark.integration
    def test_fused_heterocycle_routes_correctly(self):
        """Test fused heterocycle detected correctly."""
        result = name_compound("c1ccc2[nH]ccc2c1")
        assert result == "1H-indole"
        # If misclassified, might try to use HW systematic naming
        assert "aza" not in result


# =============================================================================
# Edge Cases and Mixed/Complex Systems
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.integration
    def test_aromatic_fused_not_bicyclo(self):
        """Test that naphthalene is NOT classified as bicyclo."""
        result = name_compound("c1ccc2ccccc2c1")
        assert result == "naphthalene"
        # Should use retained name, not bicyclo descriptor
        assert "bicyclo" not in result
