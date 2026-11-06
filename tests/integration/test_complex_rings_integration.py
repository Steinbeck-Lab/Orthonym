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
