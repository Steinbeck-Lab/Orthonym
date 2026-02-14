"""
Integration tests for Phase 57 Plan 02: Fused Heterocycle Locant Verification.

Verifies iupac_locants correctness for 4 newly activated fused heterocycles
(xanthene, phenothiazine, phenoxazine, thianthrene) through substituted naming
tests. Catches locant shift bugs (like the isoindoline bug from Phase 27).

Requirements tested:
- PRNT-05: Strip-and-match substructure approach works for substituted forms
- PRNT-06: iupac_locants data integrity for all 4 compounds
"""

import pytest
from rdkit import Chem

from src.orthonym import name_compound
from src.orthonym.data.fused_heterocycles import (
    match_fused_heterocycle_core,
    FUSED_HETEROCYCLE_DATA,
)


# =============================================================================
# Xanthene locant tests
# =============================================================================


class TestXantheneLocants:
    """Verify locant correctness for xanthene (9H-xanthene)."""

    @pytest.mark.integration
    def test_unsubstituted_xanthene(self):
        """Unsubstituted xanthene must produce '9H-xanthene'."""
        result = name_compound('c1ccc2c(c1)Cc1ccccc1O2')
        assert result == '9H-xanthene', f"Got '{result}'"

    @pytest.mark.integration
    def test_substituted_xanthene_has_locant(self):
        """Methyl-substituted xanthene must contain 'xanthene' and 'methyl' with locant."""
        # Methyl on the benzene ring (position 3 based on canonical atom mapping)
        result = name_compound('Cc1ccc2c(c1)Cc1ccccc1O2')
        assert 'xanthene' in result, f"Missing 'xanthene' in '{result}'"
        assert 'methyl' in result, f"Missing 'methyl' in '{result}'"
        # Locant should be a digit
        assert any(c.isdigit() for c in result if c != 'H'), (
            f"No locant digit found in '{result}'"
        )

    @pytest.mark.integration
    def test_xanthene_indicated_hydrogen(self):
        """Substituted xanthene must retain '9H-' indicated hydrogen."""
        result = name_compound('Cc1ccc2c(c1)Cc1ccccc1O2')
        assert '9H-' in result, f"Missing '9H-' in '{result}'"


# =============================================================================
# Phenothiazine locant tests
# =============================================================================


class TestPhenothiazineLocants:
    """Verify locant correctness for phenothiazine (10H-phenothiazine)."""

    @pytest.mark.integration
    def test_unsubstituted_phenothiazine(self):
        """Unsubstituted phenothiazine must produce '10H-phenothiazine'."""
        result = name_compound('c1ccc2c(c1)Nc1ccccc1S2')
        assert result == '10H-phenothiazine', f"Got '{result}'"

    @pytest.mark.integration
    def test_substituted_phenothiazine_has_locant(self):
        """Methyl-substituted phenothiazine must contain 'phenothiazine' and 'methyl'."""
        result = name_compound('Cc1ccc2c(c1)Nc1ccccc1S2')
        assert 'phenothiazine' in result, f"Missing 'phenothiazine' in '{result}'"
        assert 'methyl' in result, f"Missing 'methyl' in '{result}'"
        assert any(c.isdigit() for c in result if c != 'H'), (
            f"No locant digit found in '{result}'"
        )

    @pytest.mark.integration
    def test_phenothiazine_indicated_hydrogen(self):
        """Substituted phenothiazine must retain '10H-' indicated hydrogen."""
        result = name_compound('Cc1ccc2c(c1)Nc1ccccc1S2')
        assert '10H-' in result, f"Missing '10H-' in '{result}'"


# =============================================================================
# Phenoxazine locant tests
# =============================================================================


class TestPhenoxazineLocants:
    """Verify locant correctness for phenoxazine (10H-phenoxazine)."""

    @pytest.mark.integration
    def test_unsubstituted_phenoxazine(self):
        """Unsubstituted phenoxazine must produce '10H-phenoxazine'."""
        result = name_compound('c1ccc2c(c1)Nc1ccccc1O2')
        assert result == '10H-phenoxazine', f"Got '{result}'"

    @pytest.mark.integration
    def test_substituted_phenoxazine_has_locant(self):
        """Methyl-substituted phenoxazine must contain 'phenoxazine' and 'methyl'."""
        result = name_compound('Cc1ccc2c(c1)Nc1ccccc1O2')
        assert 'phenoxazine' in result, f"Missing 'phenoxazine' in '{result}'"
        assert 'methyl' in result, f"Missing 'methyl' in '{result}'"
        assert any(c.isdigit() for c in result if c != 'H'), (
            f"No locant digit found in '{result}'"
        )

    @pytest.mark.integration
    def test_phenoxazine_indicated_hydrogen(self):
        """Substituted phenoxazine must retain '10H-' indicated hydrogen."""
        result = name_compound('Cc1ccc2c(c1)Nc1ccccc1O2')
        assert '10H-' in result, f"Missing '10H-' in '{result}'"


# =============================================================================
# Thianthrene locant tests
# =============================================================================


class TestThianthreneLocants:
    """Verify locant correctness for thianthrene."""

    @pytest.mark.integration
    def test_unsubstituted_thianthrene(self):
        """Unsubstituted thianthrene must produce 'thianthrene'."""
        result = name_compound('c1ccc2c(c1)Sc1ccccc1S2')
        assert result == 'thianthrene', f"Got '{result}'"

    @pytest.mark.integration
    def test_substituted_thianthrene_has_locant(self):
        """Methyl-substituted thianthrene must contain 'thianthrene' and 'methyl'."""
        result = name_compound('Cc1ccc2c(c1)Sc1ccccc1S2')
        assert 'thianthrene' in result, f"Missing 'thianthrene' in '{result}'"
        assert 'methyl' in result, f"Missing 'methyl' in '{result}'"
        assert any(c.isdigit() for c in result), (
            f"No locant digit found in '{result}'"
        )


# =============================================================================
# Substructure core matching tests (PRNT-05)
# =============================================================================


class TestSubstructureCoreMatching:
    """Verify strip-and-match substructure approach works for all 4 compounds."""

    @pytest.mark.integration
    def test_substituted_xanthene_matched_via_core(self):
        """match_fused_heterocycle_core must find xanthene core in substituted xanthene."""
        mol = Chem.MolFromSmiles('Cc1ccc2c(c1)Cc1ccccc1O2')
        result = match_fused_heterocycle_core(mol)
        assert result is not None, "No core match for substituted xanthene"
        name, mapping, core_smiles = result
        assert '9H-xanthene' == name, f"Expected '9H-xanthene', got '{name}'"
        assert core_smiles == 'c1ccc2c(c1)Cc1ccccc1O2', (
            f"Wrong core SMILES: '{core_smiles}'"
        )

    @pytest.mark.integration
    def test_substituted_phenothiazine_matched_via_core(self):
        """match_fused_heterocycle_core must find phenothiazine core."""
        mol = Chem.MolFromSmiles('Cc1ccc2c(c1)Nc1ccccc1S2')
        result = match_fused_heterocycle_core(mol)
        assert result is not None, "No core match for substituted phenothiazine"
        name, mapping, core_smiles = result
        assert '10H-phenothiazine' == name, f"Expected '10H-phenothiazine', got '{name}'"

    @pytest.mark.integration
    def test_substituted_phenoxazine_matched_via_core(self):
        """match_fused_heterocycle_core must find phenoxazine core."""
        mol = Chem.MolFromSmiles('Cc1ccc2c(c1)Nc1ccccc1O2')
        result = match_fused_heterocycle_core(mol)
        assert result is not None, "No core match for substituted phenoxazine"
        name, mapping, core_smiles = result
        assert '10H-phenoxazine' == name, f"Expected '10H-phenoxazine', got '{name}'"

    @pytest.mark.integration
    def test_substituted_thianthrene_matched_via_core(self):
        """match_fused_heterocycle_core must find thianthrene core."""
        mol = Chem.MolFromSmiles('Cc1ccc2c(c1)Sc1ccccc1S2')
        result = match_fused_heterocycle_core(mol)
        assert result is not None, "No core match for substituted thianthrene"
        name, mapping, core_smiles = result
        assert 'thianthrene' == name, f"Expected 'thianthrene', got '{name}'"


# =============================================================================
# Locant data integrity tests (PRNT-06)
# =============================================================================


class TestLocantDataIntegrity:
    """Verify iupac_locants dict structure for all 4 compounds."""

    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,compound_name", [
        ('c1ccc2c(c1)Cc1ccccc1O2', 'xanthene'),
        ('c1ccc2c(c1)Nc1ccccc1S2', 'phenothiazine'),
        ('c1ccc2c(c1)Nc1ccccc1O2', 'phenoxazine'),
        ('c1ccc2c(c1)Sc1ccccc1S2', 'thianthrene'),
    ])
    def test_locant_count_matches_parent_atoms(self, smiles, compound_name):
        """iupac_locants must have exactly parent_atoms entries."""
        data = FUSED_HETEROCYCLE_DATA[smiles]
        locants = data['iupac_locants']
        expected = data['parent_atoms']
        assert len(locants) == expected, (
            f"{compound_name}: expected {expected} locants, got {len(locants)}"
        )

    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,compound_name", [
        ('c1ccc2c(c1)Cc1ccccc1O2', 'xanthene'),
        ('c1ccc2c(c1)Nc1ccccc1S2', 'phenothiazine'),
        ('c1ccc2c(c1)Nc1ccccc1O2', 'phenoxazine'),
        ('c1ccc2c(c1)Sc1ccccc1S2', 'thianthrene'),
    ])
    def test_indices_contiguous(self, smiles, compound_name):
        """iupac_locants indices must be contiguous 0..N-1."""
        data = FUSED_HETEROCYCLE_DATA[smiles]
        locants = data['iupac_locants']
        n = data['parent_atoms']
        expected_indices = set(range(n))
        actual_indices = set(locants.keys())
        assert actual_indices == expected_indices, (
            f"{compound_name}: indices not contiguous. "
            f"Missing: {expected_indices - actual_indices}, "
            f"Extra: {actual_indices - expected_indices}"
        )

    @pytest.mark.integration
    def test_xanthene_locant_values(self):
        """Xanthene locant set must contain expected IUPAC positions.

        Xanthene (9H-xanthene) IUPAC numbering:
        1,2,3,4,4a,4b,5,6,7,8,8a,9,9a,10 = 14 positions
        """
        data = FUSED_HETEROCYCLE_DATA['c1ccc2c(c1)Cc1ccccc1O2']
        locant_values = set(data['iupac_locants'].values())
        expected = {1, 2, 3, 4, '4a', '4b', 5, 6, 7, 8, '8a', 9, '9a', 10}
        assert locant_values == expected, (
            f"Xanthene locant values mismatch.\n"
            f"  Expected: {expected}\n"
            f"  Got:      {locant_values}\n"
            f"  Missing:  {expected - locant_values}\n"
            f"  Extra:    {locant_values - expected}"
        )

    @pytest.mark.integration
    def test_phenothiazine_locant_values(self):
        """Phenothiazine locant set must contain expected IUPAC positions.

        Phenothiazine (10H-phenothiazine) IUPAC numbering:
        1,2,3,4,4a,5,5a,6,7,8,9,9a,10,10a = 14 positions
        """
        data = FUSED_HETEROCYCLE_DATA['c1ccc2c(c1)Nc1ccccc1S2']
        locant_values = set(data['iupac_locants'].values())
        expected = {1, 2, 3, 4, '4a', 5, '5a', 6, 7, 8, 9, '9a', 10, '10a'}
        assert locant_values == expected, (
            f"Phenothiazine locant values mismatch.\n"
            f"  Expected: {expected}\n"
            f"  Got:      {locant_values}\n"
            f"  Missing:  {expected - locant_values}\n"
            f"  Extra:    {locant_values - expected}"
        )

    @pytest.mark.integration
    def test_phenoxazine_locant_values(self):
        """Phenoxazine locant set must contain expected IUPAC positions.

        Phenoxazine (10H-phenoxazine) IUPAC numbering:
        1,2,3,4,4a,5,5a,6,7,8,9,9a,10,10a = 14 positions
        """
        data = FUSED_HETEROCYCLE_DATA['c1ccc2c(c1)Nc1ccccc1O2']
        locant_values = set(data['iupac_locants'].values())
        expected = {1, 2, 3, 4, '4a', 5, '5a', 6, 7, 8, 9, '9a', 10, '10a'}
        assert locant_values == expected, (
            f"Phenoxazine locant values mismatch.\n"
            f"  Expected: {expected}\n"
            f"  Got:      {locant_values}\n"
            f"  Missing:  {expected - locant_values}\n"
            f"  Extra:    {locant_values - expected}"
        )

    @pytest.mark.integration
    def test_thianthrene_locant_values(self):
        """Thianthrene locant set must contain expected IUPAC positions.

        Thianthrene IUPAC numbering:
        1,2,3,4,4a,5,5a,6,7,8,9,9a,10,10a = 14 positions
        """
        data = FUSED_HETEROCYCLE_DATA['c1ccc2c(c1)Sc1ccccc1S2']
        locant_values = set(data['iupac_locants'].values())
        expected = {1, 2, 3, 4, '4a', 5, '5a', 6, 7, 8, 9, '9a', 10, '10a'}
        assert locant_values == expected, (
            f"Thianthrene locant values mismatch.\n"
            f"  Expected: {expected}\n"
            f"  Got:      {locant_values}\n"
            f"  Missing:  {expected - locant_values}\n"
            f"  Extra:    {locant_values - expected}"
        )

    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,compound_name", [
        ('c1ccc2c(c1)Cc1ccccc1O2', 'xanthene'),
        ('c1ccc2c(c1)Nc1ccccc1S2', 'phenothiazine'),
        ('c1ccc2c(c1)Nc1ccccc1O2', 'phenoxazine'),
        ('c1ccc2c(c1)Sc1ccccc1S2', 'thianthrene'),
    ])
    def test_locants_contain_fusion_positions(self, smiles, compound_name):
        """All 4 compounds must have fusion positions (string locants like '4a')."""
        data = FUSED_HETEROCYCLE_DATA[smiles]
        locant_values = set(data['iupac_locants'].values())
        fusion_locants = {v for v in locant_values if isinstance(v, str)}
        assert len(fusion_locants) >= 2, (
            f"{compound_name}: expected >= 2 fusion positions, "
            f"got {len(fusion_locants)}: {fusion_locants}"
        )
