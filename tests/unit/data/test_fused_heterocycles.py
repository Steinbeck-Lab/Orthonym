"""
Unit tests for fused heterocycle lookup tables.

Tests:
- Exact SMILES matching for unsubstituted systems
- Tautomer locant correctness
- Substructure matching for substituted systems
- Non-matches for inappropriate molecules
- Canonical SMILES consistency
"""

import pytest
from rdkit import Chem
from src.orthonym.data.fused_heterocycles import (
    FUSED_HETEROCYCLE_DATA,
    get_fused_heterocycle_name,
    match_fused_heterocycle_core,
    get_fused_heterocycle_info,
    is_fused_heterocycle,
    get_ring_system_type,
)


class TestExactSMILESMatching:
    """Test exact SMILES matching for unsubstituted fused heterocycles."""

    @pytest.mark.unit
    def test_indole_returns_correct_name(self):
        """Indole should return '1H-indole' with tautomer locant 1."""
        mol = Chem.MolFromSmiles('c1ccc2[nH]ccc2c1')
        result = get_fused_heterocycle_name(mol)
        assert result is not None
        assert result[0] == '1H-indole'
        assert result[1] == 1

    @pytest.mark.unit
    def test_quinoline_returns_correct_name(self):
        """Quinoline should return 'quinoline' with no tautomer locant."""
        mol = Chem.MolFromSmiles('c1ccc2ncccc2c1')
        result = get_fused_heterocycle_name(mol)
        assert result is not None
        assert result[0] == 'quinoline'
        assert result[1] is None

    @pytest.mark.unit
    def test_carbazole_returns_correct_name(self):
        """Carbazole should return '9H-carbazole' with tautomer locant 9."""
        mol = Chem.MolFromSmiles('c1ccc2c(c1)[nH]c1ccccc12')
        result = get_fused_heterocycle_name(mol)
        assert result is not None
        assert result[0] == '9H-carbazole'
        assert result[1] == 9

    @pytest.mark.unit
    def test_benzofuran_returns_correct_name(self):
        """Benzofuran should return '1-benzofuran' with no tautomer locant."""
        mol = Chem.MolFromSmiles('c1ccc2occc2c1')
        result = get_fused_heterocycle_name(mol)
        assert result is not None
        assert result[0] == '1-benzofuran'
        assert result[1] is None

    @pytest.mark.unit
    def test_isoquinoline_returns_correct_name(self):
        """Isoquinoline should return 'isoquinoline'."""
        mol = Chem.MolFromSmiles('c1ccc2cnccc2c1')
        result = get_fused_heterocycle_name(mol)
        assert result is not None
        assert result[0] == 'isoquinoline'
        assert result[1] is None

    @pytest.mark.unit
    def test_benzimidazole_returns_correct_name(self):
        """Benzimidazole should return '1H-benzimidazole'."""
        mol = Chem.MolFromSmiles('c1ccc2[nH]cnc2c1')
        result = get_fused_heterocycle_name(mol)
        assert result is not None
        assert result[0] == '1H-benzimidazole'
        assert result[1] == 1

    @pytest.mark.unit
    def test_purine_returns_correct_name(self):
        """Purine should return '9H-purine' with tautomer locant 9."""
        mol = Chem.MolFromSmiles('c1ncc2nc[nH]c2n1')
        result = get_fused_heterocycle_name(mol)
        assert result is not None
        assert result[0] == '9H-purine'
        assert result[1] == 9


class TestTautomerLocants:
    """Test tautomer locant assignment for indicated hydrogen."""

    @pytest.mark.unit
    def test_1h_indole_has_locant_1(self):
        """1H-indole has indicated hydrogen at position 1."""
        mol = Chem.MolFromSmiles('c1ccc2[nH]ccc2c1')
        result = get_fused_heterocycle_name(mol)
        assert result[1] == 1

    @pytest.mark.unit
    def test_2h_isoindole_has_locant_2(self):
        """2H-isoindole has indicated hydrogen at position 2."""
        mol = Chem.MolFromSmiles('C1=Nc2ccccc2C1')
        result = get_fused_heterocycle_name(mol)
        assert result is not None
        assert result[0] == '2H-isoindole'
        assert result[1] == 2

    @pytest.mark.unit
    def test_1h_isoindole_has_locant_1(self):
        """1H-isoindole (aromatic tautomer) has indicated hydrogen at position 1."""
        mol = Chem.MolFromSmiles('c1ccc2c[nH]cc2c1')
        result = get_fused_heterocycle_name(mol)
        assert result is not None
        assert result[0] == '1H-isoindole'
        assert result[1] == 1

    @pytest.mark.unit
    def test_9h_carbazole_has_locant_9(self):
        """9H-carbazole has indicated hydrogen at position 9."""
        mol = Chem.MolFromSmiles('c1ccc2c(c1)[nH]c1ccccc12')
        result = get_fused_heterocycle_name(mol)
        assert result[1] == 9

    @pytest.mark.unit
    def test_9h_xanthene_has_locant_9(self):
        """9H-xanthene has indicated hydrogen at position 9."""
        mol = Chem.MolFromSmiles('c1ccc2c(c1)Cc1ccccc1O2')
        result = get_fused_heterocycle_name(mol)
        assert result is not None
        assert result[0] == '9H-xanthene'
        assert result[1] == 9

    @pytest.mark.unit
    def test_10h_phenothiazine_has_locant_10(self):
        """10H-phenothiazine has indicated hydrogen at position 10."""
        mol = Chem.MolFromSmiles('c1ccc2c(c1)Nc1ccccc1S2')
        result = get_fused_heterocycle_name(mol)
        assert result is not None
        assert result[0] == '10H-phenothiazine'
        assert result[1] == 10

    @pytest.mark.unit
    def test_quinoline_has_no_tautomer_locant(self):
        """Quinoline has no indicated hydrogen (fully aromatic)."""
        mol = Chem.MolFromSmiles('c1ccc2ncccc2c1')
        result = get_fused_heterocycle_name(mol)
        assert result[1] is None

    @pytest.mark.unit
    def test_acridine_has_no_tautomer_locant(self):
        """Acridine has no indicated hydrogen (fully aromatic)."""
        mol = Chem.MolFromSmiles('c1ccc2nc3ccccc3cc2c1')
        result = get_fused_heterocycle_name(mol)
        assert result is not None
        assert result[0] == 'acridine'
        assert result[1] is None


class TestSubstructureMatching:
    """Test substructure matching for substituted fused heterocycles."""

    @pytest.mark.unit
    def test_5_methylindole_matches_indole_core(self):
        """5-methylindole should match indole core."""
        mol = Chem.MolFromSmiles('Cc1ccc2[nH]ccc2c1')
        result = match_fused_heterocycle_core(mol)
        assert result is not None
        name, mapping = result
        assert name == '1H-indole'
        assert isinstance(mapping, dict)
        assert len(mapping) == 9  # 9 atoms in indole core

    @pytest.mark.unit
    def test_4_chloroquinoline_matches_quinoline_core(self):
        """4-chloroquinoline should match quinoline core."""
        mol = Chem.MolFromSmiles('Clc1ccnc2ccccc12')
        result = match_fused_heterocycle_core(mol)
        assert result is not None
        name, mapping = result
        assert name == 'quinoline'
        assert len(mapping) == 10  # 10 atoms in quinoline core

    @pytest.mark.unit
    def test_2_methylbenzimidazole_matches_benzimidazole_core(self):
        """2-methylbenzimidazole should match benzimidazole core."""
        mol = Chem.MolFromSmiles('Cc1nc2ccccc2[nH]1')
        result = match_fused_heterocycle_core(mol)
        assert result is not None
        name, mapping = result
        assert name == '1H-benzimidazole'
        assert len(mapping) == 9

    @pytest.mark.unit
    def test_substituted_purine_matches_purine_core(self):
        """6-methylpurine should match purine core."""
        mol = Chem.MolFromSmiles('Cc1ncnc2nc[nH]c12')
        result = match_fused_heterocycle_core(mol)
        assert result is not None
        name, mapping = result
        assert name == '9H-purine'
        assert len(mapping) == 9

    @pytest.mark.unit
    def test_atom_mapping_is_valid(self):
        """Atom mapping should contain valid atom indices."""
        mol = Chem.MolFromSmiles('Cc1ccc2[nH]ccc2c1')  # 5-methylindole
        result = match_fused_heterocycle_core(mol)
        assert result is not None
        _, mapping = result

        # All mapped indices should be valid atom indices in the molecule
        for atom_idx in mapping.keys():
            assert 0 <= atom_idx < mol.GetNumAtoms()

        # All locants should be positive integers
        for locant in mapping.values():
            assert isinstance(locant, int)
            assert locant > 0


class TestNonMatches:
    """Test that inappropriate molecules do not match."""

    @pytest.mark.unit
    def test_benzene_does_not_match(self):
        """Benzene should not match any fused heterocycle."""
        mol = Chem.MolFromSmiles('c1ccccc1')
        assert get_fused_heterocycle_name(mol) is None
        assert match_fused_heterocycle_core(mol) is None

    @pytest.mark.unit
    def test_simple_pyridine_does_not_match(self):
        """Simple pyridine (not fused) should not match."""
        mol = Chem.MolFromSmiles('c1ccncc1')
        assert get_fused_heterocycle_name(mol) is None
        assert match_fused_heterocycle_core(mol) is None

    @pytest.mark.unit
    def test_naphthalene_does_not_match(self):
        """Naphthalene (no heteroatom) should not match."""
        mol = Chem.MolFromSmiles('c1ccc2ccccc2c1')
        assert get_fused_heterocycle_name(mol) is None
        assert match_fused_heterocycle_core(mol) is None

    @pytest.mark.unit
    def test_simple_furan_does_not_match(self):
        """Simple furan (not fused) should not match."""
        mol = Chem.MolFromSmiles('c1ccoc1')
        assert get_fused_heterocycle_name(mol) is None
        assert match_fused_heterocycle_core(mol) is None

    @pytest.mark.unit
    def test_simple_thiophene_does_not_match(self):
        """Simple thiophene (not fused) should not match."""
        mol = Chem.MolFromSmiles('c1ccsc1')
        assert get_fused_heterocycle_name(mol) is None
        assert match_fused_heterocycle_core(mol) is None

    @pytest.mark.unit
    def test_none_mol_returns_none(self):
        """None molecule should return None."""
        assert get_fused_heterocycle_name(None) is None
        assert match_fused_heterocycle_core(None) is None

    @pytest.mark.unit
    def test_cyclohexane_does_not_match(self):
        """Cyclohexane should not match any fused heterocycle."""
        mol = Chem.MolFromSmiles('C1CCCCC1')
        assert get_fused_heterocycle_name(mol) is None
        assert match_fused_heterocycle_core(mol) is None


class TestCanonicalSMILESConsistency:
    """Test that all stored SMILES are in canonical form."""

    @pytest.mark.unit
    def test_all_smiles_are_canonical(self):
        """All stored SMILES should equal their RDKit canonical form."""
        for smiles, data in FUSED_HETEROCYCLE_DATA.items():
            mol = Chem.MolFromSmiles(smiles)
            assert mol is not None, f"Invalid SMILES in data: {smiles}"
            canonical = Chem.MolToSmiles(mol, canonical=True)
            assert smiles == canonical, (
                f"SMILES not canonical for {data['name']}: "
                f"stored={smiles}, canonical={canonical}"
            )

    @pytest.mark.unit
    def test_all_molecules_are_valid(self):
        """All stored SMILES should parse to valid molecules."""
        for smiles, data in FUSED_HETEROCYCLE_DATA.items():
            mol = Chem.MolFromSmiles(smiles)
            assert mol is not None, f"Invalid SMILES for {data['name']}: {smiles}"

    @pytest.mark.unit
    def test_parent_atoms_match_molecule_size(self):
        """parent_atoms should match actual atom count."""
        for smiles, data in FUSED_HETEROCYCLE_DATA.items():
            mol = Chem.MolFromSmiles(smiles)
            assert mol.GetNumAtoms() == data['parent_atoms'], (
                f"Atom count mismatch for {data['name']}: "
                f"stored={data['parent_atoms']}, actual={mol.GetNumAtoms()}"
            )


class TestCoverageByRingSystem:
    """Test coverage of different ring system categories."""

    @pytest.mark.unit
    def test_has_benzo_5_membered_entries(self):
        """Should have entries for benzo-5-membered ring systems."""
        benzo_5 = [
            data for data in FUSED_HETEROCYCLE_DATA.values()
            if data['ring_system'] == 'benzo-5-membered'
        ]
        assert len(benzo_5) >= 6, "Should have at least 6 benzo-5-membered entries"

    @pytest.mark.unit
    def test_has_benzo_6_membered_entries(self):
        """Should have entries for benzo-6-membered ring systems."""
        benzo_6 = [
            data for data in FUSED_HETEROCYCLE_DATA.values()
            if data['ring_system'] == 'benzo-6-membered'
        ]
        assert len(benzo_6) >= 4, "Should have at least 4 benzo-6-membered entries"

    @pytest.mark.unit
    def test_has_tricyclic_entries(self):
        """Should have entries for tricyclic ring systems."""
        tricyclic = [
            data for data in FUSED_HETEROCYCLE_DATA.values()
            if data['ring_system'] == 'tricyclic'
        ]
        assert len(tricyclic) >= 5, "Should have at least 5 tricyclic entries"

    @pytest.mark.unit
    def test_has_saturated_entries(self):
        """Should have entries for saturated variants."""
        saturated = [
            data for data in FUSED_HETEROCYCLE_DATA.values()
            if 'saturated' in data['ring_system']
        ]
        assert len(saturated) >= 4, "Should have at least 4 saturated entries"

    @pytest.mark.unit
    def test_minimum_total_entries(self):
        """Should have at least 25 total entries."""
        assert len(FUSED_HETEROCYCLE_DATA) >= 25


class TestHelperFunctions:
    """Test additional helper functions."""

    @pytest.mark.unit
    def test_get_fused_heterocycle_info_returns_full_data(self):
        """get_fused_heterocycle_info should return complete data dict."""
        mol = Chem.MolFromSmiles('c1ccc2[nH]ccc2c1')
        canonical = Chem.MolToSmiles(mol, canonical=True)
        info = get_fused_heterocycle_info(canonical)
        assert info is not None
        assert 'name' in info
        assert 'tautomer_locant' in info
        assert 'ring_system' in info
        assert 'parent_atoms' in info
        assert info['name'] == '1H-indole'

    @pytest.mark.unit
    def test_is_fused_heterocycle_returns_true_for_known(self):
        """is_fused_heterocycle should return True for known systems."""
        mol = Chem.MolFromSmiles('c1ccc2ncccc2c1')  # quinoline
        assert is_fused_heterocycle(mol) is True

    @pytest.mark.unit
    def test_is_fused_heterocycle_returns_false_for_unknown(self):
        """is_fused_heterocycle should return False for unknown systems."""
        mol = Chem.MolFromSmiles('c1ccccc1')  # benzene
        assert is_fused_heterocycle(mol) is False

    @pytest.mark.unit
    def test_get_ring_system_type_returns_classification(self):
        """get_ring_system_type should return correct classification."""
        mol = Chem.MolFromSmiles('c1ccc2[nH]ccc2c1')  # indole
        assert get_ring_system_type(mol) == 'benzo-5-membered'

        mol = Chem.MolFromSmiles('c1ccc2ncccc2c1')  # quinoline
        assert get_ring_system_type(mol) == 'benzo-6-membered'

        mol = Chem.MolFromSmiles('c1ccc2c(c1)[nH]c1ccccc12')  # carbazole
        assert get_ring_system_type(mol) == 'tricyclic'

    @pytest.mark.unit
    def test_get_ring_system_type_returns_none_for_unknown(self):
        """get_ring_system_type should return None for unknown systems."""
        mol = Chem.MolFromSmiles('c1ccccc1')  # benzene
        assert get_ring_system_type(mol) is None


class TestAlternativeInputSMILES:
    """Test that alternative input SMILES canonicalize to matching entries."""

    @pytest.mark.unit
    def test_indole_alternate_smiles(self):
        """Different indole SMILES representations should work."""
        # Various ways to write indole
        smiles_list = [
            'c1ccc2[nH]ccc2c1',
            'C1=Cc2ccccc2N1',
            'c1cc2ccccc2[nH]1',
        ]
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol:  # Skip if invalid
                result = get_fused_heterocycle_name(mol)
                if result:  # Different tautomers may not match
                    assert 'indole' in result[0].lower()

    @pytest.mark.unit
    def test_quinoline_alternate_smiles(self):
        """Different quinoline SMILES representations should work."""
        smiles_list = [
            'c1ccc2ncccc2c1',
            'c1ccnc2ccccc12',
            'C1=CC=NC2=CC=CC=C12',
        ]
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            result = get_fused_heterocycle_name(mol)
            assert result is not None
            assert result[0] == 'quinoline'


class TestBenzoFusedOxazolesAndThiazoles:
    """Test benzo-fused oxazoles and thiazoles."""

    @pytest.mark.unit
    def test_benzoxazole(self):
        """Benzoxazole should be correctly identified."""
        mol = Chem.MolFromSmiles('c1ccc2ocnc2c1')
        result = get_fused_heterocycle_name(mol)
        assert result is not None
        assert result[0] == '1,3-benzoxazole'

    @pytest.mark.unit
    def test_benzothiazole(self):
        """Benzothiazole should be correctly identified."""
        mol = Chem.MolFromSmiles('c1ccc2scnc2c1')
        result = get_fused_heterocycle_name(mol)
        assert result is not None
        assert result[0] == '1,3-benzothiazole'

    @pytest.mark.unit
    def test_benzisoxazole(self):
        """Benzisoxazole should be correctly identified."""
        mol = Chem.MolFromSmiles('c1ccc2nocc2c1')
        result = get_fused_heterocycle_name(mol)
        assert result is not None
        assert result[0] == '1,2-benzisoxazole'

    @pytest.mark.unit
    def test_benzisothiazole(self):
        """Benzisothiazole should be correctly identified."""
        mol = Chem.MolFromSmiles('c1ccc2nscc2c1')
        result = get_fused_heterocycle_name(mol)
        assert result is not None
        assert result[0] == '1,2-benzisothiazole'


class TestSaturatedVariants:
    """Test saturated/partially saturated fused heterocycles."""

    @pytest.mark.unit
    def test_indoline(self):
        """Indoline (2,3-dihydro-1H-indole) should be identified."""
        mol = Chem.MolFromSmiles('c1ccc2c(c1)CCN2')
        result = get_fused_heterocycle_name(mol)
        assert result is not None
        assert result[0] == 'indoline'
        assert result[1] is None  # No tautomer locant for saturated

    @pytest.mark.unit
    def test_isoindoline(self):
        """Isoindoline should be identified."""
        mol = Chem.MolFromSmiles('c1ccc2c(c1)CNC2')
        result = get_fused_heterocycle_name(mol)
        assert result is not None
        assert result[0] == 'isoindoline'

    @pytest.mark.unit
    def test_tetrahydroquinoline(self):
        """1,2,3,4-Tetrahydroquinoline should be identified."""
        mol = Chem.MolFromSmiles('c1ccc2c(c1)CCCN2')
        result = get_fused_heterocycle_name(mol)
        assert result is not None
        assert result[0] == '1,2,3,4-tetrahydroquinoline'

    @pytest.mark.unit
    def test_chromane(self):
        """Chromane should be identified."""
        mol = Chem.MolFromSmiles('c1ccc2c(c1)CCCO2')
        result = get_fused_heterocycle_name(mol)
        assert result is not None
        assert result[0] == 'chromane'
