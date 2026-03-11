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
        name, mapping, core_smiles = result
        assert name == '1H-indole'
        assert isinstance(mapping, dict)
        assert len(mapping) == 9  # 9 atoms in indole core

    @pytest.mark.unit
    def test_4_chloroquinoline_matches_quinoline_core(self):
        """4-chloroquinoline should match quinoline core."""
        mol = Chem.MolFromSmiles('Clc1ccnc2ccccc12')
        result = match_fused_heterocycle_core(mol)
        assert result is not None
        name, mapping, core_smiles = result
        assert name == 'quinoline'
        assert len(mapping) == 10  # 10 atoms in quinoline core

    @pytest.mark.unit
    def test_2_methylbenzimidazole_matches_benzimidazole_core(self):
        """2-methylbenzimidazole should match benzimidazole core."""
        mol = Chem.MolFromSmiles('Cc1nc2ccccc2[nH]1')
        result = match_fused_heterocycle_core(mol)
        assert result is not None
        name, mapping, core_smiles = result
        assert name == '1H-benzimidazole'
        assert len(mapping) == 9

    @pytest.mark.unit
    def test_substituted_purine_matches_purine_core(self):
        """6-methylpurine should match purine core."""
        mol = Chem.MolFromSmiles('Cc1ncnc2nc[nH]c12')
        result = match_fused_heterocycle_core(mol)
        assert result is not None
        name, mapping, core_smiles = result
        assert name == '9H-purine'
        assert len(mapping) == 9

    @pytest.mark.unit
    def test_atom_mapping_is_valid(self):
        """Atom mapping should contain valid atom indices and locants."""
        mol = Chem.MolFromSmiles('Cc1ccc2[nH]ccc2c1')  # 5-methylindole
        result = match_fused_heterocycle_core(mol)
        assert result is not None
        _, mapping, _ = result

        # All mapped indices should be valid atom indices in the molecule
        for atom_idx in mapping.keys():
            assert 0 <= atom_idx < mol.GetNumAtoms()

        # All locants should be positive integers or fusion locants like '3a'
        for locant in mapping.values():
            if isinstance(locant, int):
                assert locant > 0
            elif isinstance(locant, str):
                # Fusion locants like '3a', '7a', '4a', '8a'
                assert locant.endswith('a')
                assert locant[:-1].isdigit()


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


class TestIUPACLocantMappings:
    """Test IUPAC peripheral locant mappings for fused heterocycles."""

    @pytest.mark.unit
    def test_indole_locant_mapping(self):
        """Verify indole IUPAC locant mapping is correct.

        Indole IUPAC numbering:
        - Position 1: N (with H)
        - Positions 2,3: 5-ring carbons
        - Position 3a: fusion atom
        - Positions 4-7: benzene ring carbons
        - Position 7a: fusion atom
        """
        from src.orthonym.data.fused_heterocycles import FUSED_HETEROCYCLE_DATA

        indole_data = FUSED_HETEROCYCLE_DATA['c1ccc2[nH]ccc2c1']
        locants = indole_data['iupac_locants']

        # Verify N is at position 1
        mol = Chem.MolFromSmiles('c1ccc2[nH]ccc2c1')
        for idx, locant in locants.items():
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetSymbol() == 'N':
                assert locant == 1, f"N should be at position 1, got {locant}"

        # Verify fusion atoms have 'a' suffix locants
        fusion_locants = [loc for loc in locants.values() if isinstance(loc, str)]
        assert '3a' in fusion_locants
        assert '7a' in fusion_locants

        # Verify 9 total positions mapped
        assert len(locants) == 9

    @pytest.mark.unit
    def test_purine_locant_mapping(self):
        """Verify purine IUPAC locant mapping is correct.

        Purine IUPAC numbering (9H-purine):
        - N1, C2, N3: pyrimidine nitrogens and carbon
        - C4, C5: fusion carbons (shared)
        - C6: pyrimidine carbon
        - N7, C8, N9: imidazole atoms (N9 has H)
        """
        from src.orthonym.data.fused_heterocycles import FUSED_HETEROCYCLE_DATA

        purine_data = FUSED_HETEROCYCLE_DATA['c1ncc2nc[nH]c2n1']
        locants = purine_data['iupac_locants']

        # Verify 9 total positions mapped
        assert len(locants) == 9

        # Check that position 9 exists (for N9-H)
        assert 9 in locants.values()

        # Verify N9 has H (the tautomer position)
        assert purine_data['tautomer_locant'] == 9

    @pytest.mark.unit
    def test_quinoline_locant_mapping(self):
        """Verify quinoline IUPAC locant mapping is correct.

        Quinoline IUPAC numbering:
        - Position 1: N
        - Positions 2,3,4: pyridine ring
        - Position 4a: fusion
        - Positions 5,6,7,8: benzene ring
        - Position 8a: fusion
        """
        from src.orthonym.data.fused_heterocycles import FUSED_HETEROCYCLE_DATA

        quinoline_data = FUSED_HETEROCYCLE_DATA['c1ccc2ncccc2c1']
        locants = quinoline_data['iupac_locants']

        # Verify N is at position 1
        mol = Chem.MolFromSmiles('c1ccc2ncccc2c1')
        for idx, locant in locants.items():
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetSymbol() == 'N':
                assert locant == 1, f"N should be at position 1, got {locant}"

        # Verify fusion atoms have 'a' suffix locants
        fusion_locants = [loc for loc in locants.values() if isinstance(loc, str)]
        assert '4a' in fusion_locants
        assert '8a' in fusion_locants

        # Verify 10 total positions mapped
        assert len(locants) == 10

    @pytest.mark.unit
    def test_substituted_indole_locant(self):
        """5-methylindole substituent should be at IUPAC position 5."""
        mol = Chem.MolFromSmiles('Cc1ccc2[nH]ccc2c1')  # 5-methylindole
        result = match_fused_heterocycle_core(mol)
        assert result is not None

        name, mapping, core_smiles = result
        assert name == '1H-indole'

        # Find the methyl carbon and its attachment point
        methyl_attached_to = None
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'C' and atom.GetDegree() == 1:  # Methyl carbon
                # Get the ring carbon it's attached to
                neighbor = list(atom.GetNeighbors())[0]
                methyl_attached_to = neighbor.GetIdx()
                break

        assert methyl_attached_to is not None
        assert mapping[methyl_attached_to] == 5, (
            f"Methyl should attach at IUPAC position 5, got {mapping[methyl_attached_to]}"
        )

    @pytest.mark.unit
    def test_6_aminopurine_locant(self):
        """6-aminopurine (adenine) amino group should be at IUPAC position 6.

        Note: When adenine is in FUSED_HETEROCYCLE_DATA as a retained name,
        it matches as 'adenine' rather than '9H-purine'. Both are valid.
        """
        mol = Chem.MolFromSmiles('Nc1ncnc2[nH]cnc12')  # adenine
        result = match_fused_heterocycle_core(mol)
        assert result is not None

        name, mapping, core_smiles = result
        # Accept either 'adenine' (retained name) or '9H-purine' (parent core)
        assert name in ('adenine', '9H-purine'), f"Expected adenine or 9H-purine, got {name}"

        # Find the amino nitrogen and its attachment point
        amino_attached_to = None
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'N' and not atom.GetIsAromatic():  # Amino N
                neighbor = list(atom.GetNeighbors())[0]
                amino_attached_to = neighbor.GetIdx()
                break

        assert amino_attached_to is not None
        # For adenine as retained name, the amino N is mapped to 'N6'
        # For purine core match, the attachment carbon maps to 6
        expected_locant = mapping.get(amino_attached_to)
        assert expected_locant in (6, 'N6'), (
            f"Amino should attach at IUPAC position 6 (or N6 for retained name), got {expected_locant}"
        )

    @pytest.mark.unit
    def test_8_chloroquinoline_locant(self):
        """8-chloroquinoline chloro group should be at IUPAC position 8."""
        mol = Chem.MolFromSmiles('Clc1cccc2cccnc12')  # 8-chloroquinoline
        result = match_fused_heterocycle_core(mol)
        assert result is not None

        name, mapping, core_smiles = result
        assert name == 'quinoline'

        # Find the chlorine and its attachment point
        cl_attached_to = None
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'Cl':
                neighbor = list(atom.GetNeighbors())[0]
                cl_attached_to = neighbor.GetIdx()
                break

        assert cl_attached_to is not None
        assert mapping[cl_attached_to] == 8, (
            f"Chloro should attach at IUPAC position 8, got {mapping[cl_attached_to]}"
        )

    @pytest.mark.unit
    def test_all_entries_have_iupac_locants(self):
        """All FUSED_HETEROCYCLE_DATA entries should have iupac_locants field."""
        from src.orthonym.data.fused_heterocycles import FUSED_HETEROCYCLE_DATA

        missing = []
        for smiles, data in FUSED_HETEROCYCLE_DATA.items():
            if 'iupac_locants' not in data:
                missing.append(data['name'])

        assert not missing, f"Missing iupac_locants for: {missing}"

        # Verify minimum count - expanded to 50+ entries in 07-04 plan
        assert len(FUSED_HETEROCYCLE_DATA) >= 50, (
            f"Expected at least 50 entries, got {len(FUSED_HETEROCYCLE_DATA)}"
        )

    @pytest.mark.unit
    def test_iupac_locants_mapping_size_matches_parent_atoms(self):
        """Each iupac_locants mapping should have same size as parent_atoms."""
        from src.orthonym.data.fused_heterocycles import FUSED_HETEROCYCLE_DATA

        mismatches = []
        for smiles, data in FUSED_HETEROCYCLE_DATA.items():
            locants = data.get('iupac_locants', {})
            parent_atoms = data['parent_atoms']

            if len(locants) != parent_atoms:
                mismatches.append(
                    f"{data['name']}: locants={len(locants)}, parent_atoms={parent_atoms}"
                )

        assert not mismatches, f"Mapping size mismatches:\n" + "\n".join(mismatches)


class TestChromeneVariants:
    """Tests for chromene/coumarin data entries (BUG-4 fix)."""

    @pytest.mark.unit
    def test_coumarin_in_data(self):
        """Coumarin should be in FUSED_HETEROCYCLE_DATA."""
        from rdkit import Chem
        from src.orthonym.data.fused_heterocycles import FUSED_HETEROCYCLE_DATA

        mol = Chem.MolFromSmiles('O=c1ccc2ccccc2o1')
        canonical = Chem.MolToSmiles(mol)
        assert canonical in FUSED_HETEROCYCLE_DATA, f"Coumarin ({canonical}) not in data"
        assert FUSED_HETEROCYCLE_DATA[canonical]['name'] == 'coumarin'

    @pytest.mark.unit
    def test_coumarin_systematic_name(self):
        """Coumarin should have correct systematic name."""
        from src.orthonym.data.fused_heterocycles import FUSED_HETEROCYCLE_DATA

        coumarin_data = FUSED_HETEROCYCLE_DATA['O=c1ccc2ccccc2o1']
        assert coumarin_data['systematic'] == '2H-chromen-2-one'
        assert coumarin_data['tautomer_locant'] == 2

    @pytest.mark.unit
    def test_coumarin_e2e(self):
        """O=c1ccc2ccccc2o1 should return coumarin or chromen-2-one."""
        from src.orthonym import name_compound
        result = name_compound('O=c1ccc2ccccc2o1')
        result_lower = result.lower()
        assert 'coumarin' in result_lower or 'chromen' in result_lower, f"Got {result}"

    @pytest.mark.unit
    def test_dihydrobenzofuran_in_data(self):
        """2,3-dihydro-1-benzofuran should be in FUSED_HETEROCYCLE_DATA."""
        from rdkit import Chem
        from src.orthonym.data.fused_heterocycles import FUSED_HETEROCYCLE_DATA

        mol = Chem.MolFromSmiles('C1Cc2ccccc2O1')
        canonical = Chem.MolToSmiles(mol)
        assert canonical in FUSED_HETEROCYCLE_DATA, f"Dihydrobenzofuran ({canonical}) not in data"
        data = FUSED_HETEROCYCLE_DATA[canonical]
        assert data['name'] == '2,3-dihydro-1-benzofuran'

    @pytest.mark.unit
    def test_dihydrobenzofuran_e2e(self):
        """C1Cc2ccccc2O1 should return 2,3-dihydro-1-benzofuran."""
        from src.orthonym import name_compound
        result = name_compound('C1Cc2ccccc2O1')
        result_lower = result.lower()
        assert 'benzofuran' in result_lower or 'dihydro' in result_lower, f"Got {result}"

    @pytest.mark.unit
    def test_chromane_already_exists(self):
        """Chromane (3,4-dihydro-2H-chromene) should already be in data."""
        from rdkit import Chem
        from src.orthonym.data.fused_heterocycles import FUSED_HETEROCYCLE_DATA

        mol = Chem.MolFromSmiles('c1ccc2c(c1)CCCO2')
        canonical = Chem.MolToSmiles(mol)
        assert canonical in FUSED_HETEROCYCLE_DATA, f"Chromane ({canonical}) not in data"
        assert FUSED_HETEROCYCLE_DATA[canonical]['name'] == 'chromane'

    @pytest.mark.unit
    def test_isochromane_already_exists(self):
        """Isochromane should already be in data."""
        from rdkit import Chem
        from src.orthonym.data.fused_heterocycles import FUSED_HETEROCYCLE_DATA

        mol = Chem.MolFromSmiles('c1ccc2c(c1)CCOC2')
        canonical = Chem.MolToSmiles(mol)
        assert canonical in FUSED_HETEROCYCLE_DATA, f"Isochromane ({canonical}) not in data"
        assert FUSED_HETEROCYCLE_DATA[canonical]['name'] == 'isochromane'

    @pytest.mark.unit
    def test_2h_chromene_exists(self):
        """2H-chromene should be in data."""
        from src.orthonym.data.fused_heterocycles import FUSED_HETEROCYCLE_DATA

        # 2H-chromene canonical SMILES
        assert 'C1=Cc2ccccc2OC1' in FUSED_HETEROCYCLE_DATA
        assert FUSED_HETEROCYCLE_DATA['C1=Cc2ccccc2OC1']['name'] == '2H-chromene'

    @pytest.mark.unit
    def test_coumarin_ring_system_type(self):
        """Coumarin should have lactone ring system classification."""
        from src.orthonym.data.fused_heterocycles import FUSED_HETEROCYCLE_DATA

        coumarin_data = FUSED_HETEROCYCLE_DATA['O=c1ccc2ccccc2o1']
        assert coumarin_data['ring_system'] == 'benzo-6-membered-lactone'
        assert coumarin_data['parent_atoms'] == 11  # Includes =O

    @pytest.mark.unit
    def test_dihydrobenzofuran_ring_system_type(self):
        """2,3-dihydro-1-benzofuran should have saturated ring system."""
        from src.orthonym.data.fused_heterocycles import FUSED_HETEROCYCLE_DATA

        data = FUSED_HETEROCYCLE_DATA['c1ccc2c(c1)CCO2']
        assert data['ring_system'] == 'benzo-5-membered-saturated'
        assert data['parent_atoms'] == 9


class TestPhase101FixedEntries:
    """Tests for entries fixed in Phase 101-01 (corrected SMILES keys)."""

    @pytest.mark.unit
    def test_phenanthridine_exact_match(self):
        """Phenanthridine with corrected SMILES should return correct name."""
        mol = Chem.MolFromSmiles('c1ccc2c(c1)cnc1ccccc12')
        result = get_fused_heterocycle_name(mol)
        assert result is not None
        assert result[0] == 'phenanthridine'
        assert result[1] is None

    @pytest.mark.unit
    def test_phenanthridine_canonical_smiles_in_dict(self):
        """Phenanthridine SMILES key must match RDKit canonical form."""
        smi = 'c1ccc2c(c1)cnc1ccccc12'
        mol = Chem.MolFromSmiles(smi)
        canon = Chem.MolToSmiles(mol)
        assert canon == smi
        assert smi in FUSED_HETEROCYCLE_DATA

    @pytest.mark.unit
    def test_phenanthridine_locant_nitrogen_at_5(self):
        """Phenanthridine N should be at IUPAC position 5."""
        mol = Chem.MolFromSmiles('c1ccc2c(c1)cnc1ccccc12')
        data = FUSED_HETEROCYCLE_DATA['c1ccc2c(c1)cnc1ccccc12']
        locants = data['iupac_locants']
        for idx, locant in locants.items():
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetSymbol() == 'N':
                assert locant == 5, f"N should be at position 5, got {locant}"

    @pytest.mark.unit
    def test_phenanthridine_has_14_atoms(self):
        """Phenanthridine should have 14 parent atoms."""
        data = FUSED_HETEROCYCLE_DATA['c1ccc2c(c1)cnc1ccccc12']
        assert data['parent_atoms'] == 14
        assert len(data['iupac_locants']) == 14

    @pytest.mark.unit
    def test_phenanthridine_fusion_locants(self):
        """Phenanthridine should have junction locants 4a, 4b, 8a, 10a."""
        data = FUSED_HETEROCYCLE_DATA['c1ccc2c(c1)cnc1ccccc12']
        locant_values = set(data['iupac_locants'].values())
        for junction in ['4a', '4b', '8a', '10a']:
            assert junction in locant_values, f"Missing junction locant {junction}"

    @pytest.mark.unit
    def test_phenanthridine_old_smiles_not_in_dict(self):
        """The old broken SMILES key should NOT be in the dictionary."""
        assert 'c1ccc2c(c1)ccc1cccnc12' not in FUSED_HETEROCYCLE_DATA

    @pytest.mark.unit
    def test_4h_quinolizine_exact_match(self):
        """4H-quinolizine with corrected SMILES should return correct name."""
        mol = Chem.MolFromSmiles('C1=CCN2C=CC=CC2=C1')
        result = get_fused_heterocycle_name(mol)
        assert result is not None
        assert result[0] == '4H-quinolizine'
        assert result[1] == 4

    @pytest.mark.unit
    def test_4h_quinolizine_canonical_smiles_in_dict(self):
        """4H-quinolizine SMILES key must match RDKit canonical form."""
        smi = 'C1=CCN2C=CC=CC2=C1'
        mol = Chem.MolFromSmiles(smi)
        canon = Chem.MolToSmiles(mol)
        assert canon == smi
        assert smi in FUSED_HETEROCYCLE_DATA

    @pytest.mark.unit
    def test_4h_quinolizine_bridgehead_type(self):
        """4H-quinolizine should be classified as bridgehead."""
        data = FUSED_HETEROCYCLE_DATA['C1=CCN2C=CC=CC2=C1']
        assert data['ring_system'] == 'bridgehead'

    @pytest.mark.unit
    def test_4h_quinolizine_old_smiles_not_in_dict(self):
        """The old broken 4H-quinolizine SMILES key should NOT be in the dictionary."""
        assert 'C1=CC2=CCC=CN2C=C1' not in FUSED_HETEROCYCLE_DATA


class TestPhase101NewEntries:
    """Tests for new entries added in Phase 101-01."""

    # --- Xanthone ---

    @pytest.mark.unit
    def test_xanthone_exact_match(self):
        """Xanthone should return correct retained name."""
        mol = Chem.MolFromSmiles('O=c1c2ccccc2oc2ccccc12')
        result = get_fused_heterocycle_name(mol)
        assert result is not None
        assert result[0] == 'xanthone'

    @pytest.mark.unit
    def test_xanthone_canonical_smiles_in_dict(self):
        """Xanthone SMILES key must match RDKit canonical form."""
        smi = 'O=c1c2ccccc2oc2ccccc12'
        mol = Chem.MolFromSmiles(smi)
        canon = Chem.MolToSmiles(mol)
        assert canon == smi
        assert smi in FUSED_HETEROCYCLE_DATA

    @pytest.mark.unit
    def test_xanthone_is_retained_name(self):
        """Xanthone should be marked as a retained name."""
        data = FUSED_HETEROCYCLE_DATA['O=c1c2ccccc2oc2ccccc12']
        assert data.get('is_retained_name') is True

    @pytest.mark.unit
    def test_xanthone_has_15_atoms(self):
        """Xanthone should have 15 parent atoms (includes =O)."""
        data = FUSED_HETEROCYCLE_DATA['O=c1c2ccccc2oc2ccccc12']
        assert data['parent_atoms'] == 15

    @pytest.mark.unit
    def test_xanthone_has_exocyclic_oxygen(self):
        """Xanthone locant map should include '=O' for exocyclic oxygen."""
        data = FUSED_HETEROCYCLE_DATA['O=c1c2ccccc2oc2ccccc12']
        assert '=O' in data['iupac_locants'].values()

    # --- Thioxanthone ---

    @pytest.mark.unit
    def test_thioxanthone_exact_match(self):
        """Thioxanthone should return correct retained name."""
        mol = Chem.MolFromSmiles('O=c1c2ccccc2sc2ccccc12')
        result = get_fused_heterocycle_name(mol)
        assert result is not None
        assert result[0] == 'thioxanthone'

    @pytest.mark.unit
    def test_thioxanthone_canonical_smiles_in_dict(self):
        """Thioxanthone SMILES key must match RDKit canonical form."""
        smi = 'O=c1c2ccccc2sc2ccccc12'
        mol = Chem.MolFromSmiles(smi)
        canon = Chem.MolToSmiles(mol)
        assert canon == smi
        assert smi in FUSED_HETEROCYCLE_DATA

    @pytest.mark.unit
    def test_thioxanthone_is_retained_name(self):
        """Thioxanthone should be marked as a retained name."""
        data = FUSED_HETEROCYCLE_DATA['O=c1c2ccccc2sc2ccccc12']
        assert data.get('is_retained_name') is True

    # --- 1,10-Phenanthroline ---

    @pytest.mark.unit
    def test_phenanthroline_exact_match(self):
        """1,10-phenanthroline should return correct name."""
        mol = Chem.MolFromSmiles('c1cnc2c(c1)ccc1cccnc12')
        result = get_fused_heterocycle_name(mol)
        assert result is not None
        assert result[0] == '1,10-phenanthroline'
        assert result[1] is None

    @pytest.mark.unit
    def test_phenanthroline_canonical_smiles_in_dict(self):
        """1,10-phenanthroline SMILES key must match RDKit canonical form."""
        smi = 'c1cnc2c(c1)ccc1cccnc12'
        mol = Chem.MolFromSmiles(smi)
        canon = Chem.MolToSmiles(mol)
        assert canon == smi
        assert smi in FUSED_HETEROCYCLE_DATA

    @pytest.mark.unit
    def test_phenanthroline_has_two_nitrogens(self):
        """1,10-phenanthroline should have N at positions 1 and 10."""
        mol = Chem.MolFromSmiles('c1cnc2c(c1)ccc1cccnc12')
        data = FUSED_HETEROCYCLE_DATA['c1cnc2c(c1)ccc1cccnc12']
        locants = data['iupac_locants']
        n_positions = []
        for idx, locant in locants.items():
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetSymbol() == 'N':
                n_positions.append(locant)
        assert sorted(n_positions) == [1, 10], f"N positions: {n_positions}"

    @pytest.mark.unit
    def test_phenanthroline_has_14_atoms(self):
        """1,10-phenanthroline should have 14 parent atoms."""
        data = FUSED_HETEROCYCLE_DATA['c1cnc2c(c1)ccc1cccnc12']
        assert data['parent_atoms'] == 14

    # --- 1H-Perimidine ---

    @pytest.mark.unit
    def test_perimidine_exact_match(self):
        """1H-perimidine should return correct name."""
        mol = Chem.MolFromSmiles('C1=Nc2cccc3cccc(c23)N1')
        result = get_fused_heterocycle_name(mol)
        assert result is not None
        assert result[0] == '1H-perimidine'
        assert result[1] == 1

    @pytest.mark.unit
    def test_perimidine_canonical_smiles_in_dict(self):
        """1H-perimidine SMILES key must match RDKit canonical form."""
        smi = 'C1=Nc2cccc3cccc(c23)N1'
        mol = Chem.MolFromSmiles(smi)
        canon = Chem.MolToSmiles(mol)
        assert canon == smi
        assert smi in FUSED_HETEROCYCLE_DATA

    @pytest.mark.unit
    def test_perimidine_has_13_atoms(self):
        """1H-perimidine should have 13 parent atoms."""
        data = FUSED_HETEROCYCLE_DATA['C1=Nc2cccc3cccc(c23)N1']
        assert data['parent_atoms'] == 13

    @pytest.mark.unit
    def test_perimidine_has_peri_junction(self):
        """1H-perimidine should have 9b peri-junction locant."""
        data = FUSED_HETEROCYCLE_DATA['C1=Nc2cccc3cccc(c23)N1']
        assert '9b' in data['iupac_locants'].values()

    # --- 9H-Thioxanthene ---

    @pytest.mark.unit
    def test_thioxanthene_exact_match(self):
        """9H-thioxanthene should return correct name."""
        mol = Chem.MolFromSmiles('c1ccc2c(c1)Cc1ccccc1S2')
        result = get_fused_heterocycle_name(mol)
        assert result is not None
        assert result[0] == '9H-thioxanthene'
        assert result[1] == 9

    @pytest.mark.unit
    def test_thioxanthene_canonical_smiles_in_dict(self):
        """9H-thioxanthene SMILES key must match RDKit canonical form."""
        smi = 'c1ccc2c(c1)Cc1ccccc1S2'
        mol = Chem.MolFromSmiles(smi)
        canon = Chem.MolToSmiles(mol)
        assert canon == smi
        assert smi in FUSED_HETEROCYCLE_DATA

    @pytest.mark.unit
    def test_thioxanthene_has_14_atoms(self):
        """9H-thioxanthene should have 14 parent atoms."""
        data = FUSED_HETEROCYCLE_DATA['c1ccc2c(c1)Cc1ccccc1S2']
        assert data['parent_atoms'] == 14

    @pytest.mark.unit
    def test_thioxanthene_has_sulfur_mapped(self):
        """9H-thioxanthene should have S mapped at same position as xanthene O."""
        mol = Chem.MolFromSmiles('c1ccc2c(c1)Cc1ccccc1S2')
        data = FUSED_HETEROCYCLE_DATA['c1ccc2c(c1)Cc1ccccc1S2']
        locants = data['iupac_locants']
        # Thioxanthene uses same locant mapping as xanthene (S replaces O)
        xanthene_data = FUSED_HETEROCYCLE_DATA['c1ccc2c(c1)Cc1ccccc1O2']
        s_locant = None
        o_locant = None
        for idx, locant in locants.items():
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetSymbol() == 'S':
                s_locant = locant
        xanthene_mol = Chem.MolFromSmiles('c1ccc2c(c1)Cc1ccccc1O2')
        for idx, locant in xanthene_data['iupac_locants'].items():
            atom = xanthene_mol.GetAtomWithIdx(idx)
            if atom.GetSymbol() == 'O':
                o_locant = locant
        assert s_locant == o_locant, (
            f"S locant ({s_locant}) should match xanthene O locant ({o_locant})"
        )

    # --- Pyrrolizine ---

    @pytest.mark.unit
    def test_pyrrolizine_exact_match(self):
        """Pyrrolizine should return correct name."""
        mol = Chem.MolFromSmiles('C1=Cn2cccc2C1')
        result = get_fused_heterocycle_name(mol)
        assert result is not None
        assert result[0] == 'pyrrolizine'
        assert result[1] is None

    @pytest.mark.unit
    def test_pyrrolizine_canonical_smiles_in_dict(self):
        """Pyrrolizine SMILES key must match RDKit canonical form."""
        smi = 'C1=Cn2cccc2C1'
        mol = Chem.MolFromSmiles(smi)
        canon = Chem.MolToSmiles(mol)
        assert canon == smi
        assert smi in FUSED_HETEROCYCLE_DATA

    @pytest.mark.unit
    def test_pyrrolizine_bridgehead_type(self):
        """Pyrrolizine should be classified as bridgehead."""
        data = FUSED_HETEROCYCLE_DATA['C1=Cn2cccc2C1']
        assert data['ring_system'] == 'bridgehead'

    @pytest.mark.unit
    def test_pyrrolizine_has_8_atoms(self):
        """Pyrrolizine should have 8 parent atoms."""
        data = FUSED_HETEROCYCLE_DATA['C1=Cn2cccc2C1']
        assert data['parent_atoms'] == 8

    @pytest.mark.unit
    def test_pyrrolizine_has_bridgehead_junctions(self):
        """Pyrrolizine should have junction locants 3a and 7a."""
        data = FUSED_HETEROCYCLE_DATA['C1=Cn2cccc2C1']
        locant_values = set(data['iupac_locants'].values())
        assert '3a' in locant_values, "Missing junction locant 3a"
        assert '7a' in locant_values, "Missing junction locant 7a"


class TestPhase101EndToEndNaming:
    """End-to-end naming tests for Phase 101 fixed/new entries."""

    @pytest.mark.unit
    def test_phenanthridine_e2e(self):
        """Phenanthridine SMILES should produce 'phenanthridine'."""
        from orthonym import name_compound
        result = name_compound('c1ccc2c(c1)cnc1ccccc12')
        assert result == 'phenanthridine'

    @pytest.mark.unit
    def test_4h_quinolizine_e2e(self):
        """4H-quinolizine SMILES should produce '4H-quinolizine'."""
        from orthonym import name_compound
        result = name_compound('C1=CCN2C=CC=CC2=C1')
        assert result == '4H-quinolizine'

    @pytest.mark.unit
    def test_xanthone_e2e(self):
        """Xanthone SMILES should produce 'xanthone'."""
        from orthonym import name_compound
        result = name_compound('O=c1c2ccccc2oc2ccccc12')
        assert result == 'xanthone'

    @pytest.mark.unit
    def test_thioxanthone_e2e(self):
        """Thioxanthone SMILES should produce 'thioxanthone'."""
        from orthonym import name_compound
        result = name_compound('O=c1c2ccccc2sc2ccccc12')
        assert result == 'thioxanthone'

    @pytest.mark.unit
    def test_phenanthroline_e2e(self):
        """1,10-phenanthroline SMILES should produce '1,10-phenanthroline'."""
        from orthonym import name_compound
        result = name_compound('c1cnc2c(c1)ccc1cccnc12')
        assert result == '1,10-phenanthroline'

    @pytest.mark.unit
    def test_perimidine_e2e(self):
        """1H-perimidine SMILES should produce '1H-perimidine'."""
        from orthonym import name_compound
        result = name_compound('C1=Nc2cccc3cccc(c23)N1')
        assert result == '1H-perimidine'

    @pytest.mark.unit
    def test_thioxanthene_e2e(self):
        """9H-thioxanthene SMILES should produce '9H-thioxanthene'."""
        from orthonym import name_compound
        result = name_compound('c1ccc2c(c1)Cc1ccccc1S2')
        assert result == '9H-thioxanthene'

    @pytest.mark.unit
    def test_pyrrolizine_e2e(self):
        """Pyrrolizine SMILES should produce 'pyrrolizine'."""
        from orthonym import name_compound
        result = name_compound('C1=Cn2cccc2C1')
        assert result == 'pyrrolizine'

    @pytest.mark.unit
    def test_6_methylphenanthridine_e2e(self):
        """6-methylphenanthridine should have correct locant from naming."""
        from orthonym import name_compound
        result = name_compound('Cc1nc2ccccc2c2ccccc12')
        assert '6-methyl' in result
        assert 'phenanthridine' in result

    @pytest.mark.unit
    def test_2_methylpyrrolizine_e2e(self):
        """2-methylpyrrolizine should have correct locant from naming."""
        from orthonym import name_compound
        result = name_compound('CC1=Cn2cccc2C1')
        assert '2-methyl' in result
        assert 'pyrrolizine' in result


class TestPhase101ComprehensiveCanonicalConsistency:
    """Comprehensive test that ALL dictionary entries have canonical SMILES keys."""

    @pytest.mark.unit
    def test_all_entries_have_canonical_smiles_keys(self):
        """Every SMILES key in FUSED_HETEROCYCLE_DATA must equal its RDKit canonical form."""
        non_canonical = []
        for smi, data in FUSED_HETEROCYCLE_DATA.items():
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                non_canonical.append(f"{data['name']}: INVALID SMILES {smi}")
                continue
            canon = Chem.MolToSmiles(mol)
            if canon != smi:
                non_canonical.append(
                    f"{data['name']}: key={smi}, canonical={canon}"
                )
        assert not non_canonical, (
            f"Non-canonical SMILES keys found:\n" + "\n".join(non_canonical)
        )

    @pytest.mark.unit
    def test_all_entries_have_required_fields(self):
        """Every entry must have name, tautomer_locant, ring_system, parent_atoms, iupac_locants."""
        required = ['name', 'tautomer_locant', 'ring_system', 'parent_atoms', 'iupac_locants']
        missing_fields = []
        for smi, data in FUSED_HETEROCYCLE_DATA.items():
            for field in required:
                if field not in data:
                    missing_fields.append(f"{data.get('name', smi)}: missing {field}")
        assert not missing_fields, (
            f"Missing required fields:\n" + "\n".join(missing_fields)
        )

    @pytest.mark.unit
    def test_no_duplicate_names(self):
        """No two entries should have the same name."""
        names = [data['name'] for data in FUSED_HETEROCYCLE_DATA.values()]
        duplicates = [n for n in names if names.count(n) > 1]
        assert not duplicates, f"Duplicate names found: {set(duplicates)}"

    @pytest.mark.unit
    def test_dictionary_has_at_least_80_entries(self):
        """Dictionary should have at least 80 entries after Phase 101 additions."""
        assert len(FUSED_HETEROCYCLE_DATA) >= 80, (
            f"Expected >= 80 entries, got {len(FUSED_HETEROCYCLE_DATA)}"
        )
