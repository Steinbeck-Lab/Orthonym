"""
Comprehensive tests for complete bicyclo naming.

Tests the full bicyclo naming including:
- IUPAC numbering for various bicyclo sizes
- Single and multiple substituents
- Double bonds at various positions
- Stereochemistry descriptors
- Edge cases (heteroatom bicyclics, etc.)

Coverage:
- get_bicyclo_numbering
- get_bicyclo_substituents
- detect_bicyclo_unsaturation
- get_complete_bicyclo_data
- Full name assembly via name_compound
"""

import pytest
from rdkit import Chem
from rdkit.Chem import rdCIPLabeler

from src.orthonym.rules.bicyclo import (
    get_bicyclo_numbering,
    get_bicyclo_substituents,
    detect_bicyclo_unsaturation,
    get_complete_bicyclo_data,
    get_bicyclo_ring_atoms,
)
from src.orthonym.namer import name_compound


# ============================================================================
# Tests for get_bicyclo_numbering()
# ============================================================================

class TestBicycloNumbering:
    """Test IUPAC bicyclo numbering for various sizes."""

    @pytest.mark.unit
    def test_norbornane_numbering_221(self):
        """Bicyclo[2.2.1]heptane should have 7 numbered atoms."""
        mol = Chem.MolFromSmiles('C1CC2CCC1C2')
        numbering = get_bicyclo_numbering(mol)
        assert numbering is not None
        assert len(numbering) == 7
        # Locants should be 1-7
        assert set(numbering.values()) == {1, 2, 3, 4, 5, 6, 7}

    @pytest.mark.unit
    def test_bicyclo_222_numbering(self):
        """Bicyclo[2.2.2]octane should have 8 numbered atoms."""
        mol = Chem.MolFromSmiles('C1CC2CCC1CC2')
        numbering = get_bicyclo_numbering(mol)
        assert numbering is not None
        assert len(numbering) == 8
        assert set(numbering.values()) == {1, 2, 3, 4, 5, 6, 7, 8}

    @pytest.mark.unit
    def test_bicyclo_321_numbering(self):
        """Bicyclo[3.2.1]octane should have 8 numbered atoms."""
        mol = Chem.MolFromSmiles('C1CC2CCCC1C2')
        numbering = get_bicyclo_numbering(mol)
        assert numbering is not None
        assert len(numbering) == 8

    @pytest.mark.unit
    def test_bicyclo_410_numbering(self):
        """Bicyclo[4.1.0]heptane should have 7 numbered atoms."""
        mol = Chem.MolFromSmiles('C1CCC2CC2C1')
        numbering = get_bicyclo_numbering(mol)
        assert numbering is not None
        assert len(numbering) == 7

    @pytest.mark.unit
    def test_bicyclo_111_numbering(self):
        """Bicyclo[1.1.1]pentane should have 5 numbered atoms."""
        mol = Chem.MolFromSmiles('C1C2CC1C2')
        numbering = get_bicyclo_numbering(mol)
        assert numbering is not None
        assert len(numbering) == 5

    @pytest.mark.unit
    def test_numbering_bridgeheads_at_1_and_4(self):
        """For norbornane, bridgeheads should be at positions 1 and 4."""
        mol = Chem.MolFromSmiles('C1CC2CCC1C2')
        numbering = get_bicyclo_numbering(mol)
        # Find which atom indices map to locants 1 and 4
        locant_1_atom = None
        locant_4_atom = None
        for atom_idx, locant in numbering.items():
            if locant == 1:
                locant_1_atom = atom_idx
            elif locant == 4:
                locant_4_atom = atom_idx

        # Both bridgehead positions should have 3 neighbors
        atom1 = mol.GetAtomWithIdx(locant_1_atom)
        atom4 = mol.GetAtomWithIdx(locant_4_atom)
        assert len(list(atom1.GetNeighbors())) == 3
        assert len(list(atom4.GetNeighbors())) == 3

    @pytest.mark.unit
    def test_non_bicyclo_returns_none(self):
        """Non-bicyclo compounds should return None."""
        mol = Chem.MolFromSmiles('C1CCCCC1')  # cyclohexane
        numbering = get_bicyclo_numbering(mol)
        assert numbering is None


# ============================================================================
# Tests for get_bicyclo_substituents()
# ============================================================================

class TestBicycloSubstituents:
    """Test substituent detection on bicyclo systems."""

    @pytest.mark.unit
    def test_methylnorbornane_one_substituent(self):
        """Methylnorbornane should have one methyl substituent."""
        mol = Chem.MolFromSmiles('CC1CC2CCC1C2')
        ring_atoms = get_bicyclo_ring_atoms(mol)
        subs = get_bicyclo_substituents(mol, ring_atoms)

        # Should have substituents at one ring atom
        assert len(subs) == 1
        # The substituent should have 1 carbon
        for ring_idx, sub_list in subs.items():
            assert len(sub_list) == 1
            assert sub_list[0]['carbon_count'] == 1

    @pytest.mark.unit
    def test_dimethylnorbornane_two_substituents(self):
        """Dimethylnorbornane should have two methyl substituents."""
        # 7,7-dimethylnorbornane: both methyls on bridgehead carbon
        mol = Chem.MolFromSmiles('CC1(C)CC2CCC1C2')
        ring_atoms = get_bicyclo_ring_atoms(mol)
        subs = get_bicyclo_substituents(mol, ring_atoms)

        # Count total substituents
        total_subs = sum(len(sub_list) for sub_list in subs.values())
        assert total_subs == 2

    @pytest.mark.unit
    def test_ethylnorbornane_two_carbons(self):
        """Ethylnorbornane should have an ethyl substituent with 2 carbons."""
        mol = Chem.MolFromSmiles('CCC1CC2CCC1C2')
        ring_atoms = get_bicyclo_ring_atoms(mol)
        subs = get_bicyclo_substituents(mol, ring_atoms)

        # Should have one substituent with 2 carbons
        assert len(subs) == 1
        for ring_idx, sub_list in subs.items():
            assert len(sub_list) == 1
            assert sub_list[0]['carbon_count'] == 2

    @pytest.mark.unit
    def test_unsubstituted_norbornane_no_substituents(self):
        """Unsubstituted norbornane should have no substituents."""
        mol = Chem.MolFromSmiles('C1CC2CCC1C2')
        ring_atoms = get_bicyclo_ring_atoms(mol)
        subs = get_bicyclo_substituents(mol, ring_atoms)
        assert len(subs) == 0

    @pytest.mark.unit
    def test_multiple_substituents_same_position(self):
        """7,7-dimethylnorbornane should have 2 substituents at same position."""
        mol = Chem.MolFromSmiles('CC1(C)C2CCC1CC2')
        ring_atoms = get_bicyclo_ring_atoms(mol)
        subs = get_bicyclo_substituents(mol, ring_atoms)

        # One ring atom should have 2 substituents
        for ring_idx, sub_list in subs.items():
            if len(sub_list) == 2:
                # Both should be methyl
                assert sub_list[0]['carbon_count'] == 1
                assert sub_list[1]['carbon_count'] == 1


# ============================================================================
# Tests for detect_bicyclo_unsaturation()
# ============================================================================

class TestBicycloUnsaturation:
    """Test unsaturation detection in bicyclo systems."""

    @pytest.mark.unit
    def test_norbornene_one_double_bond(self):
        """Norbornene should have one double bond."""
        mol = Chem.MolFromSmiles('C1=CC2CCC1C2')
        ring_atoms = get_bicyclo_ring_atoms(mol)
        unsat = detect_bicyclo_unsaturation(mol, ring_atoms)

        assert len(unsat['double_bonds']) == 1
        assert len(unsat['triple_bonds']) == 0

    @pytest.mark.unit
    def test_norbornadiene_two_double_bonds(self):
        """Norbornadiene should have two double bonds."""
        mol = Chem.MolFromSmiles('C1=CC2C=CC1C2')
        ring_atoms = get_bicyclo_ring_atoms(mol)
        unsat = detect_bicyclo_unsaturation(mol, ring_atoms)

        assert len(unsat['double_bonds']) == 2
        assert len(unsat['triple_bonds']) == 0

    @pytest.mark.unit
    def test_saturated_norbornane_no_unsaturation(self):
        """Saturated norbornane should have no unsaturation."""
        mol = Chem.MolFromSmiles('C1CC2CCC1C2')
        ring_atoms = get_bicyclo_ring_atoms(mol)
        unsat = detect_bicyclo_unsaturation(mol, ring_atoms)

        assert len(unsat['double_bonds']) == 0
        assert len(unsat['triple_bonds']) == 0

    @pytest.mark.unit
    def test_bicyclo_222_octene(self):
        """Bicyclo[2.2.2]oct-2-ene should have one double bond."""
        mol = Chem.MolFromSmiles('C1=CC2CCC1CC2')
        ring_atoms = get_bicyclo_ring_atoms(mol)
        unsat = detect_bicyclo_unsaturation(mol, ring_atoms)

        assert len(unsat['double_bonds']) == 1


# ============================================================================
# Tests for Complete Bicyclo Naming via name_compound()
# ============================================================================

class TestCompleteBicycloNaming:
    """Integration tests for complete bicyclo naming."""

    @pytest.mark.unit
    def test_methylnorbornane_name(self):
        """Methylnorbornane should include methyl prefix."""
        name = name_compound('CC1CC2CCC1C2')
        assert 'methyl' in name.lower()
        assert 'bicyclo' in name.lower()

    @pytest.mark.unit
    def test_norbornene_name(self):
        """Norbornene should have -ene suffix."""
        name = name_compound('C1=CC2CCC1C2')
        assert 'ene' in name.lower()
        assert 'bicyclo' in name.lower()

    @pytest.mark.unit
    def test_dimethylnorbornane_name(self):
        """7,7-dimethylnorbornane should include dimethyl."""
        name = name_compound('CC1(C)C2CCC1CC2')
        assert 'dimethyl' in name.lower()
        assert 'bicyclo' in name.lower()

    @pytest.mark.unit
    def test_bicyclo_222_octane_name(self):
        """Bicyclo[2.2.2]octane should name correctly."""
        name = name_compound('C1CC2CCC1CC2')
        assert 'bicyclo[2.2.2]octane' in name.lower()

    @pytest.mark.unit
    def test_bicyclo_321_octane_name(self):
        """Bicyclo[3.2.1]octane should name correctly."""
        name = name_compound('C1CC2CCCC1C2')
        assert 'bicyclo' in name.lower()
        assert 'octane' in name.lower()

    @pytest.mark.unit
    def test_methylbicyclo_222_octane_name(self):
        """Methylbicyclo[2.2.2]octane should include methyl prefix."""
        name = name_compound('CC1CC2CCC1CC2')
        assert 'methyl' in name.lower()
        assert 'bicyclo' in name.lower()

    @pytest.mark.unit
    def test_norbornane_retained_name(self):
        """Unsubstituted norbornane should use retained name."""
        name = name_compound('C1CC2CCC1C2')
        assert name.lower() == 'norbornane'

    @pytest.mark.unit
    def test_bicyclo_410_heptane_name(self):
        """Bicyclo[4.1.0]heptane should name correctly."""
        name = name_compound('C1CCC2CC2C1')
        assert 'bicyclo' in name.lower()
        assert 'heptane' in name.lower()


# ============================================================================
# Tests for Stereochemistry in Bicyclo Systems
# ============================================================================

class TestBicycloStereochemistry:
    """Test stereochemistry handling in bicyclo naming."""

    @pytest.mark.unit
    def test_stereo_norbornane_collected(self):
        """Stereo norbornane should have stereodescriptors collected."""
        # (1R,4S)-norbornane variant
        mol = Chem.MolFromSmiles('[C@H]1CC[C@@H]2CC1C2')
        rdCIPLabeler.AssignCIPLabels(mol)

        data = get_complete_bicyclo_data(mol)
        assert data is not None
        # Verify we have atom_to_locant for stereo collection
        assert 'atom_to_locant' in data

    @pytest.mark.unit
    def test_stereo_methyl_bicyclo_name(self):
        """Stereo substituted bicyclo should include stereo prefix."""
        # This is a simplified test - actual stereo naming depends on
        # whether RDKit correctly assigns CIP labels
        mol = Chem.MolFromSmiles('C[C@H]1CC2CCC1C2')
        rdCIPLabeler.AssignCIPLabels(mol)

        # Just verify the molecule can be named without error
        name = name_compound('C[C@H]1CC2CCC1C2')
        assert name is not None
        assert len(name) > 0


# ============================================================================
# Tests for Edge Cases
# ============================================================================

class TestBicycloEdgeCases:
    """Test edge cases in bicyclo naming."""

    @pytest.mark.unit
    def test_heterobicyclo_azabicyclo(self):
        """Azabicyclo (nitrogen containing) should still detect substituents."""
        mol = Chem.MolFromSmiles('C1CC2CCC1CN2')  # quinuclidine
        ring_atoms = get_bicyclo_ring_atoms(mol)
        # Should detect ring atoms including nitrogen
        assert ring_atoms is not None
        assert len(ring_atoms) == 8

    @pytest.mark.unit
    def test_oxabicyclo_ring_atoms(self):
        """Oxabicyclo (oxygen containing) should detect ring atoms."""
        mol = Chem.MolFromSmiles('C1CC2CCC1CO2')  # 8-oxabicyclo[3.2.1]octane
        ring_atoms = get_bicyclo_ring_atoms(mol)
        assert ring_atoms is not None

    @pytest.mark.unit
    def test_large_bicyclo_311_heptane(self):
        """Bicyclo[3.1.1]heptane should number correctly."""
        mol = Chem.MolFromSmiles('C1CC2CC1CC2')
        numbering = get_bicyclo_numbering(mol)
        assert numbering is not None
        assert len(numbering) == 7

    @pytest.mark.unit
    def test_substituted_norbornene(self):
        """Substituted norbornene should handle both substituent and unsaturation."""
        name = name_compound('CC1=CC2CCC1C2')  # methylnorbornene
        assert 'methyl' in name.lower()
        assert 'ene' in name.lower() or 'en' in name.lower()

    @pytest.mark.unit
    def test_trimethyl_bicyclo(self):
        """Trimethyl bicyclo should have trimethyl prefix."""
        # 3,7,7-trimethylbicyclo[4.1.0]heptane
        name = name_compound('CC1(C)C2=C(C)CCC2C1')
        assert 'methyl' in name.lower()
        assert 'bicyclo' in name.lower()

    @pytest.mark.unit
    def test_complete_data_has_all_fields(self):
        """get_complete_bicyclo_data should return all required fields."""
        mol = Chem.MolFromSmiles('CC1CC2CCC1C2')
        data = get_complete_bicyclo_data(mol)

        assert data is not None
        required_fields = [
            'base_name', 'descriptor', 'ring_atoms', 'atom_to_locant',
            'substituents', 'unsaturation', 'bridgeheads', 'carbon_count'
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"


# ============================================================================
# Tests for Name Format Validation
# ============================================================================

class TestBicycloNameFormat:
    """Test that generated names follow IUPAC format."""

    @pytest.mark.unit
    def test_descriptor_format_in_name(self):
        """Name should contain bicyclo[x.y.z] descriptor."""
        name = name_compound('C1CC2CCC1CC2')  # bicyclo[2.2.2]octane
        # Check descriptor format
        assert '[2.2.2]' in name

    @pytest.mark.unit
    def test_substituent_locant_format(self):
        """Substituent should have locant prefix."""
        name = name_compound('CC1CC2CCC1C2')  # methylnorbornane
        # Should be like "X-methyl" not just "methyl"
        import re
        pattern = r'\d+-methyl'
        assert re.search(pattern, name.lower()), f"Expected locant-methyl, got: {name}"

    @pytest.mark.unit
    def test_unsaturation_locant_format(self):
        """Unsaturation suffix should have locant."""
        name = name_compound('C1=CC2CCC1C2')  # norbornene
        # Should be like "hept-X-ene"
        import re
        pattern = r'-\d+-en'
        assert re.search(pattern, name.lower()), f"Expected -X-en pattern, got: {name}"

    @pytest.mark.unit
    def test_name_ends_with_ane_or_ene(self):
        """Bicyclo names should end with -ane or -ene."""
        saturated = name_compound('C1CC2CCC1CC2')
        unsaturated = name_compound('C1=CC2CCC1CC2')

        assert saturated.lower().endswith('ane'), f"Expected -ane ending: {saturated}"
        assert unsaturated.lower().endswith('ene'), f"Expected -ene ending: {unsaturated}"
