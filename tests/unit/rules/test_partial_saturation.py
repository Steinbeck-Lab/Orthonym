"""
Unit tests for partial saturation detection and prefix generation.

Tests cover:
1. Saturation detection for various hydrogen counts (dihydro, tetrahydro, etc.)
2. Locant ordering and formatting
3. Perhydro special case (fully saturated, no locants)
4. E2E naming integration
5. Edge cases and error handling

IUPAC 2013 Blue Book P-31.1.1: Hydro prefixes indicate the addition of
hydrogen to specified positions of an otherwise unsaturated parent structure.
"""

import pytest
from rdkit import Chem

from orthonym.rules.partial_saturation import (
    detect_partial_saturation,
    get_saturation_prefix,
    format_saturation_prefix,
    get_saturation_locants,
    analyze_saturation_for_naming,
    is_fully_saturated,
    count_ring_sp3_atoms,
    get_ring_saturation_level,
)
from orthonym.data.partial_saturation_refs import (
    AROMATIC_REFERENCES,
    get_aromatic_reference,
    get_reference_smiles,
    get_reference_ring_atoms,
    list_reference_names,
)
from orthonym.assembly.composer import (
    get_saturation_prefix_for_fused_ring,
    assemble_fused_ring_with_saturation,
)


class TestPartialSaturationDetection:
    """Tests for detect_partial_saturation function."""

    def test_tetrahydroquinoline_detection(self):
        """Test detection of 1,2,3,4-tetrahydroquinoline saturation."""
        mol = Chem.MolFromSmiles('c1ccc2c(c1)CCCN2')  # tetrahydroquinoline
        ref = 'c1ccc2ncccc2c1'  # quinoline
        result = detect_partial_saturation(mol, ref)

        assert result is not None
        assert result['sp3_count'] == 3  # 3 sp3 carbons in reduced ring
        assert result['prefix'] == 'hexahydro'  # 3 sp3 * 2H = 6H
        assert result['is_perhydro'] is False

    def test_indoline_detection(self):
        """Test detection of indoline (2,3-dihydro-1H-indole) saturation."""
        mol = Chem.MolFromSmiles('c1ccc2c(c1)CCN2')  # indoline
        ref = 'c1ccc2[nH]ccc2c1'  # indole
        result = detect_partial_saturation(mol, ref)

        assert result is not None
        assert result['sp3_count'] == 2  # 2 sp3 carbons at positions 2,3
        assert result['prefix'] == 'tetrahydro'  # 2 sp3 * 2H = 4H

    def test_dihydrofuran_detection(self):
        """Test detection of 2,3-dihydrofuran saturation."""
        mol = Chem.MolFromSmiles('C1CC=CO1')  # 2,3-dihydrofuran
        ref = 'c1ccoc1'  # furan
        result = detect_partial_saturation(mol, ref)

        assert result is not None
        assert result['sp3_count'] == 2  # 2 sp3 carbons
        assert result['prefix'] == 'tetrahydro'  # 2 sp3 * 2H = 4H

    def test_perhydro_detection_decalin(self):
        """Test perhydro detection for decalin (decahydronaphthalene)."""
        mol = Chem.MolFromSmiles('C1CCC2CCCCC2C1')  # decalin
        ref = 'c1ccc2ccccc2c1'  # naphthalene
        result = detect_partial_saturation(mol, ref)

        assert result is not None
        assert result['is_perhydro'] is True
        assert result['prefix'] == 'perhydro'

    def test_no_saturation_aromatic(self):
        """Test that fully aromatic compounds return None."""
        mol = Chem.MolFromSmiles('c1ccc2ncccc2c1')  # quinoline (aromatic)
        ref = 'c1ccc2ncccc2c1'  # quinoline
        result = detect_partial_saturation(mol, ref)

        assert result is None  # No saturation detected

    def test_hexahydro_detection(self):
        """Test hexahydro (6H) detection."""
        mol = Chem.MolFromSmiles('c1ccc2c(c1)CCCN2')  # tetrahydroquinoline
        ref = 'c1ccc2ncccc2c1'
        result = detect_partial_saturation(mol, ref)

        # 3 sp3 carbons * 2H = 6H = hexahydro
        assert result['prefix'] == 'hexahydro'

    @pytest.mark.parametrize("smiles,parent,expected_sp3", [
        ('c1ccc2c(c1)CCN2', 'c1ccc2[nH]ccc2c1', 2),  # indoline: 2 sp3
        ('C1CCC2CCCCC2C1', 'c1ccc2ccccc2c1', 10),  # decalin: 10 sp3
        ('C1CC=CO1', 'c1ccoc1', 2),  # 2,3-dihydrofuran: 2 sp3
    ])
    def test_sp3_count_parametrized(self, smiles, parent, expected_sp3):
        """Parametrized test for sp3 atom counts."""
        mol = Chem.MolFromSmiles(smiles)
        result = detect_partial_saturation(mol, parent)

        assert result is not None
        assert result['sp3_count'] == expected_sp3


class TestSaturationLocants:
    """Tests for locant ordering and formatting."""

    def test_locant_ordering_simple(self):
        """Test sorting of simple numeric locants."""
        mol = Chem.MolFromSmiles('c1ccc2c(c1)CCN2')
        indices = [6, 7]  # sp3 atom indices
        atom_to_locant = {6: 2, 7: 3}

        locants = get_saturation_locants(mol, indices, atom_to_locant)

        assert locants == [2, 3]

    def test_locant_ordering_mixed(self):
        """Test sorting of mixed int/str locants (e.g., 1, 2, '3a', '7a')."""
        mol = Chem.MolFromSmiles('c1ccc2c(c1)CCN2')
        indices = [0, 1, 2, 3]
        atom_to_locant = {0: 5, 1: '3a', 2: 1, 3: '7a'}

        locants = get_saturation_locants(mol, indices, atom_to_locant)

        # Should sort as: 1, '3a', 5, '7a'
        assert locants == [1, '3a', 5, '7a']

    def test_locant_formatting_with_locants(self):
        """Test formatting of saturation prefix with locants."""
        result = format_saturation_prefix('dihydro', [2, 3])
        assert result == '2,3-dihydro'

    def test_locant_formatting_tetrahydro(self):
        """Test formatting of tetrahydro prefix with locants."""
        result = format_saturation_prefix('tetrahydro', [1, 2, 3, 4])
        assert result == '1,2,3,4-tetrahydro'

    def test_perhydro_no_locants(self):
        """Test that perhydro prefix has no locants."""
        result = format_saturation_prefix('perhydro', [1, 2, 3, 4, 5])
        assert result == 'perhydro'  # Locants ignored for perhydro

        result2 = format_saturation_prefix('perhydro')
        assert result2 == 'perhydro'


class TestSaturationPrefixFormat:
    """Tests for saturation prefix selection and formatting."""

    @pytest.mark.parametrize("hydrogen_count,expected_prefix", [
        (2, 'dihydro'),
        (4, 'tetrahydro'),
        (6, 'hexahydro'),
        (8, 'octahydro'),
        (10, 'decahydro'),
        (12, 'dodecahydro'),
    ])
    def test_prefix_selection(self, hydrogen_count, expected_prefix):
        """Test correct prefix for various hydrogen counts."""
        result = get_saturation_prefix(hydrogen_count)
        assert result == expected_prefix

    def test_invalid_hydrogen_count(self):
        """Test that invalid hydrogen counts return None."""
        assert get_saturation_prefix(3) is None  # Odd number
        assert get_saturation_prefix(5) is None
        assert get_saturation_prefix(0) is None


class TestComposerIntegration:
    """Tests for composer helper functions."""

    def test_assemble_basic(self):
        """Test basic name assembly with saturation prefix."""
        result = assemble_fused_ring_with_saturation(
            core_name='indole',
            saturation_prefix='2,3-dihydro',
            indicated_h='1H'
        )
        assert result == '2,3-dihydro-1H-indole'

    def test_assemble_with_substituent(self):
        """Test name assembly with substituent and saturation prefix."""
        result = assemble_fused_ring_with_saturation(
            core_name='indole',
            saturation_prefix='2,3-dihydro',
            substituent_prefixes='5-methyl',
            indicated_h='1H'
        )
        assert result == '5-methyl-2,3-dihydro-1H-indole'

    def test_assemble_perhydro(self):
        """Test name assembly for perhydro compound."""
        result = assemble_fused_ring_with_saturation(
            core_name='naphthalene',
            saturation_prefix='perhydro'
        )
        assert result == 'perhydro-naphthalene'

    def test_assemble_no_saturation(self):
        """Test name assembly without saturation prefix."""
        result = assemble_fused_ring_with_saturation(
            core_name='indole',
            saturation_prefix=None,
            indicated_h='1H'
        )
        assert result == '1H-indole'

    def test_get_saturation_prefix_quinoline(self):
        """Test saturation prefix detection for tetrahydroquinoline."""
        mol = Chem.MolFromSmiles('c1ccc2c(c1)CCCN2')
        prefix = get_saturation_prefix_for_fused_ring(mol, 'quinoline')
        assert prefix is not None
        # Returns the prefix based on sp3 count

    def test_get_saturation_prefix_no_parent(self):
        """Test saturation prefix with no parent returns None."""
        mol = Chem.MolFromSmiles('c1ccc2c(c1)CCCN2')
        prefix = get_saturation_prefix_for_fused_ring(mol, 'nonexistent_parent')
        assert prefix is None


class TestE2ESaturationNaming:
    """End-to-end tests for saturation naming via name_compound."""

    def test_tetrahydroquinoline_e2e(self):
        """Test that tetrahydroquinoline gets correct retained name."""
        from orthonym import name_compound

        result = name_compound('c1ccc2c(c1)CCCN2')
        assert result == '1,2,3,4-tetrahydroquinoline'

    def test_indoline_e2e(self):
        """Test that indoline gets correct retained name."""
        from orthonym import name_compound

        result = name_compound('c1ccc2c(c1)CCN2')
        assert result == 'indoline'

    def test_tetrahydroisoquinoline_e2e(self):
        """Test tetrahydroisoquinoline naming."""
        from orthonym import name_compound

        result = name_compound('c1ccc2c(c1)CCNC2')
        assert result == '1,2,3,4-tetrahydroisoquinoline'


class TestAromaticReferences:
    """Tests for aromatic reference data module."""

    def test_reference_count(self):
        """Test that we have sufficient aromatic references."""
        assert len(AROMATIC_REFERENCES) >= 10

    def test_quinoline_reference(self):
        """Test quinoline reference data."""
        ref = AROMATIC_REFERENCES.get('quinoline')
        assert ref is not None
        assert ref['smiles'] == 'c1ccc2ncccc2c1'
        assert ref['ring_atoms'] == 10

    def test_indole_reference(self):
        """Test indole reference data."""
        ref = AROMATIC_REFERENCES.get('indole')
        assert ref is not None
        assert ref['ring_atoms'] == 9

    def test_naphthalene_reference(self):
        """Test naphthalene reference data."""
        ref = AROMATIC_REFERENCES.get('naphthalene')
        assert ref is not None
        assert ref['ring_atoms'] == 10

    def test_xanthene_reference(self):
        """Test xanthene reference data (RING-05)."""
        ref = AROMATIC_REFERENCES.get('xanthene')
        assert ref is not None
        assert ref['ring_atoms'] == 13
        assert 'o' in ref['smiles'].lower()  # Contains oxygen heteroatom

    def test_get_reference_smiles(self):
        """Test get_reference_smiles function."""
        smiles = get_reference_smiles('quinoline')
        assert smiles == 'c1ccc2ncccc2c1'

        smiles2 = get_reference_smiles('nonexistent')
        assert smiles2 is None

    def test_get_reference_ring_atoms(self):
        """Test get_reference_ring_atoms function."""
        count = get_reference_ring_atoms('naphthalene')
        assert count == 10

        count2 = get_reference_ring_atoms('nonexistent')
        assert count2 == 0

    def test_list_reference_names(self):
        """Test list_reference_names function."""
        names = list_reference_names()
        assert 'quinoline' in names
        assert 'indole' in names
        assert 'naphthalene' in names
        assert 'purine' in names


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_is_fully_saturated_true(self):
        """Test is_fully_saturated returns True for decalin."""
        mol = Chem.MolFromSmiles('C1CCC2CCCCC2C1')  # decalin
        ref = 'c1ccc2ccccc2c1'  # naphthalene
        assert is_fully_saturated(mol, ref) is True

    def test_is_fully_saturated_false(self):
        """Test is_fully_saturated returns False for partial saturation."""
        mol = Chem.MolFromSmiles('c1ccc2c(c1)CCN2')  # indoline
        ref = 'c1ccc2[nH]ccc2c1'  # indole
        assert is_fully_saturated(mol, ref) is False

    def test_count_ring_sp3_atoms(self):
        """Test counting sp3 atoms in ring systems."""
        mol = Chem.MolFromSmiles('C1CCC2CCCCC2C1')  # decalin
        count = count_ring_sp3_atoms(mol)
        assert count == 10  # All 10 ring atoms are sp3

    def test_get_ring_saturation_level_aromatic(self):
        """Test saturation level for aromatic compound."""
        mol = Chem.MolFromSmiles('c1ccc2ncccc2c1')  # quinoline
        ref = 'c1ccc2ncccc2c1'
        level = get_ring_saturation_level(mol, ref)
        assert level == 'aromatic'

    def test_get_ring_saturation_level_partial(self):
        """Test saturation level for partially saturated compound."""
        mol = Chem.MolFromSmiles('c1ccc2c(c1)CCN2')  # indoline
        ref = 'c1ccc2[nH]ccc2c1'  # indole
        level = get_ring_saturation_level(mol, ref)
        assert level == 'partially saturated'

    def test_get_ring_saturation_level_full(self):
        """Test saturation level for fully saturated compound."""
        mol = Chem.MolFromSmiles('C1CCC2CCCCC2C1')  # decalin
        ref = 'c1ccc2ccccc2c1'  # naphthalene
        level = get_ring_saturation_level(mol, ref)
        assert level == 'fully saturated'


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_none_molecule(self):
        """Test that None molecule returns None."""
        result = detect_partial_saturation(None, 'c1ccc2ncccc2c1')
        assert result is None

    def test_none_parent(self):
        """Test that None parent returns None."""
        mol = Chem.MolFromSmiles('c1ccc2c(c1)CCCN2')
        result = detect_partial_saturation(mol, None)
        assert result is None

    def test_invalid_parent_smiles(self):
        """Test that invalid parent SMILES returns None."""
        mol = Chem.MolFromSmiles('c1ccc2c(c1)CCCN2')
        result = detect_partial_saturation(mol, 'invalid_smiles')
        assert result is None

    def test_format_prefix_without_locants(self):
        """Test formatting prefix without locants (fallback)."""
        result = format_saturation_prefix('dihydro')
        assert result == 'dihydro'  # Returns prefix alone

    def test_empty_locants(self):
        """Test formatting with empty locant list."""
        result = format_saturation_prefix('tetrahydro', [])
        assert result == 'tetrahydro'


class TestIUPACCompliance:
    """Tests verifying IUPAC 2013 compliance."""

    def test_locant_always_included(self):
        """Test that locants are always included for partial saturation."""
        # IUPAC rule: 2,3-dihydro not just dihydro
        result = format_saturation_prefix('dihydro', [2, 3])
        assert '2,3-' in result

    def test_perhydro_no_locants(self):
        """Test that perhydro has no locants per IUPAC."""
        result = format_saturation_prefix('perhydro', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        assert result == 'perhydro'
        assert ',' not in result  # No locant numbers

    def test_prefix_ordering_substituent_saturation_indicatedh_parent(self):
        """Test IUPAC ordering: substituent-saturation-indicatedH-parent."""
        result = assemble_fused_ring_with_saturation(
            core_name='indole',
            saturation_prefix='2,3-dihydro',
            substituent_prefixes='5-methyl',
            indicated_h='1H'
        )
        # Order should be: 5-methyl-2,3-dihydro-1H-indole
        parts = result.split('-')
        assert parts[0] == '5'  # substituent locant
        assert 'methyl' in parts[1]  # substituent name
        assert '2,3' in result  # saturation locants
        assert 'dihydro' in result  # saturation prefix
        assert '1H' in result  # indicated H
        assert result.endswith('indole')  # parent
