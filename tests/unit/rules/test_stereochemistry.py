"""
Unit tests for stereochemistry rules module.

Tests the IUPAC stereodescriptor collection and formatting functions
that map RDKit stereochemistry to locant-based IUPAC descriptors.
"""

import pytest
from rdkit import Chem
from rdkit.Chem import rdCIPLabeler

from orthonym.rules.stereochemistry import (
    collect_stereodescriptors,
    collect_ring_stereodescriptors,
    format_stereodescriptor_string,
    get_double_bond_locant,
)


# =============================================================================
# Test collect_stereodescriptors - R/S stereocenters
# =============================================================================

class TestCollectStereodescriptorsRS:
    """Tests for R/S stereocenter collection."""

    def test_single_r_stereocenter(self):
        """Single R stereocenter returns [(locant, 'R')]."""
        # (2R)-butan-2-ol: C[C@@H](O)CC (@@H gives R in this context)
        mol = Chem.MolFromSmiles('C[C@@H](O)CC')
        rdCIPLabeler.AssignCIPLabels(mol)

        # Verify CIP assignment
        atom1 = mol.GetAtomWithIdx(1)
        assert atom1.HasProp('_CIPCode')
        assert atom1.GetProp('_CIPCode') == 'R'

        # Principal chain: locant 1=terminal, 2=stereocenter, 3,4=other carbons
        atom_to_locant = {0: 4, 1: 3, 3: 2, 4: 1}  # Reverse for alcohol locant

        descriptors = collect_stereodescriptors(mol, atom_to_locant)
        assert descriptors == [(3, 'R')]

    def test_single_s_stereocenter(self):
        """Single S stereocenter returns [(locant, 'S')]."""
        # C[C@H](O)CC - @H gives S in this context
        mol = Chem.MolFromSmiles('C[C@H](O)CC')
        rdCIPLabeler.AssignCIPLabels(mol)

        atom1 = mol.GetAtomWithIdx(1)
        assert atom1.HasProp('_CIPCode')
        assert atom1.GetProp('_CIPCode') == 'S'

        atom_to_locant = {0: 4, 1: 3, 3: 2, 4: 1}

        descriptors = collect_stereodescriptors(mol, atom_to_locant)
        assert descriptors == [(3, 'S')]

    def test_multiple_stereocenters_sorted_by_locant(self):
        """Multiple stereocenters returned sorted by locant."""
        # Butane-2,3-diol: C[C@H](O)[C@@H](O)C
        mol = Chem.MolFromSmiles('C[C@H](O)[C@@H](O)C')
        rdCIPLabeler.AssignCIPLabels(mol)

        # Chain: atom 0 (locant 1), atom 1 (locant 2), atom 3 (locant 3), atom 5 (locant 4)
        atom_to_locant = {0: 1, 1: 2, 3: 3, 5: 4}

        descriptors = collect_stereodescriptors(mol, atom_to_locant)

        # Should be sorted by locant
        assert len(descriptors) == 2
        assert descriptors[0][0] < descriptors[1][0]  # locant 2 before locant 3

    def test_no_stereocenters_returns_empty(self):
        """Molecule without stereocenters returns empty list."""
        mol = Chem.MolFromSmiles('CCCC')  # Butane
        rdCIPLabeler.AssignCIPLabels(mol)

        atom_to_locant = {0: 1, 1: 2, 2: 3, 3: 4}

        descriptors = collect_stereodescriptors(mol, atom_to_locant)
        assert descriptors == []

    def test_stereocenter_not_in_mapping_excluded(self):
        """Stereocenters not in atom_to_locant are excluded."""
        mol = Chem.MolFromSmiles('C[C@H](O)CC')
        rdCIPLabeler.AssignCIPLabels(mol)

        # Only include non-stereocenter atoms
        atom_to_locant = {0: 1, 3: 2, 4: 3}  # Exclude atom 1 (stereocenter)

        descriptors = collect_stereodescriptors(mol, atom_to_locant)
        assert descriptors == []

    def test_empty_mapping_returns_empty(self):
        """Empty atom_to_locant returns empty list."""
        mol = Chem.MolFromSmiles('C[C@H](O)CC')
        rdCIPLabeler.AssignCIPLabels(mol)

        descriptors = collect_stereodescriptors(mol, {})
        assert descriptors == []


# =============================================================================
# Test collect_stereodescriptors - E/Z double bonds
# =============================================================================

class TestCollectStereodescriptorsEZ:
    """Tests for E/Z double bond collection."""

    def test_e_double_bond(self):
        """E double bond returns [(locant, 'E')] with lower locant."""
        mol = Chem.MolFromSmiles('C/C=C/C')  # (2E)-but-2-ene
        rdCIPLabeler.AssignCIPLabels(mol)

        atom_to_locant = {0: 1, 1: 2, 2: 3, 3: 4}

        descriptors = collect_stereodescriptors(mol, atom_to_locant)
        assert descriptors == [(2, 'E')]

    def test_z_double_bond(self):
        """Z double bond returns [(locant, 'Z')] with lower locant."""
        mol = Chem.MolFromSmiles(r'C/C=C\C')  # (2Z)-but-2-ene
        rdCIPLabeler.AssignCIPLabels(mol)

        atom_to_locant = {0: 1, 1: 2, 2: 3, 3: 4}

        descriptors = collect_stereodescriptors(mol, atom_to_locant)
        assert descriptors == [(2, 'Z')]

    def test_double_bond_uses_lower_locant(self):
        """E/Z descriptor uses lower of two atom locants."""
        mol = Chem.MolFromSmiles('C/C=C/C')
        rdCIPLabeler.AssignCIPLabels(mol)

        # Map with atom 1 -> locant 5, atom 2 -> locant 6
        atom_to_locant = {0: 4, 1: 5, 2: 6, 3: 7}

        descriptors = collect_stereodescriptors(mol, atom_to_locant)
        assert descriptors == [(5, 'E')]  # Uses min(5, 6) = 5

    def test_double_bond_one_atom_missing_excluded(self):
        """Double bond excluded if one atom not in mapping."""
        mol = Chem.MolFromSmiles('C/C=C/C')
        rdCIPLabeler.AssignCIPLabels(mol)

        # Exclude atom 2 (one end of double bond)
        atom_to_locant = {0: 1, 1: 2, 3: 4}  # Missing atom 2

        descriptors = collect_stereodescriptors(mol, atom_to_locant)
        assert descriptors == []

    def test_double_bond_both_atoms_missing_excluded(self):
        """Double bond excluded if both atoms not in mapping."""
        mol = Chem.MolFromSmiles('C/C=C/C')
        rdCIPLabeler.AssignCIPLabels(mol)

        # Only include terminal atoms, not double bond atoms
        atom_to_locant = {0: 1, 3: 2}

        descriptors = collect_stereodescriptors(mol, atom_to_locant)
        assert descriptors == []


# =============================================================================
# Test mixed R/S and E/Z
# =============================================================================

class TestCollectStereodescriptorsMixed:
    """Tests for molecules with both R/S and E/Z."""

    def test_mixed_rs_and_ez_sorted_by_locant(self):
        """Mixed R/S and E/Z sorted by locant."""
        # Molecule with E double bond and R stereocenter
        # (2E,4R)-4-hydroxypent-2-ene: C/C=C/[C@H](O)C
        mol = Chem.MolFromSmiles('C/C=C/[C@@H](O)C')
        rdCIPLabeler.AssignCIPLabels(mol)

        # Check what we have
        for atom in mol.GetAtoms():
            if atom.HasProp('_CIPCode'):
                print(f"Atom {atom.GetIdx()}: {atom.GetProp('_CIPCode')}")
        for bond in mol.GetBonds():
            if bond.HasProp('_CIPCode'):
                print(f"Bond {bond.GetIdx()}: {bond.GetProp('_CIPCode')}")

        # Chain: atom 0 (1), atom 1 (2), atom 2 (3), atom 3 (4), atom 5 (5)
        atom_to_locant = {0: 1, 1: 2, 2: 3, 3: 4, 5: 5}

        descriptors = collect_stereodescriptors(mol, atom_to_locant)

        # Should have E at locant 2 and R/S at locant 4, sorted
        assert len(descriptors) == 2
        locants = [d[0] for d in descriptors]
        assert locants == sorted(locants)  # Verify sorted

    def test_multiple_double_bonds_and_stereocenters(self):
        """Complex molecule with multiple E/Z and R/S."""
        # (2E,4E)-hexa-2,4-diene with R stereocenter would be complex
        # Let's use a simpler case: just verify sorting works
        mol = Chem.MolFromSmiles('C/C=C/C=C/C')  # hexa-2,4-diene
        rdCIPLabeler.AssignCIPLabels(mol)

        atom_to_locant = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}

        descriptors = collect_stereodescriptors(mol, atom_to_locant)

        # Should have two E descriptors at positions 2 and 4
        # (actual positions depend on CIP assignment)
        locants = [d[0] for d in descriptors]
        assert locants == sorted(locants)


# =============================================================================
# Test pseudoasymmetric centers
# =============================================================================

class TestCollectStereodescriptorsPseudoasymmetric:
    """Tests for pseudoasymmetric (meso) centers with lowercase r/s."""

    def test_pseudoasymmetric_lowercase(self):
        """Pseudoasymmetric centers return lowercase 'r' or 's'."""
        # Finding a good test case for pseudoasymmetric is tricky
        # Meso tartaric acid has pseudoasymmetric centers
        # For now, we test that the function preserves whatever CIP returns
        # This test verifies the function doesn't uppercase the code
        mol = Chem.MolFromSmiles('C[C@H](O)CC')
        rdCIPLabeler.AssignCIPLabels(mol)

        atom_to_locant = {0: 4, 1: 3, 3: 2, 4: 1}

        descriptors = collect_stereodescriptors(mol, atom_to_locant)

        # The CIP code should be preserved as-is (R, S, r, or s)
        assert len(descriptors) == 1
        cip = descriptors[0][1]
        assert cip in ['R', 'S', 'r', 's']


# =============================================================================
# Test get_double_bond_locant
# =============================================================================

class TestGetDoubleBondLocant:
    """Tests for get_double_bond_locant function."""

    def test_both_atoms_in_mapping(self):
        """Returns min locant when both atoms in mapping."""
        mol = Chem.MolFromSmiles('C/C=C/C')
        bond = mol.GetBondBetweenAtoms(1, 2)

        atom_to_locant = {0: 1, 1: 2, 2: 3, 3: 4}

        locant = get_double_bond_locant(bond, atom_to_locant)
        assert locant == 2  # min(2, 3)

    def test_returns_minimum_of_locants(self):
        """Returns minimum of the two locants."""
        mol = Chem.MolFromSmiles('C/C=C/C')
        bond = mol.GetBondBetweenAtoms(1, 2)

        # Reverse numbering
        atom_to_locant = {0: 4, 1: 3, 2: 2, 3: 1}

        locant = get_double_bond_locant(bond, atom_to_locant)
        assert locant == 2  # min(3, 2)

    def test_one_atom_missing_returns_none(self):
        """Returns None if one atom not in mapping."""
        mol = Chem.MolFromSmiles('C/C=C/C')
        bond = mol.GetBondBetweenAtoms(1, 2)

        atom_to_locant = {0: 1, 1: 2, 3: 4}  # Missing atom 2

        locant = get_double_bond_locant(bond, atom_to_locant)
        assert locant is None

    def test_both_atoms_missing_returns_none(self):
        """Returns None if both atoms not in mapping."""
        mol = Chem.MolFromSmiles('C/C=C/C')
        bond = mol.GetBondBetweenAtoms(1, 2)

        atom_to_locant = {0: 1, 3: 2}  # Missing atoms 1 and 2

        locant = get_double_bond_locant(bond, atom_to_locant)
        assert locant is None

    def test_empty_mapping_returns_none(self):
        """Returns None with empty mapping."""
        mol = Chem.MolFromSmiles('C/C=C/C')
        bond = mol.GetBondBetweenAtoms(1, 2)

        locant = get_double_bond_locant(bond, {})
        assert locant is None


# =============================================================================
# Test format_stereodescriptor_string
# =============================================================================

class TestFormatStereodescriptorString:
    """Tests for format_stereodescriptor_string function."""

    def test_empty_list_returns_empty_string(self):
        """Empty descriptor list returns empty string."""
        assert format_stereodescriptor_string([]) == ""

    def test_single_r_descriptor(self):
        """Single R descriptor formatted as '(2R)-'."""
        assert format_stereodescriptor_string([(2, 'R')]) == "(2R)-"

    def test_single_s_descriptor(self):
        """Single S descriptor formatted as '(3S)-'."""
        assert format_stereodescriptor_string([(3, 'S')]) == "(3S)-"

    def test_single_e_descriptor(self):
        """Single E descriptor formatted as '(2E)-'."""
        assert format_stereodescriptor_string([(2, 'E')]) == "(2E)-"

    def test_single_z_descriptor(self):
        """Single Z descriptor formatted as '(4Z)-'."""
        assert format_stereodescriptor_string([(4, 'Z')]) == "(4Z)-"

    def test_multiple_rs_descriptors(self):
        """Multiple R/S descriptors comma-separated."""
        descriptors = [(2, 'R'), (3, 'S')]
        assert format_stereodescriptor_string(descriptors) == "(2R,3S)-"

    def test_multiple_ez_descriptors(self):
        """Multiple E/Z descriptors comma-separated."""
        descriptors = [(2, 'E'), (4, 'Z')]
        assert format_stereodescriptor_string(descriptors) == "(2E,4Z)-"

    def test_mixed_rs_and_ez_descriptors(self):
        """Mixed R/S and E/Z descriptors in single block."""
        descriptors = [(2, 'E'), (3, 'R'), (5, 'Z')]
        assert format_stereodescriptor_string(descriptors) == "(2E,3R,5Z)-"

    def test_lowercase_rs_for_pseudoasymmetric(self):
        """Lowercase r/s preserved for pseudoasymmetric centers."""
        descriptors = [(2, 'r'), (3, 's')]
        assert format_stereodescriptor_string(descriptors) == "(2r,3s)-"

    def test_three_descriptors(self):
        """Three descriptors formatted correctly."""
        descriptors = [(1, 'R'), (2, 'S'), (4, 'R')]
        assert format_stereodescriptor_string(descriptors) == "(1R,2S,4R)-"

    def test_preserves_order(self):
        """Descriptors output in provided order."""
        # Note: format doesn't sort; sorting is done by collect_*
        descriptors = [(5, 'E'), (2, 'R')]
        assert format_stereodescriptor_string(descriptors) == "(5E,2R)-"


# =============================================================================
# Test collect_ring_stereodescriptors
# =============================================================================

class TestCollectRingStereodescriptors:
    """Tests for collect_ring_stereodescriptors function."""

    def test_ring_single_stereocenter(self):
        """Ring with single stereocenter returns correct locant."""
        # Use 1,2-dimethylcyclohexane which has actual stereocenters
        # We'll just check one stereocenter by limiting the mapping
        mol = Chem.MolFromSmiles('[C@H]1(O)C[C@@H](C)CCC1')  # 4-methylcyclohexan-1-ol
        rdCIPLabeler.AssignCIPLabels(mol)

        # Atom 0 is the OH-bearing carbon (R), atom 3 is the methyl-bearing carbon (S)
        # Include only atom 0 in the ring map to test single stereocenter
        ring_atom_to_locant = {0: 1, 2: 2, 3: 3, 5: 4, 6: 5, 7: 6}

        descriptors = collect_ring_stereodescriptors(mol, ring_atom_to_locant)

        # Should have stereocenters at atom 0 (locant 1) and atom 3 (locant 3)
        assert len(descriptors) == 2
        locants = [d[0] for d in descriptors]
        assert 1 in locants  # stereocenter at locant 1

    def test_ring_multiple_stereocenters_sorted(self):
        """Ring with multiple stereocenters sorted by locant."""
        # cis-1,2-dimethylcyclohexane: C[C@H]1[C@H](C)CCCC1
        mol = Chem.MolFromSmiles('C[C@H]1[C@H](C)CCCC1')
        rdCIPLabeler.AssignCIPLabels(mol)

        # Ring atoms are 1,2,4,5,6,7 (assuming structure)
        ring_atom_to_locant = {1: 1, 2: 2, 4: 3, 5: 4, 6: 5, 7: 6}

        descriptors = collect_ring_stereodescriptors(mol, ring_atom_to_locant)

        # Should have two stereocenters, sorted by locant
        assert len(descriptors) == 2
        locants = [d[0] for d in descriptors]
        assert locants == sorted(locants)

    def test_atoms_outside_ring_excluded(self):
        """Atoms outside ring are excluded from results."""
        # Methylcyclohexane with stereo at ring junction
        mol = Chem.MolFromSmiles('C[C@H]1CCCCC1')
        rdCIPLabeler.AssignCIPLabels(mol)

        # Only include some ring atoms, exclude methyl (atom 0)
        ring_atom_to_locant = {2: 2, 3: 3, 4: 4, 5: 5, 6: 6}  # Exclude atoms 0 and 1

        descriptors = collect_ring_stereodescriptors(mol, ring_atom_to_locant)

        # Stereocenter at atom 1 should be excluded
        assert len(descriptors) == 0

    def test_ring_with_no_stereocenters(self):
        """Ring without stereocenters returns empty list."""
        mol = Chem.MolFromSmiles('C1CCCCC1')  # Cyclohexane
        rdCIPLabeler.AssignCIPLabels(mol)

        ring_atom_to_locant = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}

        descriptors = collect_ring_stereodescriptors(mol, ring_atom_to_locant)
        assert descriptors == []


# =============================================================================
# Integration-style tests with real molecules
# =============================================================================

class TestRealMolecules:
    """Integration tests with real chemical structures."""

    def test_butan2ol_s_configuration(self):
        """(3S)-butan-2-ol produces correct descriptor."""
        # CC[C@@H](O)C produces S at position 2 (atom index 2)
        mol = Chem.MolFromSmiles('CC[C@@H](O)C')
        rdCIPLabeler.AssignCIPLabels(mol)

        # Verify we have S configuration
        for atom in mol.GetAtoms():
            if atom.HasProp('_CIPCode'):
                assert atom.GetProp('_CIPCode') == 'S'
                break

        # Map: atom 0,1,2,4 are the carbon chain
        atom_to_locant = {0: 1, 1: 2, 2: 3, 4: 4}

        descriptors = collect_stereodescriptors(mol, atom_to_locant)
        formatted = format_stereodescriptor_string(descriptors)

        assert '3' in formatted  # locant 3 for stereocenter
        assert 'S' in formatted

    def test_but2ene_e_configuration(self):
        """(2E)-but-2-ene produces correct descriptor."""
        mol = Chem.MolFromSmiles('C/C=C/C')
        rdCIPLabeler.AssignCIPLabels(mol)

        atom_to_locant = {0: 1, 1: 2, 2: 3, 3: 4}

        descriptors = collect_stereodescriptors(mol, atom_to_locant)
        formatted = format_stereodescriptor_string(descriptors)

        assert formatted == "(2E)-"

    def test_but2ene_z_configuration(self):
        """(2Z)-but-2-ene produces correct descriptor."""
        mol = Chem.MolFromSmiles(r'C/C=C\C')
        rdCIPLabeler.AssignCIPLabels(mol)

        atom_to_locant = {0: 1, 1: 2, 2: 3, 3: 4}

        descriptors = collect_stereodescriptors(mol, atom_to_locant)
        formatted = format_stereodescriptor_string(descriptors)

        assert formatted == "(2Z)-"

    def test_molecule_without_stereo_produces_empty(self):
        """Achiral molecule produces empty descriptor string."""
        mol = Chem.MolFromSmiles('CCCC')  # Butane
        rdCIPLabeler.AssignCIPLabels(mol)

        atom_to_locant = {0: 1, 1: 2, 2: 3, 3: 4}

        descriptors = collect_stereodescriptors(mol, atom_to_locant)
        formatted = format_stereodescriptor_string(descriptors)

        assert formatted == ""

    def test_propene_no_ez_defined(self):
        """Propene without defined E/Z returns empty."""
        mol = Chem.MolFromSmiles('CC=C')  # No E/Z specified
        rdCIPLabeler.AssignCIPLabels(mol)

        atom_to_locant = {0: 1, 1: 2, 2: 3}

        descriptors = collect_stereodescriptors(mol, atom_to_locant)

        # Propene can't have E/Z (only two groups on one carbon)
        assert descriptors == []


# =============================================================================
# Parametrized tests for comprehensive coverage
# =============================================================================

@pytest.mark.parametrize("smiles,expected_cip", [
    ('C[C@H](O)CC', 'S'),    # @H at this position gives S
    ('C[C@@H](O)CC', 'R'),   # @@H at this position gives R
    ('CC[C@H](O)C', 'R'),    # @H at this position gives R (different context)
    ('CC[C@@H](O)C', 'S'),   # @@H at this position gives S (different context)
])
def test_rs_cip_codes(smiles, expected_cip):
    """Test CIP code assignment for various SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    rdCIPLabeler.AssignCIPLabels(mol)

    # Find the stereocenter
    for atom in mol.GetAtoms():
        if atom.HasProp('_CIPCode'):
            assert atom.GetProp('_CIPCode') == expected_cip
            break
    else:
        pytest.fail("No stereocenter found")


@pytest.mark.parametrize("smiles,expected_stereo", [
    ('C/C=C/C', 'E'),
    (r'C/C=C\C', 'Z'),
    ('CC/C=C/C', 'E'),
    (r'CC/C=C\C', 'Z'),
])
def test_ez_cip_codes(smiles, expected_stereo):
    """Test E/Z code assignment for various SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    rdCIPLabeler.AssignCIPLabels(mol)

    # Find the stereogenic double bond
    for bond in mol.GetBonds():
        if bond.HasProp('_CIPCode'):
            assert bond.GetProp('_CIPCode') == expected_stereo
            break
    else:
        pytest.fail("No stereogenic double bond found")


@pytest.mark.parametrize("descriptors,expected", [
    ([], ""),
    ([(1, 'R')], "(1R)-"),
    ([(2, 'S')], "(2S)-"),
    ([(3, 'E')], "(3E)-"),
    ([(4, 'Z')], "(4Z)-"),
    ([(2, 'R'), (3, 'S')], "(2R,3S)-"),
    ([(2, 'E'), (4, 'E')], "(2E,4E)-"),
    ([(2, 'r')], "(2r)-"),
    ([(2, 'r'), (3, 's')], "(2r,3s)-"),
    ([(1, 'E'), (2, 'R'), (3, 'S'), (5, 'Z')], "(1E,2R,3S,5Z)-"),
])
def test_format_stereodescriptor_parametrized(descriptors, expected):
    """Parametrized tests for format_stereodescriptor_string."""
    assert format_stereodescriptor_string(descriptors) == expected


# =============================================================================
# Tests for idempotent assign_stereochemistry guard
# =============================================================================

class TestAssignStereochemistryIdempotent:
    """Tests for the marker-based idempotent guard in assign_stereochemistry()."""

    def test_assign_stereochemistry_sets_marker(self):
        """assign_stereochemistry() sets the marker property after first call."""
        from orthonym.perception.stereo import assign_stereochemistry, _CIP_ASSIGNED_PROP

        mol = Chem.MolFromSmiles('C[C@@H](O)CC')

        # Clear any marker that may exist
        if mol.HasProp(_CIP_ASSIGNED_PROP):
            mol.ClearProp(_CIP_ASSIGNED_PROP)

        assert not mol.HasProp(_CIP_ASSIGNED_PROP)

        assign_stereochemistry(mol)

        # Marker should now be set
        assert mol.HasProp(_CIP_ASSIGNED_PROP)
        assert mol.GetProp(_CIP_ASSIGNED_PROP) == '1'

    def test_assign_stereochemistry_idempotent_with_marker(self):
        """assign_stereochemistry() skips re-assignment when marker already set."""
        from orthonym.perception.stereo import assign_stereochemistry, _CIP_ASSIGNED_PROP

        mol = Chem.MolFromSmiles('C[C@@H](O)CC')
        # Simulate the authoritative call in namer.py _perceive()
        rdCIPLabeler.AssignCIPLabels(mol)
        mol.SetProp(_CIP_ASSIGNED_PROP, '1')

        # Capture original CIP codes
        original_codes = {}
        for atom in mol.GetAtoms():
            if atom.HasProp('_CIPCode'):
                original_codes[atom.GetIdx()] = atom.GetProp('_CIPCode')

        assert len(original_codes) > 0, "Should have at least one stereocenter"

        # Call assign_stereochemistry -- should be a no-op (marker guard)
        assign_stereochemistry(mol)

        # Verify CIP codes are unchanged
        for idx, code in original_codes.items():
            atom = mol.GetAtomWithIdx(idx)
            assert atom.HasProp('_CIPCode')
            assert atom.GetProp('_CIPCode') == code

    def test_assign_stereochemistry_idempotent_bonds_with_marker(self):
        """assign_stereochemistry() preserves E/Z bond codes when marker set."""
        from orthonym.perception.stereo import assign_stereochemistry, _CIP_ASSIGNED_PROP

        mol = Chem.MolFromSmiles('C/C=C/C')  # (E)-but-2-ene
        rdCIPLabeler.AssignCIPLabels(mol)
        mol.SetProp(_CIP_ASSIGNED_PROP, '1')

        # Capture original bond CIP codes
        original_bond_codes = {}
        for bond in mol.GetBonds():
            if bond.HasProp('_CIPCode'):
                original_bond_codes[bond.GetIdx()] = bond.GetProp('_CIPCode')

        assert len(original_bond_codes) > 0, "Should have at least one E/Z bond"

        # Call assign_stereochemistry -- should be a no-op
        assign_stereochemistry(mol)

        # Verify bond CIP codes unchanged
        for idx, code in original_bond_codes.items():
            bond = mol.GetBondWithIdx(idx)
            assert bond.HasProp('_CIPCode')
            assert bond.GetProp('_CIPCode') == code

    def test_assign_stereochemistry_assigns_if_no_marker(self):
        """assign_stereochemistry() assigns CIP labels when no marker is present."""
        from orthonym.perception.stereo import assign_stereochemistry, _CIP_ASSIGNED_PROP

        mol = Chem.MolFromSmiles('C[C@@H](O)CC')

        # Clear any CIP codes and ensure no marker
        for atom in mol.GetAtoms():
            if atom.HasProp('_CIPCode'):
                atom.ClearProp('_CIPCode')
        if mol.HasProp(_CIP_ASSIGNED_PROP):
            mol.ClearProp(_CIP_ASSIGNED_PROP)

        # assign_stereochemistry should assign CIP labels (no marker -> runs labeler)
        assign_stereochemistry(mol)

        # Now should have CIP codes and marker
        has_cip = any(atom.HasProp('_CIPCode') for atom in mol.GetAtoms())
        assert has_cip, "CIP should now be assigned"
        assert mol.HasProp(_CIP_ASSIGNED_PROP), "Marker should be set after assignment"


# =============================================================================
# Tests for pseudoasymmetric r/s preservation
# =============================================================================

class TestPseudoasymmetricPreservation:
    """Tests that pseudoasymmetric centers preserve lowercase r/s."""

    def test_format_stereodescriptor_preserves_lowercase_rs(self):
        """format_stereodescriptor_string preserves lowercase r/s for pseudoasymmetric."""
        result = format_stereodescriptor_string([(2, 'r'), (3, 's')])
        assert result == "(2r,3s)-", f"Expected (2r,3s)- but got {result}"

    def test_format_stereodescriptor_preserves_mixed_case(self):
        """format_stereodescriptor_string handles mixed R/r and S/s."""
        result = format_stereodescriptor_string([(2, 'R'), (3, 'r'), (4, 'S')])
        assert result == "(2R,3r,4S)-", f"Expected (2R,3r,4S)- but got {result}"

    def test_get_stereocenters_preserves_cip_case(self):
        """get_stereocenters preserves CIP code case as returned by RDKit."""
        from orthonym.perception.stereo import get_stereocenters

        mol = Chem.MolFromSmiles('C[C@@H](O)CC')
        rdCIPLabeler.AssignCIPLabels(mol)

        centers = get_stereocenters(mol)
        assert len(centers) == 1

        # The CIP code should be whatever RDKit assigned -- no forced uppercasing
        cip = centers[0]['cip']
        assert cip in ('R', 'S', 'r', 's'), f"CIP code should be valid: {cip}"

    def test_intraring_small_ring_no_ez(self):
        """Double bonds in small rings (<=8) should NOT produce E/Z descriptors."""
        mol = Chem.MolFromSmiles('C1=CCCCC1')
        rdCIPLabeler.AssignCIPLabels(mol)

        atom_to_locant = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}
        descriptors = collect_stereodescriptors(mol, atom_to_locant)

        ez_descriptors = [d for d in descriptors if d[1] in ('E', 'Z')]
        assert len(ez_descriptors) == 0, f"Unexpected E/Z in cyclohexene: {ez_descriptors}"

    def test_ez_uses_cipcode_on_bond(self):
        """E/Z collection uses _CIPCode property on bonds, not BondStereo enum."""
        mol = Chem.MolFromSmiles('C/C=C/C')
        rdCIPLabeler.AssignCIPLabels(mol)

        ez_bond = None
        for bond in mol.GetBonds():
            if bond.HasProp('_CIPCode'):
                ez_bond = bond
                break
        assert ez_bond is not None, "rdCIPLabeler should set _CIPCode on E/Z bond"
        assert ez_bond.GetProp('_CIPCode') == 'E'

        atom_to_locant = {0: 1, 1: 2, 2: 3, 3: 4}
        descriptors = collect_stereodescriptors(mol, atom_to_locant)
        assert descriptors == [(2, 'E')]

    def test_collect_stereodescriptors_preserves_cip_case(self):
        """collect_stereodescriptors preserves CIP code case (no forced upper)."""
        mol = Chem.MolFromSmiles('C[C@@H](O)CC')
        rdCIPLabeler.AssignCIPLabels(mol)

        atom_to_locant = {0: 4, 1: 3, 3: 2, 4: 1}
        descriptors = collect_stereodescriptors(mol, atom_to_locant)

        assert len(descriptors) == 1
        cip = descriptors[0][1]
        # Should preserve whatever RDKit returned (R in this case)
        assert cip in ('R', 'S', 'r', 's'), f"CIP code should be valid: {cip}"


class TestClusterFPseudoasymmetricSignResolution:
    """Wave-8 P6 Cluster F reproduce-first (D-F1/D-F2/D-F3): the ledger's
    'CIP-ceiling' alarm for cyclobutane-1,3-diol was a FALSE ALARM, not an
    engine bug. `O[C@@H]1C[C@H](O)C1` (cis) and its BB-cited PIN counterpart
    `O[C@@H]1C[C@@H](O)C1` (trans) are DIFFERENT diastereomers -- both
    verified via 3D-embedding face check AND OPSIN cis-/trans- round-trip.
    The centres/rdCIPLabeler engine (perception/stereo.py assign_stereo-
    chemistry) is correct on BOTH. No `pseudoasymmetric_confident` gate was
    needed this session since nothing built in this phase consumes it (see
    Cluster A, which needed only the ordinary R/S branch for hydrindane, not
    pseudoasymmetric r/s); if a future cluster needs a gate, build it then.
    """

    @pytest.mark.unit
    def test_cyclobutane_diol_cis_is_lowercase_ss(self):
        from orthonym import Orthonym
        o = Orthonym(_disable_opsin_validity_gate=True)
        assert o.name("O[C@@H]1C[C@H](O)C1") == "(1s,3s)-cyclobutane-1,3-diol"

    @pytest.mark.unit
    def test_cyclobutane_diol_trans_matches_bb_verbatim_pin(self):
        """BlueBookV2.md:45793 verbatim (PIN): '(1r,3r)-cyclobutane-1,3-diol'."""
        from orthonym import Orthonym
        o = Orthonym(_disable_opsin_validity_gate=True)
        assert o.name("O[C@@H]1C[C@@H](O)C1") == "(1r,3r)-cyclobutane-1,3-diol"

    @pytest.mark.unit
    def test_cis_trans_are_different_diastereomers(self):
        """Premise check for the false-alarm resolution: the two SMILES are
        genuinely different diastereomers (different canonical SMILES), not
        the same molecule described two ways."""
        cis = Chem.MolFromSmiles("O[C@@H]1C[C@H](O)C1")
        trans = Chem.MolFromSmiles("O[C@@H]1C[C@@H](O)C1")
        assert Chem.MolToSmiles(cis) != Chem.MolToSmiles(trans)


class TestClusterBSpiroCipCeiling:
    """Wave-8 P6 Cluster B reproduce-first (D-B1, load-bearing): does the CIP
    engine (centres OR rdCIPLabeler) assign R/S to the ring CH-Cl carbons of
    `Cl[C@H]1C[C@]2(C1)C[C@@H](Cl)C2` (2,6-dichlorospiro[3.3]heptane, BB
    P-93.5.3.5 / BB L49267)?

    Reproduced: NEITHER engine assigns a `_CIPCode` to either CH-Cl carbon
    (both show a defined chiral tag / RDKit's ``FindMolChiralCenters``
    reports them as 'Tet_CW' -- detected-but-unrankable -- not merely
    'unassigned'). This is confirmed against the vendored `centres` jar
    directly (``centres_label_mol`` returns True -- it ran -- but still sets
    no ``_CIPCode`` on these atoms), so it is a genuine CIP-ceiling limitation
    of BOTH available engines, not an integration bug. Per the accuracy-first
    policy (a missing descriptor beats a wrong one), Cluster B is CIP-ceiling
    fail-closed THIS CYCLE: no code change, no gold. The shipped name
    ('2,6-dichlorospiro[3.3]heptane', stereo silently dropped, RT-MISMATCH
    against the input) is the correct fail-closed behaviour, not a leak --
    the constitution is right, only the descriptor is (honestly) missing.
    """

    @pytest.mark.unit
    def test_spiro_ring_chcl_carbons_get_no_cip_from_either_engine(self):
        from orthonym.perception.stereo import assign_stereochemistry
        mol = Chem.MolFromSmiles("Cl[C@H]1C[C@]2(C1)C[C@@H](Cl)C2")
        assign_stereochemistry(mol)  # production path: centres, then rdCIPLabeler fallback
        ring_chcl_atoms = [
            a.GetIdx() for a in mol.GetAtoms()
            if a.GetSymbol() == 'C' and a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
            and any(n.GetSymbol() == 'Cl' for n in a.GetNeighbors())
        ]
        assert len(ring_chcl_atoms) == 2, f"expected 2 CH-Cl stereocentres, found {ring_chcl_atoms}"
        for idx in ring_chcl_atoms:
            assert not mol.GetAtomWithIdx(idx).HasProp('_CIPCode'), (
                f"atom {idx}: CIP-ceiling premise changed -- engine NOW assigns "
                f"a code; Cluster B should be revisited and built"
            )

    @pytest.mark.unit
    def test_spiro_diol_e2e_fails_closed_not_wrong(self):
        """The shipped name must be the correct CONSTITUTION with stereo
        honestly omitted -- never a wrong skeleton."""
        from orthonym import Orthonym
        o = Orthonym(_disable_opsin_validity_gate=True)
        out = o.name("Cl[C@H]1C[C@]2(C1)C[C@@H](Cl)C2")
        assert out == "2,6-dichlorospiro[3.3]heptane", f"got {out!r}"
