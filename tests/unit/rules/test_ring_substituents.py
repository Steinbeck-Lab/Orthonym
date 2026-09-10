"""
Unit tests for ring substituent naming (IUPAC.

Tests the ring_substituents module for correct identification and naming
of rings when they become substituents on a chain parent.
"""

import pytest
from rdkit import Chem

from orthonym.rules.ring_substituents import (
    identify_ring_system,
    get_ring_substituent_name,
    get_ring_attachment_locant,
    get_ring_attachment_atom,
    RING_SUBSTITUENT_NAMES,
    POSITION_SPECIFIC_RINGS,
)


class TestIdentifyRingSystem:
    """Test ring system identification from structure."""

    def test_benzene_ring(self):
        """Benzene is a 6-membered aromatic carbocycle."""
        mol = Chem.MolFromSmiles('c1ccccc1')
        ring_atoms = tuple(mol.GetRingInfo().AtomRings()[0])
        result = identify_ring_system(mol, ring_atoms)
        assert result == 'benzene'

    def test_cyclohexane_ring(self):
        """Cyclohexane is a 6-membered saturated carbocycle."""
        mol = Chem.MolFromSmiles('C1CCCCC1')
        ring_atoms = tuple(mol.GetRingInfo().AtomRings()[0])
        result = identify_ring_system(mol, ring_atoms)
        assert result == 'cyclohexane'

    def test_cyclopentane_ring(self):
        """Cyclopentane is a 5-membered saturated carbocycle."""
        mol = Chem.MolFromSmiles('C1CCCC1')
        ring_atoms = tuple(mol.GetRingInfo().AtomRings()[0])
        result = identify_ring_system(mol, ring_atoms)
        assert result == 'cyclopentane'

    def test_cyclopropane_ring(self):
        """Cyclopropane is a 3-membered carbocycle."""
        mol = Chem.MolFromSmiles('C1CC1')
        ring_atoms = tuple(mol.GetRingInfo().AtomRings()[0])
        result = identify_ring_system(mol, ring_atoms)
        assert result == 'cyclopropane'

    def test_cyclobutane_ring(self):
        """Cyclobutane is a 4-membered carbocycle."""
        mol = Chem.MolFromSmiles('C1CCC1')
        ring_atoms = tuple(mol.GetRingInfo().AtomRings()[0])
        result = identify_ring_system(mol, ring_atoms)
        assert result == 'cyclobutane'

    def test_cycloheptane_ring(self):
        """Cycloheptane is a 7-membered saturated carbocycle."""
        mol = Chem.MolFromSmiles('C1CCCCCC1')
        ring_atoms = tuple(mol.GetRingInfo().AtomRings()[0])
        result = identify_ring_system(mol, ring_atoms)
        assert result == 'cycloheptane'

    def test_pyridine_ring(self):
        """Pyridine is a 6-membered aromatic ring with one N."""
        mol = Chem.MolFromSmiles('c1ccncc1')
        ring_atoms = tuple(mol.GetRingInfo().AtomRings()[0])
        result = identify_ring_system(mol, ring_atoms)
        assert result == 'pyridine'

    def test_furan_ring(self):
        """Furan is a 5-membered aromatic ring with one O."""
        mol = Chem.MolFromSmiles('c1ccoc1')
        ring_atoms = tuple(mol.GetRingInfo().AtomRings()[0])
        result = identify_ring_system(mol, ring_atoms)
        assert result == 'furan'

    def test_thiophene_ring(self):
        """Thiophene is a 5-membered aromatic ring with one S."""
        mol = Chem.MolFromSmiles('c1ccsc1')
        ring_atoms = tuple(mol.GetRingInfo().AtomRings()[0])
        result = identify_ring_system(mol, ring_atoms)
        assert result == 'thiophene'

    def test_pyrrole_ring(self):
        """Pyrrole is a 5-membered aromatic ring with one N."""
        mol = Chem.MolFromSmiles('c1cc[nH]c1')
        ring_atoms = tuple(mol.GetRingInfo().AtomRings()[0])
        result = identify_ring_system(mol, ring_atoms)
        assert result == 'pyrrole'

    def test_piperidine_ring(self):
        """Piperidine is a 6-membered saturated ring with one N."""
        mol = Chem.MolFromSmiles('C1CCNCC1')
        ring_atoms = tuple(mol.GetRingInfo().AtomRings()[0])
        result = identify_ring_system(mol, ring_atoms)
        assert result == 'piperidine'

    def test_morpholine_ring(self):
        """Morpholine is a 6-membered saturated ring with N and O."""
        mol = Chem.MolFromSmiles('C1COCCN1')
        ring_atoms = tuple(mol.GetRingInfo().AtomRings()[0])
        result = identify_ring_system(mol, ring_atoms)
        assert result == 'morpholine'


class TestGetRingSubstituentName:
    """Test substituent name generation from ring structure."""

    def test_benzene_to_phenyl(self):
        """Benzene as substituent is named phenyl."""
        mol = Chem.MolFromSmiles('c1ccccc1')
        ring_atoms = tuple(mol.GetRingInfo().AtomRings()[0])
        result = get_ring_substituent_name(mol, ring_atoms)
        assert result == 'phenyl'

    def test_cyclohexane_to_cyclohexyl(self):
        """Cyclohexane as substituent is named cyclohexyl."""
        mol = Chem.MolFromSmiles('C1CCCCC1')
        ring_atoms = tuple(mol.GetRingInfo().AtomRings()[0])
        result = get_ring_substituent_name(mol, ring_atoms)
        assert result == 'cyclohexyl'

    def test_cyclopentane_to_cyclopentyl(self):
        """Cyclopentane as substituent is named cyclopentyl."""
        mol = Chem.MolFromSmiles('C1CCCC1')
        ring_atoms = tuple(mol.GetRingInfo().AtomRings()[0])
        result = get_ring_substituent_name(mol, ring_atoms)
        assert result == 'cyclopentyl'

    def test_cyclopropane_to_cyclopropyl(self):
        """Cyclopropane as substituent is named cyclopropyl."""
        mol = Chem.MolFromSmiles('C1CC1')
        ring_atoms = tuple(mol.GetRingInfo().AtomRings()[0])
        result = get_ring_substituent_name(mol, ring_atoms)
        assert result == 'cyclopropyl'

    def test_cyclobutane_to_cyclobutyl(self):
        """Cyclobutane as substituent is named cyclobutyl."""
        mol = Chem.MolFromSmiles('C1CCC1')
        ring_atoms = tuple(mol.GetRingInfo().AtomRings()[0])
        result = get_ring_substituent_name(mol, ring_atoms)
        assert result == 'cyclobutyl'

    def test_pyridine_to_pyridyl(self):
        """Pyridine as substituent is named pyridyl."""
        mol = Chem.MolFromSmiles('c1ccncc1')
        ring_atoms = tuple(mol.GetRingInfo().AtomRings()[0])
        result = get_ring_substituent_name(mol, ring_atoms)
        assert result == 'pyridyl'

    def test_furan_to_furyl(self):
        """Furan as substituent is named furyl."""
        mol = Chem.MolFromSmiles('c1ccoc1')
        ring_atoms = tuple(mol.GetRingInfo().AtomRings()[0])
        result = get_ring_substituent_name(mol, ring_atoms)
        assert result == 'furyl'

    def test_thiophene_to_thienyl(self):
        """Thiophene as substituent is named thienyl."""
        mol = Chem.MolFromSmiles('c1ccsc1')
        ring_atoms = tuple(mol.GetRingInfo().AtomRings()[0])
        result = get_ring_substituent_name(mol, ring_atoms)
        assert result == 'thienyl'


class TestRingAttachmentLocant:
    """Test finding chain position where ring attaches."""

    def test_phenyl_at_position_4(self):
        """4-phenylbutanoic acid: phenyl attaches at position 4."""
        # c1ccc(CCCC(=O)O)cc1 = 4-phenylbutanoic acid
        # Structure: benzene ring atoms (0,1,2,3,10,11), chain atoms 4-5-6-7
        # Ring atom 3 connects to chain atom 4
        mol = Chem.MolFromSmiles('c1ccc(CCCC(=O)O)cc1')

        # Find benzene ring (aromatic 6-membered)
        ring_info = mol.GetRingInfo()
        ring_atoms = None
        for ring in ring_info.AtomRings():
            if len(ring) == 6:
                if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
                    ring_atoms = tuple(ring)
                    break

        # Chain: atoms 4-5-6-7 (4 carbons of butanoic acid)
        # Numbered from COOH end: 7=C1, 6=C2, 5=C3, 4=C4
        chain_atoms = [7, 6, 5, 4]  # Numbered from COOH end
        atom_to_locant = {7: 1, 6: 2, 5: 3, 4: 4}

        locant = get_ring_attachment_locant(mol, ring_atoms, chain_atoms, atom_to_locant)
        assert locant == 4

    def test_phenyl_at_position_3(self):
        """3-phenylpropanoic acid: phenyl attaches at position 3."""
        # c1ccc(CCC(=O)O)cc1 = 3-phenylpropanoic acid
        # Structure: benzene ring connects to chain at position 3
        mol = Chem.MolFromSmiles('c1ccc(CCC(=O)O)cc1')

        ring_info = mol.GetRingInfo()
        ring_atoms = None
        for ring in ring_info.AtomRings():
            if len(ring) == 6:
                if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
                    ring_atoms = tuple(ring)
                    break

        # Chain: 3 carbons of propanoic acid
        # Atoms 4-5-6 (ring connects to 4)
        chain_atoms = [6, 5, 4]  # Numbered from COOH end
        atom_to_locant = {6: 1, 5: 2, 4: 3}

        locant = get_ring_attachment_locant(mol, ring_atoms, chain_atoms, atom_to_locant)
        assert locant == 3

    def test_cyclohexyl_at_position_4(self):
        """4-cyclohexylbutanoic acid: cyclohexyl attaches at position 4."""
        # C1CCCCC1CCCC(=O)O = 4-cyclohexylbutanoic acid
        # Ring atoms 0-5, chain atoms 6-7-8-9
        mol = Chem.MolFromSmiles('C1CCCCC1CCCC(=O)O')

        ring_info = mol.GetRingInfo()
        ring_atoms = tuple(ring_info.AtomRings()[0])

        # Chain atoms 6-7-8-9 (4 carbons)
        # Numbered from COOH: 9=C1, 8=C2, 7=C3, 6=C4
        chain_atoms = [9, 8, 7, 6]  # Numbered from COOH end
        atom_to_locant = {9: 1, 8: 2, 7: 3, 6: 4}

        locant = get_ring_attachment_locant(mol, ring_atoms, chain_atoms, atom_to_locant)
        assert locant == 4


class TestGetRingAttachmentAtom:
    """Test finding the ring atom that connects to chain."""

    def test_find_attachment_atom_phenyl(self):
        """Find which benzene carbon attaches to the chain."""
        mol = Chem.MolFromSmiles('c1ccc(CCCC(=O)O)cc1')

        ring_info = mol.GetRingInfo()
        ring_atoms = None
        for ring in ring_info.AtomRings():
            if len(ring) == 6:
                if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
                    ring_atoms = tuple(ring)
                    break

        # Chain atoms 4-5-6-7
        chain_atoms = [4, 5, 6, 7]

        attachment = get_ring_attachment_atom(mol, ring_atoms, chain_atoms)
        assert attachment is not None
        # The attachment should be a ring atom bonded to chain atom 4
        assert attachment in ring_atoms


class TestRingSubstituentData:
    """Test the data dictionaries for completeness."""

    def test_common_rings_have_substituent_names(self):
        """Common rings should have substituent name mappings."""
        required_rings = [
            'benzene', 'cyclohexane', 'cyclopentane', 'cyclopropane',
            'cyclobutane', 'pyridine', 'furan', 'thiophene', 'pyrrole'
        ]
        for ring in required_rings:
            assert ring in RING_SUBSTITUENT_NAMES, f"{ring} not in RING_SUBSTITUENT_NAMES"

    def test_phenyl_is_benzene_substituent(self):
        """Benzene -> phenyl mapping exists."""
        assert RING_SUBSTITUENT_NAMES['benzene'] == 'phenyl'

    def test_cyclohexyl_is_cyclohexane_substituent(self):
        """Cyclohexane -> cyclohexyl mapping exists."""
        assert RING_SUBSTITUENT_NAMES['cyclohexane'] == 'cyclohexyl'

    def test_position_specific_rings_exist(self):
        """Position-specific rings have correct entries."""
        assert 'pyridine' in POSITION_SPECIFIC_RINGS
        assert 'naphthalene' in POSITION_SPECIFIC_RINGS
