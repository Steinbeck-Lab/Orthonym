"""
Unit tests for ring boundary detection - get_complete_ring_atom_set().

Tests that all atoms in fused, bridged, spiro, and single ring systems
are correctly identified, and that exocyclic atoms are excluded.
"""

import pytest
from rdkit import Chem

from orthonym.perception.rings import get_complete_ring_atom_set


class TestGetCompleteRingAtomSet:
    """Tests for the single-source-of-truth ring boundary function."""

    def test_fused_bicyclic_naphthalene(self):
        """Fused bicyclic (naphthalene) returns all 10 ring atoms."""
        mol = Chem.MolFromSmiles("C1=CC2=CC=CC=C2C=C1")
        result = get_complete_ring_atom_set(mol)
        # Naphthalene has exactly 10 ring atoms
        assert len(result) == 10
        # All atoms should be ring atoms
        assert isinstance(result, frozenset)

    def test_fused_bicyclic_naphthalene_aromatic(self):
        """Fused bicyclic (naphthalene, aromatic SMILES) returns all 10 ring atoms."""
        mol = Chem.MolFromSmiles("c1ccc2ccccc2c1")
        result = get_complete_ring_atom_set(mol)
        assert len(result) == 10

    def test_bridged_bicyclic_norbornane(self):
        """Bridged bicyclic (norbornane) returns all 7 ring atoms."""
        mol = Chem.MolFromSmiles("C1CC2CC1CC2")
        result = get_complete_ring_atom_set(mol)
        assert len(result) == 7

    def test_spiro_compound(self):
        """Spiro[4.5]decane returns all 10 ring atoms (including spiro atom)."""
        mol = Chem.MolFromSmiles("C1CCCC11CCCCC1")
        result = get_complete_ring_atom_set(mol)
        # spiro[4.5]decane has 10 atoms total, all in rings
        assert len(result) == 10

    def test_single_ring_cyclohexane(self):
        """Single ring (cyclohexane) returns 6 ring atoms."""
        mol = Chem.MolFromSmiles("C1CCCCC1")
        result = get_complete_ring_atom_set(mol)
        assert len(result) == 6

    def test_exocyclic_oxygen_not_in_ring_set(self):
        """Exocyclic =O on ring (cyclohexanone) - =O atom NOT in ring set."""
        mol = Chem.MolFromSmiles("O=C1CCCCC1")
        result = get_complete_ring_atom_set(mol)
        # 6 ring carbons, oxygen is exocyclic
        assert len(result) == 6
        # Find the oxygen atom index
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "O":
                assert atom.GetIdx() not in result, "Exocyclic =O should NOT be in ring set"

    def test_disconnected_ring_systems_biphenyl(self):
        """Biphenyl (two disconnected rings) returns all 12 ring atoms."""
        mol = Chem.MolFromSmiles("c1ccc(-c2ccccc2)cc1")
        result = get_complete_ring_atom_set(mol)
        assert len(result) == 12

    def test_caffeine_fused_rings(self):
        """Caffeine returns exactly the 9 ring atoms (fused imidazole + pyrimidine)."""
        mol = Chem.MolFromSmiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
        result = get_complete_ring_atom_set(mol)
        # Caffeine has a fused 5+6 ring system = 9 unique ring atoms
        # Count ring atoms from RDKit to verify
        ri = mol.GetRingInfo()
        expected_ring_atoms = set()
        for ring in ri.AtomRings():
            expected_ring_atoms.update(ring)
        assert result == frozenset(expected_ring_atoms)
        assert len(result) == 9

    def test_no_rings(self):
        """Acyclic molecule returns empty frozenset."""
        mol = Chem.MolFromSmiles("CCCCC")
        result = get_complete_ring_atom_set(mol)
        assert result == frozenset()
        assert len(result) == 0

    def test_returns_frozenset(self):
        """Return type is frozenset for immutability."""
        mol = Chem.MolFromSmiles("C1CCCCC1")
        result = get_complete_ring_atom_set(mol)
        assert isinstance(result, frozenset)

    def test_exocyclic_hydroxyl_not_in_ring_set(self):
        """Exocyclic -OH on ring NOT in ring set (phenol)."""
        mol = Chem.MolFromSmiles("Oc1ccccc1")
        result = get_complete_ring_atom_set(mol)
        # 6 ring carbons; O and H are exocyclic
        assert len(result) == 6

    def test_indole_fused_heterocyclic(self):
        """Indole (fused 5+6 heterocyclic) returns all 9 ring atoms."""
        mol = Chem.MolFromSmiles("c1ccc2[nH]ccc2c1")
        result = get_complete_ring_atom_set(mol)
        assert len(result) == 9

    def test_purine_fused_heterocyclic(self):
        """Purine (fused 5+6 with multiple N) returns all 9 ring atoms."""
        mol = Chem.MolFromSmiles("c1ncc2[nH]cnc2n1")
        result = get_complete_ring_atom_set(mol)
        assert len(result) == 9


class TestMolecularFeaturesAllRingAtoms:
    """Test that MolecularFeatures.all_ring_atoms is populated during perception."""

    def test_all_ring_atoms_populated(self):
        """MolecularFeatures.all_ring_atoms is populated after _perceive()."""
        from orthonym.namer import Orthonym
        namer = Orthonym()
        mol = Chem.MolFromSmiles("c1ccc2ccccc2c1")  # naphthalene
        smiles = "c1ccc2ccccc2c1"
        canonical = Chem.MolToSmiles(mol)
        features = namer._perceive(mol, smiles, canonical)
        assert hasattr(features, "all_ring_atoms")
        assert isinstance(features.all_ring_atoms, frozenset)
        assert len(features.all_ring_atoms) == 10

    def test_all_ring_atoms_empty_for_acyclic(self):
        """MolecularFeatures.all_ring_atoms is empty for acyclic molecules."""
        from orthonym.namer import Orthonym
        namer = Orthonym()
        mol = Chem.MolFromSmiles("CCCCCC")
        smiles = "CCCCCC"
        canonical = Chem.MolToSmiles(mol)
        features = namer._perceive(mol, smiles, canonical)
        assert features.all_ring_atoms == frozenset()
