"""
Unit tests for IUPAC ring numbering in fusion naming context.

Tests that _get_iupac_ring_order() correctly maps RDKit atom indices
to IUPAC positions for monocyclic components:
- Heterocyclic rings start at highest-priority heteroatom (O > S > N)
- Direction chosen to give lowest locants to remaining heteroatoms
- Carbocyclic rings preserve input order (no reordering needed)
"""

import pytest
from rdkit import Chem

from orthonym.rules.fusion_descriptors import _get_iupac_ring_order


def _get_ring_and_order(smiles):
    """Get mol, ring atoms, and IUPAC-ordered atoms from a SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Invalid SMILES: {smiles}"
    ri = mol.GetRingInfo()
    rings = ri.AtomRings()
    assert len(rings) >= 1, f"No rings in {smiles}"
    ring = list(rings[0])
    order = _get_iupac_ring_order(mol, ring)
    return mol, ring, order


def _atom_symbol(mol, idx):
    """Get element symbol for atom index."""
    return mol.GetAtomWithIdx(idx).GetSymbol()


# ============================================================================
# Heterocyclic ring ordering
# ============================================================================

class TestHeterocyclicRingOrder:
    """Tests for heterocyclic ring IUPAC numbering."""

    def test_furan_starts_at_O(self):
        """Furan: IUPAC position 1 should be oxygen."""
        mol, ring, order = _get_ring_and_order('c1ccoc1')
        assert _atom_symbol(mol, order[0]) == 'O', \
            f"Expected O at position 1, got {_atom_symbol(mol, order[0])}"

    def test_furan_order_length(self):
        """Furan IUPAC order should have 5 atoms."""
        mol, ring, order = _get_ring_and_order('c1ccoc1')
        assert len(order) == 5

    def test_thiophene_starts_at_S(self):
        """Thiophene: IUPAC position 1 should be sulfur."""
        mol, ring, order = _get_ring_and_order('c1ccsc1')
        assert _atom_symbol(mol, order[0]) == 'S', \
            f"Expected S at position 1, got {_atom_symbol(mol, order[0])}"

    def test_pyrrole_starts_at_N(self):
        """Pyrrole: IUPAC position 1 should be nitrogen."""
        mol, ring, order = _get_ring_and_order('c1cc[nH]c1')
        assert _atom_symbol(mol, order[0]) == 'N', \
            f"Expected N at position 1, got {_atom_symbol(mol, order[0])}"

    def test_pyridine_starts_at_N(self):
        """Pyridine: IUPAC position 1 should be nitrogen."""
        mol, ring, order = _get_ring_and_order('c1ccncc1')
        assert _atom_symbol(mol, order[0]) == 'N', \
            f"Expected N at position 1, got {_atom_symbol(mol, order[0])}"

    def test_pyridine_order_length(self):
        """Pyridine IUPAC order should have 6 atoms."""
        mol, ring, order = _get_ring_and_order('c1ccncc1')
        assert len(order) == 6


class TestDiazineRingOrder:
    """Tests for IUPAC numbering of 2-heteroatom rings."""

    def test_pyrimidine_N_at_1_and_3(self):
        """Pyrimidine: N at IUPAC positions 1 and 3."""
        mol, ring, order = _get_ring_and_order('c1ccncn1')
        sym_1 = _atom_symbol(mol, order[0])
        sym_3 = _atom_symbol(mol, order[2])
        assert sym_1 == 'N', f"Position 1 should be N, got {sym_1}"
        assert sym_3 == 'N', f"Position 3 should be N, got {sym_3}"

    def test_pyrazine_N_at_1_and_4(self):
        """Pyrazine: N at IUPAC positions 1 and 4."""
        mol, ring, order = _get_ring_and_order('c1cnccn1')
        sym_1 = _atom_symbol(mol, order[0])
        sym_4 = _atom_symbol(mol, order[3])
        assert sym_1 == 'N', f"Position 1 should be N, got {sym_1}"
        assert sym_4 == 'N', f"Position 4 should be N, got {sym_4}"

    def test_pyridazine_N_at_1_and_2(self):
        """Pyridazine: N at IUPAC positions 1 and 2."""
        mol, ring, order = _get_ring_and_order('c1ccnnc1')
        sym_1 = _atom_symbol(mol, order[0])
        sym_2 = _atom_symbol(mol, order[1])
        assert sym_1 == 'N', f"Position 1 should be N, got {sym_1}"
        assert sym_2 == 'N', f"Position 2 should be N, got {sym_2}"

    def test_imidazole_N_at_1_and_3(self):
        """Imidazole: N at IUPAC positions 1 and 3."""
        mol, ring, order = _get_ring_and_order('c1cnc[nH]1')
        sym_1 = _atom_symbol(mol, order[0])
        sym_3 = _atom_symbol(mol, order[2])
        assert sym_1 == 'N', f"Position 1 should be N, got {sym_1}"
        assert sym_3 == 'N', f"Position 3 should be N, got {sym_3}"

    def test_pyrazole_N_at_1_and_2(self):
        """Pyrazole: N at IUPAC positions 1 and 2."""
        mol, ring, order = _get_ring_and_order('c1cc[nH]n1')
        sym_1 = _atom_symbol(mol, order[0])
        sym_2 = _atom_symbol(mol, order[1])
        assert sym_1 == 'N', f"Position 1 should be N, got {sym_1}"
        assert sym_2 == 'N', f"Position 2 should be N, got {sym_2}"


class TestMixedHeteroatomOrder:
    """Tests for rings with different heteroatom types."""

    def test_oxazole_O_at_1(self):
        """Oxazole: O is higher priority than N, should be at position 1."""
        mol, ring, order = _get_ring_and_order('c1cocn1')
        sym_1 = _atom_symbol(mol, order[0])
        assert sym_1 == 'O', f"Position 1 should be O, got {sym_1}"

    def test_thiazole_S_at_1(self):
        """Thiazole: S is higher priority than N, should be at position 1."""
        mol, ring, order = _get_ring_and_order('c1cscn1')
        sym_1 = _atom_symbol(mol, order[0])
        assert sym_1 == 'S', f"Position 1 should be S, got {sym_1}"


# ============================================================================
# Carbocyclic ring ordering
# ============================================================================

class TestCarbocyclicRingOrder:
    """Tests for carbocyclic ring numbering (no reordering)."""

    def test_benzene_returns_atoms_in_order(self):
        """Benzene: all-C ring should return atoms in order (no heteroatom to start from)."""
        mol, ring, order = _get_ring_and_order('c1ccccc1')
        assert len(order) == 6
        # All atoms should be C
        for idx in order:
            assert _atom_symbol(mol, idx) == 'C'

    def test_benzene_preserves_all_atoms(self):
        """All ring atoms should be present in the order."""
        mol, ring, order = _get_ring_and_order('c1ccccc1')
        assert set(order) == set(ring)


# ============================================================================
# Edge cases
# ============================================================================

class TestEdgeCases:
    """Edge case tests for ring ordering."""

    def test_order_covers_all_atoms(self):
        """IUPAC order must include all ring atoms."""
        for smiles in ['c1ccoc1', 'c1ccsc1', 'c1cc[nH]c1', 'c1ccncc1',
                       'c1ccncn1', 'c1cnccn1', 'c1ccnnc1']:
            mol, ring, order = _get_ring_and_order(smiles)
            assert set(order) == set(ring), \
                f"Order missing atoms for {smiles}: order={order}, ring={ring}"

    def test_order_has_no_duplicates(self):
        """IUPAC order must not contain duplicate atoms."""
        for smiles in ['c1ccoc1', 'c1ccncc1', 'c1ccncn1', 'c1cnccn1']:
            mol, ring, order = _get_ring_and_order(smiles)
            assert len(order) == len(set(order)), \
                f"Duplicate atoms in order for {smiles}: {order}"

    def test_consecutive_atoms_are_bonded(self):
        """Consecutive atoms in IUPAC order should be bonded in the molecule."""
        for smiles in ['c1ccoc1', 'c1ccncc1', 'c1ccncn1']:
            mol, ring, order = _get_ring_and_order(smiles)
            for i in range(len(order)):
                a1 = order[i]
                a2 = order[(i + 1) % len(order)]
                bond = mol.GetBondBetweenAtoms(a1, a2)
                assert bond is not None, \
                    f"Atoms {a1} and {a2} not bonded in {smiles} order={order}"
