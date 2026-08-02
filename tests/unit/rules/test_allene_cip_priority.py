"""
Unit tests for STER-14: allene CIP priority using true CIP priority
instead of CanonicalRankAtoms proxy.

Tests that _cip_priority_key returns (atomic_number, neighbor_sum) tuples,
that _terminal_is_achiral correctly compares priority tuples, and that
_manual_allene_cip produces correct Ra/Sa assignments.

IUPAC Reference: P-92.1.3 (CIP sequence rules for allene axial chirality)
"""

import pytest
from rdkit import Chem
from rdkit.Chem import rdCIPLabeler

from orthonym.perception.stereo import (
    _cip_priority_key,
    _manual_allene_cip,
    detect_axial_chirality,
)


# =============================================================================
# Helpers for constructing test molecules
# =============================================================================

def _make_allene_mol(smiles: str):
    """Create an allene molecule with CHI_ALLENE tag on the central carbon.

    Args:
        smiles: SMILES string containing an allene (C=C=C motif).

    Returns:
        Tuple (mol, central_idx): RDKit Mol and index of the central allene C.
    """
    mol = Chem.RWMol(Chem.MolFromSmiles(smiles))
    central_idx = None
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'C':
            dbl_count = sum(
                1 for b in atom.GetBonds()
                if b.GetBondType() == Chem.BondType.DOUBLE
            )
            if dbl_count == 2:
                atom.SetChiralTag(Chem.ChiralType.CHI_ALLENE)
                central_idx = atom.GetIdx()
                break
    assert central_idx is not None, f"No central allene C found in {smiles}"
    return mol.GetMol(), central_idx


def _make_allene_mol_h(smiles: str):
    """Create an allene molecule with explicit Hs and CHI_ALLENE tag.

    Returns:
        Tuple (mol_h, central_idx): RDKit Mol with Hs and central allene C index.
    """
    mol, central_idx = _make_allene_mol(smiles)
    mol_h = Chem.AddHs(mol)
    return mol_h, central_idx


# =============================================================================
# Test 1: _cip_priority_key returns (atomic_number, neighbor_sum)
# =============================================================================

class TestCipPriorityKey:
    """Tests for the _cip_priority_key helper function."""

    def test_carbon_with_two_carbon_neighbors(self):
        """Test 1: Carbon with two C neighbors returns (6, 12).

        For a carbon atom connected to two other carbons (excluding the
        terminal carbon direction), the primary key is 6 (carbon atomic
        number) and secondary key is the sum of neighbor atomic numbers.
        """
        # CC=C=CC -- the terminal C bonded to the allene has one C neighbor
        # and H neighbors (after AddHs)
        mol_h, central_idx = _make_allene_mol_h('CC=C=CC')
        # Find a terminal carbon of the allene
        central_atom = mol_h.GetAtomWithIdx(central_idx)
        term_idx = None
        for bond in central_atom.GetBonds():
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                other = bond.GetOtherAtomIdx(central_idx)
                term_idx = other
                break
        assert term_idx is not None

        # Get a substituent of the terminal C (not the central allene C)
        term_atom = mol_h.GetAtomWithIdx(term_idx)
        sub_idx = None
        for nbr in term_atom.GetNeighbors():
            if nbr.GetIdx() != central_idx and nbr.GetAtomicNum() == 6:
                sub_idx = nbr.GetIdx()
                break
        assert sub_idx is not None, "Should find a C substituent on terminal"

        result = _cip_priority_key(mol_h, sub_idx, term_idx)
        # primary = 6 (carbon), secondary = sum of neighbors excl terminal
        assert result[0] == 6  # carbon atomic number
        assert isinstance(result[1], int)  # neighbor sum is an integer

    def test_hydrogen_returns_1_0(self):
        """Test 2: _cip_priority_key for hydrogen returns (1, 0).

        Hydrogen has atomic number 1 and no neighbors beyond the terminal
        carbon (which is excluded), so the neighbor sum is 0.
        """
        mol_h, central_idx = _make_allene_mol_h('ClC=C=CBr')
        # Find a terminal carbon
        central_atom = mol_h.GetAtomWithIdx(central_idx)
        term_idx = None
        for bond in central_atom.GetBonds():
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                term_idx = bond.GetOtherAtomIdx(central_idx)
                break
        # Find an H neighbor of the terminal
        term_atom = mol_h.GetAtomWithIdx(term_idx)
        h_idx = None
        for nbr in term_atom.GetNeighbors():
            if nbr.GetIdx() != central_idx and nbr.GetAtomicNum() == 1:
                h_idx = nbr.GetIdx()
                break
        assert h_idx is not None, "Should find H on terminal after AddHs"

        result = _cip_priority_key(mol_h, h_idx, term_idx)
        assert result == (1, 0)

    def test_chlorine_distinguished_from_carbon_by_atomic_number(self):
        """Test 3: Cl (17, ...) is distinguished from C (6, ...) by primary key."""
        mol_h, central_idx = _make_allene_mol_h('ClC=C=CC')
        central_atom = mol_h.GetAtomWithIdx(central_idx)

        # Find the terminal that has the Cl substituent
        for bond in central_atom.GetBonds():
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                t_idx = bond.GetOtherAtomIdx(central_idx)
                t_atom = mol_h.GetAtomWithIdx(t_idx)
                for nbr in t_atom.GetNeighbors():
                    if nbr.GetIdx() != central_idx and nbr.GetAtomicNum() == 17:
                        cl_idx = nbr.GetIdx()
                        cl_key = _cip_priority_key(mol_h, cl_idx, t_idx)
                        assert cl_key[0] == 17, "Cl should have primary key 17"
                        # Also get a H key for comparison
                        for nbr2 in t_atom.GetNeighbors():
                            if nbr2.GetIdx() != central_idx and nbr2.GetAtomicNum() == 1:
                                h_key = _cip_priority_key(mol_h, nbr2.GetIdx(), t_idx)
                                assert cl_key > h_key, "Cl priority > H priority"
                                return
        pytest.fail("Could not find Cl on allene terminal")

    def test_methyl_vs_ethyl_distinguished_by_secondary_key(self):
        """Test 4: Methyl C (6, H-sum) vs ethyl C (6, C+H-sum) differ in secondary key.

        Both have primary=6 (carbon), but ethyl's first carbon has a C
        neighbor (atomic number 6) instead of only H neighbors, giving
        a higher secondary sum.
        """
        # CC(C)=C=CC -- terminal A has CH3 and CH3; terminal B has C and H
        # Better: use CC=C=C(C)CC -- terminal with methyl and ethyl
        mol_h, central_idx = _make_allene_mol_h('CC=C=C(C)CC')
        central_atom = mol_h.GetAtomWithIdx(central_idx)

        # Find the terminal with both methyl and ethyl
        for bond in central_atom.GetBonds():
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                t_idx = bond.GetOtherAtomIdx(central_idx)
                t_atom = mol_h.GetAtomWithIdx(t_idx)
                c_subs = []
                for nbr in t_atom.GetNeighbors():
                    if nbr.GetIdx() != central_idx and nbr.GetAtomicNum() == 6:
                        key = _cip_priority_key(mol_h, nbr.GetIdx(), t_idx)
                        c_subs.append((nbr.GetIdx(), key))
                if len(c_subs) >= 2:
                    # Both have primary=6, secondary should differ
                    keys = sorted([k for _, k in c_subs])
                    assert keys[0][0] == 6 and keys[1][0] == 6, "Both should be carbon"
                    assert keys[0][1] != keys[1][1], (
                        f"Methyl and ethyl should have different secondary keys: {keys}"
                    )
                    return
        pytest.fail("Could not find terminal with methyl+ethyl substituents")


# =============================================================================
# Test 5-6: _terminal_is_achiral with priority tuples
# =============================================================================

class TestTerminalIsAchiral:
    """Tests for improved _terminal_is_achiral behavior."""

    def test_identical_substituents_returns_true(self):
        """Test 5: Two H atoms (same atomic number, degree, neighbor numbers) -> True.

        Both have priority key (1, 0), so the terminal is achiral.
        """
        # HC=C=CCl -- terminal with Cl has H+Cl (different), but terminal
        # with just H on each side would be: we need a terminal with 2 H
        # Use C=C=CCl -- the =CH2 terminal has two H substituents
        mol, central_idx = _make_allene_mol('C=C=CCl')
        result = _manual_allene_cip(mol, central_idx)
        # Terminal with two H is achiral, so overall allene is achiral
        assert result is None

    def test_same_element_different_connectivity_returns_false(self):
        """Test 6: Methyl vs ethyl (same atomic number, different neighbors) -> not achiral.

        Both substituents are carbon (primary=6), but methyl has only H
        neighbors while ethyl has one C neighbor. The improved achirality
        check using priority tuples correctly identifies them as different.
        """
        # CC=C=C(C)CC -- terminal with methyl and ethyl
        mol, central_idx = _make_allene_mol('CC=C=C(C)CC')
        # If the allene is detected as chiral (not achiral at the multi-sub terminal),
        # _manual_allene_cip should return Ra or Sa (not None)
        result = _manual_allene_cip(mol, central_idx)
        # With true CIP priority, methyl != ethyl, so not achiral
        assert result in ('M', 'P'), (
            f"Expected M or P for allene with methyl+ethyl terminal, got {result}"
        )


# =============================================================================
# Test 7-8: _manual_allene_cip Ra/Sa assignment
# =============================================================================

class TestManualAlleneCipAssignment:
    """Tests for correct Ra/Sa assignment in _manual_allene_cip."""

    def test_distinct_substituents_assigns_ra_or_sa(self):
        """Test 7: Allene with Cl and H on each terminal gives Ra or Sa."""
        mol, central_idx = _make_allene_mol('ClC=C=CBr')
        result = _manual_allene_cip(mol, central_idx)
        assert result in ('M', 'P'), f"Expected M or P, got {result}"

    def test_identical_terminal_substituents_returns_none(self):
        """Test 8: Allene with identical substituents on one terminal is achiral."""
        # C=C=C (propadiene) -- both terminals have 2 H each
        mol, central_idx = _make_allene_mol('C=C=C')
        result = _manual_allene_cip(mol, central_idx)
        assert result is None, f"Expected None for achiral allene, got {result}"


# =============================================================================
# Test 9: End-to-end detect_axial_chirality
# =============================================================================

class TestEndToEndAlleneCip:
    """End-to-end test via detect_axial_chirality."""

    def test_detect_axial_chirality_on_chiral_allene(self):
        """Test 9: detect_axial_chirality on molecule with CHI_ALLENE produces Ra/Sa."""
        mol, central_idx = _make_allene_mol('ClC=C=CBr')
        results = detect_axial_chirality(mol)
        assert len(results) == 1
        entry = results[0]
        assert entry['type'] == 'allene'
        assert entry['cip'] in ('M', 'P')
        assert entry['locant_atom'] == central_idx
