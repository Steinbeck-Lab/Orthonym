"""
TDD tests for fragment capping in the decomposition engine.

Tests that cleave_and_cap() produces valid H-capped and OH-capped
fragment SMILES from a molecule and bond info list.
"""

import pytest
from rdkit import Chem

from orthonym.decomposition.bond_cleavage import find_cleavable_bonds
from orthonym.decomposition.fragment_capping import cleave_and_cap


# ---------------------------------------------------------------------------
# Basic H-capping
# ---------------------------------------------------------------------------

class TestBasicHCapping:
    """Tests for basic fragment capping (H-cap mode)."""

    def test_ester_cleavage_produces_two_fragments(self):
        """Cleaving ethyl acetate at ester bond produces 2 fragments."""
        mol = Chem.MolFromSmiles("CC(=O)OCC")
        bonds = find_cleavable_bonds(mol)
        ester_bonds = [b for b in bonds if b["type"] == "ester"]
        assert len(ester_bonds) == 1

        fragments = cleave_and_cap(mol, ester_bonds, acid_side_oh=False)
        assert len(fragments) == 2

    def test_fragments_are_valid_smiles(self):
        """All fragment SMILES must be parseable by RDKit."""
        mol = Chem.MolFromSmiles("CC(=O)OCC")
        bonds = find_cleavable_bonds(mol)
        ester_bonds = [b for b in bonds if b["type"] == "ester"]

        fragments = cleave_and_cap(mol, ester_bonds, acid_side_oh=False)
        for frag in fragments:
            parsed = Chem.MolFromSmiles(frag["smiles"])
            assert parsed is not None, f"Invalid SMILES: {frag['smiles']}"

    def test_no_dummy_atoms_in_fragments(self):
        """Fragment SMILES should NOT contain dummy atoms (atomic num 0)."""
        mol = Chem.MolFromSmiles("CC(=O)OCC")
        bonds = find_cleavable_bonds(mol)
        ester_bonds = [b for b in bonds if b["type"] == "ester"]

        fragments = cleave_and_cap(mol, ester_bonds, acid_side_oh=False)
        for frag in fragments:
            parsed = Chem.MolFromSmiles(frag["smiles"])
            for atom in parsed.GetAtoms():
                assert atom.GetAtomicNum() != 0, \
                    f"Dummy atom found in fragment: {frag['smiles']}"

    def test_no_isotope_labels_in_fragments(self):
        """Fragment SMILES should NOT contain isotope labels."""
        mol = Chem.MolFromSmiles("CC(=O)OCC")
        bonds = find_cleavable_bonds(mol)
        ester_bonds = [b for b in bonds if b["type"] == "ester"]

        fragments = cleave_and_cap(mol, ester_bonds, acid_side_oh=False)
        for frag in fragments:
            parsed = Chem.MolFromSmiles(frag["smiles"])
            for atom in parsed.GetAtoms():
                assert atom.GetIsotope() == 0, \
                    f"Isotope label found in fragment: {frag['smiles']}"


# ---------------------------------------------------------------------------
# OH-capping for acid-side fragments
# ---------------------------------------------------------------------------

class TestOHCapping:
    """Tests for OH-capping of acid-side ester fragments."""

    def test_ester_acid_side_produces_carboxylic_acid(self):
        """Cleaving ethyl acetate with acid_side_oh=True: acid fragment
        should be acetic acid (CC(=O)O or CC(O)=O)."""
        mol = Chem.MolFromSmiles("CC(=O)OCC")
        bonds = find_cleavable_bonds(mol)
        ester_bonds = [b for b in bonds if b["type"] == "ester"]

        fragments = cleave_and_cap(mol, ester_bonds, acid_side_oh=True)
        acid_frags = [f for f in fragments if f["side"] == "acid"]
        assert len(acid_frags) == 1
        # Acetic acid canonical SMILES
        acid_smiles = acid_frags[0]["smiles"]
        acid_mol = Chem.MolFromSmiles(acid_smiles)
        expected_mol = Chem.MolFromSmiles("CC(=O)O")
        assert Chem.MolToSmiles(acid_mol) == Chem.MolToSmiles(expected_mol), \
            f"Expected acetic acid, got {acid_smiles}"

    def test_ester_alkyl_side_produces_alcohol(self):
        """Cleaving ethyl acetate with acid_side_oh=True: alkyl fragment
        should be ethanol (CCO)."""
        mol = Chem.MolFromSmiles("CC(=O)OCC")
        bonds = find_cleavable_bonds(mol)
        ester_bonds = [b for b in bonds if b["type"] == "ester"]

        fragments = cleave_and_cap(mol, ester_bonds, acid_side_oh=True)
        alkyl_frags = [f for f in fragments if f["side"] == "alkyl"]
        assert len(alkyl_frags) == 1
        alkyl_smiles = alkyl_frags[0]["smiles"]
        alkyl_mol = Chem.MolFromSmiles(alkyl_smiles)
        expected_mol = Chem.MolFromSmiles("CCO")
        assert Chem.MolToSmiles(alkyl_mol) == Chem.MolToSmiles(expected_mol), \
            f"Expected ethanol, got {alkyl_smiles}"


# ---------------------------------------------------------------------------
# Sanitization
# ---------------------------------------------------------------------------

class TestSanitization:
    """Tests for fragment sanitization and canonicalization."""

    def test_all_fragments_are_canonical_smiles(self):
        """Returned SMILES should be canonical (round-trip equals self)."""
        mol = Chem.MolFromSmiles("CC(=O)OCC")
        bonds = find_cleavable_bonds(mol)
        ester_bonds = [b for b in bonds if b["type"] == "ester"]

        fragments = cleave_and_cap(mol, ester_bonds, acid_side_oh=True)
        for frag in fragments:
            parsed = Chem.MolFromSmiles(frag["smiles"])
            canonical = Chem.MolToSmiles(parsed)
            assert frag["smiles"] == canonical, \
                f"Non-canonical SMILES: {frag['smiles']} vs {canonical}"

    def test_amide_cleavage_fragments_valid(self):
        """Cleaving N-methylacetamide produces valid sanitized fragments."""
        mol = Chem.MolFromSmiles("CC(=O)NC")
        bonds = find_cleavable_bonds(mol)
        amide_bonds = [b for b in bonds if b["type"] == "amide"]
        assert len(amide_bonds) >= 1

        fragments = cleave_and_cap(mol, amide_bonds, acid_side_oh=True)
        for frag in fragments:
            parsed = Chem.MolFromSmiles(frag["smiles"])
            assert parsed is not None, f"Invalid SMILES: {frag['smiles']}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Tests for edge cases in fragment capping."""

    def test_single_bond_cleavage_two_fragments(self):
        """Single bond cleavage produces at least 2 fragments."""
        mol = Chem.MolFromSmiles("CC(=O)OCC")
        bonds = find_cleavable_bonds(mol)
        ester_bonds = [b for b in bonds if b["type"] == "ester"]
        assert len(ester_bonds) == 1

        fragments = cleave_and_cap(mol, ester_bonds)
        assert len(fragments) >= 2

    def test_multiple_bond_cleavage_three_fragments(self):
        """Cleaving 2 ester bonds produces 3 fragments."""
        mol = Chem.MolFromSmiles("CC(=O)OCCC(=O)OCC")
        bonds = find_cleavable_bonds(mol)
        ester_bonds = [b for b in bonds if b["type"] == "ester"]
        assert len(ester_bonds) == 2

        fragments = cleave_and_cap(mol, ester_bonds)
        assert len(fragments) == 3

    def test_fragment_has_side_key(self):
        """Each fragment dict should have a 'side' key."""
        mol = Chem.MolFromSmiles("CC(=O)OCC")
        bonds = find_cleavable_bonds(mol)
        ester_bonds = [b for b in bonds if b["type"] == "ester"]

        fragments = cleave_and_cap(mol, ester_bonds)
        for frag in fragments:
            assert "side" in frag, f"Missing 'side' key in fragment: {frag}"

    def test_fragment_has_smiles_key(self):
        """Each fragment dict should have a 'smiles' key."""
        mol = Chem.MolFromSmiles("CC(=O)OCC")
        bonds = find_cleavable_bonds(mol)
        ester_bonds = [b for b in bonds if b["type"] == "ester"]

        fragments = cleave_and_cap(mol, ester_bonds)
        for frag in fragments:
            assert "smiles" in frag, f"Missing 'smiles' key in fragment: {frag}"


# ---------------------------------------------------------------------------
# DECO-21: Middle fragment OH capping
# ---------------------------------------------------------------------------

class TestMiddleFragmentOHCapping:
    """Tests for middle fragment OH capping in multi-bond cleavage.

    When a molecule is cleaved at multiple bonds, middle fragments
    (between two cleavage points) have both label-1 and label-2 dummies.
    Label-2 dummies on middle fragments correspond to the acid side of
    the original bond, so they should receive OH capping.
    """

    def test_middle_fragment_oh_capping(self):
        """Middle fragment in mixed ester+amide should receive OH capping on label-2 dummies.

        CC(=O)OCCC(=O)NC: ester bond + amide bond creates a middle fragment
        with BOTH label-1 (acid side of amide) and label-2 (alkyl side of ester).
        Label-2 dummies should be OH-capped to reconstruct acid-side FG.

        Before fix: middle fragment gets H-capped on label-2 -> O=CCCO (2 O)
        After fix: middle fragment gets OH-capped on label-2 -> OC(=O)CCO (3 O)
        """
        mol = Chem.MolFromSmiles("CC(=O)OCCC(=O)NC")
        bonds = find_cleavable_bonds(mol)
        assert len(bonds) >= 2, f"Expected >= 2 cleavable bonds, got {len(bonds)}"

        fragments = cleave_and_cap(mol, bonds, acid_side_oh=True)
        middle_frags = [f for f in fragments if f["side"] == "middle"]
        assert len(middle_frags) == 1, (
            f"Expected 1 middle fragment, got sides: "
            f"{[f['side'] for f in fragments]}"
        )

        # After OH-capping fix, middle fragment should have 3 oxygen atoms:
        # carbonyl O, the ester-derived O, and the new OH from label-2 capping
        middle_mol = Chem.MolFromSmiles(middle_frags[0]["smiles"])
        assert middle_mol is not None
        o_count = sum(1 for a in middle_mol.GetAtoms() if a.GetAtomicNum() == 8)
        assert o_count >= 3, (
            f"Middle fragment '{middle_frags[0]['smiles']}' should have >= 3 "
            f"oxygen atoms (including OH from label-2 capping), got {o_count}"
        )

    def test_acid_fragment_still_oh_capped(self):
        """Acid-side fragments should still receive OH capping (no regression)."""
        mol = Chem.MolFromSmiles("CC(=O)OCC")
        bonds = find_cleavable_bonds(mol)
        ester_bonds = [b for b in bonds if b["type"] == "ester"]

        fragments = cleave_and_cap(mol, ester_bonds, acid_side_oh=True)
        acid_frags = [f for f in fragments if f["side"] == "acid"]
        assert len(acid_frags) == 1
        acid_mol = Chem.MolFromSmiles(acid_frags[0]["smiles"])
        # Should be acetic acid (has C(=O)O)
        has_oxygen = sum(1 for a in acid_mol.GetAtoms() if a.GetAtomicNum() == 8)
        assert has_oxygen == 2, (
            f"Acid fragment should have 2 oxygen atoms (C(=O)OH), "
            f"got {has_oxygen} in '{acid_frags[0]['smiles']}'"
        )

    def test_alkyl_fragment_still_h_capped(self):
        """Alkyl-side fragments should still receive H capping (no regression)."""
        mol = Chem.MolFromSmiles("CC(=O)OCC")
        bonds = find_cleavable_bonds(mol)
        ester_bonds = [b for b in bonds if b["type"] == "ester"]

        fragments = cleave_and_cap(mol, ester_bonds, acid_side_oh=True)
        alkyl_frags = [f for f in fragments if f["side"] == "alkyl"]
        assert len(alkyl_frags) == 1
        alkyl_mol = Chem.MolFromSmiles(alkyl_frags[0]["smiles"])
        # Should be ethanol (CCO) -- has 1 oxygen from the ester O being kept
        # Actually, for alkyl side, the dummy gets H-capped, producing ethane (CC)
        # NOT ethanol. The ester oxygen goes with the acid side.
        expected = Chem.MolToSmiles(Chem.MolFromSmiles("CCO"))
        actual = Chem.MolToSmiles(alkyl_mol)
        assert actual == expected, (
            f"Alkyl fragment should be ethanol (CCO), got '{actual}'"
        )
