"""Isotopic substitution decorator (Wave-2 P2 Tasks 1-4).

BB P-82.2.1 (BlueBookV2.md:43718): the nuclide symbol(s) in parentheses,
preceded by any necessary locants, are inserted before the isotopically
substituted part; a preceding locant takes a hyphen after the parenthesis;
polysubstitution count is a right subscript. P-45.4.1/.4.2/.4.3
(BlueBookV2.md:22212-22232): lowest locants to modified positions, then to
higher atomic number, then to higher mass number.
"""
import pytest
from rdkit import Chem

from orthonym.rules.isotopes import (
    has_isotopes,
    strip_isotopes,
)


class TestHasIsotopes:
    @pytest.mark.parametrize("smiles,expected", [
        ("[2H]C([2H])([2H])CO", True),   # trideuterio-ethanol
        ("[14CH3]CO", True),             # 14C ethanol
        ("[13CH3]CO", True),
        ("[3H]C([3H])([3H])CO", True),   # tritium
        ("CCO", False),                  # unlabeled — must be inert
        ("c1ccccc1", False),
        ("[Na+].[O-]C(=O)C", False),     # charged, unlabeled
    ])
    def test_detection(self, smiles, expected):
        mol = Chem.MolFromSmiles(smiles)
        assert has_isotopes(mol) is expected

    def test_none_mol_is_false(self):
        assert has_isotopes(None) is False


class TestStripIsotopes:
    def test_strip_returns_unlabeled_copy_and_map(self):
        mol = Chem.MolFromSmiles("[14CH3]CO")
        stripped, label_map = strip_isotopes(mol)
        # every isotope cleared on the copy
        assert all(a.GetIsotope() == 0 for a in stripped.GetAtoms())
        # original untouched (copy semantics)
        assert any(a.GetIsotope() != 0 for a in mol.GetAtoms())
        # canonical skeleton is plain ethanol
        assert Chem.MolToSmiles(stripped) == Chem.MolToSmiles(Chem.MolFromSmiles("CCO"))
        # the one 14C atom is recorded with its mass number
        assert list(label_map.values()) == [14]

    def test_multi_label_map(self):
        mol = Chem.MolFromSmiles("[2H]C([2H])([2H])CO")
        stripped, label_map = strip_isotopes(mol)
        assert sorted(label_map.values()) == [2, 2, 2]
        assert Chem.MolToSmiles(stripped) == Chem.MolToSmiles(Chem.MolFromSmiles("CCO"))
