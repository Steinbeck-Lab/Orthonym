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


from orthonym.rules.isotopes import (
    nuclide_symbol,
    format_isotope_descriptor,
)


class TestNuclideSymbol:
    @pytest.mark.parametrize("mass,element,out", [
        (14, "C", "14C"),
        (2, "H", "2H"),
        (3, "H", "3H"),
        (13, "C", "13C"),
        (18, "O", "18O"),
        (12, "C", "12C"),
        (81, "Br", "81Br"),
    ])
    def test_symbol(self, mass, element, out):
        assert nuclide_symbol(mass, element) == out


class TestFormatIsotopeDescriptor:
    def test_trideuterio_with_locant(self):
        # (2,2,2-2H3)  three 2H at locant 2 -> single grouped token, count 3
        # BB P-84 (BlueBookV2.md:44506): (2,2,2-2H3)ethan-1-ol
        groups = [(2, 2, "H", 3)]
        assert format_isotope_descriptor(groups) == "(2,2,2-2H3)"

    def test_single_14c_with_locant(self):
        # BB:43740 (2-13C); here (2-14C)
        assert format_isotope_descriptor([(2, 14, "C", 1)]) == "(2-14C1)"

    def test_deuterio_no_locant_single_position_ring_substituent(self):
        # BB:43730 (2H3)methoxybenzene — descriptor at front, count 3, no locant
        assert format_isotope_descriptor([(None, 2, "H", 3)]) == "(2H3)"

    def test_12c_methane_no_locant(self):
        # BB:43724 trichloro(12C)methane — single position; Orthonym emits the
        # count-subscript form (12C1) which OPSIN also parses.
        assert format_isotope_descriptor([(None, 12, "C", 1)]) == "(12C1)"

    def test_deuterio_methane_count_one(self):
        # BB:43726 (2H1)methane — count subscript kept even for count 1
        assert format_isotope_descriptor([(None, 2, "H", 1)]) == "(2H1)"

    def test_two_nuclides_same_place_alphabetical_then_mass(self):
        # P-82.2.1: cited alphabetically by element, then by mass number.
        # elements alphabetical C < H, so 13C first.
        groups = [(1, 2, "H", 1), (1, 13, "C", 1)]
        assert format_isotope_descriptor(groups) == "(1-13C1,1-2H1)"
