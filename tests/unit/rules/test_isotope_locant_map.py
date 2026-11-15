"""Isotope descriptors from OPSIN's locants of the unlabelled name (``rules/isotope_locant_map.py``, lane I).

 "nuclide descriptors" (the Blue Book): the nuclide symbols with their locants, in parentheses,
before the part of the name they refer to ('trichloro(12C)methane (PIN)'); an integer locant is repeated per atom
('(1,1,1,3,3-2H5)pentan-2-one',:44208). A descriptor inside a substituent steps its marks up,
'1,2-di[(13C)methyl]benzene',:7492). The decorator runs after the offset search finds nothing; every name reads back
to the full standard InChIKey (isotopic layer included) and is systematic_verified.
"""
import pytest
from rdkit import Chem

from orthonym.rules.isotope_locant_map import locant_map_descriptor
from tests.support.rt_assert import name_best_effort

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles, name", [
    ("[2H]C1=CC2=C(C=CC(=N2)N(CO)C3=CC=CC=C3)N=C1", "{N-[(7-2H)-1,5-naphthyridin-2-yl]anilino}methanol"),
    ("[2H]c1c([2H])c([2H])c2c(CS(N)(=O)=O)coc2c1[2H]", "1-[(4,5,6,7-2H4)-1-benzofuran-3-yl]methanesulfonamide"),
])
def test_a_label_the_offset_search_misses(smiles, name):
    r = name_best_effort(smiles)
    assert r["name"] == name and r["tier"] == "systematic_verified" and not r["is_pin"], r


def test_the_offset_search_keeps_its_names():
    r = name_best_effort("OCC([2H])([2H])C")
    assert r["name"] == "(2,2-2H2)propan-1-ol" and r["tier"] == "pin_verified"


def test_the_descriptor_repeats_a_locant_per_atom_and_counts_a_lone_deuteron_on_ch3():
    assert locant_map_descriptor(Chem.MolFromSmiles("OCC([2H])([2H])C"), "propan-1-ol") == "(2,2-2H2)"
    assert locant_map_descriptor(Chem.MolFromSmiles("[2H]CCO"), "ethan-1-ol") == "(2-2H1)"
    assert locant_map_descriptor(Chem.MolFromSmiles("ClC(Cl)(Cl)[13CH3]"), "1,1,1-trichloroethane") == "(2-13C)"


def test_a_skeleton_name_of_another_molecule_gives_no_name():
    from orthonym.rules.isotope_locant_map import decorate_by_locant_map
    assert decorate_by_locant_map(Chem.MolFromSmiles("OCC([2H])([2H])C"), "butan-1-ol") is None
    assert decorate_by_locant_map(Chem.MolFromSmiles("OCC([2H])([2H])C"), "propan-2-ol") is None
