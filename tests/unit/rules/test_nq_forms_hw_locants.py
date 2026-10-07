"""Lane L2 proper fix: in the Hantzsch-Widman builder, from the heteroatom
set -- no rewrite of a built name.

 (the Blue Book) "All locants are omitted for parent Hantzsch-Widman
names if there is only one heteroatom or if there is no ambiguity if locants are
omitted"; '1H-tetrazole (PIN) (not 1H-1,2,3,4-tetrazole)' (:2989).
"""
import pytest

from orthonym import Orthonym
from orthonym.rules.heterocycles import build_hw_name
from tests.support.rt_assert import name_is_rt_exact


@pytest.mark.parametrize("heteroatoms,size,saturated,aromatic,expected", [
    ([(1, "N"), (2, "N"), (3, "N"), (4, "N")], 5, False, False, "tetrazole"),
    ([(1, "N"), (2, "N"), (3, "N"), (4, "N"), (5, "N")], 5, False, True, "pentazole"),
    ([(1, "O"), (2, "O")], 3, True, False, "dioxirane"),
    ([(1, "N"), (2, "N")], 3, False, False, "diazirine"),
    # an isomer exists: the locants stay
    ([(1, "N"), (2, "N"), (3, "N")], 5, False, True, "1,2,3-triazole"),
    ([(1, "O"), (3, "O")], 5, True, False, "1,3-dioxolane"),
    ([(1, "O"), (2, "N")], 3, True, False, "1,2-oxaziridine"),
])
def test_hw_locants_are_omitted_when_no_isomer_exists(heteroatoms, size, saturated,
                                                      aromatic, expected):
    assert build_hw_name(heteroatoms, size, saturated, aromatic) == expected


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", [
    ("C1N=NN=N1", "5H-tetrazole"),
    ("CC1(C)N=NN=N1", "5,5-dimethyl-5H-tetrazole"),
    ("OC(=O)C1(C)N=NN=N1", "5-methyl-5H-tetrazole-5-carboxylic acid"),
    ("C1OO1", "dioxirane"),
    ("C1N=N1", "3H-diazirine"),
    ("c1nn[nH]n1", "2H-tetrazole"),            # without the deleted table row
    ("c1nnn[nH]1", "1H-tetrazole"),
])
def test_the_pin_tier_names_cite_no_redundant_heteroatom_locant(smiles, expected):
    row = Orthonym().name_tiered(smiles)
    assert (row.get("name"), row.get("tier")) == (expected, "pin_verified"), row
    assert name_is_rt_exact(expected, smiles)
