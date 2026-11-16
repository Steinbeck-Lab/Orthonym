"""The additive branch records its halido names for the read-back of their 'chloro' spelling
(``validation.halido_twin.note_bonded_halido_name``).

 names the halide ligands 'chlorido' ('trichlorido(methyl)titanium'); OPSIN reads the 'chloro' spelling of the
same structure, so the name ships through that twin (label ``opsin_twin``). No PIN.
"""
import pytest

from tests.support.rt_assert import name_best_effort

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,expected", [
    ("Cl[Zr](C)(C)Cl", "dichloridobis(methyl)zirconium"),
    ("Cl[Hf](C)(C)Cl", "dichloridobis(methyl)hafnium"),
])
def test_additive_halido_name_ships_through_its_twin(smiles, expected):
    res = name_best_effort(smiles)
    assert (res["name"], res["tier"], res["verified"]) == (expected, "systematic_verified", "opsin_twin"), res


def test_second_naming_gives_the_same_name():
    first = name_best_effort("Cl[Zr](C)(C)Cl")["name"]
    assert name_best_effort("Cl[Zr](C)(C)Cl")["name"] == first


def test_titanium_unchanged():
    assert name_best_effort("C[Ti](Cl)(Cl)Cl")["name"] == "trichlorido(methyl)titanium"
