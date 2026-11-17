"""A ligand of a metal admitted to bonded names and named directly, at the best-effort tier
(``rules.organometallics._bonded_metal_ligand_name``).

 "Definitions of terms" (the Blue Book): "names for substituent groups as defined in Section can
also be used as ligand names";: ligands "cited in alphanumerical order", identical ligands multiplied
('trichlorido(methyl)titanium'). The metal's oxidation number as drawn must be one of its charges in
``data/metal_scope.py`` (``perception.metals.bonded_valence_ok``). No PIN.
"""
import pytest

from orthonym.cli import _tiered_namer
from tests.support.rt_assert import name_best_effort

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,expected", [
    ("C[Ti](C)(C)C[Si](C)(C)C", "trimethyl[(trimethylsilyl)methyl]titanium"),
    ("CC(=O)O[Zr](C)(C)C", "(acetyloxy)tri(methyl)zirconium"),
    ("CC(C)CN(CC(C)C)[Hg]C1=CC=CC=C1", "[bis(2-methylpropyl)amino](phenyl)mercury"),
    ("C[Hg]SOC1=CC=CC=C1C(=O)O", "[(2-carboxyphenoxy)sulfanyl](methyl)mercury"),
])
def test_ligand_named_directly(smiles, expected):
    res = name_best_effort(smiles)
    assert (res["name"], res["tier"], res["is_pin"]) == (expected, "systematic_verified", False), res
    assert _tiered_namer("pin", "pin").name_tiered(smiles)["tier"] == "abstain"


def test_ligands_in_alphanumerical_order():
    #: the locants of '3,3-difluoropropyl' do not count; 'chlorido' sorts before 'difluoropropyl'
    assert name_best_effort("FC(F)CC[Ti](Cl)(Cl)Cl")["name"] == "trichlorido(3,3-difluoropropyl)titanium"


def test_valence_outside_the_table_gets_no_metal_name():
    """Aluminium(1) is not one of its charges, so no organometallic name; the drawing is a radical on Al, named on
    its parent hydride "General method", the Blue Book, '-ylidene' with elision of the final 'e')."""
    assert name_best_effort("C(C[Al])C(F)F")["name"] == "(3,3-difluoropropyl)alumanylidene"


def test_group14_parent_keeps_its_senior_group():
    res = name_best_effort("C[Si](C)(C)CC(=O)O")
    assert res["name"] == "(trimethylsilyl)acetic acid" and res["tier"] == "pin_verified", res


@pytest.mark.parametrize("smiles,expected", [
    ("C[Ti](Cl)(Cl)Cl", "trichlorido(methyl)titanium"),
    ("CC[Ti](CC)(CC)CC", "tetraethyltitanium"),
])
def test_names_made_before_are_unchanged(smiles, expected):
    assert name_best_effort(smiles)["name"] == expected
