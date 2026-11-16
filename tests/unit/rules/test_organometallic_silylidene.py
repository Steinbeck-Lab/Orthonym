"""A silylidene or germylidene ligand on a metal admitted to bonded names (``rules.organometallics.
_multiply_bonded_ligand_name``).

 "Definitions of terms" (the Blue Book): "names for substituent groups as defined in Section can
also be used as ligand names"; "Specific method and retained names" (:40476) names the Group 14 radical
with two hydrogen atoms removed from one atom '-ylidene' ('silylidene'). The metal's oxidation number must be one of
its charges in ``data/metal_scope.py`` (``perception.metals.bonded_valence_ok``). No PIN.
"""
import pytest
from rdkit import Chem

from orthonym.cli import _tiered_namer
from orthonym.perception.metals import bonded_valence_ok
from tests.support.rt_assert import name_best_effort, name_is_rt_exact

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,expected", [
    ("C[Si](=[Zr])C", "(dimethylsilylidene)zirconium"),
    ("C[Si](=[Zr](Cl)Cl)C", "dichlorido(dimethylsilylidene)zirconium"),
    ("C[Ge](=[Zr+2])C", "(dimethylgermylidene)zirconium(2+)"),
    ("[SiH2]=[Ti]", "silylidenetitanium"),
])
def test_silylidene_ligand(smiles, expected):
    res = name_best_effort(smiles)
    assert (res["name"], res["tier"], res["is_pin"]) == (expected, "systematic_verified", False), res
    assert name_is_rt_exact(res["name"], smiles) or res["verified"] == "opsin_twin", res
    assert _tiered_namer("pin", "pin").name_tiered(smiles)["tier"] == "abstain"


@pytest.mark.parametrize("smiles,expected", [
    ("C[Ti](Cl)(Cl)Cl", "trichlorido(methyl)titanium"),
    ("C=[W]", "methylidenetungsten"),
])
def test_names_made_before_are_unchanged(smiles, expected):
    assert name_best_effort(smiles)["name"] == expected


def test_valence_outside_the_table_is_refused():
    mol = Chem.MolFromSmiles("C[Si](=[Ti+4])C")     # titanium(6) is not one of its charges
    metal = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "Ti")
    assert bonded_valence_ok(mol, metal) is False
    assert name_best_effort("C[Si](=[Ti+4])C")["name"] is None


def test_dative_bonds_do_not_count_in_the_oxidation_number():
    mol = Chem.MolFromSmiles("N->[Ti](C)(Cl)(Cl)Cl")
    assert bonded_valence_ok(mol, next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "Ti")) is True


def test_valence_in_the_table():
    mol = Chem.MolFromSmiles("C[Si](=[Zr])C")
    assert bonded_valence_ok(mol, next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "Zr")) is True
