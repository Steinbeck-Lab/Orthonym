"""The tetralin-class conservation veto (``partial_sat_sp3_substituent_drop``) keeps its documented scope.

 (the Blue Book), "Fused ring systems and mancude ring assemblies composed of fused ring
systems": "Preferred IUPAC names for the partially saturated and fully saturated compounds are formed by
using 'hydro' prefixes" ('decahydronaphthalene (PIN)':24233). The veto stops a name that drops an sp3-ring
substituent of an aromatic + saturated fused system ('NC1CCc2ccccc2C1' -> '4-butylcyclohexan-1-amine').
Two defects made it suppress correct names:

* its shape guard let a system with NO aromatic ring through (decahydroazulene triols: dev2000 row
  CC(C)(O)[C@@H]1CC[C@](C)(O)[C@H]2CC[C@](C)(O)[C@H]2C1), against the docstring's "fully-aromatic ring +
  fully-saturated ring";
* when the tetralin producer refused (it does not spell an ester), the veto answered "dropped" for any
  name, including the ester handler's correct 'methyl 1,2,3,4-tetrahydronaphthalene-2-carboxylate'.
  The refusal now defers to the structure the shipped name describes (its read-back), so the wrong
  molecule is still vetoed and the right one ships.
"""
import pytest
from rdkit import Chem

from orthonym.perception import structure_conservation as sc
from tests.support.read_back import same_key

pytestmark = pytest.mark.opsin_gate

TRIOL = "CC(C)(O)[C@@H]1CC[C@](C)(O)[C@H]2CC[C@](C)(O)[C@H]2C1"


def test_the_saturated_bicycle_is_named(namer):
    row = namer.name_tiered(TRIOL)
    expected = "(1S,3aS,4S,7R,8aS)-7-(2-hydroxypropan-2-yl)-1,4-dimethyldecahydroazulene-1,4-diol"
    assert row["name"] == expected and row["tier"] == "pin_verified", row
    assert same_key(TRIOL, expected)


@pytest.mark.parametrize("smiles,expected", [
    ("COC(=O)C1CCc2ccccc2C1", "methyl 1,2,3,4-tetrahydronaphthalene-2-carboxylate"),
    ("COC(=O)c1ccc2c(c1)C(O)CCC2", "methyl 8-hydroxy-5,6,7,8-tetrahydronaphthalene-2-carboxylate"),
])
def test_tetralin_esters_are_named(namer, smiles, expected):
    row = namer.name_tiered(smiles)
    assert row["name"] == expected and row["tier"] == "pin_verified", row
    assert same_key(smiles, expected)


@pytest.mark.parametrize("smiles,expected", [
    ("NC1CCc2ccccc2C1", "1,2,3,4-tetrahydronaphthalen-2-amine"),
    ("OC(=O)C1CCc2ccccc2C1", "1,2,3,4-tetrahydronaphthalene-2-carboxylic acid"),
    ("OC1CCC2CCCCC2C1", "decahydronaphthalen-2-ol"),
])
def test_neighbours_keep_their_names(namer, smiles, expected):
    assert namer.name(smiles) == expected


def test_a_wrong_molecule_is_still_vetoed_when_the_producer_refuses(monkeypatch):
    # the case the 'produced is None' branch exists for: the producer refused and another path spelled the
    # aromatic half as a butyl chain on a monocycle. Its read-back is a different constitution: vetoed.
    import orthonym.rules.polycyclics as pc
    monkeypatch.setattr(pc, "name_partially_saturated_carbocycle_with_locants", lambda mol: None)
    mol = Chem.MolFromSmiles("NC1CCc2ccccc2C1")
    assert sc.partial_sat_sp3_substituent_drop(mol, "4-butylcyclohexan-1-amine") is True
    # the same refusal, a name that reads back to the molecule: not a drop
    assert sc.partial_sat_sp3_substituent_drop(mol, "1,2,3,4-tetrahydronaphthalen-2-amine") is False


def test_no_aromatic_ring_is_out_of_scope():
    mol = Chem.MolFromSmiles("OC1CCC2CCCCC2C1")
    assert sc.partial_sat_sp3_substituent_drop(mol, "decahydronaphthalen-2-ol") is False


def test_without_a_read_back_the_veto_holds(monkeypatch):
    # no jar / no parse: nothing proves the name, so a refusal of the producer still vetoes (fail closed)
    import orthonym.validation.opsin_roundtrip as rt
    import orthonym.rules.polycyclics as pc
    monkeypatch.setattr(pc, "name_partially_saturated_carbocycle_with_locants", lambda mol: None)
    monkeypatch.setattr(rt, "opsin_parse", lambda name, *a, **k: None)
    mol = Chem.MolFromSmiles("COC(=O)C1CCc2ccccc2C1")
    assert sc.partial_sat_sp3_substituent_drop(mol, "methyl 1,2,3,4-tetrahydronaphthalene-2-carboxylate") is True


def test_the_hydroxy_aminotetralin_is_named(namer):
    smiles = "NC1CCc2cc(O)ccc2C1"
    row = namer.name_tiered(smiles)
    assert row["name"] == "6-amino-5,6,7,8-tetrahydronaphthalen-2-ol" and row["tier"] == "pin_verified", row
    assert same_key(smiles, row["name"])
