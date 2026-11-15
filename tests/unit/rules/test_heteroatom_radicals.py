"""Heteroatom-centred radicals named by the all-organic program (lane C).

* A Group 15 centre with five bonds and electrons is the λ5 parent hydride radical: 'Monovalent
  radicals' (the Blue Book), 'tert-butoxytri(phenyl)-λ5-phosphanyl (PIN)' (:40442).
* An acyl group on the centre that the acyl helper does not name (an alkoxycarbonyl) is named by the trusted ligand
  namer (``rules.centre_ligands``).
* A monovalent S, Se or Te radical whose group the chalcogen producer does not name is a substituted mononuclear
  centre: (:40721) "named on the basis of preselected parent radical names, such as 'sulfanyl', 'selanyl'".
Every such name ships through the strict radical read-back and is systematic_verified, never a PIN.
"""
import pytest

from tests.support.rt_assert import name_best_effort

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles, name", [
    ("C[As](C)(C)C", "tetramethyl-λ5-arsanyl"),
    ("C[P](C)(C)C", "tetramethyl-λ5-phosphanyl"),
    ("CC[As](C)(C)O", "ethyl(hydroxy)di(methyl)-λ5-arsanyl"),
    ("CC(C)(C)OC(=O)[As]C1=CC=CC=C1", "(tert-butoxycarbonyl)(phenyl)arsanyl"),
    ("C(#N)[Te]", "cyanotellanyl"),
])
def test_heteroatom_radical(smiles, name):
    r = name_best_effort(smiles)
    assert r["name"] == name, r
    assert r["tier"] == "systematic_verified" and not r["is_pin"], r


@pytest.mark.parametrize("smiles", ["C[As]C", "CC[Te]", "CC(=O)[As]C"])
def test_the_radicals_named_before_keep_their_label(smiles):
    assert name_best_effort(smiles)["tier"] == "pin_verified"
