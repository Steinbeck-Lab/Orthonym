"""A stereoparent candidate names every decoration of the input, or the producer declines.

 (the Blue Book, "SUBSTITUTIVE NOMENCLATURE"): every substituent is cited as a prefix or
a suffix. Two producer paths dropped decorations and returned the name of a different molecule,
which only the exit round trip stopped:
- a scaffold without a numbering map returned its bare name for a decorated input
  ('aporphine' for the 1,2-diol, 'berberine' for tetrahydropalmatine);
- the ester/conjugate assembler takes no N-alkyl, epoxy, methoxy or glycosyloxy decoration and no
  skeletal modification ('morphinan-6-yl acetate' for an epoxy/methoxy/N-methyl morphinan ester).
The best-effort tier keeps an RT-exact name for each of them (systematic path).
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots
from orthonym.rules.natural_products import name_natural_product
from tests.support.rt_assert import name_is_rt_exact

DROPPED = [
    # (input, the candidate the producer used to return)
    ("CC(=O)OC1CCC2C3Cc4ccc(OC)c5OC1C2(CCN3C)c45", "morphinan-6-yl acetate"),
    ("CC(=O)Oc1ccc2c(c1)[C@@]13CCCC[C@@H]3[C@@H](C2)N(C)CC1", "(9R,13S,14S)-morphinan-3-yl acetate"),
    ("CN1CCc2cc(O)c(O)c3c2C1Cc1ccccc1-3", "aporphine"),
    ("COc1cc2c(cc1OC)[C@H]1Cc3ccc(OC)c(OC)c3CN1CC2", "berberine"),
    ("C=C[C@H]1CN2CC[C@@H]1C[C@H]2[C@H](O)c1ccnc2ccc(OC)cc12", "cinchonane"),
]
KEPT = [  # the paths still name what they can name in full
    ("CC(=O)Oc1ccc2c(c1)[C@@]13CCCC[C@@H]3[C@@H](C2)NCC1", "(9R,13S,14S)-morphinan-3-yl acetate"),
    ("Oc1ccc2c(c1)[C@@]13CCCC[C@@H]3[C@@H](C2)N(C)CC1", "(9R,13S,14S)-17-methylmorphinan-3-ol"),
]


@pytest.mark.parametrize("smiles,dropped", DROPPED)
def test_a_candidate_that_drops_a_decoration_is_declined(smiles, dropped):
    name = name_natural_product(Chem.MolFromSmiles(smiles))
    assert name != dropped, name
    assert name is None, name


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", KEPT)
def test_a_candidate_that_carries_every_decoration_is_kept(smiles, name):
    assert name_natural_product(Chem.MolFromSmiles(smiles)) == name
    assert name_is_rt_exact(name, smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,dropped", DROPPED)
def test_best_effort_still_names_them_rt_exact(smiles, dropped):
    with jvm_slots(1, purpose="np-decorations"):
        row = Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(smiles)
    assert row["tier"] != "abstain", row
    assert name_is_rt_exact(row["name"], smiles), row["name"]
