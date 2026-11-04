"""Ester names and substituent prefixes the best-effort tier used to build below the PIN.

* "All preferred IUPAC names for esters are named by functional class nomenclature": the ring cascade of
  a wider tier named an ester molecule with the ester as a prefix ('[(phenylacetyl)oxy]', 'ethoxycarbonyl'), and the
  decomposition engine took that name as good enough and did not split the ester (``decomposition.engine``).
* "Retained names" (the Blue Book): 'phenoxy' is "fully substitutable": '(4-chlorophenoxy)methyl', not
  '[(4-chlorophenyl)oxy]methyl' (``rules.ring_substituents._compound_ring_on_chain_substituent``).
* The acid word of an ester left the =O of a ring of an acid-side substituent out of that substituent, which then
  named a different molecule ('...cyclopentyl]heptanoate' for '...-5-oxocyclopentyl]heptanoate'; ``rules.esters``).
* The alkyl of an ester is numbered on the chain of its prefix: its stereodescriptors were read off a
  longest path through a ring of a substituent of it ('(7R,8S)-...' for '(1R,2S)-...undecyl').
* An unsaturated von Baeyer ring-yl gives the free valence lower locants than the 'ene' endings,
   (c) before (e)): 'bicyclo[3.1.1]hept-3-en-2-yl', not 'hept-2-en-4-yl'.
"""
import pytest

from orthonym import name_compound
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

_CASES = [
    ("C=C1C2CCC(C2)C1(C)CCC=C(C)COC(=O)Cc1ccccc1",
     "2-methyl-5-(2-methyl-3-methylidenebicyclo[2.2.1]heptan-2-yl)pent-2-en-1-yl phenylacetate"),
    ("C1=CC=C(C=C1)N2C(=NN=C2SCCC(=O)NC3=CC=CC4=CC=CC=C43)COC5=CC=C(C=C5)Cl",
     "3-({5-[(4-chlorophenoxy)methyl]-4-phenyl-4H-1,2,4-triazol-3-yl}sulfanyl)-N-(naphthalen-1-yl)propanamide"),
    ("Clc1ccc(OCc2ccccn2)cc1", "2-[(4-chlorophenoxy)methyl]pyridine"),
    ("CCCCCCCCCCOC(=O)CCCCCC[C@@H]1[C@H](CCC1=O)CCC(CCCCC)O",
     "decyl 7-[(1R,2S)-2-(3-hydroxyoctyl)-5-oxocyclopentyl]heptanoate"),
    ("CCCCC(Cl)CCCC[C@H](C)[C@@H](OC(C)=O)c1cc(O)cc(O)c1",
     "(1R,2S)-7-chloro-1-(3,5-dihydroxyphenyl)-2-methylundecyl acetate"),
    ("CC1=C[C@@H](C2CC1C2(C)C)OC(=O)C", "(2S)-4,6,6-trimethylbicyclo[3.1.1]hept-3-en-2-yl acetate"),
]


@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles,name", _CASES)
def test_the_pin_at_the_pin_tier_and_the_default(smiles, name, tier):
    got = name_compound(smiles, emit_tier=tier)
    assert got == name
    assert_full_rt(got, smiles)


def test_an_ester_named_as_a_prefix_is_kept_when_no_split_names_it():
    """The ring-cascade name stays the name of a molecule the decomposition engine does not name: a lactone's ring
    ester is named as the ring '-one', never split."""
    assert name_compound("O=C1CCCCO1") == "oxan-2-one"


@pytest.mark.parametrize("smiles,name", [
    # (the Blue Book) "a ring or ring system has seniority over a chain": the =O of a ring ester stays
    # out of the chain acid's prefixes, so the chain acid is no parent
    ("COC(=O)CCc1ccc(C(=O)OC)cc1", "methyl 4-(3-methoxy-3-oxopropyl)benzoate"),
    ("COC(=O)CCC1CCCC1C(=O)OC", "methyl 2-(3-methoxy-3-oxopropyl)cyclopentane-1-carboxylate"),
    # two arms of one length on the alkyl's chain: the general numbering decides, as before
    ("CC(=O)O[C@@H](c1cc(O)cc(O)c1)C(CC)[C@H](C)Cl",
     "(1R,3S)-3-chloro-1-(3,5-dihydroxyphenyl)-2-ethylbutyl acetate"),
])
def test_the_pin_tier_keeps_these_pins(smiles, name):
    assert name_compound(smiles, emit_tier="pin") == name
