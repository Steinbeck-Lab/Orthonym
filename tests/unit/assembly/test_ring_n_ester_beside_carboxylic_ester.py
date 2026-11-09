"""The ester of a ring-nitrogen carboxylic acid beside a carboxylic ester, and beside a lactone (2026-11-09).

: of the esters of a polyester the senior one is the parent, by the seniority of rings and chains
 to; 'phenyl 3-(benzoyloxy)benzoate (PIN)', the Blue Book); a ring-nitrogen acid is a carboxylic
acid ranked by its ring (``decomposition.ester_in_alcohol``). The ester route named the acetate and labelled
'1-(tert-butoxycarbonyl)piperidin-4-yl acetate' a PIN. Two acids on one ring system are one diacid, whose mixed
diester is named on that diacid ('ethyl methyl benzene-1,3-dicarboxylate (PIN)',:5068).
A lactone is a heterocyclic pseudoketone, not an ester "Lactones",:32084).
"""
import pytest

from orthonym import Orthonym

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles, name", [
    ("CC(C)(C)OC(=O)N1CCC(CC1)OC(=O)C", "tert-butyl 4-(acetyloxy)piperidine-1-carboxylate"),
    ("CC(C)(C)OC(=O)N1CCC(CC1)OC(=O)c1ccccc1", "tert-butyl 4-(benzoyloxy)piperidine-1-carboxylate"),
    ("CC(C)(C)OC(=O)N1CCC(CC1)COC(=O)C", "tert-butyl 4-[(acetyloxy)methyl]piperidine-1-carboxylate"),
    ("CC(C)(C)OC(=O)N1CCC(CC1)CC(=O)OC", "tert-butyl 4-(2-methoxy-2-oxoethyl)piperidine-1-carboxylate"),
    ("CC(C)(C)OC(=O)N1CCC2(CC1)CCOC2=O", "tert-butyl 1-oxo-2-oxa-8-azaspiro[4.5]decane-8-carboxylate"),
])
def test_the_ring_nitrogen_ester_is_the_parent(smiles, name):
    row = Orthonym().name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, "pin_verified")


@pytest.mark.parametrize("smiles, name", [
    ("CC(C)(C)OC(=O)N1CCC(CC1)C(=O)OC", "1-tert-butyl 4-methyl piperidine-1,4-dicarboxylate"),
    ("CC(C)(C)OC(=O)N1CCCC1C(=O)OC", "1-tert-butyl 2-methyl pyrrolidine-1,2-dicarboxylate"),
])
def test_two_acids_on_one_ring_are_one_diacid(smiles, name):
    """never 'tert-butyl 4-(methoxycarbonyl)piperidine-1-carboxylate': the mixed diester of the one diacid
    (test_ring_n_mixed_diester.py)."""
    row = Orthonym().name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, "pin_verified")
