"""Rows that guard the numbering of rings that carry a principal group and an added-carbon
group, against an anchor that cites the wrong suffix.

 (the Blue Book,:18184) ranks esters above amides, so on a ring that carries an
ester the ester is the suffix ('-carboxylate') and a carboxamide, nitrile or formyl group is
a prefix. The names below are main's; the hydro-heterocycle rows pin that no anchor makes the
amide the cited suffix of a ring whose principal group is an ester.
"""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,name", [
    ("COC(=O)C1CCC(C(N)=O)C(C)C1", "methyl 4-carbamoyl-3-methylcyclohexane-1-carboxylate"),
    ("CCOC(=O)C1CC(C(N)=O)CCC1", "ethyl 3-carbamoylcyclohexane-1-carboxylate"),
    ("COC(=O)C1CCC(C#N)C(C)C1", "methyl 4-cyano-3-methylcyclohexane-1-carboxylate"),
])
def test_a_carbocycle_keeps_the_principal_groups_suffix_at_locant_1(smiles, name):
    row = Orthonym(style="pin").name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row


@pytest.mark.parametrize("smiles,name", [
    ("COC(=O)C1=CCCN(C)C1C(N)=O",
     "6-carbamoyl-5-(methoxycarbonyl)-1-methyl-1,2,3,6-tetrahydropyridine"),
    ("COC(=O)C1=CCCOC1C(N)=O", "5-(methoxycarbonyl)-3,6-dihydro-2H-pyran-6-carboxamide"),
    ("O=C1OCC(C(N)=O)C1C", "3-methyl-2-oxooxolane-4-carboxamide"),
])
def test_a_hydro_heterocycle_with_an_ester_and_an_amide_keeps_main_s_name(smiles, name):
    # the amide is never the cited suffix of a ring whose principal group is an ester
    #, the Blue Book above:18184); main's names, below the PIN at every tier
    assert Orthonym(style="pin").name_tiered(smiles)["tier"] == "abstain"
    row = Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, "systematic_verified"), row
