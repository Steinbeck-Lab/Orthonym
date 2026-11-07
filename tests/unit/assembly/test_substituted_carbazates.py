"""Esters of substituted hydrazinecarboxylic acids, R-O-C(=O)-N(R1)-N(R2)R3 (``assembly.carbazate_names``).

'hydrazinecarboxylic acid (PIN) carbonohydrazidic acid (carboxylic acids expressed by suffixes are preferred to carbonic
acid analogues; see and ' (the Blue Book). The hydrazine nitrogens are numbered from the carbonyl,
as the book's carboxamides of the same acid: '2-(propan-2-ylidene)hydrazine-1-carboxamide (PIN)' (:5096),
'N,1-dimethylhydrazine-1-carboxamide (PIN)' (:38637). Being a carboxylic acid, its ester is senior to a sulfonate ester
 class 7a,:18172), as the ring-nitrogen carboxylates are.
"""
import pytest

from tests.support.pin_tiers import tier_row

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,expected", [
    ("CC(C)(C)OC(=O)NN", "tert-butyl hydrazinecarboxylate"),
    ("CC(C)(C)OC(=O)NNC", "tert-butyl 2-methylhydrazine-1-carboxylate"),
    ("CC(C)(C)OC(=O)NNCCO", "tert-butyl 2-(2-hydroxyethyl)hydrazine-1-carboxylate"),
    ("CC(C)(C)OC(=O)NNc1ccccc1", "tert-butyl 2-phenylhydrazine-1-carboxylate"),
    ("CC(C)(C)OC(=O)NNC(C)=O", "tert-butyl 2-acetylhydrazine-1-carboxylate"),
    ("CC(C)(C)OC(=O)N(C)N", "tert-butyl 1-methylhydrazine-1-carboxylate"),
    ("CC(C)(C)OC(=O)N(C)NC", "tert-butyl 1,2-dimethylhydrazine-1-carboxylate"),
    ("COC(=O)N(C)N", "methyl 1-methylhydrazine-1-carboxylate"),            # was declined (test_l17_03 _UNCHANGED)
    ("OCCOC(=O)NNC", "2-hydroxyethyl 2-methylhydrazine-1-carboxylate"),
    # the carboxylic ester of the hydrazine is senior to the sulfonate ester
    ("CC(C)(C)OC(=O)NNCCOS(C)(=O)=O", "tert-butyl 2-{2-[(methanesulfonyl)oxy]ethyl}hydrazine-1-carboxylate"),
])
def test_pin(smiles, expected):
    for tier in ("pin", "best-effort"):
        row = tier_row(smiles, tier)
        assert (row["name"], row["tier"]) == (expected, "pin_verified"), row


@pytest.mark.parametrize("smiles", [
    "CC(C)(C)OC(=O)NNN",              # a third nitrogen: the triazane is the longer parent (:24910)
    "CC(C)(C)OC(=O)NNNC(C)=O",
    "CC(C)(C)OC(=O)NNP(=O)(O)O",      # a free acid is senior to the ester
    "CC(C)(C)OC(=O)NNS(=O)(=O)O",
])
def test_no_carbazate_pin(smiles):
    assert tier_row(smiles, "pin")["tier"] != "pin_verified"


@pytest.mark.parametrize("smiles,expected", [
    # an N=O on the nitrogen is a nitrous amide, not a 'hydrazinone' (:24922): the carbamate names stay
    ("COC(=O)NN=O", "methyl nitrosocarbamate"),
    ("CC(C)(C)OC(=O)N(C)N=O", "tert-butyl methyl(nitroso)carbamate"),
    ("CC(C)(C)OC(=O)NN=NC", "tert-butyl (methyldiazenyl)carbamate"),
])
def test_other_nitrogen_groups_keep_their_names(smiles, expected):
    assert tier_row(smiles, "pin")["name"] == expected


def test_a_nitrate_beside_a_carbamate_is_no_pin():
    # nitric acid (class 7d, is junior to carbamic acid (class 7b,: '2-[(tert-butoxycarbonyl)amino]ethyl
    # nitrate' is no PIN; best-effort names the carbamate parent
    smiles = "CC(C)(C)OC(=O)NCCO[N+](=O)[O-]"
    assert tier_row(smiles, "pin")["tier"] != "pin_verified"
    assert tier_row(smiles, "best-effort")["name"] == "tert-butyl [2-(nitrooxy)ethyl]carbamate"
