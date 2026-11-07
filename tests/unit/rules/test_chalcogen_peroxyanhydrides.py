"""Acyl anhydrides joined by -SS-, -SeSe-, -TeTe- (``rules.chalcogen_peroxyanhydrides``).

 (the Blue Book): junctions '-SS-, -OS-, -SSe-, etc. are named as 'dithioperoxyanhydrides',
'thioperoxyanhydrides', 'selenothioperoxyanhydrides''; 'acetic dithioperoxyanhydride (PIN)' (:32354);
'N-ethylcarbamimidic N-methylcarbamimidic dithioperoxyanhydride (PIN)',:34418). Both were named
substitutively below the PIN ('1-(acetyldisulfanyl)-1-oxoethane', '3,6-diimino-4,5-dithia-2,7-diazanonane').
"""
import pytest

from tests.support.pin_tiers import tier_row

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,expected", [
    ("CC(=O)SSC(C)=O", "acetic dithioperoxyanhydride"),
    ("CCNC(=N)SSC(=N)NC", "N-ethylcarbamimidic N-methylcarbamimidic dithioperoxyanhydride"),
    ("CCC(=O)SSC(C)=O", "acetic propanoic dithioperoxyanhydride"),
    ("CC(=O)[Se][Se]C(C)=O", "acetic diselenoperoxyanhydride"),
])
def test_pin(smiles, expected):
    for tier in ("pin", "best-effort"):
        row = tier_row(smiles, tier)
        assert (row["name"], row["tier"]) == (expected, "pin_verified"), row


def test_a_senior_acid_keeps_the_substitutive_name():
    # the junction folds into the anhydride class only when no senior group is present
    assert tier_row("OC(=O)CCC(=O)SSC(C)=O", "pin")["name"] == "4-(acetyldisulfanyl)-4-oxobutanoic acid"


def test_a_mixed_junction_is_not_named_as_an_anhydride():
    # OPSIN 2.9.0 reads no 'thioperoxyanhydride'; the molecule keeps its substitutive name
    from rdkit import Chem
    from orthonym.rules.chalcogen_peroxyanhydrides import name_chalcogen_peroxy_anhydride
    assert name_chalcogen_peroxy_anhydride(Chem.MolFromSmiles("CC(=O)OSC(C)=O")) is None


@pytest.mark.parametrize("smiles,expected", [
    # the hypohalous acids, the Blue Book-:35528); 'benzoic hypoiodous anhydride (PIN)'
    #:35998), was '[(iodanyloxy)(oxo)methyl]benzene'
    ("O=C(OI)c1ccccc1", "benzoic hypoiodous anhydride"),
    ("CC(=O)OCl", "acetic hypochlorous anhydride"),
])
def test_hypohalous_anhydride_pin(smiles, expected):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (expected, "pin_verified"), row


@pytest.mark.parametrize("smiles,expected", [
    # (the Blue Book): '(CH3)2B-O-O-B(CH3)2 dimethylborinic peroxyanhydride (PIN)' (:36000), was
    # '{[(dimethylboranyl)peroxy](methyl)boranyl}methane'
    ("CB(C)OOB(C)C", "dimethylborinic peroxyanhydride"),
    ("CP(C)(=O)OOP(C)(C)=O", "dimethylphosphinic peroxyanhydride"),
])
def test_noncarbon_peroxyanhydride_pin(smiles, expected):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (expected, "pin_verified"), row


def test_sulfonyl_thioacyl_thioanhydride_pin():
    # 'Thioanhydrides and other chalcogen analogues' (the Blue Book): 'benzenecarbothioic
    # ethanesulfonic thioanhydride (PIN)' (:32296), was '(3,3-dioxo-1-sulfanylidene-2,3-dithiapentan-1-yl)benzene'
    for tier in ("pin", "best-effort"):
        row = tier_row("CCS(=O)(=O)SC(=S)c1ccccc1", tier)
        assert (row["name"], row["tier"]) == ("benzenecarbothioic ethanesulfonic thioanhydride", "pin_verified"), row


@pytest.mark.parametrize("smiles,expected", [
    # (the Blue Book): O > S > Se > Te; the other links are named substitutively
    ("CC(=O)OC(=O)CC(=O)SSC(C)=O", "acetic 3-(acetyldisulfanyl)-3-oxopropanoic anhydride"),
    # a compound N-substituent takes enclosing marks, 'bis[...]' around them
    ("ClCCNC(=N)SSC(=N)NCCCl", "bis[N-(2-chloroethyl)carbamimidic] dithioperoxyanhydride"),
    # (:30977): the -O-CN group is 'cyanato' (was '[(cyano)oxy]', and before that the false
    # '3-(acetyloxy)-3-oxopropanoic cyanic anhydride', a dianhydride named with one link)
    ("CC(=O)OC(=O)CC(=O)OC#N", "acetic 3-cyanato-3-oxopropanoic anhydride"),
    ("N#COc1ccc(C(=O)O)cc1", "4-cyanatobenzoic acid"),
])
def test_review_rows(smiles, expected):
    assert tier_row(smiles, "pin")["name"] == expected


def test_a_second_link_is_no_hypohalous_anhydride():
    # (:32380): a dianhydride, whose name OPSIN 2.9.0 does not read; never the one-link name as a PIN
    assert tier_row("CC(=O)OC(=O)CC(=O)OCl", "pin")["tier"] != "pin_verified"


def test_an_imidic_side_keeps_its_route():
    # only the carbamimidic side H2N-C(=NH)- is folded into the anhydride class
    assert tier_row("CC(=N)SSC(C)=N", "pin")["name"] == "1-(ethanimidoyldisulfanyl)ethan-1-imine"


def test_a_carbamothioic_side_claims_no_pin():
    # disulfiram: the acid namer spells the side '(diethylamino)methanethioic', not the book's 'diethylcarbamothioic'
    #; the writer declines rather than ship that acid word as a PIN (end check P18/end1)
    assert tier_row("CCN(CC)C(=S)SSC(=S)N(CC)CC", "pin")["tier"] != "pin_verified"
