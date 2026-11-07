"""A carbamate ester on a benzene ring nitrogen as the prefix of a senior group (``n_acyl_amido._carbamate_amino_group``).

 "Esters cited as prefixes" (the Blue Book) names -CO-OR' 'alkoxycarbonyl'; the book concatenates
'(benzyloxy)carbonyl (preferred prefix)' (:18116) and cites 'N2-(tert-butoxycarbonyl)-L-lysine' (:54417). The chain
and heteroarene parents already named the group ('4-[(ethoxycarbonyl)amino]pyridine-3-carboxylic acid'); the benzene
writer did not, so the PIN tier abstained and best-effort shipped the same name as pin_unverified.
"""
import pytest

from tests.support.pin_tiers import tier_row

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,expected", [
    ("CCOC(=O)Nc1ccccc1C(=O)O", "2-[(ethoxycarbonyl)amino]benzoic acid"),
    ("CC(C)(C)OC(=O)Nc1ccccc1C(=O)O", "2-[(tert-butoxycarbonyl)amino]benzoic acid"),
    ("O=C(Nc1ccccc1C(=O)O)OCc1ccccc1", "2-{[(benzyloxy)carbonyl]amino}benzoic acid"),
    ("O=C(Nc1ccccc1C(=O)O)OCC1c2ccccc2-c2ccccc21", "2-({[(9H-fluoren-9-yl)methoxy]carbonyl}amino)benzoic acid"),
    ("CCOC(=O)Nc1ccc(cc1)S(=O)(=O)O", "4-[(ethoxycarbonyl)amino]benzene-1-sulfonic acid"),
    ("CC(C)(C)OC(=O)Nc1cc(ccc1C)C(=O)O", "3-[(tert-butoxycarbonyl)amino]-4-methylbenzoic acid"),
])
def test_pin(smiles, expected):
    for tier in ("pin", "best-effort"):
        row = tier_row(smiles, tier)
        assert (row["name"], row["tier"]) == (expected, "pin_verified"), row


@pytest.mark.parametrize("smiles,expected", [
    ("CCOC(=O)Nc1ccccc1", "ethyl phenylcarbamate"),                  # the carbamate is the principal group
    ("CC(=O)Nc1ccccc1C(=O)O", "2-acetamidobenzoic acid"),            # a carbon acyl keeps its amido prefix
    ("NC(=O)Nc1ccccc1C(=O)O", "2-(carbamoylamino)benzoic acid"),     # a urea keeps its carbamoyl prefix
])
def test_neighbours_unchanged(smiles, expected):
    assert tier_row(smiles, "pin")["name"] == expected


@pytest.mark.parametrize("smiles,expected", [
    #: a decorated phenyl on oxygen is the substitutable 'phenoxy' (BB 17796), not '(2-methoxyphenyl)oxy'
    ("COc1ccccc1OC(=O)NCCC(=O)O", "3-{[(2-methoxyphenoxy)carbonyl]amino}propanoic acid"),
    ("O=C(Oc1ccc(Cl)cc1)NCC(=O)O", "{[(4-chlorophenoxy)carbonyl]amino}acetic acid"),
    ("CC(C)Oc1ccccc1OC(=O)NCCC(=O)O", "3-[({2-[(propan-2-yl)oxy]phenoxy}carbonyl)amino]propanoic acid"),
    # a biphenyl stays a ring assembly on oxygen
    ("O=C(NCC(=O)O)Oc1ccc(cc1)-c1ccccc1", "({[([1,1'-biphenyl]-4-yl)oxy]carbonyl}amino)acetic acid"),
])
def test_aryloxycarbonyl_amino(smiles, expected):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (expected, "pin_verified"), row


@pytest.mark.parametrize("smiles,expected", [
    # a carbamate on the alcohol side of a senior ester is the '[(alkoxycarbonyl)amino]' prefix of its alkyl group, and
    # the carbamate's own tert-butyl is no second ester of that alcohol (no polyol)
    ("CC(C)(C)OC(=O)NCCOS(C)(=O)=O", "2-[(tert-butoxycarbonyl)amino]ethyl methanesulfonate"),
    ("CC(C)(C)OC(=O)NCCOC(C)=O", "2-[(tert-butoxycarbonyl)amino]ethyl acetate"),
    ("O=C(NCCCOS(C)(=O)=O)OCc1ccccc1", "3-{[(benzyloxy)carbonyl]amino}propyl methanesulfonate"),
    ("CC(C)(C)OC(=O)NCC(C)OS(C)(=O)=O", "1-[(tert-butoxycarbonyl)amino]propan-2-yl methanesulfonate"),
    ("CC(C)(C)OC(=O)NCCSC(C)=O", "S-{2-[(tert-butoxycarbonyl)amino]ethyl} ethanethioate"),
])
def test_carbamate_on_the_alcohol_of_a_senior_ester(smiles, expected):
    for tier in ("pin", "best-effort"):
        row = tier_row(smiles, tier)
        assert (row["name"], row["tier"]) == (expected, "pin_verified"), row
