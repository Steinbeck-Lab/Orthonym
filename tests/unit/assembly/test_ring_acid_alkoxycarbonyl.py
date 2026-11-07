"""An ester group as the 'alkoxycarbonyl' prefix of a ring acid other than benzene, at the PIN tier.

 "Esters cited as prefixes" (the Blue Book) names -CO-OR' 'alkoxycarbonyl';
'2-chloro-6-(ethoxycarbonyl)benzoic acid (PIN)' (:31935). The PIN-tier functional-group namer did not take the branch, so
the heteroarene and saturated-ring acids abstained there and best-effort shipped the same name as pin_unverified.
"""
import pytest

from tests.support.pin_tiers import tier_row

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,expected", [
    ("COC(=O)c1ccncc1C(=O)O", "4-(methoxycarbonyl)pyridine-3-carboxylic acid"),
    ("COC(=O)c1cccnc1C(=O)O", "3-(methoxycarbonyl)pyridine-2-carboxylic acid"),
    ("COC(=O)c1ccsc1C(=O)O", "3-(methoxycarbonyl)thiophene-2-carboxylic acid"),
    ("CC(C)(C)OC(=O)c1ccncc1C(=O)O", "4-(tert-butoxycarbonyl)pyridine-3-carboxylic acid"),
    ("CCCCCOC(=O)c1ccncc1C(=O)O", "4-[(pentyloxy)carbonyl]pyridine-3-carboxylic acid"),
    ("COC(=O)C1CCNCC1C(=O)O", "4-(methoxycarbonyl)piperidine-3-carboxylic acid"),
])
def test_pin(smiles, expected):
    for tier in ("pin", "best-effort"):
        row = tier_row(smiles, tier)
        assert (row["name"], row["tier"]) == (expected, "pin_verified"), row


@pytest.mark.parametrize("smiles,expected", [
    # -CO-SR' is '(R-sulfanyl)carbonyl' by concatenation: '(acetylsulfanyl)carbonyl (preferred prefix)' (:18128),
    # '4-[(hydroxysulfanyl)carbonyl]cyclohexanecarboxylic acid (PIN)' (:30363); not '(methylsulfanyl)(oxo)methyl'
    ("CSC(=O)c1ccccc1C(=O)O", "2-[(methylsulfanyl)carbonyl]benzoic acid"),
    ("CSC(=O)c1ccncc1C(=O)O", "4-[(methylsulfanyl)carbonyl]pyridine-3-carboxylic acid"),
    ("CCSC(=O)C1CCC(CC1)C(=O)O", "4-[(ethylsulfanyl)carbonyl]cyclohexane-1-carboxylic acid"),
    ("CC(C)SC(=O)c1ccccc1C(=O)O", "2-{[(propan-2-yl)sulfanyl]carbonyl}benzoic acid"),
    ("CC(=O)SC(=O)c1ccccc1C(=O)O", "2-[(acetylsulfanyl)carbonyl]benzoic acid"),
])
def test_s_ester_prefix(smiles, expected):
    for tier in ("pin", "best-effort"):
        row = tier_row(smiles, tier)
        assert (row["name"], row["tier"]) == (expected, "pin_verified"), row


@pytest.mark.parametrize("smiles,expected", [
    # an acyloxy group with a ring in its acid side on a chain branch, '3-(benzoyloxy)propanoic acid
    # (PIN)':31711, '3-[(pyridine-3-carbonyl)oxy]propanoic acid (PIN)':31723), consumed before the ring branches
    ("O=C(OCCc1ccc(cc1)C(=O)O)c1ccccc1", "4-[2-(benzoyloxy)ethyl]benzoic acid"),
    ("O=C(OCC1CCC(CC1)C(=O)O)c1ccccc1", "4-[(benzoyloxy)methyl]cyclohexane-1-carboxylic acid"),
    ("O=C(OCCc1ccc(cc1)C(=O)O)c1cccnc1", "4-{2-[(pyridine-3-carbonyl)oxy]ethyl}benzoic acid"),
    ("OC(=O)c1ccc(CCOC(=O)C2CCCCC2)cc1", "4-{2-[(cyclohexanecarbonyl)oxy]ethyl}benzoic acid"),
])
def test_ring_acyl_acyloxy_on_a_chain_branch(smiles, expected):
    for tier in ("pin", "best-effort"):
        row = tier_row(smiles, tier)
        assert (row["name"], row["tier"]) == (expected, "pin_verified"), row
