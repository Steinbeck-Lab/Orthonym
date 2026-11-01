"""A silyl or germyl ether, or its chalcogen analogue, as a prefix on a parent with a principal group.

 "ETHERS AND CHALCOGEN ANALOGUES", "Definitions and general methodology" (the Blue Book, the sentence at:27627):
"In substitutive nomenclature, when R is different from R', RH is chosen as parent hydride and R'-O- is cited
as a substituent to it". "Systematic names" (:27631), (:27633): "Substituent prefix names for
R'-O- groups are formed by concatenation, i.e., by adding the prefix 'oxy' to the substituent prefix name for
the group R'." (its example 'H3Si-O silyloxy (preselected prefix) (not siloxy)',:27647), and:
"Substituent prefixes for the substituent groups R'S-, R'Se-, and R'Te-, are formed by substitution of the
groups HS-, 'sulfanyl', HSe-, 'selanyl'; and HTe-, 'tellanyl'" (:27649). The group R' of -O-Si(CH3)3 is the silyl group
of the parent hydride silane,:40414), so it is '(trimethylsilyl)oxy'; a compound prefix takes enclosing
marks, as '(butan-2-yl)oxy (preferred prefix)' (:27639) does.

The book prints no example of a silyl ether prefix; the names below are what these rules construct, and OPSIN
reads each back to the full InChIKey of its structure (the ``opsin_gate`` marker puts the engine's own gate
on). The generic fragment namer capped the group to the silanol, 'hydroxytrimethylsilane', and cut 'yl' from
it ('2-(hydroxytrimethylsilyl)ethan-1-ol'), which OPSIN reads as another structure; the PIN tier declined
every molecule with such a group on a chain or a benzene ring that carries a principal group
(12 of the 110 silicon or germanium rows of dev2000 and druglike2000).

Changed asserted values, none: every row is new. The molecules whose parent is the silane (no principal
group,:18917) are the controls and keep the names they had.
"""
import pytest

import orthonym.assembly.substituent_naming as substituent_naming
import orthonym.rules.benzene as benzene
from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.assembly.group14_chalcogen_prefix import UNHANDLED

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


def _tiered(smiles: str):
    row = Orthonym().name_tiered(smiles)
    return row["name"], row["tier"]


@pytest.mark.parametrize("smiles, expected", [
    # the principal group,:18875) decides the parent, the silyl ether is a prefix
    ("OCCO[Si](C)(C)C", "2-[(trimethylsilyl)oxy]ethan-1-ol"),
    ("NCCCO[Si](C)(C)C", "3-[(trimethylsilyl)oxy]propan-1-amine"),
    ("OC(=O)CCO[Si](C)(C)C", "3-[(trimethylsilyl)oxy]propanoic acid"),
    ("O=CCCO[Si](C)(C)C", "3-[(trimethylsilyl)oxy]propanal"),
    ("OC1CCCCC1O[Si](C)(C)C", "2-[(trimethylsilyl)oxy]cyclohexan-1-ol"),
    # a substituted silyl is a compound prefix of its own: the marks escalate
    ("OCCO[Si](C)(C)C(C)(C)C", "2-{[tert-butyldi(methyl)silyl]oxy}ethan-1-ol"),
    ("OCCO[Si](Cl)(C)C", "2-{[chlorodi(methyl)silyl]oxy}ethan-1-ol"),
    ("OCCO[Si](C)(C)c1ccccc1", "2-{[dimethyl(phenyl)silyl]oxy}ethan-1-ol"),
    ("CC(C)(C)[Si](C)(C)OCCCC#CCCCO", "8-{[tert-butyldi(methyl)silyl]oxy}oct-4-yn-1-ol"),
    ("CC(C)(C)[Si](C)(C)OCCCC/C=C\\C#N", "(2Z)-7-{[tert-butyldi(methyl)silyl]oxy}hept-2-enenitrile"),
    # germanium: the same rule
    ("OCCO[Ge](C)(C)C", "2-[(trimethylgermyl)oxy]ethan-1-ol"),
])
def test_a_silyl_or_germyl_ether_prefix_on_a_chain_or_ring(smiles, expected):
    assert _tiered(smiles) == (expected, "pin_verified")


@pytest.mark.parametrize("smiles, expected", [
    #: the chalcogen analogues take 'sulfanyl' / 'selanyl'
    ("OCCS[Si](C)(C)C", "2-[(trimethylsilyl)sulfanyl]ethan-1-ol"),
    ("OCC[Se][Si](C)(C)C", "2-[(trimethylsilyl)selanyl]ethan-1-ol"),
    ("OC(=O)c1ccc(S[Si](C)(C)C)cc1", "4-[(trimethylsilyl)sulfanyl]benzoic acid"),
])
def test_the_chalcogen_analogues_take_the_sulfanyl_family(smiles, expected):
    assert _tiered(smiles) == (expected, "pin_verified")


@pytest.mark.parametrize("smiles, expected", [
    # the benzene parents: 'benzoic acid', 'phenol', 'aniline' and 'benzaldehyde' are retained PIN parents
    ("OC(=O)c1ccc(O[Si](C)(C)C)cc1", "4-[(trimethylsilyl)oxy]benzoic acid"),
    ("Oc1ccc(O[Si](C)(C)C)cc1", "4-[(trimethylsilyl)oxy]phenol"),
    ("Nc1ccc(O[Si](C)(C)C)cc1", "4-[(trimethylsilyl)oxy]aniline"),
    ("O=Cc1ccc(O[Si](C)(C)C)cc1", "4-[(trimethylsilyl)oxy]benzaldehyde"),
    ("COc1ccc(O[Si](C)(C)C)cc1C(=O)O", "2-methoxy-5-[(trimethylsilyl)oxy]benzoic acid"),
    ("OC(=O)c1ccc(O[Si](C)(C)C(C)(C)C)cc1", "4-{[tert-butyldi(methyl)silyl]oxy}benzoic acid"),
])
def test_a_silyl_ether_prefix_on_a_benzene_parent(smiles, expected):
    assert _tiered(smiles) == (expected, "pin_verified")


@pytest.mark.parametrize("smiles, expected", [
    # no principal group: the silane is the senior parent,:18917) and names the ether as 'oxy'
    ("CC(C)O[Si](C)(C)C", "trimethyl[(propan-2-yl)oxy]silane"),
    ("c1ccccc1O[Si](C)(C)C", "trimethyl(phenoxy)silane"),
    # a C-O-C ether keeps the alkoxy prefix of the carbon group
    ("OCCOC(C)(C)C", "2-tert-butoxyethan-1-ol"),
])
def test_the_controls_keep_their_names(smiles, expected):
    assert _tiered(smiles) == (expected, "pin_verified")


# The prefix route is the carbon parent's: "The senior parent structure has the maximum number of substituents
# corresponding to the principal characteristic group (suffix)" (:18875), and with no principal group anywhere
# (:18917) "N > P > As > Sb > Bi > Si > Ge >... > O > S > Se > Te > C": the silane is the parent, 'tert-butyldi(methyl)
# (oxiranylmethoxy)silane (PIN)' (:18929, "Si is senior to O"), '2-(2-carboxyethyl)disilane-1-carboxylic acid (PIN)' (:18931).
# A hydroxy, amino or sulfanyl group on the Si or Ge is a principal group of the silane or germane too. None of these
# is a PIN as '[(trimethylsilyl)sulfanyl]benzene' or '2-{[dihydroxy(methyl)silyl]oxy}ethan-1-ol'; the PIN tier declines them
# (the second needs the silanediol parent, not built) and the best-effort tier keeps the name it always had.
@pytest.mark.parametrize("smiles", [
    "C[Si](C)(C)Sc1ccccc1",             # no principal group: the silane is the parent (S-linked)
    "CCCCS[Si](C)(C)C",
    "Cc1ccc(S[Si](C)(C)C)cc1",
    "Clc1ccc(S[Si](C)(C)C)cc1",
    "C[Ge](C)(C)Sc1ccccc1",
    "CC[Se][Si](C)(C)C",
    "OCCO[Si](C)(O)O",                  # the silicon holds principal groups of its own
    "OCCO[Si](C)(C)O",
    "NCCO[Si](C)(C)N",
    "Oc1ccc(O[Si](C)(C)O)cc1",
    "OCCS[Ge](C)(C)O",
])
def test_the_silane_or_germane_is_the_parent_where_the_carbon_side_is_not_senior(smiles):
    name, tier = _tiered(smiles)
    assert tier != "pin_verified", (smiles, name)


@pytest.mark.parametrize("smiles, expected", [
    ("c1ccccc1O[Si](C)(C)C", "trimethyl(phenoxy)silane"),      # the O-linked twin: the organometallic producer's PIN
    ("OCCO[Si](Cl)(C)C", "2-{[chlorodi(methyl)silyl]oxy}ethan-1-ol"),   # a halogen on the silicon is no group
])
def test_the_senior_parent_controls(smiles, expected):
    assert _tiered(smiles) == (expected, "pin_verified")


def test_a_stereocentre_in_the_group_leaves_the_name_to_the_old_path():
    """The group is not named here, so the generic cascade runs as before: the best-effort name keeps its label."""
    smiles = "OCCO[Si](C)(C)[C@H](C)CC"
    assert _tiered(smiles)[1] == "abstain"
    row = Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(smiles)
    assert row["tier"] == "pin_unverified", row


@pytest.mark.parametrize("smiles", [
    "OCCO[Si](C)(C)O[Si](C)(C)C",       # a siloxane: _name_group14_substituent declines, nothing is guessed
    "OCCO[Si](C)(C)[C@H](C)CC",         # a stereocentre in the group: the group is not named without it
])
def test_a_group_the_namer_declines_is_not_guessed(smiles):
    name, tier = _tiered(smiles)
    assert tier == "abstain", (smiles, name)


def test_the_tests_fail_without_the_hooks(monkeypatch):
    """The mutation check: with both entry points switched off the rows are declined again."""
    monkeypatch.setattr(substituent_naming, "group14_chalcogen_prefix", lambda *a, **k: UNHANDLED)
    monkeypatch.setattr(benzene, "ring_group14_chalcogen_substituent", lambda *a, **k: None)
    for smiles in ("OCCO[Si](C)(C)C", "OC(=O)c1ccc(O[Si](C)(C)C)cc1"):
        assert _tiered(smiles)[1] == "abstain", smiles
