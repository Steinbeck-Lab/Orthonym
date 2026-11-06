"""A carbamate ester whose molecule has a junior group is named as the ester, and a carbazate ester as a hydrazinecarboxylate
(lane/l17 item 3 fix round).

 "SENIORITY ORDER FOR CLASSES" (the Blue Book), Table 4.1: class 9 "Esters" (:18182) outranks class 11 amides, 12
hydrazides, 14 nitriles, 15 aldehydes, 16 ketones, 17 hydroxy compounds and 19 amines. (:31663): "All preferred
IUPAC names for esters are named by functional class nomenclature". The exemplar of (:30766) is a carbamate ester whose
R group holds a hydroxy group: '2-hydroxypropyl (2-aminoethyl)carbamate (PIN)'.

A carbamate is "functional class only" in ``rules.seniority``, so the principal group of a molecule that has one is the first
group *below* the esters (the alcohol, the ketone, the hydrazide that perception also reads on the N-C(=O) of a carbazate), and the
carbamate handler stood aside for it: the PIN tier declined the molecule and the wider tiers named it on that group
('1-(carbamoyloxy)propan-2-ol', '1-(2-hydroxypropoxy)methanehydrazide' -- a hydrazide, class 12, for an ester, class 9).

 (:30670, its Note, example:30682): the replacement of the hydrogen of formic acid by -NH-NH2 gives
'hydrazinecarboxylic acid (PIN)' (not carbazic acid), a carboxylic acid on the parent hydride hydrazine, so the
ester of it is 'R hydrazinecarboxylate' (derived from; the book has no such ester as an example). OPSIN reads it.
"""
import pytest

from orthonym.errors import is_failure_name
from tests.support.rt_assert import assert_full_rt
from tests.support.pin_tiers import tier_row

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

_TIERS = ["pin", "best-effort"]


def _abstains(row):
    return row["tier"] == "abstain" or not row["name"] or is_failure_name(row["name"])


# carbamate and carbazate esters whose only senior group is the carbamate: the junior group is a prefix of R (or of the N group)
_ESTERS = [
    # the verbatim Blue Book exemplar of (:30766), a carbamate ester with a hydroxy group in R and an amino group in N
    ("CC(O)COC(=O)NCCN", "2-hydroxypropyl (2-aminoethyl)carbamate"),
    # the rows of the review of a436fc3dbe
    ("CC(COC(=O)NN)O", "2-hydroxypropyl hydrazinecarboxylate"),
    ("C(COC(=O)NN)O", "2-hydroxyethyl hydrazinecarboxylate"),
    # a carbazate ester with no junior group: the hydrazide was the principal group of its name
    ("COC(=O)NN", "methyl hydrazinecarboxylate"),
    ("NNC(=O)OCC", "ethyl hydrazinecarboxylate"),
    ("NNC(=O)OCC(=O)C", "2-oxopropyl hydrazinecarboxylate"),
    ("NNC(=O)OCCCl", "2-chloroethyl hydrazinecarboxylate"),
    ("C[C@@H](O)COC(=O)NN", "(2R)-2-hydroxypropyl hydrazinecarboxylate"),
    ("[13CH3]OC(=O)NN", "(13C)methyl hydrazinecarboxylate"),
    # a carbamate ester with a hydroxy, oxo, cyano, amino, amide or sulfonamide group
    ("OCCOC(N)=O", "2-hydroxyethyl carbamate"),
    ("CC(COC(N)=O)O", "2-hydroxypropyl carbamate"),
    ("C[C@H](O)COC(N)=O", "(2S)-2-hydroxypropyl carbamate"),
    ("OC1CCCCC1OC(N)=O", "2-hydroxycyclohexyl carbamate"),
    ("Oc1ccc(OC(N)=O)cc1", "4-hydroxyphenyl carbamate"),
    ("CC(=O)COC(N)=O", "2-oxopropyl carbamate"),
    ("N#CCOC(N)=O", "cyanomethyl carbamate"),
    ("NCCOC(N)=O", "2-aminoethyl carbamate"),
    ("NNC(=O)CCOC(N)=O", "3-hydrazinyl-3-oxopropyl carbamate"),
    ("NS(=O)(=O)CCOC(N)=O", "2-sulfamoylethyl carbamate"),
    # the junior group on the N side of the carbamate
    ("CC(C)(C)OC(=O)NCCO", "tert-butyl (2-hydroxyethyl)carbamate"),
    ("CC(C)(C)OC(=O)NC(CO)CO", "tert-butyl (1,3-dihydroxypropan-2-yl)carbamate"),
    ("OCCNC(=O)OCCO", "2-hydroxyethyl (2-hydroxyethyl)carbamate"),
    ("Oc1ccccc1NC(=O)OC", "methyl (2-hydroxyphenyl)carbamate"),
    ("CC(=O)CNC(=O)OC", "methyl (2-oxopropyl)carbamate"),
    ("OC(CC)COC(=O)N(CC)CC", "2-hydroxybutyl diethylcarbamate"),
    ("CCOC(=O)NCC(N)=O", "ethyl (2-amino-2-oxoethyl)carbamate"),
]


@pytest.mark.parametrize("tier", _TIERS)
@pytest.mark.parametrize("smiles,name", _ESTERS)
def test_a_carbamate_ester_with_a_junior_group_is_named_as_the_ester(smiles, name, tier):
    """ class 9 > 11/12/14/16/17/19;; the exemplar of (:30766)."""
    row = tier_row(smiles, tier)
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)


# the carbamate is not the parent of the name here (a senior group, or a unit this handler does not build): the names are as
# they were -- the PIN tier declines the carbazate of a substituted hydrazine, a ring-N carboxylate and a diester (no 'bis'
# multiplier here), and a senior acid keeps the parent
_UNCHANGED = [
    ("COC(=O)N(C)N", "PIN abstains"),
    ("OCCOC(=O)NNC", "PIN abstains"),
    ("OC(=O)CCOC(=O)NN", "PIN abstains"),
    # COC(=O)N1CCCC1 is 'methyl pyrrolidine-1-carboxylate' (PIN) since program 7 (2026-11-06): the ester of the ring-
    # nitrogen acid 'pyrrolidine-1-carboxylic acid (PIN)', the Blue Book), test_ring_n_carboxylate.py
    ("NC(=O)OCCOC(N)=O", "PIN abstains"),
]


@pytest.mark.parametrize("smiles,_why", _UNCHANGED)
def test_a_carbamate_the_handler_does_not_build_keeps_its_declined_pin_tier(smiles, _why):
    assert _abstains(tier_row(smiles, "pin"))
    row = tier_row(smiles, "best-effort")
    assert row["name"] and row["tier"] != "abstain", row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("tier", _TIERS)
def test_a_senior_acid_keeps_the_parent_and_cites_the_carbamate_as_a_prefix(tier):
    """ class 7 (acids) outranks class 9: the carbamate is the prefix 'carbamoyloxy'."""
    row = tier_row("OC(=O)COC(N)=O", tier)
    assert (row["name"], row["tier"]) == ("(carbamoyloxy)acetic acid", "pin_verified"), row
    assert_full_rt(row["name"], "OC(=O)COC(N)=O")


@pytest.mark.parametrize("tier", _TIERS)
def test_the_hydrazide_name_of_the_ester_is_not_emitted(tier):
    """Class 12 (a hydrazide) is below class 9: no name of a carbazate ester expresses 'methanehydrazide' / 'carbamoyl'."""
    for smiles in ("CC(COC(=O)NN)O", "COC(=O)NN", "NNC(=O)OCCO"):
        name = tier_row(smiles, tier)["name"]
        assert "hydrazide" not in name and "aminocarbamate" not in name and "carbamoyl" not in name, name


# --- fix a performance pass ------------------------------------------------------------------------------------------------------------
# Esters rank by their acid "Esters cited as prefixes",:31696; rules.seniority.ESTER_ACID_ORDER): carbamic acid is a
# carbonic acid, class 7b "CLASS 7b. CARBON ACIDS WITH NO SUBSTITUTABLE HYDROGEN ATOMS",:18314), below the carboxylic,
# sulfonic and sulfinic acids of class 7a "CLASS 7a. ACIDS EXPRESSED BY SUFFIXES",:18302) and with their imidic and chalcogen analogues, above phosphonic and
# phosphinic acid "CLASS 7c. NONCARBON ACIDS HAVING SUBSTITUTABLE HYDROGEN ATOMS ON THE CENTRAL ATOM",:18318) and
# phosphoric and sulfuric acid "CLASS 7d",:18375).

def test_the_acid_order_of_the_class_9_esters_is_the_one_of_p_42():
    from orthonym.rules.seniority import ester_acid_rank
    ranks = [ester_acid_rank(name) for name in (
        "ester", "thioester", "sulfonic_ester", "iminoester", "carbamate", "phosphonate_diester", "phosphinate_ester",
        "phosphate_triester", "sulfate_diester")]
    assert ranks == [0, 0, 0, 0, 1, 2, 2, 3, 3]
    assert ester_acid_rank("primary_alcohol") is None


# a carbamate with the ester of a senior acid (class 7a): the ester of the senior acid is the parent word
_SENIOR_ACID_ESTERS = [
    ("CCSC(=O)CCNC(=O)OC", "S-ethyl 3-[(methoxycarbonyl)amino]propanethioate"),
    ("COS(=O)(=O)CCNC(=O)OC", "methyl 2-[(methoxycarbonyl)amino]ethane-1-sulfonate"),
    ("COS(=O)CCNC(=O)OC", "methyl 2-[(methoxycarbonyl)amino]ethane-1-sulfinate"),
    ("CCOS(=O)(=O)CCOC(N)=O", "ethyl 2-(carbamoyloxy)ethane-1-sulfonate"),
    ("C[Si](C)(C)OC(=O)CCNC(=O)OC", "trimethylsilyl 3-[(methoxycarbonyl)amino]propanoate"),
    ("CCOC(=N)CCNC(=O)OC", "ethyl 3-[(methoxycarbonyl)amino]propanimidate"),
]


@pytest.mark.parametrize("tier", _TIERS)
@pytest.mark.parametrize("smiles,name", _SENIOR_ACID_ESTERS)
def test_the_ester_of_a_senior_acid_keeps_the_parent_over_the_carbamate(smiles, name, tier):
    """ (:31696) keeps the ester of the senior acid as the parent: the carbamate (class 7b) is its prefix."""
    row = tier_row(smiles, tier)
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)


# the esters of the phosphorus and sulfur acids of classes 7c and 7d rank below the carbamate: the carbamate is the parent word.
# The default tier cannot spell the P/S group prefix of R ('dimethoxyphosphoryl' is a wider-tier prefix), so it declines; the
# functional-class name on the phosphorus or sulfur acid is not the PIN and no tier labels it so.
_JUNIOR_ACID_ESTERS = [
    ("COP(=O)(OC)c1ccc(NC(=O)OC)cc1", "methyl [4-(dimethoxyphosphoryl)phenyl]carbamate", "pin_unverified",
     "dimethyl {4-[(methoxycarbonyl)amino]phenyl}phosphonate"),
    ("COP(=O)(OC)CCNC(=O)OC", "methyl [2-(dimethoxyphosphoryl)ethyl]carbamate", "pin_unverified",
     "dimethyl {2-[(methoxycarbonyl)amino]ethyl}phosphonate"),
    ("COC(=O)NCCP(=O)(OCC)OCC", "methyl [2-(diethoxyphosphoryl)ethyl]carbamate", "pin_unverified",
     "diethyl {2-[(methoxycarbonyl)amino]ethyl}phosphonate"),
    ("COP(=O)(OC)CCOC(=O)NC", "2-(dimethoxyphosphoryl)ethyl methylcarbamate", "pin_unverified",
     "dimethyl {2-[(methylcarbamoyl)oxy]ethyl}phosphonate"),
    ("COP(=O)(C)CCNC(=O)OC", "methyl {2-[methoxy(methyl)phosphoryl]ethyl}carbamate", "systematic_verified",
     "methyl {2-[(methoxycarbonyl)amino]ethyl}(methyl)phosphinate"),
    ("CCOP(=O)(OCC)OCCNC(=O)OC", "methyl {2-[(diethoxyphosphoryl)oxy]ethyl}carbamate", "pin_unverified",
     "diethyl 2-[(methoxycarbonyl)amino]ethyl phosphate"),
    ("COS(=O)(=O)OCCNC(=O)OC", "methyl {2-[(methoxysulfonyl)oxy]ethyl}carbamate", "pin_unverified",
     "2-[(methoxycarbonyl)amino]ethyl methyl sulfate"),
]


@pytest.mark.parametrize("smiles,name,label,junior_parent_name", _JUNIOR_ACID_ESTERS)
def test_a_carbamate_ester_outranks_the_ester_of_a_phosphorus_or_sulfur_acid(smiles, name, label, junior_parent_name):
    """ class 7b > class 7c > class 7d; class 9 ranks esters by their acid,:31696)."""
    assert _abstains(tier_row(smiles, "pin"))
    row = tier_row(smiles, "best-effort")
    assert (row["name"], row["tier"]) == (name, label), row
    assert row["name"] != junior_parent_name
    assert_full_rt(row["name"], smiles)
    # the name of the junior acid's ester reads back to the molecule and is no PIN: it is not shipped as one at any tier
    assert_full_rt(junior_parent_name, smiles)
    for tier in _TIERS:
        assert (tier_row(smiles, tier)["name"], tier_row(smiles, tier)["tier"]) != (junior_parent_name, "pin_verified")


def test_a_phosphonate_ester_keeps_the_parent_when_the_other_group_is_junior_to_it():
    """The control: an amine (class 19) is junior to the ester of phosphonic acid, so its PIN is unchanged."""
    for tier in _TIERS:
        row = tier_row("COP(=O)(OC)CCNC", tier)
        assert (row["name"], row["tier"]) == ("dimethyl [2-(methylamino)ethyl]phosphonate", "pin_verified"), row


def test_a_phosphorus_substituent_of_a_carbon_chain_is_not_read_off_a_name_on_a_carbon_skeleton():
    """ (:15762): the parent substituent group carries the free valence. A P(=O)(OCH3)2 capped with hydrogen is named on a
    carbon skeleton ('...oxy}methane'), whose '-yl' puts the free valence on that carbon: another group."""
    from rdkit import Chem

    from orthonym.assembly.hetero_chain_prefix import fragment_parent_to_prefix
    mol = Chem.MolFromSmiles("NCCP(=O)(OC)OC")
    frag = [3, 4, 5, 6, 7, 8]
    assert fragment_parent_to_prefix(mol, frag, 3, "{[methoxy(oxo)-λ5-phosphanyl]oxy}methane", 2) is None
    assert fragment_parent_to_prefix(Chem.MolFromSmiles("NCCP(C)C"), [3, 4, 5], 3, "dimethylphosphane", 2) == "dimethylphosphanyl"


# --- the esters of carbamothioic and carbamodithioic acid,:30768;,:32023) ------------------------
_CARBAMOTHIOATES = [
    # "Esters of carbonic acid, cyanic acid, and polycarbonic acids modified by functional replacement" (:32023):
    # "Names of acids modified by functional replacement are used to generate preferred IUPAC names of corresponding esters"
    ("OCCOC(=S)NC", "O-(2-hydroxyethyl) methylcarbamothioate"),
    ("OCCSC(=O)NC", "S-(2-hydroxyethyl) methylcarbamothioate"),
    ("COC(=S)NC", "O-methyl methylcarbamothioate"),
    ("CSC(=O)NC", "S-methyl methylcarbamothioate"),
    ("CSC(=S)NC", "methyl methylcarbamodithioate"),
    ("COC(=S)N", "O-methyl carbamothioate"),
    ("CSC(N)=O", "S-methyl carbamothioate"),
    ("CSC(N)=S", "methyl carbamodithioate"),
    ("COC(=S)N(C)C", "O-methyl dimethylcarbamothioate"),
    ("CCOC(=S)NCCO", "O-ethyl (2-hydroxyethyl)carbamothioate"),
    ("NCCOC(=S)NC", "O-(2-aminoethyl) methylcarbamothioate"),
]


@pytest.mark.parametrize("tier", _TIERS)
@pytest.mark.parametrize("smiles,name", _CARBAMOTHIOATES)
def test_the_ester_of_carbamothioic_acid_is_named_on_the_carbamothioate(smiles, name, tier):
    """ (:30756): "Substitution of formic acid is not recommended for generation of these names", so
    '(methylamino)methanethioate' and '[(methylcarbamothioyl)oxy]ethan-1-ol' (the hydroxy group as the parent) are no names of these
    esters (class 9 ranks above class 17,,:18182,:18190)."""
    row = tier_row(smiles, tier)
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)


# --- M6: the carbamate-first name is not refused as under-naming its amides ------------------------------------------------------
def test_the_carbamate_first_name_of_a_molecule_with_carbamoyl_amides_is_kept():
    """The N group of the carbamate cites the two amides as '(...carbamoyl)'; the multi-amide check of the decomposition engine
    counted none of them,:30867: carbamoyl is the prefix of carbamic acid) and handed the molecule to a producer
    that spelled '2-(1H-indol-3-yl)ethan-1-yl'."""
    smiles = "COC(=O)Nc1coc(C(=O)Nc2coc(C(=O)NCCc3c[nH]c4ccccc34)c2)c1"
    row = tier_row(smiles, "best-effort")
    assert row["name"] == ("methyl {5-[(5-{[2-(1H-indol-3-yl)ethyl]carbamoyl}furan-3-yl)carbamoyl]furan-3-yl}carbamate"), row
    assert "ethan-1-yl" not in row["name"]
    assert_full_rt(row["name"], smiles)


# --- M9: estramustine ----------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("tier", _TIERS)
def test_estramustine_is_named_as_the_carbamate_on_the_systematic_fused_parent(tier):
    """The carbamate (class 9,,:18182) outranks the 17-hydroxy group (class 17,:18190). "INTRODUCTION"
    (:50943): "Preferred IUPAC names (PINs) are not identified for the compounds in this Chapter", so the semisystematic
    'estra-1,3,5(10)-trien-3-yl' is not the name to pin; the systematic fused name is the PIN."""
    smiles = "C[C@]12CC[C@H]3[C@@H](CCc4cc(OC(=O)N(CCCl)CCCl)ccc34)[C@@H]1CC[C@@H]2O"
    row = tier_row(smiles, tier)
    assert (row["name"], row["tier"]) == (
        "(8R,9S,13S,14S,17S)-17-hydroxy-13-methyl-7,8,9,11,12,13,14,15,16,17-decahydro-6H-cyclopenta[a]phenanthren-3-yl "
        "bis(2-chloroethyl)carbamate", "pin_verified"), row
    assert_full_rt(row["name"], smiles)


# --- ions: the ester is not the parent of an ion's name ---------------------------------------------------------------------
# (the Blue Book-:18167) ranks "4 Anions", "5 Zwitterions" and "6 Cations" above "7 Acids" and "9 Esters":
# the name of an ion cites its charge in the parent ('-aminium'), and the ion routes name the neutral analogue in a nested
# call, which needs the senior group of the ion as that call's parent too. The names are the ones of the base 5cd9d81c3a.
_IONS = [
    ("COC(=O)NCC[NH3+]", "2-[(methoxycarbonyl)amino]ethan-1-aminium", "pin_verified"),
    ("COC(=O)NCC[O-]", "2-[(methoxycarbonyl)amino]ethoxide", "pin_verified"),
    ("COC(=O)NCC[N+](C)(C)C", "2-[(methoxycarbonyl)amino]-N,N,N-trimethylethan-1-aminium", "pin_verified"),
    ("C=CCNC(=S)SCC[NH3+]", "2-{[(prop-2-en-1-yl)carbamothioyl]sulfanyl}ethan-1-aminium", "pin_unverified"),
]


@pytest.mark.parametrize("smiles,name,label", _IONS)
def test_an_ion_keeps_its_charged_parent_over_the_carbamate_ester(smiles, name, label):
    row = tier_row(smiles, "best-effort")
    assert (row["name"], row["tier"]) == (name, label), row
    assert_full_rt(row["name"], smiles)
