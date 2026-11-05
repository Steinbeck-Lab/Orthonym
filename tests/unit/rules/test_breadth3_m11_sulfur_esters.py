"""Breadth job 3, class M11 (sulfur acid esters).

1. A sulfuric-acid MONOESTER under a senior parent is the preselected prefix
   'sulfooxy': (the Blue Book) "If a sulfur-containing group is
   attached by oxygen... to a compound that contains also another substituent having
   priority... the sulfur-containing group is named by an appropriate prefix" --
   '3-(sulfooxy)propanoic acid (PIN)' (:36488); 'HO-SO2-O- sulfooxy (preselected
   prefix)' (:31342). The PIN tier dropped the group (no prefix form) and built a
   different molecule ('hexanoic acid' for 5-(sulfooxy)hexanoic acid).
2. On benzene the same group was spelled 'sulfonyloxy' (-O-SO2- with a free valence on
   S, a different molecule). Under a junior parent (phenol) the name on the phenol is correct but
   is not the PIN: the ester outranks the -OH): the molecule is named by the functional
   class ester, '2-hydroxyphenyl hydrogen sulfate' (lane/l17 fix round).
3. The acyl chain of a thio/seleno ester keeps its multiple bonds ('dodeca-2,8-diene',
   ; it was always written '-ane') and its stereodescriptors stand in
   front of the acid word, as for every ester ('ethyl (2E)-but-2-enoate';,
   the Blue Book).

Every name is read back by a FRESH OPSIN call that does not go through the engine
(tests.support.rt_assert._independent_parse) and compared by full InChIKey.
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from tests.support.rt_assert import _independent_parse
from tests.support.default_tier import (  # noqa: E402
    declined_pin_row,
    default_tier_rule_applies,
)

# Default tier: the paper, Methods, "Tiers" (L73): "The default configuration emits a
# name only when the pipeline can build the preferred IUPAC name (PIN); otherwise, it
# declines." User decision 2026-09-30 ("Ship it in 1.0.2"): a name the code records
# as not the PIN is declined at the default tier with NO_VERIFIED_PIN; for the
# molecules below the test asserts that decline, the strict path's name and label,
# and the same name at the best-effort tier (tests/support/default_tier.py).
# (Lane/l17 fix round: the twelve molecules of the junior-parent tests below, whose name was a name the code records as not the
# PIN and so declined at the default tier, are named by the functional class ester, the PIN; the set is empty.)
DEFAULT_TIER_DECLINES = frozenset()
#... whose best-effort name is another one (it reads back exactly)
BEST_EFFORT_NAMES_IT_OTHERWISE = frozenset({
    "CC(=O)NCCSC(=O)/C=C/CCC[C@H](O)/C=C/CC(C)O",
})


def _declined_pin_row(smiles):
    return declined_pin_row(
        smiles, best_effort_same=smiles not in BEST_EFFORT_NAMES_IT_OTHERWISE)


pytestmark = [pytest.mark.opsin_gate]


def _key(smiles):
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return Chem.MolToInchiKey(mol) if mol is not None else ""


def _assert_named(smiles, expected, tiers):
    row = (_declined_pin_row(smiles)
           if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies()
           else Orthonym().name_tiered(smiles))
    assert row["name"] == expected, row
    assert row["tier"] in tiers, row
    assert _key(_independent_parse(expected, smiles)) == _key(smiles)


@pytest.mark.parametrize("smiles,expected", [
    ("OC(=O)CCOS(=O)(=O)O", "3-(sulfooxy)propanoic acid"),          # BB:36488
    ("OC(=O)COS(=O)(=O)O", "(sulfooxy)acetic acid"),
    ("OC(=O)CCCC(C)OS(=O)(=O)O", "5-(sulfooxy)hexanoic acid"),
    # the milestone1500 M11 row
    ("CCCCCC[C@H](C/C=C/CCCCCCCC(=O)O)OS(=O)(=O)O",
     "(9E,12R)-12-(sulfooxy)octadec-9-enoic acid"),
])
def test_sulfooxy_prefix_under_a_senior_acid(smiles, expected):
    _assert_named(smiles, expected, {"pin_verified"})


# change-asserted-value (lane/l17 fix round), the old -> new pairs, all read back by a fresh OPSIN call to the input's key:
# 2-methoxy-6-(sulfooxy)phenol -> 2-hydroxy-3-methoxyphenyl hydrogen sulfate; 4-nitro-2-(sulfooxy)phenol -> 2-hydroxy-5-nitrophenyl
# hydrogen sulfate; 2-(sulfooxy)phenol -> 2-hydroxyphenyl hydrogen sulfate; 2-(sulfooxy)ethan-1-ol -> 2-hydroxyethyl hydrogen sulfate;
# 4-(sulfooxy)benzaldehyde -> 4-formylphenyl hydrogen sulfate; 2-(sulfooxy)benzaldehyde -> 2-formylphenyl hydrogen sulfate;
# 2-hydroxy-4-(sulfooxy)benzaldehyde -> 4-formyl-3-hydroxyphenyl hydrogen sulfate; (sulfooxy)acetaldehyde -> 2-oxoethyl hydrogen
# sulfate; 4-(sulfooxy)aniline -> 4-aminophenyl hydrogen sulfate; 3-(sulfooxy)propanehydrazide -> 3-hydrazinyl-3-oxopropyl hydrogen
# sulfate; N-hydroxy-2-(sulfooxy)ethan-1-imine -> 2-(hydroxyimino)ethyl hydrogen sulfate; 4-(phosphonooxy)benzaldehyde ->
# 4-formylphenyl dihydrogen phosphate.
# Quote: (the Blue Book-18192) puts "9 Esters" above "12 Hydrazides", "14 Nitriles", "15 Aldehydes", "17 Hydroxy
# compounds" and "19 Amines"; (:36484) cites the sulfur group by a prefix only "to a compound that contains also another
# substituent having priority over the sulfur-containing group for citation as principal group"; (:35916) 'methyl hydrogen
# sulfate (PIN)' (:35968), 'methyl dihydrogen phosphate (PIN)' (:35940); the book's own name of a hydroxy organyl of an acid ester
# is '(2S)-2,3-dihydroxypropyl dihydrogen phosphate' (:55148,, no PIN label there,; the prefixes of the groups below
# the ester are the book's 'formyl' ('4-formylbenzoic acid (PIN)',:2158), 'hydrazinyl'+'oxo' ('3-hydrazinyl-3-oxopropanoic acid
# (PIN)',:34018), 'hydroxyimino' ('3-(hydroxyimino)butanal (PIN)',:38487). Independent check: the old names (the sulfooxy prefix
# under a junior parent) violate Table 4.1, and a fresh OPSIN call reads every new name back to the full InChIKey of the input.
# Mutation: with ``rules/seniority.py`` at 3ddf7c467e these rows name the old spelling and the tests fail (see the lane report).
# Direct book support for the oxo group: "Esters of mononuclear noncarbon oxoacids" prints 'CH3-CO-CH2-CH2-O-BrO2
# 3-oxobutyl bromate (PIN)' (:35974), a ketone group cited as a prefix of the organyl group of an ester of an inorganic acid.
# UNVERIFIED: the book prints no PIN example of an organyl group that carries a hydroxy, formyl or amino group beside an acid
# ester of sulfuric acid; for those the label rests on the class order of Table 4.1 and on 'Esters are named in the
# same way as esters of organic acids'.
_JUNIOR_PARENT_PINS = [
    ("COc1cccc(OS(=O)(=O)O)c1O", "2-hydroxy-3-methoxyphenyl hydrogen sulfate"),
    ("O=[N+]([O-])c1ccc(O)c(OS(=O)(=O)O)c1", "2-hydroxy-5-nitrophenyl hydrogen sulfate"),
    ("Oc1ccccc1OS(=O)(=O)O", "2-hydroxyphenyl hydrogen sulfate"),
    ("OCCOS(=O)(=O)O", "2-hydroxyethyl hydrogen sulfate"),
    ("O=Cc1ccc(OS(=O)(=O)O)cc1", "4-formylphenyl hydrogen sulfate"),
    ("O=Cc1ccccc1OS(=O)(=O)O", "2-formylphenyl hydrogen sulfate"),
    ("Oc1cc(OS(=O)(=O)O)ccc1C=O", "4-formyl-3-hydroxyphenyl hydrogen sulfate"),
    ("O=CCOS(=O)(=O)O", "2-oxoethyl hydrogen sulfate"),
    ("Nc1ccc(OS(=O)(=O)O)cc1", "4-aminophenyl hydrogen sulfate"),
    ("NNC(=O)CCOS(=O)(=O)O", "3-hydrazinyl-3-oxopropyl hydrogen sulfate"),
    ("ON=CCOS(=O)(=O)O", "2-(hydroxyimino)ethyl hydrogen sulfate"),
    ("O=Cc1ccc(OP(=O)(O)O)cc1", "4-formylphenyl dihydrogen phosphate"),
]


@pytest.mark.parametrize("smiles,expected", _JUNIOR_PARENT_PINS)
def test_an_oxoacid_ester_under_a_junior_group_is_the_functional_class_pin(smiles, expected):
    #: the ester outranks the groups below class 9, so it is the parent (functional class nomenclature,
    # and those groups are cited as prefixes of its organyl group, never the sulfooxy prefix of a name on the junior group
    _assert_named(smiles, expected, {"pin_verified"})


@pytest.mark.parametrize("smiles,expected", [
    ("c1ccccc1OS(=O)(=O)O", "phenyl hydrogen sulfate"),
    ("COS(=O)(=O)O", "methyl hydrogen sulfate"),                    # BB:35968
])
def test_sulfate_monoester_parent_unchanged(smiles, expected):
    _assert_named(smiles, expected, {"pin_verified"})


@pytest.mark.parametrize("smiles", [
    # acyl sulfates are mixed anhydrides, not sulfate esters
    "CC(=O)OS(=O)(=O)O",
    "OCC(=O)OS(=O)(=O)O",
])
def test_acyl_sulfate_is_not_claimed_as_sulfooxy_at_pin(smiles):
    row = Orthonym().name_tiered(smiles)
    assert row["tier"] == "abstain", row


@pytest.mark.parametrize("smiles,expected", [
    ("CCSC(=O)/C=C/C", "S-ethyl (2E)-but-2-enethioate"),
    ("CCSC(=O)C#CC", "S-ethyl but-2-ynethioate"),
    ("CCSC(=O)C=C", "S-ethyl prop-2-enethioate"),
    ("CC[Se]C(=O)/C=C\\C", "Se-ethyl (2Z)-but-2-eneselenoate"),
    ("CCSC(=O)[C@H](C)O", "S-ethyl (2S)-2-hydroxypropanethioate"),
    ("CCSC(=O)CCC", "S-ethyl butanethioate"),
])
def test_chalcogen_ester_acyl_chain(smiles, expected):
    _assert_named(smiles, expected, {"pin_verified"})


def test_thioester_dev_set_row():
    # a dev split / milestone1500 M11 row (it was refused: the diene was dropped). DT3 step C certifies the re-run name as the
    # PIN (assembly.pin_rerun.rerun_name_certified); a blind Blue Book reading agreed: (the Blue Book)
    # "Structural specification for esters of thio-... carboxylic acids... is provided by the appropriate italic element
    # symbol, such as S, O, or SO, prefixed to the name of the organyl group" ('S-ethyl hexanethioate (PIN)',:31989)
    _assert_named(
        "CC(=O)NCCSC(=O)/C=C/CCC[C@H](O)/C=C/CC(C)O",
        "S-(2-acetamidoethyl) (2E,7S,8E)-7,11-dihydroxydodeca-2,8-dienethioate",
        {"pin_verified"})


# Review fix (finding 1): the oxoacid-ester prefix is the PIN form only under a parent
# whose class is senior to the ester. (the Blue Book): "If a
# sulfur-containing group is attached by oxygen... to a compound that contains also
# another substituent having priority over the sulfur-containing group for citation as
# principal group, then the sulfur-containing group is named by an appropriate prefix";
# (:18160-18194) ranks acids (7) and esters (9) above hydrazides (12),
# aldehydes (15), hydroxy compounds (17) and amines (19). The names below round-trip
# (fresh OPSIN call, full key) but are not the PIN: since the lane/l17 fix round the engine names these molecules by the
# functional class ester (``_JUNIOR_PARENT_PINS``), and these spellings are the names the guard below reads as non-PIN.
_JUNIOR_HEAD_ROWS = [
    ("O=Cc1ccc(OS(=O)(=O)O)cc1", "4-(sulfooxy)benzaldehyde"),
    ("O=Cc1ccccc1OS(=O)(=O)O", "2-(sulfooxy)benzaldehyde"),
    ("Oc1cc(OS(=O)(=O)O)ccc1C=O", "2-hydroxy-4-(sulfooxy)benzaldehyde"),
    ("O=CCOS(=O)(=O)O", "(sulfooxy)acetaldehyde"),
    ("Nc1ccc(OS(=O)(=O)O)cc1", "4-(sulfooxy)aniline"),
    ("NNC(=O)CCOS(=O)(=O)O", "3-(sulfooxy)propanehydrazide"),
    # oximes are named substitutively, 'N-hydroxy' among the prefixes of the imine,
    # the Blue Book 'propanal oxime... N-hydroxypropan-1-imine (PIN)')
    ("ON=CCOS(=O)(=O)O", "N-hydroxy-2-(sulfooxy)ethan-1-imine"),
    ("O=Cc1ccc(OP(=O)(O)O)cc1", "4-(phosphonooxy)benzaldehyde"),
]


@pytest.mark.parametrize("smiles,expected", _JUNIOR_HEAD_ROWS)
def test_the_sulfooxy_spelling_under_a_junior_head_is_a_valid_name_below_the_pin(smiles, expected):
    # the spelling is read back by a fresh OPSIN call to its molecule; the name of the engine for the molecule is the functional
    # class ester of ``_JUNIOR_PARENT_PINS``, and the vocabulary guard below still reads this spelling as below the PIN
    assert _key(_independent_parse(expected, smiles)) == _key(smiles)


@pytest.mark.parametrize("name,token", [
    (name, "phosphonooxy" if "phosphono" in name else "sulfooxy")
    for _smi, name in _JUNIOR_HEAD_ROWS] + [
    ("4-(acetyloxy)benzaldehyde", "acetyloxy"),         # the carboxylic ester prefix
    ("4-(sulfooxy)benzoyl chloride", "sulfooxy"),       # acid halides are class 10
])
def test_guard_reads_the_head_class_from_p41(name, token):
    from orthonym.rules.pin_vocabulary import non_pin_vocabulary
    assert non_pin_vocabulary(name) == token


@pytest.mark.parametrize("name", [
    "3-(sulfooxy)propanoic acid",                        # BB:36488, acid (7)
    "3-(sulfooxy)propanoate",                            # anion (4)
    "sodium 3-(sulfooxy)propanoate",                     # salt (4)
    "2-(acetyloxy)-N,N,N-trimethylethan-1-aminium",      # cation (6)
    "3-(acetyloxy)-3-oxopropanoic butanedioic dianhydride",  # anhydride (8), BB PIN
    "4-(sulfooxy)phenyl",                                # a prefix string: no head
])
def test_guard_leaves_senior_heads_alone(name):
    from orthonym.rules.pin_vocabulary import non_pin_vocabulary
    assert non_pin_vocabulary(name) is None


# Review fix (finding 8): each word of a chalcogen ester carries its own descriptors.
# "NAMING OF STEREOISOMERS" (the Blue Book): "stereodescriptors are placed
# immediately at the front of the part of the name to which they relate... When they
# relate to substituent groups, they are cited at the front of the corresponding
# prefix"; '(2S)-butan-2-yl (4S)-4-chlorohexanoate (PIN)' (:46715). With stereo on the
# organyl side the acyl descriptors went in front of the whole name ('(2E)-S-[(3S)-3-
# hydroxybutyl] but-2-enethioate'), and an organyl-only descriptor left its word
# ('(2R)-S-[butan-2-yl] ethanethioate').
@pytest.mark.parametrize("smiles,expected", [
    ("C[C@H](O)CCSC(=O)/C=C/C", "S-[(3S)-3-hydroxybutyl] (2E)-but-2-enethioate"),
    ("C[C@H](O)SC(=O)/C=C/C", "S-[(1R)-1-hydroxyethyl] (2E)-but-2-enethioate"),
    ("C[C@@H](O)C(=O)SC[C@H](C)O",
     "S-[(2S)-2-hydroxypropyl] (2R)-2-hydroxypropanethioate"),
    ("C/C=C/CSC(=O)/C=C/C", "S-[(2E)-but-2-en-1-yl] (2E)-but-2-enethioate"),
    ("C[C@H](CC)SC(=O)C", "S-[(2R)-butan-2-yl] ethanethioate"),
    ("C/C=C/CSC(=O)C", "S-[(2E)-but-2-en-1-yl] ethanethioate"),
    ("CC[C@H](C)SC(=O)[C@@H](C)N", "S-[(2S)-butan-2-yl] (2R)-2-aminopropanethioate"),
    ("C[C@H](CC)[Se]C(=O)/C=C\\C", "Se-[(2R)-butan-2-yl] (2Z)-but-2-eneselenoate"),
])
def test_chalcogen_ester_descriptors_stand_in_their_own_word(smiles, expected):
    _assert_named(smiles, expected, {"pin_verified"})
