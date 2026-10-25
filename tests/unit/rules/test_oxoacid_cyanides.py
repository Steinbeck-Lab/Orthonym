""" -- the cyanides of the sulfonic, sulfinic, selenonic, seleninic, telluronic and
tellurinic acids are named by functional class, ``<acyl> cyanide``, and that name is the PIN.

Governing rule, the Blue Book, heading " Cyanides corresponding to the sulfur,
selenium, and tellurium acids": "Cyanides formally formed by replacing the -OH group of sulfonic
acids, sulfinic acids, and similar Se and Te acids are named by functional class nomenclature."
Its two examples are ``CH3-SO2-CN methanesulfonyl cyanide (PIN)`` (:34811) and
``C6H5-SeO-CN benzeneseleninyl cyanide (PIN)`` (:34813). The grammar of the name is
"Acyl pseudohalides from suffix acids" (:31460: the name of the acyl group, then ``cyanide`` as a
separate word) and the acyl group is (:31260: "changing the 'ic acid' ending of the
suffix to 'yl'").

Before this class the -E(=O)-C#N unit of every member but the sulfonyl one was read as a nitrile on
a sulfoxide / selenoxide / selenone / telluroxide / tellurone and named ``(benzenesulfinyl)
methanenitrile`` and the like, labelled ``pin_verified``: a round-trip-correct name that is not the
Blue Book PIN. Every positive expectation below was read back by OPSIN 2.9.0 and OPSIN main when it
was recorded; the gate-on tests run the read-back themselves (a ``pin_verified`` label is earned by
it).
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym, name_compound
from orthonym.cli import _tiered_namer
from orthonym.data.oxoacid_cyanides import (
    ACID_ACYL_ENDINGS,
    ACYL_ENDING_OF_ACID_ENDING,
    OXOACID_CYANIDE_ENDINGS,
    OXOACID_CYANIDE_GROUPS,
)
from orthonym.errors import is_failure_name
from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check
from orthonym.validation.pin_spelling import check_pin_spelling
from orthonym.perception.functional_groups import FUNCTIONAL_GROUP_SMARTS, detect_functional_groups
from orthonym.rules.seniority import PREFIX_FORMS, SENIORITY_ORDER, SUFFIX_FORMS

# (SMILES, expected name, governing citation)
_BB_VERBATIM = [
    ("CS(=O)(=O)C#N", "methanesulfonyl cyanide",
     'P-66.5.1.3.2 "Cyanides corresponding to the sulfur, selenium, and tellurium acids", :34811'),
    ("N#C[Se](=O)c1ccccc1", "benzeneseleninyl cyanide",
     'P-66.5.1.3.2 "Cyanides corresponding to the sulfur, selenium, and tellurium acids", :34813'),
]

# The rest of the class: the rule's "similar Se and Te acids" with the acyl name of (:31260).
_CLASS = [
    ("N#CS(=O)c1ccccc1", "benzenesulfinyl cyanide"),
    ("CS(=O)C#N", "methanesulfinyl cyanide"),
    ("N#C[Se](=O)(=O)c1ccccc1", "benzeneselenonyl cyanide"),
    ("N#C[Te](=O)c1ccccc1", "benzenetellurinyl cyanide"),
    ("N#C[Te](=O)(=O)c1ccccc1", "benzenetelluronyl cyanide"),
    ("CC[Se](=O)C#N", "ethaneseleninyl cyanide"),
    ("C[Te](=O)(=O)C#N", "methanetelluronyl cyanide"),
    ("N#CS(=O)c1ccc(Cl)cc1", "4-chlorobenzene-1-sulfinyl cyanide"),
    ("N#CS(=O)c1ccccn1", "pyridine-2-sulfinyl cyanide"),
    ("CC(C)[Se](=O)(=O)C#N", "propane-2-selenonyl cyanide"),
    ("N#CCCS(=O)C#N", "2-cyanoethane-1-sulfinyl cyanide"),
    # the sulfonyl members keep their names,:34811)
    ("N#CS(=O)(=O)c1ccccc1", "benzenesulfonyl cyanide"),
    ("CCS(=O)(=O)C#N", "ethanesulfonyl cyanide"),
]

# A stereogenic S or Se: the CIP descriptor is the one of the cyanide, not the one of the parent acid
# (RDKit labels C[S@](=O)O R and C[S@](=O)C#N S: the ligand that replaces -OH changes the priorities).
# A locanted acyl name cites the descriptor of E at its front too: "The chirality symbols 'R' and
# 'S'" (the Blue Book), 'ethyl (R)-4-nitrobenzene-1-sulfinate (PIN)' -- E is a centre of the acid group of
# the parent, not of a sulfoxide prefix.
_STEREO = [
    ("C[S@](=O)C#N", "(S)-methanesulfinyl cyanide"),
    ("C[S@@](=O)C#N", "(R)-methanesulfinyl cyanide"),
    ("C[Se@](=O)C#N", "(S)-methaneseleninyl cyanide"),
    ("CC(C)[S@@](=O)C#N", "(R)-propane-2-sulfinyl cyanide"),
    ("N#C[S@](=O)c1ccc(Cl)cc1", "(R)-4-chlorobenzene-1-sulfinyl cyanide"),
]

# Several cyanide groups: (:31460) "followed by the name(s) of the class(es) as separate words,
# preceded by a multiplicative prefix, as needed" -- the acid carries the multiplied suffix ('benzene-1,4-disulfinic
# acid') and so does the acyl group.
_MULTIPLIED = [
    ("N#CS(=O)c1ccc(S(=O)C#N)cc1", "benzene-1,4-disulfinyl dicyanide"),
    ("N#CS(=O)(=O)CCS(=O)(=O)C#N", "ethane-1,2-disulfonyl dicyanide"),
    ("N#C[Se](=O)CC[Se](=O)C#N", "ethane-1,2-diseleninyl dicyanide"),
]


@pytest.mark.unit
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected,cite", _BB_VERBATIM)
def test_blue_book_examples_are_pins(smiles, expected, cite):
    row = Orthonym().name_tiered(smiles)
    assert row["name"] == expected, cite
    assert row["tier"] == "pin_verified", cite


@pytest.mark.unit
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", _CLASS + _STEREO + _MULTIPLIED)
def test_class_members_are_functional_class_pins(smiles, expected):
    row = Orthonym().name_tiered(smiles)
    assert row["name"] == expected
    assert row["tier"] == "pin_verified" and row["is_pin"] is True
    assert row["opsin"] == "verified"


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", _CLASS)
def test_class_names_without_the_gate(smiles, expected):
    """The name itself does not depend on the read-back gate (the gate only vets it)."""
    assert name_compound(smiles) == expected


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", _CLASS)
def test_no_member_is_named_as_a_nitrile(smiles, expected):
    """The old defect: '(benzenesulfinyl)methanenitrile' / 'benzeneseleninylmethanenitrile'."""
    assert "nitrile" not in name_compound(smiles).replace("cyanide", "")
    assert name_compound(smiles).endswith(" cyanide")


@pytest.mark.unit
@pytest.mark.opsin_gate
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
def test_two_stereocentres_never_ship_a_name_without_the_chalcogen_descriptor(tier):
    """A chiral carbon AND a chiral S: the handler cannot cite both, so the name is declined rather than
    shipped without the descriptor of S (the read-back gate sees the stereo layer). Whatever a tier does
    emit for the molecule reads back to it, the stereo layer included (OPSIN 2.9.0)."""
    smiles = "C[C@H](Cl)[S@](=O)C#N"
    row = _tiered_namer("pin", tier).name_tiered(smiles)
    assert row["tier"] != "pin_verified"
    if row["name"] and not is_failure_name(row["name"]) and row["tier"] != "abstain":
        assert opsin_roundtrip_check(smiles, row["name"])["passed"], row["name"]


@pytest.mark.unit
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", [
    "N#CS(=O)CCS(=O)(=O)C#N",   # a sulfinyl and a sulfonyl cyanide: two acids in one name
    "N#CS(=O)N(C)C",
])
def test_unbuilt_shapes_do_not_ship_a_pin(smiles):
    assert Orthonym().name_tiered(smiles)["tier"] != "pin_verified"


# --- a nuclide on E, the cyanide carbon or the cyanide nitrogen ----------------------------------


@pytest.mark.unit
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,best_effort_name", [
    ("CS(=O)[13C]#N", "(methanesulfinyl)(13C)methanenitrile"),
    ("CS(=O)C#[15N]", "(methanesulfinyl)(15N)methanenitrile"),
    ("CS(=O)(=O)[13C]#N", "(methanesulfonyl)(13C)methanenitrile"),
    ("c1ccccc1[Se](=O)[13C]#N", "(benzeneseleninyl)(13C)methanenitrile"),
    ("C[34S](=O)C#N", "[(34S)methanesulfinyl]methanenitrile"),
])
def test_a_labelled_unit_keeps_a_name_below_the_pin(smiles, best_effort_name):
    """ (the Blue Book): the nuclide descriptor stands "before the part of the compound that is
    isotopically substituted"; the class word 'cyanide' has no place for the label of the cyanide carbon or
    nitrogen, and 'methanesulfinyl cyanide' none for a nuclide of S at its 'sulfinyl'. The handler therefore
    builds no functional-class name for a labelled unit (it sees the labelled molecule through the decorator's
    ambient scope; the stripped skeleton has no isotope), and offers the substitutive '(acyl)methanenitrile'.
    That name has a place for the label, reads back, and is below the PIN: the default tier declines it, the
    best-effort tier keeps it (it was 'pin_verified' before, and for the cyanide carbon and nitrogen of a
    sulfonyl cyanide it was nothing at any tier)."""
    default = _tiered_namer("pin", "pin").name_tiered(smiles)
    assert default["tier"] == "abstain"
    wide = _tiered_namer("pin", "best-effort").name_tiered(smiles)
    assert wide["name"] == best_effort_name
    assert wide["tier"] == "systematic_verified" and wide["is_pin"] is False
    assert opsin_roundtrip_check(smiles, wide["name"])["passed"]


@pytest.mark.unit
@pytest.mark.opsin_gate
def test_a_label_on_the_organyl_group_keeps_the_pin():
    """A nuclide of the organyl group is cited in front of the part it modifies, 'methane':."""
    row = Orthonym().name_tiered("[2H]C([2H])([2H])S(=O)C#N")
    assert row["name"] == "(2H3)methanesulfinyl cyanide" and row["tier"] == "pin_verified"


# --- the acid halides and cyanides of the six acids: one table, the order -------------------


@pytest.mark.unit
def test_the_handler_reads_one_table_of_endings():
    """The halide and the cyanide branches and the substituent-prefix route read the one table
    ,:31260)."""
    from orthonym.rules import sulfur
    assert set(ACID_ACYL_ENDINGS) == {"sulfonyl_halide", "sulfinyl_halide"} | set(OXOACID_CYANIDE_GROUPS)
    for group, (acid, acyl) in ACID_ACYL_ENDINGS.items():
        assert ACYL_ENDING_OF_ACID_ENDING[acid] == acyl, group
    assert sulfur._OXIDE_ACID_SUFFIX == {"sulfinyl": "sulfinic acid", "sulfonyl": "sulfonic acid"}


@pytest.mark.unit
def test_seniority_of_the_acid_halides_and_cyanides_follows_p_41():
    """ class 10 (the Blue Book): "Acid halides and pseudohalides [first in the order of the
    corresponding acid given above, then in the order of the halogen atoms...; then... pseudohalogen groups";
    class 7a (:18172): the acids "carboxylic..., sulfonic, sulfinic, selenonic, seleninic, telluronic,
    tellurinic". The halide and the cyanide of one acid are adjacent, the halide first."""
    order = [g for g in SENIORITY_ORDER if g in ACID_ACYL_ENDINGS]
    assert order == [
        "sulfonyl_halide", "sulfonyl_cyanide",
        "sulfinyl_halide", "sulfinyl_cyanide",
        "selenonyl_cyanide", "seleninyl_cyanide",
        "telluronyl_cyanide", "tellurinyl_cyanide",
    ]
    position = {g: SENIORITY_ORDER.index(g) for g in order}
    assert all(position[a] + 1 == position[b] for a, b in zip(order, order[1:]))


# --- the guard: a carbon or a ring nitrogen beside the cyanide carbon ----------------------------


@pytest.mark.unit
@pytest.mark.parametrize("smiles,group", [
    ("N#CS(=O)(=O)N1CCCCC1", "sulfonyl_cyanide"),
    ("N#CS(=O)N1CCCCC1", "sulfinyl_cyanide"),
    ("N#C[Se](=O)N1CCCCC1", "seleninyl_cyanide"),
])
def test_a_ring_nitrogen_is_a_parent_hydride_position(smiles, group):
    """ (:31160) names the acids by a suffix on a parent hydride; a ring nitrogen takes it:
    'pyrrolidine-1-sulfonamide (PIN)', "Sulfonamides, sulfinamides, and related selenium and
    tellurium amides" (:32744,:32758)."""
    assert len(_groups(smiles)[group]) == 1


@pytest.mark.unit
@pytest.mark.parametrize("smiles", [
    "N#CS(=O)(=O)N(C)C", "N#CS(=O)(=O)N", "N#CS(=O)(=O)OC", "N#CS(=O)(=O)SC",
])
def test_sulfonyl_cyanide_guard_is_the_one_of_the_other_groups(smiles):
    """The old $([SX4][#6]) guard was satisfied by the cyanide carbon itself, so a sulfamoyl, a sulfonate and a
    thiosulfonate cyanide were claimed and left to fail at the acid name."""
    assert "sulfonyl_cyanide" not in _groups(smiles)


# --- the spelling checks: a nitrile spelling of the cyanide of an acid, or of a heteroatom, is no PIN ---


def _rules(smiles, name):
    return [f.rule for f in check_pin_spelling(Chem.MolFromSmiles(smiles), name, strict=True)]


@pytest.mark.unit
@pytest.mark.parametrize("smiles,name,rule", [
    ("N#CS(=O)c1ccccc1", "(benzenesulfinyl)methanenitrile", "P-66.5.1.3.2"),
    ("N#C[Se](=O)c1ccccc1", "benzeneseleninylmethanenitrile", "P-66.5.1.3.2"),
    ("N#CS(=O)(=O)C", "(methanesulfonyl)methanenitrile", "P-66.5.1.3.2"),
    ("N#CN=O", "nitrosomethanenitrile", "P-67.1.2.5.1"),
    ("N#CB(C)C", "(dimethylboranyl)methanenitrile", "P-66.5.1.1.3"),
    ("N#C[Si](C)(C)c1ccccc1", "[dimethyl(phenyl)silyl]methanenitrile", "P-66.5.1.1.3"),
])
def test_nitrile_spelling_of_a_cyanide_is_flagged(smiles, name, rule):
    assert rule in _rules(smiles, name)


@pytest.mark.unit
@pytest.mark.parametrize("smiles,name", [
    ("N#CS(=O)c1ccccc1", "benzenesulfinyl cyanide"),
    ("N#CB(C)C", "dimethylboranecarbonitrile"),
    ("N#C[Si](C)(C)c1ccccc1", "dimethyl(phenyl)silanecarbonitrile"),
    ("N#Cc1ccccc1", "benzonitrile"),
    ("CC#N", "acetonitrile"),
    ("N#CCC#N", "propanedinitrile"),
    # a 'nitrile' that belongs to another group: the acid cyanide is a prefix, no spelling check speaks
    ("N#CCCS(=O)C#N", "3-(cyanosulfinyl)propanenitrile"),
    # a ring atom is a position of a heterocycle: (the Blue Book) "-carbonitrile... attached
    # to a ring or ring system"; the cyclic esters are named as heterocycles, (:32177)
    ("C1CCP(=O)(C#N)C1", "1-oxo-1\u03bb5-phospholane-1-carbonitrile"),
    ("N#CP1(=O)CCCCC1", "1-oxo-1\u03bb5-phosphinane-1-carbonitrile"),
    ("O=P1(C#N)OCCO1", "2-oxo-1,3,2\u03bb5-dioxaphospholane-2-carbonitrile"),
    ("O=P1(C#N)OCCCO1", "2-oxo-1,3,2\u03bb5-dioxaphosphinane-2-carbonitrile"),
    ("C1CC[As](=O)(C#N)C1", "1-oxo-1\u03bb5-arsolane-1-carbonitrile"),
])
def test_other_names_are_not_flagged(smiles, name):
    rules = _rules(smiles, name)
    assert "P-66.5.1.3.2" not in rules and "P-67.1.2.5.1" not in rules and "P-66.5.1.1.3" not in rules


# --- the cyanides on a boron or silicon parent hydride ---------------------------------------------

# (the Blue Book): "The suffix 'carbonitrile' is always used to name nitriles having the -CN
# group attached to a ring or ring system or to an acyclic heteroatom" ('silanecarbonitrile (PIN)',:34722);
# "Suffix nomenclature" (:37387): "Suffixes containing a carbon atom generate preferred IUPAC names",
# 'H2B-CN boranecarbonitrile (PIN)' (:37393); (:35706): the pseudohalides of the boron acids and of
# silicic acid "are formed on the basis of the parent hydride names borane and silane".
_HUB_NITRILES = [
    ("N#C[BH2]", "boranecarbonitrile"),
    ("N#CB(C)C", "dimethylboranecarbonitrile"),
    ("N#CB(c1ccccc1)c1ccccc1", "diphenylboranecarbonitrile"),
    ("N#C[SiH3]", "silanecarbonitrile"),
    ("N#C[Si](C)(C)c1ccccc1", "dimethyl(phenyl)silanecarbonitrile"),
]


@pytest.mark.unit
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", _HUB_NITRILES)
def test_carbonitrile_on_a_boron_or_silicon_parent_hydride(smiles, expected):
    row = Orthonym().name_tiered(smiles)
    assert row["name"] == expected and row["tier"] == "pin_verified"


@pytest.mark.unit
@pytest.mark.opsin_gate
def test_nitrosyl_cyanide_is_not_a_pin():
    """NC-N=O is the cyanide of nitrous acid (HO-NO, 'preselected name',,:35431): by
    (:35706) an acid pseudohalide ('nitrous cyanide'), senior to a nitrile classes 10 and 14); the book
    prints no example and the engine builds none, so 'nitrosomethanenitrile' is a valid name below the PIN."""
    row = Orthonym().name_tiered("N#CN=O")
    assert row["tier"] != "pin_verified"


@pytest.mark.unit
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", [
    "OC(=O)CS(=O)C#N", "OC(=O)C[Se](=O)C#N", "CCOC(=O)CS(=O)C#N", "COC(=O)c1ccc(S(=O)C#N)cc1",
])
def test_demoted_cyanide_is_not_a_pin(smiles):
    """ "Acyl halides and pseudohalides as substituent groups" (:31517): with a carboxylic acid
    present the cyanide is a prefix, (1) acid-derived ('2-sulfurocyanidoylcyclohexane-1-carboxylic
    acid (PIN)',:31535) or (2) a halo + divalent-acyl compound prefix ('(cyanosulfonyl)', same line; cf.
     "Acyl groups",:36474 '-SO2-CN cyanosulfonyl (preferred prefix) sulfurocyanidoyl').
    This engine builds neither for the sulfinyl / seleninyl / tellurinyl groups: the demoted group has no
    prefix form (fails closed, like 'sulfonyl_cyanide'), so the old 'cyano(sulfinyl)acetic acid', which
    is neither of the two, is no longer labelled a PIN."""
    row = Orthonym().name_tiered(smiles)
    assert row["tier"] != "pin_verified"
    assert not (row["name"] or "").startswith(("cyano(", "ethyl cyano("))
    assert " cyano(" not in (row["name"] or "")


# --- perception -------------------------------------------------------------------------------


def _groups(smiles):
    return {k: v for k, v in detect_functional_groups(Chem.MolFromSmiles(smiles)).items() if v}


@pytest.mark.unit
@pytest.mark.parametrize("smiles,group,rival", [
    ("N#CS(=O)c1ccccc1", "sulfinyl_cyanide", "sulfoxide"),
    ("N#C[Se](=O)c1ccccc1", "seleninyl_cyanide", "selenoxide"),
    ("N#C[Se](=O)(=O)c1ccccc1", "selenonyl_cyanide", "selenone"),
    ("N#C[Te](=O)c1ccccc1", "tellurinyl_cyanide", "telluroxide"),
    ("N#C[Te](=O)(=O)c1ccccc1", "telluronyl_cyanide", "tellurone"),
])
def test_group_owns_its_unit(smiles, group, rival):
    """The unit E(=O)x-C#N is one group: the nitrile and the sulfoxide-type reads on its atoms go."""
    found = _groups(smiles)
    assert len(found[group]) == 1
    assert "nitrile" not in found and rival not in found


@pytest.mark.unit
def test_suppression_is_atom_scoped():
    """A nitrile and a sulfoxide elsewhere in the molecule keep their own reads."""
    found = _groups("CS(=O)CCS(=O)C#N")
    assert len(found["sulfinyl_cyanide"]) == 1
    assert len(found["sulfoxide"]) == 1          # the CH3-S(=O)-CH2 one
    found = _groups("N#CCCS(=O)C#N")
    assert len(found["nitrile"]) == 1            # the CH2-C#N one
    assert len(found["sulfinyl_cyanide"]) == 1


@pytest.mark.unit
@pytest.mark.parametrize("smiles", [
    "N#CS(=O)N(C)C",     # sulfinamide-type: no carbon organyl on S
    "N#CS(=O)OC",        # sulfinate-type
    "N#C[Se](=O)OC",
    "N#C[Se](=O)(=O)N",
    "N#C[Te](=O)OC",
    "N#CS(=O)SC",
])
def test_hetero_atom_on_the_chalcogen_is_not_this_class(smiles):
    """The recursive guard needs a SECOND carbon on E (the organyl R); the cyanide carbon alone does not
    make an organyl sulfinic / seleninic / tellurinic acid cyanide."""
    found = _groups(smiles)
    assert not (set(OXOACID_CYANIDE_GROUPS) - {"sulfonyl_cyanide"}) & set(found)


@pytest.mark.unit
def test_a_dialkyl_selenoxide_is_not_a_cyanide():
    found = _groups("C[Se](=O)CC#N")
    assert "seleninyl_cyanide" not in found and "selenoxide" in found and "nitrile" in found


# --- the tables agree -------------------------------------------------------------------------


@pytest.mark.unit
def test_every_group_is_perceived_ranked_and_without_a_static_form():
    """A group in the data table but not in the seniority order is inert; one without its None forms
    would reach the generic suffix / prefix path, which has no spelling for it."""
    for group in OXOACID_CYANIDE_GROUPS:
        assert group in FUNCTIONAL_GROUP_SMARTS, group
        assert group in SENIORITY_ORDER, group
        assert SUFFIX_FORMS[group] is None, group
        assert PREFIX_FORMS[group] is None, group


@pytest.mark.unit
def test_acid_and_acyl_endings_follow_p_65_3_2_2_2():
    """:31260 -- the acyl group is the acid name with 'ic acid' changed to 'yl'."""
    for group, (acid, acyl) in OXOACID_CYANIDE_ENDINGS.items():
        assert acid.endswith("ic acid") and acyl == acid[: -len("ic acid")] + "yl", group


@pytest.mark.unit
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,ring_pin", [
    ("C1CCP(=O)(C#N)C1", "1-oxo-1\u03bb5-phospholane-1-carbonitrile"),
    ("O=P1(C#N)OCCO1", "2-oxo-1,3,2\u03bb5-dioxaphospholane-2-carbonitrile"),
    ("C1CC[As](=O)(C#N)C1", "1-oxo-1\u03bb5-arsolane-1-carbonitrile"),
])
def test_ring_carbonitrile_keeps_its_pin_label(smiles, ring_pin):
    """The spelling check does not demote the -carbonitrile of a ring atom,:34720)."""
    row = Orthonym().name_tiered(smiles)
    assert row["name"] == ring_pin and row["tier"] == "pin_verified"


# --- the multiplicative producer and the units that carry the cyanide of an acid -----------------------


@pytest.mark.unit
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", [
    "N#C[Se](=O)c1ccc(cc1)Oc1ccc(cc1)[Se](=O)C#N",
    "N#C[Te](=O)(=O)c1ccc(cc1)Oc1ccc(cc1)[Te](=O)(=O)C#N",
])
def test_multiplicative_producer_names_no_selenium_or_tellurium_cyanide_unit(smiles):
    """The unit namer builds no name for the selenonyl, seleninyl, telluronyl and tellurinyl cyanides, so the
    multiplicative producer has no unit to multiply and declines; no name of the molecule is labelled a PIN.
    (The sulfinyl cyanide unit is multiplied with its class word: ``test_multiplied_class_words.py``.)"""
    from rdkit import Chem
    from orthonym.rules.multiplicative import name_multiplicative
    assert name_multiplicative(Chem.MolFromSmiles(smiles)) is None
    row = Orthonym().name_tiered(smiles)
    assert row["tier"] != "pin_verified"
    assert "oxydibenzene" not in (row["name"] or "")
