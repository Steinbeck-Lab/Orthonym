"""Leads program L4, item 30: three spellings that shipped at pin_verified.

(a) Identical branches of a phosphoryl-oxy prefix are multiplied, not repeated and
    enclosed. "General methodology" (the Blue Book), point (a): "Simple
    components are... unsubstituted prefixes, such as ethyl or tert-butyl... All of these
    are multiplied by the multiplicative prefixes di..." (:7031); 'dimethoxyphosphoryl' is
    the preferred prefix (:55874); '3-[(dimethoxyphosphoryl)sulfanyl]propanoic acid (PIN)'
     "Compound and complex substituent groups",:36327/:36335).
    (:7272): "the first cited substituent never has enclosing marks unless it includes a
    locant. The second and further substituents are each enclosed with parentheses even
    for simple substituents."
(b) An oxoacid with a substitutable hydrogen on its central atom is a functional parent:
     (:35453), Note (:35457) "treat the acid as a suffix (like sulfonic acid)
    leading to names such as benzenephosphonic acid. This suggestion has been rejected";
    'ethylphosphonic acid (PIN) (not ethanephosphonic acid)' (:35461). Label rows here;
    the producer rows are in ``test_leads_l4_30_oxoacid_prefix.py``.
(c) Simple prefixes on a ring are cited in alphanumerical order of their Roman letters:
     "ALPHANUMERICAL ORDER" (:3436), "Nonitalic Roman letters are considered first"
    (:3442); (:3495); '1-(butan-2-yl)-3-tert-butylbenzene (PIN)' (:3507).
"""
import pytest
from rdkit import Chem

from orthonym.rules.phosphorus import name_phosphoanhydride_oxy_substituent
from orthonym.rules.pin_vocabulary import non_pin_vocabulary
from tests.support.pin_tiers import assert_not_pin_labelled, name_breadth, name_default
from tests.support.rt_assert import name_is_rt_exact

pytestmark = pytest.mark.opsin_gate


def _aryl_phosphoryloxy(smiles: str) -> str:
    """The prefix ``name_phosphoanhydride_oxy_substituent`` writes for the first ``c-O-P``."""
    mol = Chem.MolFromSmiles(smiles)
    patt = Chem.MolFromSmarts("c-O-P")
    c, o, _p = mol.GetSubstructMatches(patt)[0]
    return name_phosphoanhydride_oxy_substituent(mol, o, c)


# ---- (a) --------------------------------------------------------------------------
PREFIX_ROWS = [
    ("c1ccccc1OP(=O)(OC)OC", "(dimethoxyphosphoryl)oxy"),
    ("c1ccccc1OP(=O)(OCC)OCC", "(diethoxyphosphoryl)oxy"),
    ("c1ccccc1OP(=O)(OCCCl)OCCCl", "[bis(2-chloroethoxy)phosphoryl]oxy"),
    ("c1ccccc1OP(=O)(OC(C)C)OC(C)C", "{bis[(propan-2-yl)oxy]phosphoryl}oxy"),
    # two different branches: the first cited bare, the second enclosed
    ("c1ccccc1OP(=O)(OCC)OC", "[ethoxy(methoxy)phosphoryl]oxy"),
    ("c1ccccc1OP(=O)(OC)O", "[hydroxy(methoxy)phosphoryl]oxy"),
    ("c1ccccc1OP(=O)(OCC)O", "[ethoxy(hydroxy)phosphoryl]oxy"),
    # a locant-bearing first branch is enclosed and a 'hydroxy' after it stays bare: the Blue
    # Book's own lipid spelling '(2-aminoethoxy)hydroxyphosphoryl' (:55180)
    ("c1ccccc1OP(=O)(OCCCl)O", "[(2-chloroethoxy)hydroxyphosphoryl]oxy"),
    ("c1ccccc1OP(=O)(O)O", "phosphonooxy"),
]


@pytest.mark.parametrize("smiles,expect", PREFIX_ROWS)
def test_phosphoryloxy_prefix_multiplies_identical_branches(smiles, expect):
    assert _aryl_phosphoryloxy(smiles) == expect


PHOSPHATE_ROWS = [
    ("COC(=O)c1ccc(cc1)OP(=O)(OC)OC", "methyl 4-[(dimethoxyphosphoryl)oxy]benzoate"),
    ("COC(=O)c1ccc(cc1)OP(=O)(OCC)OC", "methyl 4-{[ethoxy(methoxy)phosphoryl]oxy}benzoate"),
]


@pytest.mark.parametrize("smiles,pin", PHOSPHATE_ROWS)
def test_phosphate_prefix_names(smiles, pin):
    assert name_is_rt_exact(pin, smiles), pin
    d = name_default(smiles)
    assert (d.get("name"), d.get("tier")) == (pin, "pin_verified"), d
    b = name_breadth(smiles)
    assert (b.get("name"), b.get("tier")) == (pin, "pin_verified"), b


# The prefix form keeps its text (OPSIN reads it back) but names the junior class when the
# ester, class 9, outranks the group the name is built on: "SENIORITY ORDER FOR CLASSES"
# (the Blue Book), Table 4.1: "9 Esters (functional class names are given to noncyclic
# esters; lactones and other cyclic esters are named as heterocycles; see 16 below)" (:18182);
# "11 Amides" (:18184), "16 Ketones... pseudoketones" (:18189), "17 Hydroxy compounds"
# (:18190), "19 Amines" (:18192). The PIN of the coumarin diethyl phosphate is the ester
# '3-chloro-4-methyl-2-oxo-2H-1-benzopyran-7-yl diethyl phosphate'. The label rule is the
# prefix test of ``pin_vocabulary.non_pin_vocabulary`` (a phosphoryl-oxy prefix under a head
# junior to the esters); a PIN tier that cannot build the ester declines the row.
PREFIX_UNDER_JUNIOR_HEAD = [
    ("CCOP(=O)(OCC)Oc1ccc2c(C)c(Cl)c(=O)oc2c1",
     "3-chloro-7-[(diethoxyphosphoryl)oxy]-4-methyl-2H-1-benzopyran-2-one"),
    ("CCOP(=O)(OCC)Oc1ccc2ccc(=O)oc2c1", "7-[(diethoxyphosphoryl)oxy]-2H-1-benzopyran-2-one"),
    ("NC(=O)COP(=O)(OC)OC", "2-[(dimethoxyphosphoryl)oxy]ethanamide"),
    ("N[C@@H](C)COP(=O)(O)OC", "(2S)-1-{[hydroxy(methoxy)phosphoryl]oxy}propan-2-amine"),
    ("OC[C@H](O)COP(=O)(OC)OC", "(2S)-3-[(dimethoxyphosphoryl)oxy]propane-1,2-diol"),
]


@pytest.mark.parametrize("smiles,prefix_form", PREFIX_UNDER_JUNIOR_HEAD)
def test_phosphate_prefix_under_a_junior_class_is_not_labelled_pin(smiles, prefix_form):
    assert name_is_rt_exact(prefix_form, smiles), prefix_form
    assert_not_pin_labelled(smiles, prefix_form)


def test_coumarin_phosphate_breadth_name_is_the_ester():
    """The breadth tier writes the functional-class ester, which OPSIN reads back, and does
    not label it a verified PIN either (the owner is built by a breadth producer)."""
    smiles = "CCOP(=O)(OCC)Oc1ccc2c(C)c(Cl)c(=O)oc2c1"
    b = name_breadth(smiles)
    assert b.get("name") == "3-chloro-4-methyl-2-oxo-2H-1-benzopyran-7-yl diethyl phosphate", b
    assert name_is_rt_exact(b["name"], smiles), b


# ---- (b) label --------------------------------------------------------------------
NON_PIN_OXOACID = [
    "4-(dimethylsulfamoyl)benzene-1-phosphonic acid",
    "2-hydroxyethane-1-phosphonic acid",
    "(1E,4S)-4-hydroxypent-1-ene-1-boronic acid",
    "4-arsonobutane-1-phosphonic acid",
    "ethane-1,1-diphosphonic acid",
    "diethyl benzene-1-phosphonate",
]
PIN_OXOACID = [
    "phenylphosphonic acid",
    "[4-(dimethylsulfamoyl)phenyl]phosphonic acid",
    "(2-hydroxyethyl)phosphonic acid",
    "benzene-1,4-diylbis(phosphonic acid)",
    "(1-hydroxyethane-1,1-diyl)bis(phosphonic acid)",
    "[(1E,4S)-4-hydroxypent-1-en-1-yl]boronic acid",
    "ethylphosphonic acid",
]


NON_PIN_PHOSPHORYLOXY = [name for _, name in PREFIX_UNDER_JUNIOR_HEAD] + [
    "3-chloro-7-{[(ethoxy)(ethoxy)phosphoryl]oxy}-4-methyl-2H-1-benzopyran-2-one",
    "(2S)-1-{[ethoxy(methoxy)phosphoryl]oxy}propan-2-amine",
    "4-[(diethoxyphosphoryl)oxy]butan-2-one",
]
# the same prefix under a head of a class senior to the esters, or an ester head, is the PIN
# spelling: 'phosphonic acid is senior', "Compound and complex substituent groups"
# (the Blue Book): '(phosphonooxy)acetic acid (PIN)' (:36333), '3-[(dimethoxyphosphoryl)
# sulfanyl]propanoic acid (PIN)' (:36335); 'methyl [(dimethoxyphosphoryl)oxy]
# phosphonate (PIN)' (:36989); the Blue Book's lipid spelling (:55180).
PIN_PHOSPHORYLOXY = [
    "methyl 4-[(dimethoxyphosphoryl)oxy]benzoate",
    "4-[(dimethoxyphosphoryl)oxy]benzoic acid",
    "methyl [(dimethoxyphosphoryl)oxy]phosphonate",
    "(2R)-3-{[(2-aminoethoxy)hydroxyphosphoryl]oxy}propane-1,2-diyl dihexadecanoate",
    "2-({[2-(hexadecanoyloxy)ethoxy]hydroxyphosphoryl}oxy)-N,N,N-trimethylethan-1-aminium",
]


@pytest.mark.parametrize("name", NON_PIN_PHOSPHORYLOXY)
def test_phosphoryloxy_prefix_under_a_junior_head_is_not_pin_vocabulary(name):
    assert non_pin_vocabulary(name), name


@pytest.mark.parametrize("name", PIN_PHOSPHORYLOXY)
def test_phosphoryloxy_prefix_under_a_senior_head_is_pin_vocabulary(name):
    assert non_pin_vocabulary(name) is None, name


@pytest.mark.parametrize("name", NON_PIN_OXOACID)
def test_oxoacid_suffix_form_is_not_pin_vocabulary(name):
    assert non_pin_vocabulary(name), name


@pytest.mark.parametrize("name", PIN_OXOACID)
def test_oxoacid_functional_parent_form_is_pin_vocabulary(name):
    assert non_pin_vocabulary(name) is None, name


# ---- (c) --------------------------------------------------------------------------
ORDER_ROWS = [
    ("CCOc1cc(C(C)(C)C)ccc1P(=O)(OCC)OCC", "diethyl (4-tert-butyl-2-ethoxyphenyl)phosphonate"),
    ("CCOP(=O)(OCC)C(=Cc1cc(C(C)(C)C)c(O)c(C(C)(C)C)c1)P(=O)(OCC)OCC",
     "diethyl [2-(3,5-di-tert-butyl-4-hydroxyphenyl)-1-(diethoxyphosphoryl)ethen-1-yl]phosphonate"),
    ("CCOP(=O)(OCC)c1ccc(C(C)(C)C)c(O)c1", "diethyl (4-tert-butyl-3-hydroxyphenyl)phosphonate"),
    ("CCOP(=O)(OCC)c1ccc(C(C)(C)C)c(OC)c1", "diethyl (4-tert-butyl-3-methoxyphenyl)phosphonate"),
    # controls: the italic part never moves a prefix that the Roman letters already order
    ("CCOP(=O)(OCC)c1ccc(C(C)(C)C)c(Br)c1", "diethyl (3-bromo-4-tert-butylphenyl)phosphonate"),
    ("CCOP(=O)(OCC)c1ccc(C(C)(C)C)c(N)c1", "diethyl (3-amino-4-tert-butylphenyl)phosphonate"),
]


@pytest.mark.parametrize("smiles,expected", ORDER_ROWS)
def test_ring_prefixes_alphabetised_by_roman_letters(smiles, expected):
    assert name_is_rt_exact(expected, smiles), expected
    b = name_breadth(smiles)
    assert b.get("name") == expected, b
