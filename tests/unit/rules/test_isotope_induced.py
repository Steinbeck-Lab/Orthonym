""": the stereodescriptors of the centres a nuclide makes, for any name of the isotope decorator.

Cited rules (the Blue Book, with their section headings):

* STEREOISOMERIC ISOTOPICALLY SUBSTITUTED COMPOUNDS: "Two types of stereoisomeric isotopically
  substituted compounds are possible: (1) those in which the stereoisomerism results from isotopic
  modification; (2) those whose analogous unmodified compounds are themselves stereoisomers."... "When they
  must be inserted into the name at the same place as isotopic descriptors, the stereodescriptors are cited
  first." Its example '(1R)-(1-2H1)ethan-1-ol (PIN)'.
* NAMING OF STEREOISOMERS (:44643): "In preferred IUPAC names, stereodescriptors are placed
  immediately at the front of the part of the name to which they relate. They are placed at the front of the
  complete name when related to the parent structure... When they relate to substituent groups, they are
  cited at the front of the corresponding prefix."
  Examples that put a descriptor of the acid part of an ester after the alkyl word: 'ethyl (R)-4-nitrobenzene-
  1-sulfinate (PIN)' The chirality symbols 'R/S',:46268) and 'methyl (S)-[methyl(phenyl)
  phosphinate] (PIN)' Phosphates, phosphonates, and related compounds,:46022).
*: the descriptors of one group are cited in ascending order of their locants
  ('(2R,3R)-' and a centre at 5 give '(2R,3R,5S)-').
* Multiple types of enclosing marks,: the parentheses of a descriptor count in the
  nesting order.

The expected names are what OPSIN reads back to the input (each is checked again here by an independent
OPSIN call, ``name_is_rt_exact``); the PIN is claimed only where the label says so.
"""
import pytest
from rdkit import Chem

from orthonym.rules import isotope_induced as II
from tests.support.rt_assert import name_best_effort, name_is_rt_exact

pytestmark = pytest.mark.unit


# -- what is a descriptor, which atoms a nuclide makes stereogenic (no OPSIN) ------------------------------


@pytest.mark.parametrize("name,found", [
    ("(2R)-(1-2H1)propan-2-yl 4-bromobenzene-1-sulfonate", ["(1-2H1)"]),
    ("(2S)-butan-2-ol", []),                                  # a stereodescriptor reads like '2S'
    ("(3Z)-hex-3-ene", []),
    ("(O-2H)acetic acid", ["(O-2H)"]),
    ("2-(2,3-2H2,15N)amino", ["(2,3-2H2,15N)"]),
    ("1-(phosphonooxy)-3-[(2H1)phosphonooxy]propan-2-ol", ["(2H1)"]),
])
def test_isotope_descriptors_are_told_from_stereodescriptors(name, found):
    assert [name[s:e] for s, e in II.descriptor_spans(name)] == found


@pytest.mark.parametrize("smiles,count", [
    ("[H][C@]1([C@@](C(=O)N1c1ccccc1)(C)C=C)[2H]", 1),        # a CHD of a ring
    ("[2H]C[C@@H](C)OS(=O)(=O)c1ccc(Br)cc1", 1),              # a carbon beside a CH2D
    ("C[C@H](O)[2H]", 1),                                     #: (1R)-(1-2H1)ethan-1-ol
    ("[2H]O[C@@H](C)C(=O)O", 0),                              # a centre of the molecule itself
    ("[2H]C([2H])([2H])[C@H](N)C(=O)O", 0),
])
def test_induced_centres_are_the_centres_only_the_nuclides_make(smiles, count):
    mol = Chem.MolFromSmiles(smiles)
    assert len(II.induced_centres(mol)) == count


def test_a_double_bond_a_nuclide_makes_is_not_built():
    # CH3 and CD3 on one carbon of C=C make its double bond stereogenic: not a tetrahedral centre
    assert II.induced_centres(Chem.MolFromSmiles("C/C=C(/C)C([2H])([2H])[2H]")) is None


# -- where the descriptors stand (no OPSIN) ----------------------------------------------------------------


def test_a_centre_of_a_substituent_is_cited_at_the_front_of_that_prefix():
    name = "methyl 3-[5-oxo-4,5,6,7-tetrahydro(7-2H1)-1H-pyrrolo[2,3-c]pyridin-3-yl]propanoate"
    scopes = II._scopes(name)
    # inside the prefix first, then the front of the acid word, then the front of the name
    assert scopes == [name.index("[") + 1, name.index(" ") + 1, 0]
    assert II._cited(name, {scopes[0]: [(7, "R")]}) == (
        "methyl 3-[(7R)-5-oxo-4,5,6,7-tetrahydro(7-2H1)-1H-pyrrolo[2,3-c]pyridin-3-yl]propanoate")


def test_the_centre_of_the_acid_of_an_ester_is_cited_after_the_alkyl_word():
    name = "ethyl 2-(4-tert-butylphenyl)(2-2H1)acetate"
    assert II._scopes(name) == [name.index(" ") + 1, 0]       # the front of the acid word, then the front
    assert II._cited(name, {name.index(" ") + 1: [(2, "S")]}) == (
        "ethyl (2S)-2-(4-tert-butylphenyl)(2-2H1)acetate")
    # a centre of the first word has no other scope than the front
    assert II._scopes("(1-2H1)propan-2-yl 4-bromobenzene-1-sulfonate") == [0]
    # a centre of a substituent of the acid word: the group first, then the word, then the front
    scopes = II._scopes("methyl 3-[5-oxo-(7-2H1)pyridin-3-yl]propanoate")
    assert len(scopes) == 3 and scopes[0] > scopes[1] > scopes[2] == 0


def test_a_front_descriptor_is_misplaced_when_the_centre_is_in_a_later_word_or_in_marks():
    assert II._front_is_misplaced("methyl (2-2H1)propanoate")
    assert II._front_is_misplaced("3-[5-oxo(7-2H1)pyridin-3-yl]propanoic acid")
    assert not II._front_is_misplaced("(1-2H1)ethan-1-ol")
    assert not II._front_is_misplaced("(1-2H1)propan-2-yl 4-bromobenzene-1-sulfonate")


def test_a_descriptor_joins_the_group_the_scope_opens_with_in_locant_order():
    name = "[(2R,3R)-2-ethyl(5-3H1)oxolan-3-yl]oxy"
    off = name.index("(2R")
    assert II._cited(name, {off: [(5, "S")]}) == "[(2R,3R,5S)-2-ethyl(5-3H1)oxolan-3-yl]oxy"


def test_the_marks_of_a_prefix_that_gains_a_group_step_up():
    name = "2-[(2-ethyl(5-3H1)oxolan-3-yl)oxy]ethan-1-ol"
    off = name.index("(2-ethyl") + 1
    assert II._cited(name, {off: [(5, "R")]}) == "2-{[(5R)-2-ethyl(5-3H1)oxolan-3-yl]oxy}ethan-1-ol"


# -- names (OPSIN) -----------------------------------------------------------------------------------------

#: (smiles, name, rule); the production gate  is on, as ``name`` reads in a run
CASES = [
    ("[H][C@]1([C@@](C(=O)N1C2=CC=CC=C2)(C)C=C)[2H]",
     "(3R,4R)-3-ethenyl-3-methyl-2-oxo-1-phenyl(4-2H1)azetidine",
     "P-82.4: the descriptor of the CHD joins the group of the centre the molecule has"),
    ("[2H]C[C@@H](C)OS(=O)(=O)C1=CC=C(C=C1)Br",
     "(2R)-(1-2H1)propan-2-yl 4-bromobenzene-1-sulfonate",
     "P-82.4: a centre beside a CH2D, cited at the front of the first word"),
    ("[H][C@]1(C2=C(CC(=O)N1)C(=CN2)CCC(=O)OC)[2H]",
     "methyl 3-[(7R)-5-oxo-4,5,6,7-tetrahydro(7-2H1)-1H-pyrrolo[2,3-c]pyridin-3-yl]propanoate",
     "P-91.3: the centre belongs to the substituent, so its descriptor stands at the front of the prefix"),
    ("[H][C@@]1(C[C@H]([C@H](O1)COP(=S)(OCCC#N)O[C@@H]2C[C@@](O[C@@H]2CC)([H])[3H])O)[3H]",
     "3-[({[(2R,3R,5R)-2-ethyl(5-3H1)oxolan-3-yl]oxy}{[(2R,3R,5S)-3-hydroxy(5-3H1)oxolan-2-yl]methoxy}"
     "phosphorothioyl)oxy]propanenitrile",
     "P-91.3 and P-93.4.1.2: two centres in two prefixes, each descriptor in its own group"),
    ("[H][C@]([2H])(CC(=O)OC)[C@]([H])([2H])N(CC1=CC=CC=C1)C(=O)OC(C)(C)C",
     "methyl (3S,4S)-4-[benzyl(tert-butoxycarbonyl)amino](3,4-2H2)butanoate",
     "P-91.3 and P-93.3.4.1: two centres of the acid of an ester, cited at the front of the acid word"),
]

#: the family of one spelling: the centre is in the acid word of an ester or an anhydride, found by the new pass
#: (the first two) and by the one-descriptor enumeration (the others); all pin_verified at both tiers
ESTERS = [
    ("CCOC(=O)[C@@H]([2H])c1ccc(cc1)C(C)(C)C", "ethyl (2S)-2-(4-tert-butylphenyl)(2-2H1)acetate"),
    ("CCCCOC(=O)[C@@H]([2H])c1ccc(Br)cc1", "butyl (2S)-2-(4-bromophenyl)(2-2H1)acetate"),
    ("CCOC(=O)[C@@H]([2H])c1ccccc1", "ethyl (2S)-phenyl(2-2H1)acetate"),
    ("C[C@@H]([2H])C(=O)OC", "methyl (2R)-(2-2H1)propanoate"),
    ("CC(=O)OC(=O)[C@H](C)[2H]", "acetic (2S)-(2-2H1)propanoic anhydride"),
    ("COC(=O)[C@H](N)[C@@H](C)[2H]", "methyl (2R,3R)-2-amino(3-2H1)butanoate"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected,rule", CASES, ids=[c[2][:40] for c in CASES])
def test_the_stereodescriptors_of_the_induced_centres_are_cited(smiles, expected, rule):
    res = name_best_effort(smiles)
    assert res["name"] == expected, rule
    assert name_is_rt_exact(res["name"], smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", ESTERS, ids=[e[1][:40] for e in ESTERS])
def test_the_descriptor_of_the_acid_of_an_ester_stands_in_the_acid_word_at_both_tiers(smiles, expected):
    """: 'stereodescriptors are placed immediately at the front of the part of the name to which they
    relate'. The check of the placement (``validation/spelling/checks_placement.py``) declines a front
    descriptor only when its locant is higher than the positions of the first word, so a name with the
    descriptor in front of 'ethyl' or 'butyl' passed and the same in front of 'methyl' did not: the
    producers now put it in the word of its centre, and the labels agree."""
    from orthonym import Orthonym
    from orthonym.cli import _emit_tier_flags
    for flags in ({}, _emit_tier_flags("best-effort")):
        row = Orthonym(style="pin", **flags).name_tiered(smiles)
        assert (row["name"], row["tier"]) == (expected, "pin_verified")
        assert name_is_rt_exact(row["name"], smiles)
