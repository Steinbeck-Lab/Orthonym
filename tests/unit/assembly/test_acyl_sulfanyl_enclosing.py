""" /: the enclosing marks of an S-acyl sulfanyl prefix.

An acyl group on sulfur ('R-CO-S-', the S side of a thioester that is not the principal group) is
cited as the compound prefix '(acyl)sulfanyl'. The acyl part is a compound prefix in its own right
, section " Parentheses", the Blue Book: "Parentheses are used around
compound (see and complex (see prefixes"). When it already carries marks (a
stereodescriptor, a substituent in parentheses) the marks of every group around it follow the
nesting order of (heading:7444, sentence:7446): "When multiple types of enclosing marks
are required, the nesting order is as follows: {[({})]}, etc." (:7478): "In
determining the nesting order in a name the parentheses used to indicate compound locants, phane
amplification locants, stereochemistry,... are taken into consideration", so the parentheses of a
stereodescriptor count. (:7509): "When the nesting order given in results in
consecutive enclosing marks of the same level, the next level of enclosing mark is used."

The book prints the pattern for an acyl group on sulfur: '1-[(acetylperoxy)sulfanyl]ethan-1-one
(PIN)' (section " Diacyl derivatives of trioxidane and chalcogen analogues", rule
 :32358, example:32368), '1-{[(methylsulfanyl)oxy]selanyl}propan-1-one (PIN)'
 :39473, example:39497): a group holding a parenthesis is a bracket, the group around
it a brace, never a parenthesis directly inside a parenthesis. So '(2E,4E)-hexa-2,4-dienoyl' (its
descriptor in round marks) is enclosed in square brackets, '[(2E,4E)-hexa-2,4-dienoyl]sulfanyl',
and the prefix as a whole takes braces: '3-{[(2E,4E)-hexa-2,4-dienoyl]sulfanyl}'.

``get_sulfanyl_prefix`` used to write the acyl part's marks as a literal ``f"({acyl})"``, which put
a parenthesis directly inside a parenthesis, '3-{((2E,4E)-hexa-2,4-dienoyl)sulfanyl}'. OPSIN reads
that name to the right structure, but the grammar check of the default tier
(``OpsinGrammar._check_bracket_hierarchy_strict``: directly nested same-type marks) rejects it, so
the default tier declined these rows and the best-effort tier shipped the malformed spelling.
Every name asserted here is read back by a FRESH OPSIN call (``assert_full_rt``, full InChIKey).
"""
import pytest

from tests.support.rt_assert import assert_full_rt
from tests.support.pin_tiers import tier_row

pytestmark = pytest.mark.opsin_gate  # the names below are built with the validity gate on

HEXADIENOYL_CYSTEINE = r"C/C=C/C=C/C(=O)SC[C@H](NC(=O)/C=C\C=C\C)C(=O)O"
LACTOYL_GLUTATHIONE = "C[C@@H](O)C(=O)SC[C@H](NC(=O)CC[C@H](N)C(=O)O)C(=O)NCC(=O)O"
ARYLPROPANOYL_THIOACETIC = "O=C(CCc1ccc(Cl)cc1)SCC(=O)O"


def _best_effort(smiles):
    return tier_row(smiles, "best-effort")["name"]


def _no_mark_directly_inside_the_same_mark(name):
    return not any(pair in name for pair in ("((", "[[", "{{"))


@pytest.mark.parametrize("smiles,expected", [
    # the acyl group's descriptor is in round marks: the acyl group takes '', the prefix '{ }'
    (HEXADIENOYL_CYSTEINE,
     "(2R)-2-[(2Z,4E)-hexa-2,4-dienamido]-3-{[(2E,4E)-hexa-2,4-dienoyl]sulfanyl}propanoic acid"),
    # a substituent in parentheses inside the acyl group, no descriptor
    (ARYLPROPANOYL_THIOACETIC,
     "{[3-(4-chlorophenyl)propanoyl]sulfanyl}acetic acid"),
])
def test_an_acyl_group_with_marks_of_its_own_is_enclosed_by_the_next_mark_of_the_order(
        smiles, expected):
    name = _best_effort(smiles)
    assert name == expected
    assert_full_rt(name, smiles)


def test_the_acyl_prefix_of_a_thioester_inside_a_longer_name_keeps_the_nesting_order():
    # lactoylglutathione: the S-acyl prefix sits inside the marks of the N-acyl glycine group
    name = _best_effort(LACTOYL_GLUTATHIONE)
    assert "{[(2R)-2-hydroxypropanoyl]sulfanyl}" in name, name
    assert _no_mark_directly_inside_the_same_mark(name), name
    assert_full_rt(name, LACTOYL_GLUTATHIONE)


@pytest.mark.parametrize("smiles,expected", [
    # acyl groups without marks of their own: unchanged
    ("CC(=O)SCC(=O)O", "(acetylsulfanyl)acetic acid"),
    ("O=C(SCC(=O)O)C1CCCCC1", "[(cyclohexanecarbonyl)sulfanyl]acetic acid"),
    ("[H]C(=O)C(C)C(=O)SC[C@H](NC(C)=O)C(=O)O",
     "(2R)-2-acetamido-3-[(2-methyl-3-oxopropanoyl)sulfanyl]propanoic acid"),
])
def test_an_acyl_group_without_marks_of_its_own_keeps_its_parentheses(smiles, expected):
    name = _best_effort(smiles)
    assert name == expected
    assert_full_rt(name, smiles)
