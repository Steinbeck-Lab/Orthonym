""" NONALPHANUMERICAL ORDER: where the letters of two prefixes are identical, a letter is preferred to a mark.

 (the Blue Book): "there are the occasional names in which the alphanumerical character numberings are
identical. In such cases, other characters, such as enclosing marks or punctuation marks, must be compared with
letters and/or numbers, or with each other to select a preferred IUPAC name. Letters or numbers are always
preferred to any other character. When necessary the order of priority for other characters is as follows, listed
in decreasing priority order: braces > square brackets > parentheses (round brackets) > periods > commas >
semi-colons > colons > hyphens." Its example (:3545): "3-[amino(methyl)silyl]-3-[(aminomethyl)silyl]cyclopentan-1-ol
(PIN) {not 3-[(aminomethyl)silyl]-3-[amino(methyl)silyl]cyclopentan-1-ol; alphanumerical characters are identical;
at the fourth character of the name, the letter 'a' is preferred to an open parenthesis}".

The last tier of the prefix keys (``AlphaKey``, ``prefix_citation_sort_key``) compared the names as strings, so
'(' (code 40) stood before every letter and the book's "not" order was the engine's order. A name of the lambda
convention ('λ5-phosphanyl') stood before its plain twin only through that same accident (the noise-stripped
'(phosphanyl)' against 'phosphanyl'), which would reverse. The book states that order in one example,
 (h) (:3320): "1-(λ5-phosphanyl)-3-phosphanylpropan-2-ol (PIN) (λ5-phosphanyl is cited before phosphanyl and
is given the lower locant)", so the key has a tier of its own for it, before.
"""
import pytest

from orthonym.assembly.naming_utils import (
    _lambda_key,
    _own_locants,
    _nonalphanumerical_key,
    alpha_sort_key,
    prefix_citation_sort_key,
)
from tests.support.pin_tiers import assert_pin_at_both_tiers, name_breadth
from tests.support.rt_assert import name_is_rt_exact

pytestmark = pytest.mark.unit

A, B = "amino(methyl)silyl", "(aminomethyl)silyl"


@pytest.mark.parametrize("names", [[A, B], [B, A]])
def test_a_letter_is_preferred_to_an_open_parenthesis(names):
    """:3545 -- whichever way the two arrive, 'amino(methyl)silyl' is cited before '(aminomethyl)silyl'."""
    assert sorted(names, key=alpha_sort_key) == [A, B]
    assert sorted(names, key=prefix_citation_sort_key) == [A, B]


def test_the_marks_are_in_the_priority_order_of_the_book():
    """:3537 -- braces > square brackets > parentheses > periods > commas > semi-colons > colons > hyphens."""
    marks = ["{", "[", "(", ".", ",", ";", ":", "-"]
    texts = [f"a{m}b" for m in marks]
    assert sorted(reversed(texts), key=_nonalphanumerical_key) == texts
    # and a letter or a number before every one of them
    assert sorted([f"a{m}b" for m in marks] + ["aab", "a1b"], key=_nonalphanumerical_key)[:2] == ["a1b", "aab"]


def test_the_alphanumerical_tiers_still_come_first():
    """ decides only what leaves tied: 'ethyl' stays before 'methyl' whatever their marks."""
    assert sorted(["(methylsilyl)", "ethylsilyl"], key=alpha_sort_key) == ["ethylsilyl", "(methylsilyl)"]
    assert sorted(["propyl", "methyl", "ethyl"], key=alpha_sort_key) == ["ethyl", "methyl", "propyl"]


@pytest.mark.parametrize("plain,lam", [("phosphanyl", "λ5-phosphanyl"), ("phosphanyl", "(λ5-phosphanyl)"),
                                       ("phosphanyl", "[λ5-phosphanyl]")])
@pytest.mark.parametrize("flip", [False, True])
def test_a_nonstandard_bonding_number_is_cited_first(plain, lam, flip):
    """:3334 -- 'λ5-phosphanyl is cited before phosphanyl', with the marks the group is cited in or without."""
    names = [lam, plain] if not flip else [plain, lam]
    assert sorted(names, key=alpha_sort_key) == [lam, plain]
    assert sorted(names, key=prefix_citation_sort_key) == [lam, plain]


def test_the_lambda_tier_follows_the_alphanumerical_ones():
    """The tier is after the letters: 'λ5-phosphanylmethyl' still stands after 'phosphanyl' (P-14.5.2, :3477 'phosphanyl' is
    an initial segment of 'phosphanylmethyl'), as in '3-(λ5-phosphanyl)-2-(phosphanylmethyl)propanoic acid (PIN)' (:22182)."""
    assert sorted(["phosphanylmethyl", "λ5-phosphanyl"], key=alpha_sort_key) == ["λ5-phosphanyl", "phosphanylmethyl"]
    assert sorted(["λ5-phosphanylmethyl", "phosphanyl"], key=alpha_sort_key) == ["phosphanyl", "λ5-phosphanylmethyl"]
    # between two nonstandard bonding numbers the higher one first, as (h) numbers them
    assert _lambda_key("λ6-sulfanyl") < _lambda_key("λ4-sulfanyl") < _lambda_key("sulfanyl")


# (the Blue Book): "When two or more prefixes consist of identical Roman letters, priority for order of
# citation is given to the group that contains the lowest locant(s) at the first point of difference"; the
# preamble (:3442): "When all the Roman letters are identical, the set of locants for all initial locants for primary
# substituents... are compared. Absence of locants is most preferred", Greek letters left out (:3446). The lambda
# convention and break only a tie that remains AFTER the locants of the prefixes.
@pytest.mark.parametrize("names,expected", [
    # '1' < '2': the lambda tier may not put '2-(λ5-phosphanyl)ethyl' first (the key text drops a leading locant)
    (["2-(λ5-phosphanyl)ethyl", "1-phosphanylethyl"], ["1-phosphanylethyl", "2-(λ5-phosphanyl)ethyl"]),
    (["[2-(λ5-phosphanyl)ethyl]", "(1-phosphanylethyl)"], ["(1-phosphanylethyl)", "[2-(λ5-phosphanyl)ethyl]"]),
    (["(1-phosphanylethyl)", "[2-(λ5-phosphanyl)ethyl]"], ["(1-phosphanylethyl)", "[2-(λ5-phosphanyl)ethyl]"]),
    (["[1-(λ5-phosphanyl)ethyl]", "(2-phosphanylethyl)"], ["[1-(λ5-phosphanyl)ethyl]", "(2-phosphanylethyl)"]),
    # a stereo prefix in a square bracket is ordered by its locants, not by the type of its bracket
    (["[(1R)-1-chloroethyl]", "(2-chloroethyl)"], ["[(1R)-1-chloroethyl]", "(2-chloroethyl)"]),
    (["(2-chloroethyl)", "[(1R)-1-chloroethyl]"], ["[(1R)-1-chloroethyl]", "(2-chloroethyl)"]),
    (["[(2S)-3-methylbutan-2-yl]", "(2-methylbutan-2-yl)"], ["(2-methylbutan-2-yl)", "[(2S)-3-methylbutan-2-yl]"]),
    # absence of locants first; the book's own pairs (:3523,:3531)
    (["2-methylbutyl", "methylbutyl"], ["methylbutyl", "2-methylbutyl"]),
    (["3-methylbutyl", "2-methylbutyl"], ["2-methylbutyl", "3-methylbutyl"]),
    (["pentan-3-yl", "pentan-2-yl"], ["pentan-2-yl", "pentan-3-yl"]),
    # preamble (:3442): "Absence of locants is most preferred, followed by italic Roman letter locants, Greek
    # letter locants..., and arabic numerals in order from lowest to highest"
    (["2,2-dimethyl", "N1-methyl"], ["N1-methyl", "2,2-dimethyl"]),
    (["N1-methyl", "2,2-dimethyl"], ["N1-methyl", "2,2-dimethyl"]),
])
def test_the_own_locants_come_before_the_lambda_and_the_marks(names, expected):
    assert sorted(names, key=alpha_sort_key) == expected
    assert sorted(names, key=prefix_citation_sort_key) == expected


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,pin,why", [
    # verbatim: (h),:3334
    ("OC(C[PH4])CP", "1-(λ5-phosphanyl)-3-phosphanylpropan-2-ol", ":3334 verbatim (PIN)"),
    # then (g) (:3307, "lowest locants for the substituent cited first as a prefix"): the group cited first,
    # 'chloro(methyl)silyl', takes the lower locant -- an application of the rules of:3537 and:3545 to two other molecules
    ("OC(=O)c1cc([SiH](Cl)C)cc([SiH2]CCl)c1", "3-[chloro(methyl)silyl]-5-[(chloromethyl)silyl]benzoic acid",
     "derived: P-14.6 + P-14.4 (g)"),
    ("OC(=O)c1cc([SiH2]CCl)cc([SiH](Cl)C)c1", "3-[chloro(methyl)silyl]-5-[(chloromethyl)silyl]benzoic acid",
     "derived: P-14.6 + P-14.4 (g), the other drawing"),
    ("OCC([SiH](Cl)C)[SiH2]CCl", "2-[chloro(methyl)silyl]-2-[(chloromethyl)silyl]ethan-1-ol",
     "derived: P-14.6, equal locants"),
])
def test_the_pin_follows_the_book_order(smiles, pin, why):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,pin,why", [
    # the locants of the two prefixes decide, the lambda convention does not preamble,
    ("OC(=O)c1cc(CC[PH4])cc(C(C)P)c1", "3-(1-phosphanylethyl)-5-[2-(λ5-phosphanyl)ethyl]benzoic acid",
     "'1' < '2', the lambda group second"),
    ("OC(=O)c1cc(C(C)[PH4])cc(CCP)c1", "3-[1-(λ5-phosphanyl)ethyl]-5-(2-phosphanylethyl)benzoic acid",
     "'1' < '2', the lambda group first"),
    ("c1cc(CC[PH4])ccc1NC(C)P", "N-(1-phosphanylethyl)-4-[2-(λ5-phosphanyl)ethyl]aniline", "'1' < '2' on an aniline"),
    # a stereo prefix in a bracket against a prefix in parentheses
    ("ClCCc1ccc(N[C@@H](C)Cl)cc1", "N-[(1R)-1-chloroethyl]-4-(2-chloroethyl)aniline", "'1' < '2', the stereo prefix first"),
    ("C[C@@H](C(C)C)c1ccc(NC(C)(C)CC)cc1", "N-(2-methylbutan-2-yl)-4-[(2S)-3-methylbutan-2-yl]aniline",
     "(2,2) < (2,3,2), the stereo prefix second"),
    ("CC(C)(CC)c1ccc(N[C@@H](C)C(C)C)cc1", "4-(2-methylbutan-2-yl)-N-[(2S)-3-methylbutan-2-yl]aniline",
     "(2,2) < (2,3,2), the stereo prefix second, the other drawing"),
    # verbatim:,:3523 and:3531 (PIN)
    ("CCC(C)Cc1ccc(NCCC(C)C)cc1", "4-(2-methylbutyl)-N-(3-methylbutyl)aniline", ":3523 verbatim (PIN)"),
    ("CCCC(C)c1ccc(C(CC)CC)cc1", "1-(pentan-2-yl)-4-(pentan-3-yl)benzene", ":3531 verbatim (PIN)"),
])
def test_the_pin_follows_the_locants_of_the_prefixes(smiles, pin, why):
    assert_pin_at_both_tiers(smiles, pin)


# (the Blue Book): "In these subsections the principles of alphanumerical order do not include Greek letters
# (except in conjunctive names) or isotopic or stereochemical descriptors". The '2' of '(2R)' and the locant and mass
# number of an isotopic descriptor are no locants of the prefix at the tier.
@pytest.mark.parametrize("described,plain", [
    ("(2R)-2,3-dihydroxypropyl", "2,3-dihydroxypropyl"),
    ("[(2S)-3-methylbutan-2-yl]", "(3-methylbutan-2-yl)"),
    ("rel-(1R,2S)-2-methylcyclohexyl", "2-methylcyclohexyl"),
    ("[1-13C]ethyl", "ethyl"),
    ("(1-2H1)ethyl", "ethyl"),
    ("[2-hydroxy-3-(2H3)methoxypropyl]", "(2-hydroxy-3-methoxypropyl)"),
])
def test_a_descriptor_carries_no_own_locant(described, plain):
    assert _own_locants(described) == _own_locants(plain)
    assert prefix_citation_sort_key(described)[1] == prefix_citation_sort_key(plain)[1]


def test_a_stereodescriptor_does_not_order_two_prefixes_of_equal_locants():
    """The keys tie through the locant tier; '(2R)-2,3-...' no longer reads (2,2,3) against (2,3)."""
    assert alpha_sort_key("(2R)-2,3-dihydroxypropyl") == alpha_sort_key("2,3-dihydroxypropyl")


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", ["O=P([O-])(OCC(O)CO)OC[C@H](O)CO"])
def test_the_ester_name_is_not_ordered_by_the_stereodescriptor(smiles):
    """ (:3446): the descriptor '(2R)' does not put '(2R)-2,3-dihydroxypropyl' first."""
    assert_pin_at_both_tiers(smiles, "2,3-dihydroxypropyl (2R)-2,3-dihydroxypropyl phosphate")


@pytest.mark.opsin_gate
def test_the_book_example_is_spelled_in_the_book_order():
    """:3545 verbatim: the molecule of the example, named at the best-effort tier, is the book's string, never the
    'not' order (the default tier declines this molecule, so the string carries no PIN label there)."""
    smiles = "C[SiH](N)C1([SiH2]CN)CCC(O)C1"
    expected = "3-[amino(methyl)silyl]-3-[(aminomethyl)silyl]cyclopentan-1-ol"
    assert name_breadth(smiles).get("name") == expected
    assert name_is_rt_exact(expected, smiles)


# (the Blue Book): "The name of a prefix for a substituent is considered to begin with the first letter of its
# complete name"; (:3448) lets only the multiplier of identical SEPARATE prefixes drop out. The name of a
# catenated hydride group keeps its multiplier ('3-disilanyl-2-silylhexasilane',:21681: 'disilanyl' at 'd'): the book
# multiplies such a group as 'bis(phosphanyl)',:7188), so 'di' + 'phosphanyl' is the chain, not a multiplier.
@pytest.mark.parametrize("name,key", [
    ("diphosphanyl", "diphosphanyl"), ("disilanyl", "disilanyl"), ("trisulfanyl", "trisulfanyl"),
    ("disulfanyl", "disulfanyl"), ("tetraphosphanyl", "tetraphosphanyl"), ("diazanyl", "diazanyl"),
    # the multiplier of identical separate prefixes still drops out
    ("dimethyl", "methyl"), ("diphenyl", "phenyl"), ("trichloro", "chloro"), ("bis(phosphanyl)", "phosphanyl"),
    ("tris(sulfanyl)", "sulfanyl"), ("bis(2-chloroethyl)", "chloroethyl"),
])
def test_a_multiplier_that_is_part_of_the_name_of_a_hydride_group_is_alphabetized(name, key):
    assert alpha_sort_key(name) == key


def test_the_catenated_hydride_group_stands_at_its_multiplier():
    assert sorted(["phosphanyl", "diphosphanyl"], key=alpha_sort_key) == ["diphosphanyl", "phosphanyl"]
    assert sorted(["sulfanyl", "trisulfanyl", "methylsulfanyl"], key=alpha_sort_key) == [
        "methylsulfanyl", "sulfanyl", "trisulfanyl"]
    assert sorted(["trisulfanyl", "sulfanyl"], key=prefix_citation_sort_key) == ["sulfanyl", "trisulfanyl"]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,pin", [
    ("OC(=O)c1cc(SS)cc(SC)c1", "3-disulfanyl-5-(methylsulfanyl)benzoic acid"),
    ("OC(=O)c1cc(SSS)cc(SC)c1", "3-(methylsulfanyl)-5-trisulfanylbenzoic acid"),
    ("OC(=O)c1cc(SS)cc(S)c1", "3-disulfanyl-5-sulfanylbenzoic acid"),
])
def test_the_pin_alphabetizes_a_catenated_hydride_group_at_its_multiplier(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", [
    ("c1cc(PP)cc([PH4])c1", "1-diphosphanyl-3-(λ5-phosphanyl)benzene"),
    ("OC(=O)c1cc(PP)cc([PH4])c1", "3-diphosphanyl-5-(λ5-phosphanyl)benzene-1-carboxylic acid"),
])
def test_the_best_effort_name_cites_diphosphanyl_first(smiles, name):
    """'diphosphanyl' at 'd' before 'phosphanyl' (here the λ5 group, whose letters are those of 'phosphanyl')."""
    got = name_breadth(smiles).get("name")
    assert got == name, got
    assert name_is_rt_exact(name, smiles)
