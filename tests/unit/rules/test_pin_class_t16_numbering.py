"""PIN class program, Task 16: lowest locants and citation order (CHAIN_NUMBERING, PREFIX_CITATION_ORDER).

 "Numbering" (the Blue Book), criterion (f) (:3301): "detachable alphabetized prefixes, all
considered together in a series of increasing numerical order". Every prefix counts, so a chain atom that
carries two prefixes gives its locant twice: '2-bromo-1-chloro-1-fluoroethane' has the locant set
{1,1,2}, the other numbering {1,2,2}. The chain numbering (``rules/locants.py:orient_chain``, criterion
(d)) and the choice among chains of one length (``perception/chains.py:find_principal_chain``, criterion
9 and the orientation cascade) used to count one locant for each chain atom.

 (:21698) and (:21791, under " CRITERIA RELATED TO NUMBER AND LOCATION OF
SUBSTITUENT GROUPS",:21596): the senior parent has the lower set of locants for the prefixes, and when
the sets tie, the lower locants in their order of citation in the name:
'1-bromo-5-(bromomethyl)-1,6-dichlorohexane (PIN) [not 1,6-dibromo-1-chloro-5-(chloromethyl)hexane; the
locant sets are the same in both names, i.e., '1,1,5,6' but in their order of appearance in the name, the
locant set '1,5,1,6' is lower than '1,6,1,5']' (:22128), '1,5-dibromo-4-(2-bromo-1-iodoethyl)-1,6-
dichlorohexane (PIN)' (:22170).

 (:25501, "If there is a choice, low locants are assigned to detachable prefixes considered
together, and, if there is a further choice, in alphanumerical order"): '(2R)-2-bromo-1-chloro-1,1-
difluoro-2-iodoethane (PIN)' (:25516).

 (:3517): "When two or more prefixes consist of identical Roman letters, priority for order of
citation is given to the group that contains the lowest locant(s) at the first point of difference":
'1-(2-methylpentan-3-yl)-1-(3-methylpentan-2-yl)cyclopentane (PIN) [not 1-(3-methylpentan-2-yl)-1-(2-
methylpentan-3-yl)cyclopentane; the locant set '2,3' is lower than '3,2']' (:3535). The alphanumerical key
of a prefix (``assembly/naming_utils.py:AlphaKey``) compared the key text, which has dropped the leading
locant, before the locants in their order of appearance.

 (c) (:3256, "principal characteristic groups and free valences (suffixes)"): the principal group
takes the lowest locant before any prefix does. On a fused polycyclic parent the numbering anchored every
suffix-type substituent, so a junior group that is cited as a prefix ('carbamoyl', 'cyano') took part
in the choice: (:32922) '3-carbamoylnaphthalene-2-carboxylic acid (PIN)' (:32944) and
'6-carbamoylnaphthalene-2-sulfonic acid (PIN)' (:32951).

Not built here (named in internal notes): the choice between two identical ring
parents by the locants of their prefixes,:6267), the amide chain pair whose sets differ in
'N',:21730) and the isotopically modified prefix of (:22212).
"""
import pytest

from tests.support.pin_tiers import assert_pin_at_both_tiers

pytestmark = pytest.mark.opsin_gate

# (SMILES, the Blue Book PIN, the rule and line)
PIN_ROWS = [
    # (f), (:25516)
    ("FC(F)(Cl)[C@H](Br)I", "(2R)-2-bromo-1-chloro-1,1-difluoro-2-iodoethane"),
    # (:22128), (:22170)
    ("ClCC(CBr)CCCC(Cl)Br", "1-bromo-5-(bromomethyl)-1,6-dichlorohexane"),
    ("ClCC(Br)C(CCC(Cl)Br)C(I)CBr", "1,5-dibromo-4-(2-bromo-1-iodoethyl)-1,6-dichlorohexane"),
    # (:3535)
    ("CCC(C)C(C)C1(C(CC)C(C)C)CCCC1", "1-(2-methylpentan-3-yl)-1-(3-methylpentan-2-yl)cyclopentane"),
    # (c) on a fused parent (:32944), (:32951)
    ("NC(=O)c1cc2ccccc2cc1C(=O)O", "3-carbamoylnaphthalene-2-carboxylic acid"),
    ("NC(=O)c1ccc2cc(S(=O)(=O)O)ccc2c1", "6-carbamoylnaphthalene-2-sulfonic acid"),
]

# The same classes, other members, each OPSIN-checked by assert_pin_at_both_tiers.
PIN_ROWS_MORE = [
    # (f): a geminal pair against a single prefix, every orientation of the chain
    ("FC(Cl)CBr", "2-bromo-1-chloro-1-fluoroethane"),
    ("FC(F)(F)C(Cl)Br", "2-bromo-2-chloro-1,1,1-trifluoroethane"),
    ("BrCC(Cl)CC(I)F", "4-bromo-3-chloro-1-fluoro-1-iodobutane"),
    ("CC(C)(C)CC(C)C", "2,2,4-trimethylpentane"),
    #: the other chain of a pair of the book, an alcohol
    ("OCC(CCBr)CCCl", "2-(2-bromoethyl)-4-chlorobutan-1-ol"),
    # (:3521), (:3533) and the book's own pair of the first example
    ("CCC(C)Cc1ccc(NCCC(C)C)cc1", "4-(2-methylbutyl)-N-(3-methylbutyl)aniline"),
    ("CCCC(C)c1ccc(C(CC)CC)cc1", "1-(pentan-2-yl)-4-(pentan-3-yl)benzene"),
    ("CCCC(C)(C)C1(C(C)(CC)CC)CCCC1", "1-(2-methylpentan-2-yl)-1-(3-methylpentan-3-yl)cyclopentane"),
    ("CCC(C)C(C)c1ccc(cc1)C(CC)C(C)C", "1-(2-methylpentan-3-yl)-4-(3-methylpentan-2-yl)benzene"),
    # (c): the cyano and the formyl group of a naphthalene are prefixes beside an acid
    ("N#Cc1cc2ccccc2cc1C(=O)O", "3-cyanonaphthalene-2-carboxylic acid"),
    ("OC(=O)c1cc2ccccc2cc1C=O", "3-formylnaphthalene-2-carboxylic acid"),
    ("NC(=O)c1ccc2cc(C(=O)O)ccc2c1", "6-carbamoylnaphthalene-2-carboxylic acid"),
    # a nitrile is senior to an aldehyde: the nitrile takes the lowest locant
    ("O=Cc1ccc2cc(C#N)ccc2c1", "6-formylnaphthalene-2-carbonitrile"),
]

# The BB (PIN) examples that already held and that the change has to keep.
CONTROL_ROWS = [
    ("OC(=O)c1cc2ccccc2cc1Cl", "3-chloronaphthalene-2-carboxylic acid"),
    ("Nc1ccc2cc(C(=O)O)ccc2c1", "6-aminonaphthalene-2-carboxylic acid"),
    ("OC(=O)CCC(C(F)C(C)Br)C(Br)C(F)C", "5-bromo-4-(2-bromo-1-fluoropropyl)-6-fluoroheptanoic acid"),
    ("OC(=O)C(C(O)CBr)C(Br)CO", "3-bromo-2-(2-bromo-1-hydroxyethyl)-4-hydroxybutanoic acid"),
    ("CC(Br)(Cl)CC", "2-bromo-2-chlorobutane"),
]


@pytest.mark.parametrize("smiles,pin", PIN_ROWS + PIN_ROWS_MORE + CONTROL_ROWS)
def test_pin_at_both_tiers(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)
