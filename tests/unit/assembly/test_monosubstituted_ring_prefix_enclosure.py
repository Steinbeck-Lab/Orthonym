"""A compound prefix on a monosubstituted benzene or cycloalkane takes enclosing marks.

 (the Blue Book) "Parentheses are used around compound (see and complex
(see prefixes", and (:7444) nests them {}. (:27802) prints the
class on monosubstituted rings: '(cyclopentylselanyl)benzene (PIN)' (:27834),
'[(penta-1,4-dien-3-yl)sulfanyl]cyclobutane (PIN)' (:27836), '(methylsulfanyl)benzene (PIN)'
(:27848), and on a chain '[(methylsulfanyl)oxy]ethane (PIN)' (:27914).

The polysubstituted branches already enclosed by the union of needs_brackets,
is_complex_substituent and an inner enclosing mark (format_substituent_prefix); the
monosubstituted branches of the benzene and the cycloalkane prefix paths asked
is_complex_substituent alone, so a chalcogen prefix on a ring substituent
('cyclohexylsulfanylbenzene') and a prefix that carries an enclosed substituent
('(trifluoromethyl)sulfanylcyclohexane', '(methylsulfanyl)methylcyclohexane') were cited bare
with a PIN label. Both branches now call enclose_if_compound. OPSIN reads the bare and the
enclosed spellings alike, so the spelling rests on this test; every expected name is read back
here by an independent OPSIN call to the input's full InChIKey.
"""
import pytest

from tests.support.pin_tiers import assert_pin_at_both_tiers

ENCLOSED = [
    ("c1ccccc1SC1CCCCC1", "(cyclohexylsulfanyl)benzene"),
    ("c1ccccc1SC1CCCC1", "(cyclopentylsulfanyl)benzene"),
    ("FC(F)(F)SC1CCCCC1", "[(trifluoromethyl)sulfanyl]cyclohexane"),
    ("CSCC1CCCCC1", "[(methylsulfanyl)methyl]cyclohexane"),
    ("CSOC1CCCCC1", "[(methylsulfanyl)oxy]cyclohexane"),
]

# The same two branches, names that were right before and stay as they are: simple prefixes
# bare, 'tert-butyl' bare (b)), a fully enclosed or stereo-led prefix enclosed once.
UNCHANGED = [
    ("CSC1CCCCC1", "(methylsulfanyl)cyclohexane"),
    ("C=CC(C=C)SC1CCC1", "[(penta-1,4-dien-3-yl)sulfanyl]cyclobutane"),
    ("COSC1CCCCC1", "(methoxysulfanyl)cyclohexane"),
    ("C1CCCCC1CC1CCCC1", "(cyclopentylmethyl)cyclohexane"),
    ("CC(C)(C)C1CCCCC1", "tert-butylcyclohexane"),
    ("C[C@@H](Cl)CCC1CCCCC1", "[(3R)-3-chlorobutyl]cyclohexane"),
    ("CC(C)Cc1ccccc1", "(2-methylpropyl)benzene"),
    ("C[C@@H](CC)c1ccccc1", "[(2S)-butan-2-yl]benzene"),
    ("CC(C)(C)c1ccccc1", "tert-butylbenzene"),
    ("CSCOc1ccccc1", "[(methylsulfanyl)methoxy]benzene"),
    ("CS(=O)(=O)c1ccccc1", "(methanesulfonyl)benzene"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", ENCLOSED)
def test_a_compound_prefix_on_a_monosubstituted_ring_is_enclosed(smiles, name):
    assert_pin_at_both_tiers(smiles, name)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", UNCHANGED)
def test_the_other_monosubstituted_ring_names_keep_their_spelling(smiles, name):
    assert_pin_at_both_tiers(smiles, name)
