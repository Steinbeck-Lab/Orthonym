"""Leads program L3, item N8f (label backstop): the principal chain of a substituent prefix.

The names of acyclic substituent prefixes depended on the order of the atoms of the input and
shipped ``pin_verified`` for chains the Blue Book criteria do not choose. 'THE PRINCIPAL
SUBSTITUENT CHAIN' (the Blue Book) applies its criteria "successively in the order given":
(b) the longest chain; (d) the greater number of multiple bonds regardless of type, then of double
bonds,:22682); (i) the lowest locants for multiple bonds, then for double bonds
,:22726); (k) the greatest number of substituents,:22740); (l) the lowest
locants for substituents,:22768; '4-hydroxy-3-(2-hydroxyethyl)pentan-2-yl (preferred
prefix)',:22780). For a parent chain the same criteria stand in (:21033),
(:21066), (:21366), (:21604) and (:21698).

``principal_chain_check`` read only the saturated alkyl branch and the counts,.
It now reads one alternative chain through an unbranched carbon-chain prefix as long as the tail it
replaces and compares the two by (d), (i), (k), (l). Every name below, the wrong and the preferred
one, is read back by a fresh OPSIN call to the same molecule: the round trip proves the molecule,
never the choice of chain. The producer-side tests are in ``test_leads_l3_n8f_chain_selector.py``.
"""
import pytest
from rdkit import Chem

from orthonym.validation.pin_spelling import check_pin_spelling
from tests.support.rt_assert import name_is_rt_exact


def _failures(smiles, name):
    return check_pin_spelling(Chem.MolFromSmiles(smiles), name, strict=True)


def _rules(smiles, name):
    return [f.rule for f in _failures(smiles, name)]


#: (structure, a name whose chain the criteria do not choose, the rule it breaks, the PIN)
WRONG_CHAIN = [
    # (l) lowest locants for substituents: {3,5} against {3,4}; the book's own pair,:22780
    ("C[CH]C(CCO)C(C)O", "5-hydroxy-3-(1-hydroxyethyl)pentan-2-yl", "P-45.2.2",
     "4-hydroxy-3-(2-hydroxyethyl)pentan-2-yl"),
    ("OC(=O)c1ccc(cc1)C(C)C(CCO)C(C)O", "4-[5-hydroxy-3-(1-hydroxyethyl)pentan-2-yl]benzoic acid",
     "P-45.2.2", "4-[4-hydroxy-3-(2-hydroxyethyl)pentan-2-yl]benzoic acid"),
    # (d) the greater number of double bonds, as many multiple bonds in each chain
    ("C#CCC([CH]C)CC=C", "3-(prop-2-en-1-yl)hex-5-yn-2-yl", "P-46.1.4",
     "3-(prop-2-yn-1-yl)hex-5-en-2-yl"),
    ("OC(=O)c1ccc(cc1)C(C)C(CC#C)CC=C", "4-[3-(prop-2-en-1-yl)hex-5-yn-2-yl]benzoic acid", "P-46.1.4",
     "4-[3-(prop-2-yn-1-yl)hex-5-en-2-yl]benzoic acid"),
    # (d) in a parent chain: 'the greater number of double bonds'
    ("C#CCC(CCO)CC=C", "3-(prop-2-en-1-yl)hex-5-yn-1-ol", "P-44.4.1.2",
     "3-(prop-2-yn-1-yl)hex-5-en-1-ol"),
    # (k) the maximum number of substituents::22740, the book's pair at:22746
    ("C[CH]CCC(CC(C)Cl)C(Cl)C(C)Cl", "7-chloro-5-(1,2-dichloropropyl)octan-2-yl", "P-45.2.1",
     "6,7-dichloro-5-(2-chloropropyl)octan-2-yl"),
    # (i) the lowest locants for the multiple bonds as a set: the free valence ties at 4
    ("C#CC[CH]CC=C", "hept-6-en-1-yn-4-yl", "P-14.4", "hept-1-en-6-yn-4-yl"),
]


@pytest.mark.parametrize("smiles, wrong, rule, pin", WRONG_CHAIN)
def test_a_chain_the_criteria_do_not_choose_fails(smiles, wrong, rule, pin):
    # both names are the molecule (the round trip proves that and nothing about the choice)
    assert name_is_rt_exact(wrong, smiles) and name_is_rt_exact(pin, smiles)
    assert rule in _rules(smiles, wrong), _failures(smiles, wrong)
    assert _rules(smiles, pin) == [], _failures(smiles, pin)


#: names that must pass: ties (the two chains are the same name), the book's PINs, chains in which
#: the criteria choose the chain the name has (each read back by OPSIN to the structure)
PASSING = [
    ("C[CH]C(CCO)C(C)O", "4-hydroxy-3-(2-hydroxyethyl)pentan-2-yl"),                    #:22780
    ("CC(O)C(C)CC(C)O", "3-methylhexane-2,5-diol"),
    ("CCC(CC)C(C)C", "3-ethyl-2-methylpentane"),
    ("CCCC(CC)C(C)C", "3-ethyl-2-methylhexane"),                                        #:21649
    ("C#CC[CH]CC=C", "hept-1-en-6-yn-4-yl"),                                            #:17256
    ("C=CC[CH]CC=C", "hepta-1,6-dien-4-yl"),
    ("C#CCC(CC=C)CCO", "3-(prop-2-yn-1-yl)hex-5-en-1-ol"),
    ("CC(O)C(C)CCC", "3-methylhexan-2-ol"),
    ("CC(C)CC(CC)C(C)C", "3-ethyl-2,5-dimethylhexane"),
    ("C[CH]CCC(CC(C)Cl)C(Cl)C(C)Cl", "6,7-dichloro-5-(2-chloropropyl)octan-2-yl"),      #:22746
    ("OCC(CO)(C)CC", "2-ethyl-2-methylpropane-1,3-diol"),
    ("OCCC(CCO)CC", "3-ethylpentane-1,5-diol"),
]


@pytest.mark.parametrize("smiles, name", PASSING)
def test_names_with_the_chosen_chain_pass(smiles, name):
    assert name_is_rt_exact(name, smiles), name
    assert _rules(smiles, name) == [], _failures(smiles, name)


@pytest.mark.parametrize("smiles, name", [
    ("CCC(C)CC", "3-methylpentane"),
    ("CC(C)CC", "2-methylbutane"),
    ("CCC(CC)CC", "3-ethylpentane"),
])
def test_a_tie_between_two_chains_of_one_name_passes(smiles, name):
    # the alternative chain through the prefix is the same chain: nothing to choose
    assert name_is_rt_exact(name, smiles), name
    assert _rules(smiles, name) == []


def test_a_branch_the_reader_does_not_hold_with_certainty_is_not_read():
    # a ring is no chain: no alternative chain through 'cyclohexylmethyl'
    smiles, name = "CCCCC(CC)CC1CCCCC1", "3-(cyclohexylmethyl)heptane"
    assert name_is_rt_exact(name, smiles), name
    assert _rules(smiles, name) == []
