"""The Roman letters that compares are read as every other key reads them.

 (the Blue Book, ``## **** CRITERIA RELATED TO ALPHANUMERICAL ORDER OF
NAMES``): "all Roman letters are considered before any italic letters, unless the latter are
used as locants or are a part of a compound or composite locant, for example, 'N' and '4a'";
 (:3442,:3446) leaves out Greek letters and isotopic and stereochemical descriptors;
 (:3495) considers 'sec' and 'tert' after the Roman letters, their absence first
('3-(as-indacen-3-yl)-5-(s-indacen-1-yl)pyridine (PIN)',:3513). The chain check read
``str.islower`` letters instead: 'tert-butyl' gave 'tertbutyl', a compound locant '4a' gave
'a', a Greek letter 'λ'.
"""
import pytest

from orthonym import Orthonym
from orthonym.perception.chains import _cited_prefix_italics, _cited_prefix_letters

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("prefixes,letters,italics", [
    (['tert-butyl', 'chloro'], 'butylchloro', ('tert',)),
    (['sec-butyl'], 'butyl', ('sec',)),
    (['(1,2,3,4,4a,5,6,7-octahydronaphthalen-2-yl)'], 'octahydronaphthalenyl', ()),
    (['(λ5-phosphanyl)'], 'phosphanyl', ()),
    (['[(1s,4s)-4-methylcyclohexyl]'], 'methylcyclohexyl', ()),
    (['thieno[2,3-b]pyridin-2-yl'], 'thienopyridinyl', ()),
    (['s-indacen-1-yl'], 'indacenyl', ()),
    (['cis-4-methylcyclohexyl'], 'methylcyclohexyl', ()),
    # unchanged
    (['bromo', 'bromo'], 'dibromo', ()),
    (['(4-chlorophenyl)'] * 2, 'bischlorophenyl', ()),
    (['hydroxy', '(3-nitrophenyl)'], 'hydroxynitrophenyl', ()),
])
def test_the_letters_and_italic_prefixes_of_a_cited_prefix(prefixes, letters, italics):
    assert _cited_prefix_letters(prefixes) == letters
    assert _cited_prefix_italics(prefixes) == italics


def test_a_tert_butyl_prefix_does_not_move_the_letters_behind_chloromethyl():
    # by the book 'butylchloro' < 'chloromethylmethyl' (b before c); 'tertbutylchloro' put the
    # 't' of the italic prefix first
    assert (_cited_prefix_letters(['tert-butyl', 'chloro'])
            < _cited_prefix_letters(['chloromethyl', 'methyl']))


def test_roman_letters_tie_is_decided_by_the_italic_prefix_with_absence_first():
    assert _cited_prefix_letters(['butyl']) == _cited_prefix_letters(['tert-butyl'])
    assert _cited_prefix_italics(['butyl']) < _cited_prefix_italics(['tert-butyl'])
    assert _cited_prefix_italics(['sec-butyl']) < _cited_prefix_italics(['tert-butyl'])


@pytest.mark.parametrize("smiles,name", [
    ("OC(=O)C(CCl)Cc1ccccc1", "2-benzyl-3-chloropropanoic acid"),
    ("OC(=O)C(CO)Cc1ccc(Cl)cc1", "2-[(4-chlorophenyl)methyl]-3-hydroxypropanoic acid"),
])
def test_the_pin_rows_keep_their_label(smiles, name):
    row = Orthonym(style="pin").name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row


def test_the_senior_chain_is_built_and_certified():
    # Leads item N8b. The chain handed to parent selection is the one makes senior
    # (ring branches counted as prefixes), so this row is the PIN and is certified; it was the
    # name below the PIN ('3-hydroxy-2-[(3-nitrophenyl)methyl]propanenitrile') and not certified
    # while the ring atoms were left out of the prefix count. (the Blue Book,
    # '**** The preferred IUPAC name is based on the senior parent structure that has the
    # lower locant or set of locants for substituents cited as prefixes... in their order of
    # citation in the name'): the two chains hold two prefixes at {2,3}, and the order of
    # citation gives (2,3) for 'hydroxymethyl' then 'nitrophenyl' against (3,2) for 'hydroxy'
    # then '(nitrophenyl)methyl'.
    from tests.support.rt_assert import name_is_rt_exact
    smiles = "N#CC(CO)Cc1cccc([N+](=O)[O-])c1"
    pin = "2-(hydroxymethyl)-3-(3-nitrophenyl)propanenitrile"
    assert name_is_rt_exact(pin, smiles), pin
    row = Orthonym(style="pin").name_tiered(smiles)
    assert (row["name"], row["tier"]) == (pin, "pin_verified"), row
