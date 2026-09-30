"""Breadth job 3, class M34 (small parents and spellings).

1. More than twenty identical simple prefixes: the private multiplier tables of the
   substituent namer stopped at 'hexa' and fell back to the digits ('21fluoro'); the
   term is composed by (the Blue Book-2813), 21 'henicosa' (:2820).
2. A ring that is itself a substituent cites -CHO / -COOH as prefixes, not by their
   ring-suffix words ('3-carbaldehydephenyl'): 'formyl',:30404; 'HCO-
   formyl (preferred prefix)':30444).
3. -CO-NR2 terminating a chain under a senior acid is 'oxo' + the amino side:
    (:30392) "The combination of the prefixes 'anilino' and 'oxo' is used for
   describing -CO-NH-C6H5 at the end of an acyclic chain", 'anilino(oxo)acetic acid
   (PIN)' (:30398). The N,N-disubstituted amide was cited twice ('(dimethylamino)
   (dimethylcarbamoyl)acetic acid').
4. -N(OH)2 is the prefix 'dihydroxyamino' (the amino group substituted by two hydroxy;
   'hydroxyamino (preselected prefix)':38416); it was dropped.

Every name is read back by a FRESH OPSIN call that does not go through the engine
(tests.support.rt_assert._independent_parse) and compared by full InChIKey.
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from tests.support.rt_assert import _independent_parse

pytestmark = [pytest.mark.opsin_gate]


def _key(smiles):
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return Chem.MolToInchiKey(mol) if mol is not None else ""


def _assert_named(smiles, expected, tiers=("pin_verified",)):
    row = Orthonym().name_tiered(smiles)
    assert row["name"] == expected, row
    assert row["tier"] in tiers, row
    assert _key(_independent_parse(expected)) == _key(smiles)


def test_henicosa_multiplier_in_an_ester_alkyl():
    # the milestone1500 M34 row
    _assert_named(
        "C=CC(=O)OCC(C(C(C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)"
        "(F)F)(F)F",
        "2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,11-henicosafluoroundecyl "
        "prop-2-enoate")


@pytest.mark.parametrize("smiles,expected", [
    # the a dev split M34 row
    ("N[C@@H](Cc1cccc(C=O)c1)C(=O)O", "(2S)-2-amino-3-(3-formylphenyl)propanoic acid"),
    ("OC(=O)CCc1ccc(C=O)cc1", "3-(4-formylphenyl)propanoic acid"),
    ("OC(=O)CCc1ccncc1C=O", "3-(3-formylpyridin-4-yl)propanoic acid"),
    ("OC(=O)CCc1ccc(C#N)cc1", "3-(4-cyanophenyl)propanoic acid"),       # unchanged
])
def test_ring_substituent_decoration_is_a_prefix(smiles, expected):
    _assert_named(smiles, expected)


@pytest.mark.parametrize("smiles,expected", [
    # the milestone1500 M34 row
    ("CN(C)C(=O)C(=O)O", "(dimethylamino)(oxo)acetic acid"),
    ("CCN(CC)C(=O)C(=O)O", "(diethylamino)(oxo)acetic acid"),
    ("CN(C)C(=O)CC(=O)O", "3-(dimethylamino)-3-oxopropanoic acid"),
    ("CCN(C)C(=O)CCC(=O)O", "4-[ethyl(methyl)amino]-4-oxobutanoic acid"),
    ("O=C(O)CCC(=O)N1CCCC1", "4-oxo-4-(pyrrolidin-1-yl)butanoic acid"),
    # the secondary sibling is unchanged
    ("CNC(=O)C(=O)O", "(methylamino)(oxo)acetic acid"),
    ("c1ccccc1NC(=O)C(=O)O", "anilino(oxo)acetic acid"),                # BB:30398
])
def test_chain_end_amide_is_oxo_plus_amino(smiles, expected):
    _assert_named(smiles, expected)


@pytest.mark.parametrize("smiles,expected", [
    ("CCCC(N(O)O)C(=O)O", "2-(dihydroxyamino)pentanoic acid"),
    ("OC(=O)CCN(O)O", "3-(dihydroxyamino)propanoic acid"),
    # the milestone1500 M34 row
    ("CSCCCCCCC[C@@H](C(=O)[O-])N(O)O",
     "(2S)-2-(dihydroxyamino)-9-(methylsulfanyl)nonanoate"),
    ("CCCC(NO)C(=O)O", "2-(hydroxyamino)pentanoic acid"),                # unchanged
])
def test_dihydroxyamino_prefix(smiles, expected):
    _assert_named(smiles, expected)
