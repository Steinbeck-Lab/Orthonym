"""SL lane G, G3: the general engine's two-carbon chain parent is named on its retained word.

 "Retained names as preferred IUPAC names" (the Blue Book): "All can be functionalized,
but only acetic acid, benzoic acid, and oxamic acid can be substituted"; the row:29725 reads 'CH3-COOH
acetic acid (PIN) ethanoic acid'; (:34750-:34754) 'acetonitrile (PIN) ethanenitrile'
"with unlimited substitution"; (:32691) 'acetamide (PIN)'; (:34964)
acetaldehyde. A substituted two-carbon acid, nitrile or aldehyde has one substitutable position, so
its locants are omitted,:3031) and 'ethanoic acid' / 'ethanenitrile' are the non-preferred
spellings; the amide keeps its locants,:7304).

The best-effort tier named every such molecule on the general engine's chain parent
(``assembly/general_engine.py``, ``_assemble``) as '2-X-ethanoic acid': 8 of the 12 distinct
molecules whose best-effort name carried 'ethanoic acid' in the SF final evals (dev2000, the Blue
Book rows, the sample). The Blue Book's own spellings of two of them are reproduced byte for
byte. Every name below reads back to the input's full InChIKey (OPSIN 2.9.0). Left as they were:
'isothiocyanatomethanoic acid' and 'bis(cyanomethanoic) anhydride' (formic acid,, is no
acetic acid), the dev2000 furanylidene row (a stereo double bond ON the two-carbon chain, the
polyfunctional producer keeps its systematic spelling), and the PB-fixed '[(5Z)-6-hydroxyoct-5-
enamido]acetic acid'. Kept on purpose: a descriptor of C-2 (cited with its locant by the systematic
name) and an aldehyde with two prefixes (their enclosing follows:29852).
"""
import pytest

from tests.support.rt_assert import assert_full_rt, name_best_effort

pytestmark = pytest.mark.opsin_gate

ROWS = [
    # Blue Book rows (the Blue Book names printed for these structures)
    ("O=C(O)Cc1cc2ccccc2nc1CCCO", "[2-(3-hydroxypropyl)quinolin-3-yl]acetic acid"),
    ("COc1ccc2c(c1)C(CC(=O)O)=C(C)C2C(=O)c1ccc(Cl)cc1",
     "[1-(4-chlorobenzoyl)-5-methoxy-2-methyl-1H-inden-3-yl]acetic acid"),
    # a heteroatom or hydrazine-like substituent of the engine's own spelling
    # the hydrazinyl unit cites its locants, the Blue Book; '3-[2-(hydrazinecarbonyl)hydrazin-1-yl]
    # propanoic acid',:38712, is the book's second spelling of this group, the PIN prefix is
    # 'hydrazinecarbohydrazido'): a non-PIN name, '(aminocarbamoyl)' is the fragment namer's spelling of
    # H2N-NH-CO- (UNVERIFIED: not the book's 'hydrazinecarbonyl'); was '(hydrazinecarboxamidoamino)'
    ("NNC(=O)NNCC(=O)O", "[2-(aminocarbamoyl)hydrazin-1-yl]acetic acid"),
    ("NNC=NNCC(=O)O", "({[(aminoamino)methylidene]amino}amino)acetic acid"),
    ("COS(=O)OS(=O)OS(=O)OCC(=O)O", "{[({[(methoxysulfinyl)oxy]sulfinyl}oxy)sulfinyl]oxy}acetic acid"),
    # dev2000 / rows
    # '(diaminomethylidene)amino (preferred prefix)', (the Blue Book;
    # '4-[(diaminomethylidene)amino]butanoic acid (PIN)',:34282): the group is a compound prefix,
    # enclosed, so the enclosing marks of the name nest one level deeper
    ("CCc1c(O)cc([C@H](N=C(N)N)C(=O)NCC(=O)O)cc1O",
     "({(2S)-2-[(diaminomethylidene)amino]-2-(4-ethyl-3,5-dihydroxyphenyl)-1-oxoethyl}amino)acetic acid"),
    ("CC(C)N1C(CC2=C1C=C(C=C2)C(F)(F)F)CC(=O)O",
     "[1-(propan-2-yl)-6-(trifluoromethyl)-2,3-dihydro-1H-indol-2-yl]acetic acid"),
    # two prefixes: the first bare, the later one enclosed,:7312)
    ("OC(=O)C(Cl)(Br)c1cc2ccccc2nc1CCCO", "bromo(chloro)[2-(3-hydroxypropyl)quinolin-3-yl]acetic acid"),
    ("OC(=O)C(c1ccccn1)c1cc2ccccc2nc1CCCO", "[2-(3-hydroxypropyl)quinolin-3-yl](pyridin-2-yl)acetic acid"),
    # the nitrile, the aldehyde and the amide of the same shape
    ("N#CCc1cc2ccccc2nc1CCCO", "[2-(3-hydroxypropyl)quinolin-3-yl]acetonitrile"),
    ("O=CCc1cc2ccccc2nc1CCCO", "[2-(3-hydroxypropyl)quinolin-3-yl]acetaldehyde"),
    ("NC(=O)Cc1cc2ccccc2nc1CCCO", "2-[2-(3-hydroxypropyl)quinolin-3-yl]acetamide"),
    ("CN(C)C(=O)Cc1cc2ccccc2nc1CCCO", "2-[2-(3-hydroxypropyl)quinolin-3-yl]-N,N-dimethylacetamide"),
]

# the systematic spelling stays where the retained one is not certain
KEPT = [
    ("O=C(O)[C@H](Cl)c1cc2ccccc2nc1CCCO", "(2R)-2-chloro-2-[2-(3-hydroxypropyl)quinolin-3-yl]ethanoic acid"),
    ("O=CC(Cl)c1cc2ccccc2nc1CCCO", "2-chloro-2-[2-(3-hydroxypropyl)quinolin-3-yl]ethanal"),
]


@pytest.mark.parametrize("smiles,expected", ROWS)
def test_the_substituted_two_carbon_parent_is_named_on_its_retained_word(smiles, expected):
    res = name_best_effort(smiles)
    name = assert_full_rt(res.get("name"), smiles)
    assert name == expected, name
    assert "ethanoic" not in name and "ethanenitrile" not in name and "ethanal" not in name


@pytest.mark.parametrize("smiles,expected", KEPT)
def test_the_systematic_spelling_stays_where_the_retained_one_is_not_certain(smiles, expected):
    res = name_best_effort(smiles)
    name = assert_full_rt(res.get("name"), smiles)
    assert name == expected, name
