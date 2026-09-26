"""An N-substituent with locants keeps its own enclosing marks inside the '(...amino)'
unit of a secondary amide cited as 'oxo' + 'amino' (fix a performance pass, wp7).

 / (the Blue Book,:7446): '2-[di(butan-2-yl)amino]butan-2-ol
(PIN)' (:28170) encloses the located substituent inside the amino prefix. The
polyfunctional producer glued the name on bare and shipped, at pin_verified,
'3-(propan-2-ylamino)-...', '3-(2-methyldecan-2-ylamino)-...',
'2-amino-5-(2-hydroxyethylamino)-...'. Every expected name is OPSIN 2.9.0 full-InChIKey
exact (fresh java run outside the engine).
"""
import pytest

from orthonym import name_compound

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("smiles,expected", [
    ("CC(C)NC(=O)CC(=O)O", "3-oxo-3-[(propan-2-yl)amino]propanoic acid"),
    ("CCC(C)NC(=O)CC(=O)O", "3-[(butan-2-yl)amino]-3-oxopropanoic acid"),
    ("CCCCCCCCC(C)(C)NC(=O)CC(=O)O", "3-[(2-methyldecan-2-yl)amino]-3-oxopropanoic acid"),
    ("NC(CCC(=O)NCCO)C(=O)O", "2-amino-5-[(2-hydroxyethyl)amino]-5-oxopentanoic acid"),
    ("CCC(CC)(C#C)NC(=O)/C=C\\C(=O)O",
     "(2Z)-4-[(3-ethylpent-1-yn-3-yl)amino]-4-oxobut-2-enoic acid"),
    # controls: a simple N-substituent stays bare inside the parentheses; aniline stays
    ("CNC(=O)CC(=O)O", "3-(methylamino)-3-oxopropanoic acid"),
    ("c1ccccc1NC(=O)CCCC(=O)O", "5-anilino-5-oxopentanoic acid"),
])
def test_amino_prefix_inner_marks(smiles, expected):
    assert name_compound(smiles) == expected
