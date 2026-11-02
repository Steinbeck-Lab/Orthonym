"""A substituted metallocene drawn as a metal(2+) and two cyclopentadienide anions is named with its ocenyl prefix
through its cyclopentyl drawing (``assembly.ocene_drawing``); OPSIN reads 'ferrocen-1-yl...' to the same drawing.
 (alphanumerical order) guards the rewrite; encloses the prefix, which has a locant."""
import pytest

from orthonym.assembly.ocene_drawing import rewrite_surrogate_prefix
from tests.support.rt_assert import assert_full_rt, name_best_effort

pytestmark = pytest.mark.unit


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", [
    ("[CH-]1C=CC=C1.C1=C[C-](C=C1)OC(=O)C2=CC=NC=C2.[Fe+2]", "ferrocen-1-yl pyridine-4-carboxylate"),
    ("CC1=CC(=CS1)[C-]2C=CC=C2.[CH-]1C=CC=C1.[Fe+2]", "4-(ferrocen-1-yl)-2-methylthiophene"),
    ("COC1=CC=C(C=C1)C(=O)C#CC2=CC=C[CH-]2.[CH-]1C=CC=C1.[Fe+2]",
     "3-(ferrocen-1-yl)-1-(4-methoxyphenyl)prop-2-yn-1-one"),
    ("C1CCN(CC1)C[C-]2C=CC=C2.[CH-]1C=CC=C1.[Fe+2]", "1-[(ferrocen-1-yl)methyl]piperidine"),
    ("C1CN(CCN1CC2=C[CH-]C=C2)C3=NC=CC=N3.[CH-]1C=CC=C1.[Fe+2]",
     "2-{4-[(ferrocen-1-yl)methyl]piperazin-1-yl}pyrimidine"),
])
def test_ocenyl_names(smiles, name):
    row = name_best_effort(smiles)
    assert (row["name"], row["tier"]) == (name, "systematic_verified"), row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("name,expected", [
    ("4-cyclopentyl-2-methylthiophene", "4-(ferrocen-1-yl)-2-methylthiophene"),
    ("cyclopentyl pyridine-4-carboxylate", "ferrocen-1-yl pyridine-4-carboxylate"),
    # a word that sorts between 'cyclopentyl' and 'ferrocenyl' (a compound prefix with its multiplier): declined
    ("2-cyclopentyl-3-(2,6-dimethylanilino)pyridine", None),
    # inside a compound prefix: the marks around it step up, the Blue Book)
    ("(cyclopentylmethyl)benzene", "[(ferrocen-1-yl)methyl]benzene"),
    ("2-[4-(cyclopentylmethyl)piperazin-1-yl]pyrimidine", "2-{4-[(ferrocen-1-yl)methyl]piperazin-1-yl}pyrimidine"),
    ("1-bromo-4-(cyclopentylsulfanyl)benzene", "1-bromo-4-[(ferrocen-1-yl)sulfanyl]benzene"),
    ("1-cyclopentyloxybenzene", None),                   # an unenclosed compound prefix
    ("cyclopentylidenemethanol", None),                  # another free valence
    ("{[(cyclopentylmethyl)amino]methyl}benzene", None), # the marks would need a level above { }
])
def test_the_rewrite_guards(name, expected):
    assert rewrite_surrogate_prefix(name, "ferrocen") == expected
