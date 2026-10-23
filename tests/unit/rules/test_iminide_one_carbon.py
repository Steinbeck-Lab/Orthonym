"""The anion of an imidic ester, R-C(=[N-])-X-R', is named as an iminide of the one-carbon imine.

 "Anions derived from amines and imines" (the Blue Book): the anion R2C=[N-] takes the suffix 'iminide'
('1,1-diphenylmethaniminide'). The charged route converts the 'imine' suffix of the neutral name, and the neutral of
R-C(=NH)-OR' or R-C(=NH)-SR' is the ester 'methyl benzenecarboximidothioate': no 'imine' suffix, no name. The
route now names the carbon and its two groups itself (rules.iminide_one_carbon); the name is read back to the full
InChIKey, the Blue Book giving no example of this anion it is recorded below the PIN.
"""
import pytest

from tests.support.pin_tiers import name_default
from tests.support.rt_assert import name_best_effort, name_is_rt_exact

ROWS = [
    ("C1=CC=C(C=C1)C(=[N-])SC", "(methylsulfanyl)(phenyl)methaniminide"),
    ("C1=CC=C(C=C1)C(=[N-])OC", "methoxy(phenyl)methaniminide"),
    ("C1=CC=C(C=C1)C(=[N-])SC2=C(C(=C(C(=C2F)F)F)F)F", "[(pentafluorophenyl)sulfanyl](phenyl)methaniminide"),
    ("COC(=[N-])OC", "dimethoxymethaniminide"),
    # a carbon chain on the carbon is part of the parent: the alkane of one more carbon,
    ("COC(=[N-])C", "1-methoxyethan-1-iminide"),
    ("CCSC(=[N-])CC", "1-(ethylsulfanyl)propan-1-iminide"),
    ("COC(=[N-])CCCCCCC", "1-methoxyoctan-1-iminide"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", ROWS)
def test_imidic_ester_anion_is_an_iminide(smiles, expected):
    res = name_best_effort(smiles)
    assert res["name"] == expected and res["tier"] != "pin_verified", res
    assert name_is_rt_exact(expected, smiles)
    assert name_default(smiles).get("tier") != "pin_verified"


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", ["COC(=[N-])C(C)C", "COC(=[N-])C=C", "COC(=[N-])C(F)(F)F", "COC(=[N-])CC1CC1"])
def test_a_branched_or_substituted_chain_is_never_a_group_on_methane(smiles):
    #: the chain is the parent; 'methyl' or 'isopropyl' on a one-carbon parent is not a name of this anion
    res = name_best_effort(smiles)
    assert "methaniminide" not in (res["name"] or ""), res
    assert not res["name"] or name_is_rt_exact(res["name"], smiles), res


def test_diphenylmethaniminide_keeps_its_name():
    assert name_best_effort("[N-]=C(c1ccccc1)c1ccccc1")["name"] == "1,1-diphenylmethaniminide"


def test_molybdenum_amides_with_an_imidothioate_anion():
    smiles = ("[Mo+4].CC1=CC(=CC(=C1)[N-]C(C)(C)C)C.CC1=CC(=CC(=C1)[N-]C(C)(C)C)C.CC1=CC(=CC(=C1)[N-]C(C)(C)C)C."
              "C1=CC=C(C=C1)C(=[N-])SC2=C(C(=C(C(=C2F)F)F)F)F")
    res = name_best_effort(smiles)
    assert res["name"] and name_is_rt_exact(res["name"], smiles), res
