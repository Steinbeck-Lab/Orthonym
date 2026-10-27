"""A single amine anion whose neutral parent names another nitrogen is named on the anion itself.

The charged route names the neutral molecule and converts the parent's own 'amine' suffix to 'aminide'
, the Blue Book). When the molecule has a second amine group the neutral parent can be
the one the anion is not on (the tie of two amine suffixes is decided by the substituents, not by the
charge): the 'aminide' then charges the wrong nitrogen, a different molecule. The anion is then named as the
azanide of its substituent group,:41069 'acetylazanide (PIN)'; the group cites no locant,
, which is read back to the full InChIKey like the other routes.
"""
import pytest

from tests.support.rt_assert import name_best_effort, name_is_rt_exact

ROWS = [
    ("[NH-]C(OC(N)C)c1ccccc1", "[(1-aminoethoxy)(phenyl)methyl]azanide"),
    ("CC(C)(C)C1=CC=C(C=C1)C([NH-])OC(C2CCC(CC2)C3CCC(CC3)C=C)N",
     "({amino[4-(4-ethenylcyclohexyl)cyclohexyl]methoxy}(4-tert-butylphenyl)methyl)azanide"),
]


@pytest.mark.parametrize("smiles,expected", ROWS)
def test_amine_anion_beside_another_amine_is_the_azanide(smiles, expected):
    res = name_best_effort(smiles)
    assert res["name"] == expected, res
    assert name_is_rt_exact(expected, smiles)


def test_the_aminide_stays_where_the_anion_is_the_parent_amine():
    # the parent amine is the anion's: the suffix seam's name is read back and kept
    res = name_best_effort("[NH-]C(OC(N)C1CCCCC1)c1ccccc1")
    assert res["name"] == "1-[amino(cyclohexyl)methoxy]-1-phenylmethanaminide", res


# the label: (the Blue Book) gives the PIN of an amine anion with the suffix 'aminide'; the azanide
# of its groups is not the PIN ('acetylazanide (PIN)',,:41069, is the anion of an amide)
from tests.support.pin_tiers import name_default  # noqa: E402


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", ["C[N-]CCN", "NCCC[N-]C", "[Li+].C[N-]CCN(C)C"])
def test_azanide_of_an_amine_anion_is_not_a_pin(smiles):
    assert name_default(smiles).get("tier") != "pin_verified"
    res = name_best_effort(smiles)
    assert res["name"] and "azanide" in res["name"] and res["tier"] != "pin_verified", res
    assert name_is_rt_exact(res["name"], smiles)


# a symmetric diamine anion: the charged nitrogen is equivalent to the neutral parent's own, so the composer's
# aminide on it is a name of the anion and is the PIN; the azanide of the groups must not pre-empt it
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", [
    ("C[N-]CCNC", "N-methyl-2-(methylamino)ethan-1-aminide"),
    ("CC[N-]CCNCC", "N-ethyl-2-(ethylamino)ethan-1-aminide"),
    ("C[N-]CCCNC", "N-methyl-3-(methylamino)propan-1-aminide"),
])
def test_symmetric_diamine_anion_keeps_its_aminide(smiles, expected):
    for res in (name_best_effort(smiles), name_default(smiles)):
        assert res["name"] == expected and res["tier"] == "pin_verified", res


# (b) simple groups cited twice take 'di', compound groups 'bis',:4857;
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", [
    ("NCC[N-]CCN", "bis(2-aminoethyl)azanide"),
    ("OCC[N-]CCO", "bis(2-hydroxyethyl)azanide"),
    ("Cc1ccccc1[N-]c1ccccc1C", "bis(2-methylphenyl)azanide"),
    ("C=CC[N-]CC=C", "di(prop-2-en-1-yl)azanide"),
])
def test_an_arm_cited_twice_takes_its_multiplier(smiles, expected):
    res = name_best_effort(smiles)
    assert res["name"] == expected, res
    assert name_is_rt_exact(expected, smiles)
    #... and it is the azanide of an amine, not the PIN: the suffix 'aminide' gives the PIN of an amine anion
    # "Anions derived from amines and imines", the Blue Book)
    assert res["tier"] != "pin_verified", res
    assert name_default(smiles).get("tier") != "pin_verified"


# the aminide is built where the parent is a ring of one kind of position (no locant for the suffix,
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", [
    ("c1ccccc1[N-]c1ccccc1", "N-phenylbenzenaminide"),
    ("[Li+].c1ccccc1[N-]c1ccccc1", "lithium N-phenylbenzenaminide"),
    ("C[N-]c1ccccc1", "N-methylbenzenaminide"),
    ("C1CCCCC1[N-]C", "N-methylcyclohexanaminide"),
])
def test_secondary_amine_anion_on_a_ring_is_the_aminide(smiles, expected):
    for res in (name_best_effort(smiles), name_default(smiles)):
        assert res["name"] == expected and res["tier"] == "pin_verified", res
    assert name_is_rt_exact(expected, smiles)


# (the Blue Book): 'acetylazanide (PIN)', the anion of an amide: its azanide stays the PIN.
# (:7272): on a mononuclear parent "the first cited substituent never has enclosing marks unless it includes
# a locant", the later ones are enclosed: 'acetyl(methyl)chloranium (PIN)' (:41488), so 'acetyl(methyl)azanide' and not
# '(acetyl)(methyl)azanide' (the value of this row before the fix round of lane ION; it fails on that code)
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", [
    ("[Na+].CC(=O)[N-]C(C)=O", "sodium diacetylazanide"),
    ("CC(=O)[N-]C", "acetyl(methyl)azanide"),
])
def test_the_azanide_of_an_amide_is_still_a_pin(smiles, expected):
    for res in (name_best_effort(smiles), name_default(smiles)):
        assert res["name"] == expected and res["tier"] == "pin_verified", res
