""": the second and further prefixes of a mononuclear group are enclosed.

 (the Blue Book, section " Parentheses"), (:7272): "For
mononuclear parent hydrides with two or more substituents the first cited substituent never has
enclosing marks unless it includes a locant. The second and further substituents are each enclosed
with parentheses even for simple substituents." (:7304) extends it to a parent with one
substitutable position: 'cyclopropyl(phenyl)methanol (PIN)' (:7302), 'bromo(nitro)(phenyl)acetic acid
(PIN)' (:7312). A first prefix that carries a locant or marks of its own is cited as it is and the
later ones are enclosed all the same: '(cyclopenta-1,4-dien-1-yl)(hydroxy)methyl' in
'3-[(S)-(cyclopenta-1,4-dien-1-yl)(hydroxy)methyl]cyclopenta-2,4-dien-1-ide (PIN)' (:44964),
'(Z)-(4-chlorophenyl)(phenyl)methanone oxime' (:47666).

 (:7446) gives the marks their order '{[({})]}'; the parentheses of a stereodescriptor
count,:7488), so a ring prefix with its own descriptor is '[(1S)-cyclohex-2-en-1-yl]'
and the group that holds it is cited in braces.

Every name asserted here is read back by a FRESH OPSIN call (``assert_full_rt``, full InChIKey).
"""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from tests.support.rt_assert import assert_full_rt

pytestmark = pytest.mark.opsin_gate  # the names below are built with the validity gate on

# the prefix of the parent 5-methyl-2-oxopyrrolidine (an oxo-substituted ring parent, so the
# one-carbon group is a prefix of it)
PYRROLIDINONE = "C1(C)CCC(=O)N1"


def _best_effort(smiles):
    return Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(smiles)["name"]


@pytest.mark.parametrize("smiles,expected", [
    # the first prefix has a locant: the second is enclosed (:44964)
    ("OC(C1C=CCCC1)" + PYRROLIDINONE,
     "5-[(cyclohex-2-en-1-yl)(hydroxy)methyl]-5-methyl-2-oxopyrrolidine"),
    ("OC(c1ccc(Cl)cc1)" + PYRROLIDINONE,
     "5-[(4-chlorophenyl)(hydroxy)methyl]-5-methyl-2-oxopyrrolidine"),
    # a nitrile is senior to the lactam, a pseudoketone (Table 4.1 classes 14 and 16, the Blue Book-18189)
    ("N#CC(c1ccc(Cl)cc1)" + PYRROLIDINONE,
     "(4-chlorophenyl)(2-methyl-5-oxopyrrolidin-2-yl)acetonitrile"),
    # the second and the third prefix, each enclosed
    ("OC(c1ccccc1)(c1ccc(Cl)cc1)" + PYRROLIDINONE,
     "5-[(4-chlorophenyl)(hydroxy)(phenyl)methyl]-5-methyl-2-oxopyrrolidine"),
    ("OC(C1CCCCC1)(C1CCCO1)" + PYRROLIDINONE,
     "5-[cyclohexyl(hydroxy)(oxolan-2-yl)methyl]-5-methyl-2-oxopyrrolidine"),
    # the first prefix is simple and stays bare, the later one with marks of its own stays as it is
    ("OC(c1ccccc1)" + PYRROLIDINONE,
     "5-[hydroxy(phenyl)methyl]-5-methyl-2-oxopyrrolidine"),
])
def test_the_prefixes_of_a_one_carbon_substituent_after_the_first_are_enclosed(smiles, expected):
    name = _best_effort(smiles)
    assert name == expected
    assert_full_rt(name, smiles)


@pytest.mark.parametrize("smiles,expected", [
    # the ring prefix opens with its own descriptor: '[(1S)-...]', and the group is cited in braces
    ("OC([C@@H]1C=CCCC1)" + PYRROLIDINONE,
     "5-{[(1S)-cyclohex-2-en-1-yl](hydroxy)methyl}-5-methyl-2-oxopyrrolidine"),
    ("OC([C@@H]1CCCO1)" + PYRROLIDINONE,
     "5-{hydroxy[(2S)-oxolan-2-yl]methyl}-5-methyl-2-oxopyrrolidine"),
    # the one-carbon group has a descriptor of its own: it stands inside the marks of the group,
    # and the group holds a '[...]', so its marks are braces, not '[' directly inside '['
    ("O[C@H]([C@@H]1CCCO1)" + PYRROLIDINONE,
     "5-{(1S)-hydroxy[(2S)-oxolan-2-yl]methyl}-5-methyl-2-oxopyrrolidine"),
    ("OC(c1ccc(C(F)(F)F)cc1)" + PYRROLIDINONE,
     "5-{hydroxy[4-(trifluoromethyl)phenyl]methyl}-5-methyl-2-oxopyrrolidine"),
])
def test_a_ring_prefix_with_a_descriptor_takes_the_next_mark_of_the_nesting_order(smiles, expected):
    name = _best_effort(smiles)
    assert name == expected
    assert_full_rt(name, smiles)


def test_the_substituent_of_an_omuralide_type_ring_cites_the_second_prefix_in_parentheses():
    # a ChEBI row of the salinosporamide family (a 5-oxopyrrolidine-2-carboxylic acid with the
    # (cyclohexenyl)(hydroxy)methyl group); the group has a stereocentre of its own and a ring
    # prefix with one
    smiles = "CCCCCC[C@H]1C(=O)N[C@](C(=O)O)([C@@H](O)[C@@H]2C=CCCC2)[C@@]1(C)O"
    name = _best_effort(smiles)
    assert "{(1S)-[(1S)-cyclohex-2-en-1-yl](hydroxy)methyl}" in name, name
    assert_full_rt(name, smiles)


@pytest.mark.parametrize("smiles,expected", [
    # (:7304): methanone, methanol: the second prefix is enclosed, the first with its
    # locant stays as it is
    ("O=C(c1ccccc1)c1ccc(Cl)cc1", "(4-chlorophenyl)(phenyl)methanone"),
    ("OC(c1ccccc1)c1ccc(Cl)cc1", "(4-chlorophenyl)(phenyl)methanol"),
    ("OC(c1ccccc1)C1CC=CCC1", "(cyclohex-3-en-1-yl)(phenyl)methanol"),
    ("O=C(c1ccccc1)c1cc(C)n(C)n1", "(1,5-dimethyl-1H-pyrazol-3-yl)(phenyl)methanone"),
    # 'cyclopropyl(phenyl)methanol (PIN)' (:7302): both prefixes simple, unchanged
    ("OC(c1ccccc1)C1CC1", "cyclopropyl(phenyl)methanol"),
    # a multiplied prefix keeps its multiplier outside the marks (:7272, last sentence)
    ("OC(c1ccc(Cl)cc1)(c1ccc(Cl)cc1)C1CC1", "bis(4-chlorophenyl)(cyclopropyl)methanol"),
    # unchanged: two prefixes of a retained parent, 'bromo(chloro)fluoromethane'-type lists
    ("OC(=O)C(O)(c1ccccc1)c1ccc(Cl)cc1", "(4-chlorophenyl)(hydroxy)(phenyl)acetic acid"),
])
def test_the_whole_molecule_assembler_encloses_the_second_prefix_of_a_one_position_parent(
        smiles, expected):
    name = Orthonym(style="pin").name_tiered(smiles)["name"]
    assert name == expected
    assert_full_rt(name, smiles)
