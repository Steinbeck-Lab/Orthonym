"""The substitutive N-hydroxy imine is the PIN of a configured oxime and of a bis-oxime.

 'Oximes' (the Blue Book): "In these recommendations preferred IUPAC names for oximes are
generated substitutively as N-hydroxy derivatives of imines rather than by functional class nomenclature as
in previous recommendations" (':38468 N-hydroxypentan-2-imine (PIN)', ':38472 N2,N3-dihydroxybutane-2,3-diimine
(PIN)', ':47668 (2E,3Z)-N2,N3-dihydroxypentane-2,3-diimine (PIN)'). The builder used to leave an oxime with a
configured C=N, a stereocentre or a second oxime group to the functional class name, which is not the PIN
(``test_oxime_pin_label``).

The descriptor of the C=N stands at the front of the name,:44643): "preceded by a numerical or letter
locant to describe the position of the stereogenic unit when such locants are present". The name cites a locant
for the carbon of '(2Z)-N-hydroxybutan-2-imine' and '(1E)-N-hydroxy-1-phenylmethanimine' (:50452 prints
'(1E)-1-[...]methanimine') and none for that of ethanimine: '(E)-N-hydroxyethanimine', as the book prints
'(Z)-N-hydroxy(4-chlorophenyl)(phenyl)methanimine (PIN)' (:47666).

The standard InChIKey sees the configuration of the C=N of an oxime (the four configurations of the
bis-oxime below have four keys), so the full-key read-back of the engine catches a flipped descriptor; the
isomeric read-back of these tests, OPSIN's reading of the name against the input's isomeric SMILES, is the
independent check of every descriptor, with the other configuration's name as the known negative.
"""
import pytest
from rdkit import Chem

from tests.support.pin_tiers import assert_pin_at_both_tiers, name_breadth, name_default
from tests.support.rt_assert import _canon, _independent_parse, name_is_rt_exact

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

#: (SMILES, the PIN)
ROWS = [
    ("C/C=N/O", "(E)-N-hydroxyethanimine"),
    ("C/C=N\\O", "(Z)-N-hydroxyethanimine"),
    ("C/C(CC)=N/O", "(2Z)-N-hydroxybutan-2-imine"),
    ("C/C(CC)=N\\O", "(2E)-N-hydroxybutan-2-imine"),
    ("ClCC/C=N/O", "(1E)-3-chloro-N-hydroxypropan-1-imine"),
    ("CC(C)/C=N/O", "(1E)-N-hydroxy-2-methylpropan-1-imine"),
    ("O/N=C/c1ccccc1", "(1E)-N-hydroxy-1-phenylmethanimine"),
    ("O/N=C\\c1ccccc1", "(1Z)-N-hydroxy-1-phenylmethanimine"),
    ("CC(=NO)C(C)=NO", "N2,N3-dihydroxybutane-2,3-diimine"),
    ("CCC(=N/O)/C(C)=N/O", "(2E,3Z)-N2,N3-dihydroxypentane-2,3-diimine"),
    ("CCC(=N\\O)/C(C)=N/O", "(2E,3E)-N2,N3-dihydroxypentane-2,3-diimine"),
    ("CCC(=N\\O)/C(C)=N\\O", "(2Z,3E)-N2,N3-dihydroxypentane-2,3-diimine"),
    ("CCC(=N/O)/C(C)=N\\O", "(2Z,3Z)-N2,N3-dihydroxypentane-2,3-diimine"),
]


def _isomeric_read_back(name, smiles):
    """True iff OPSIN reads ``name`` to the input's isomeric canonical SMILES."""
    read = _independent_parse(name, smiles)
    return bool(read) and _canon(read) == _canon(smiles)


@pytest.mark.parametrize("smiles, pin", ROWS, ids=[r[1] for r in ROWS])
def test_the_n_hydroxy_imine_is_the_pin_at_both_tiers(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)
    assert _isomeric_read_back(pin, smiles), (smiles, pin)


def test_the_isomeric_read_back_sees_each_configuration():
    """The check discriminates: the name of the other configuration reads back to another isomeric SMILES
    (a known positive and a known negative for every bond of the bis-oxime)."""
    smiles = "CCC(=N/O)/C(C)=N/O"
    assert _isomeric_read_back("(2E,3Z)-N2,N3-dihydroxypentane-2,3-diimine", smiles)
    for other in ("(2E,3E)", "(2Z,3E)", "(2Z,3Z)"):
        assert not _isomeric_read_back(other + "-N2,N3-dihydroxypentane-2,3-diimine", smiles), other
    assert not _isomeric_read_back("(Z)-N-hydroxyethanimine", "C/C=N/O")
    assert not _isomeric_read_back("N-hydroxyethanimine", "C/C=N/O")      # states none where the input has one


@pytest.mark.parametrize("smiles, pin", [ROWS[0], ROWS[2], ROWS[6], ROWS[9]],
                         ids=["ethanimine", "butan-2-imine", "phenylmethanimine", "bis-oxime"])
def test_every_spelling_gets_the_pin(smiles, pin):
    mol = Chem.MolFromSmiles(smiles)
    spellings = list(dict.fromkeys([smiles, Chem.MolToSmiles(mol),
                                    *Chem.MolToRandomSmilesVect(mol, 5, randomSeed=7)]))
    assert len(spellings) >= 3
    assert {(name_default(s).get("name"), name_breadth(s).get("name")) for s in spellings} == {(pin, pin)}


@pytest.mark.parametrize("smiles, name", [
    ("C/C=N/O", "(E)-N-hydroxyethanimine"),
    ("C/C(CC)=N/O", "(2Z)-N-hydroxybutan-2-imine"),
])
def test_the_descriptor_of_the_name_is_the_one_of_the_input(smiles, name):
    """The full key sees the C=N of an oxime: the flipped descriptor reads back to another molecule."""
    assert name_is_rt_exact(name, smiles)
    flipped = name.replace("(E)", "(Z)") if "(E)" in name else name.replace("(2Z)", "(2E)")
    assert not name_is_rt_exact(flipped, smiles)


@pytest.mark.parametrize("smiles", [
    "C[C@@]12CC[C@@H](C1(C)C)CC2=NO",           # the imine of a bicyclic parent has no PIN at the default tier
    "O/N=C(/c1ccccc1)c1ccc(Cl)cc1",             # the book's order of the prefixes (:47666) is not 's
    "CC(=NO)C=C",                               # an unsaturated parent
])
def test_a_parent_the_builder_does_not_cite_stays_below_the_pin(smiles):
    d = name_default(smiles)
    assert d.get("tier") == "abstain" and d.get("limit_code") == "NO_VERIFIED_PIN", d
    b = name_breadth(smiles)
    assert b.get("tier") == "systematic_verified" and b.get("is_pin") is False, b
    assert name_is_rt_exact(b["name"], smiles)
