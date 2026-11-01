"""A hydroperoxide on a chain that carries a ring, and on a monosubstituted ring, (c)).

 "Hydroperoxides" (the Blue Book, the sentence at:27937): "The suffix 'peroxol' is now introduced to name the group -OOH
... Method (1) leads to preferred IUPAC names", with 'pyrrolidine-1-peroxol (PIN)' and
'1,2,3,4-tetrahydronaphthalene-1-peroxol (PIN)' (:27948,:27959); the suffix decorates a skeletal atom as '-ol' does, so
the CH2 between a ring and the -OOH is the parent ('cyclohexylmethanol (PIN)',:6692, with one more oxygen:
'cyclohexylmethaneperoxol'). (c) (:2913): "The locant '1' is omitted... in monosubstituted homogeneous
monocyclic rings", 'cyclohexanethiol (PIN)': 'cyclohexaneperoxol'.

Before: the hydroperoxide was no skeletal suffix, so the ring was the parent and the CH2 was dropped
('cyclopropane-1-peroxol' for C1CC1COO, a different molecule that the read-back refused), and the ring peroxol
kept its '1' ('cyclohexane-1-peroxol', refused by the spelling check of.

Changed asserted values: 'cyclohexane-1-peroxol' -> 'cyclohexaneperoxol' in test_heterol_suffix.py and
test_w2e_p1fg_ring_peroxol.py (quote above; OPSIN reads 'cyclohexaneperoxol' back to the structure; the rows
below fail without the change); the 'N-(2-hydroperoxyethyl)aniline' row of test_sl_g2_amine_ring_parent.py moved
to its PIN '2-anilinoethane-1-peroxol'.
"""
import pytest

import orthonym.rules.parent_selection as parent_selection
from orthonym import Orthonym

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


def _tiered(smiles: str):
    row = Orthonym().name_tiered(smiles)
    return row["name"], row["tier"]


@pytest.mark.parametrize("smiles, expected", [
    # the -OOH is on the chain carbon: the chain is the parent, the ring a substituent
    ("C1CC1COO", "cyclopropylmethaneperoxol"),
    ("OOCC1CCCCC1", "cyclohexylmethaneperoxol"),
    ("OOCc1ccccc1", "phenylmethaneperoxol"),
    ("OOCc1ccccn1", "(pyridin-2-yl)methaneperoxol"),
    ("OOCC1CCOCC1", "(oxan-4-yl)methaneperoxol"),
    ("OOCCC1CC1", "2-cyclopropylethane-1-peroxol"),
    ("OOC(C)c1ccccc1", "1-phenylethane-1-peroxol"),
    ("OOC(C)(C)c1ccccc1", "2-phenylpropane-2-peroxol"),
    ("OOC(C)C1CC1", "1-cyclopropylethane-1-peroxol"),
    # a group senior to the hydroperoxide stays the principal group
    ("OOCC(=O)c1ccccc1", "2-hydroperoxy-1-phenylethan-1-one"),     # the book's example,:27953
    # the hydroperoxide ranks above the amine class 18 before 19)
    ("OOCCNc1ccccc1", "2-anilinoethane-1-peroxol"),
])
def test_the_chain_carbon_with_the_peroxol_is_the_parent(smiles, expected):
    assert _tiered(smiles) == (expected, "pin_verified")


@pytest.mark.parametrize("smiles, expected", [
    ("OOC1CCCCC1", "cyclohexaneperoxol"),
    ("OOC1CC1", "cyclopropaneperoxol"),
    ("OOC1CCCC1", "cyclopentaneperoxol"),
    ("OOC1CCCCCC1", "cycloheptaneperoxol"),
    # the locant stays where the ring has another substituent or is not homogeneous (:27959)
    ("CC1CCCCC1OO", "2-methylcyclohexane-1-peroxol"),
    ("OOC1CCCCC1OO", "cyclohexane-1,2-diperoxol"),
    ("OOC1CCCN1", "pyrrolidine-2-peroxol"),
    ("OOC1CCCc2ccccc12", "1,2,3,4-tetrahydronaphthalene-1-peroxol"),  # the book's example,:27948
])
def test_the_ring_peroxol_locant_follows_the_monosubstituted_ring_licence(smiles, expected):
    assert _tiered(smiles) == (expected, "pin_verified")


@pytest.mark.parametrize("smiles, expected", [
    ("CCOO", "ethaneperoxol"),                                          #:24316
    ("CC(C)(CCN(C)C)OO", "4-(dimethylamino)-2-methylbutane-2-peroxol"),  #:27955
    ("OCC1CC1", "cyclopropylmethanol"),
])
def test_the_controls_keep_their_names(smiles, expected):
    assert _tiered(smiles) == (expected, "pin_verified")


def test_the_tests_fail_without_the_skeletal_suffix(monkeypatch):
    """The mutation check: with the hydroperoxide off the list the ring is the parent again."""
    monkeypatch.setattr(parent_selection, "SKELETAL_SUFFIX_PGS",
                        parent_selection.SKELETAL_SUFFIX_PGS - {"hydroperoxide"})
    assert _tiered("C1CC1COO") != ("cyclopropylmethaneperoxol", "pin_verified")
