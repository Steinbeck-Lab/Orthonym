"""Hydroperoxides on rings: the '-peroxol' suffix and the 'hydroperoxy' prefix.

 Hydroperoxides (the Blue Book): "The prefix 'hydroperoxy' is formed by concatenation to describe the
group –OOH as a substituent in the presence of a characteristic group having priority for citation as a suffix";
'2-hydroperoxyethan-1-ol (PIN)' (:28013), '1,2,3,4-tetrahydronaphthalene-1-peroxol (PIN)' (:27948) and 'CH3-CH2-OOH
ethaneperoxol (PIN)' THE SUFFIX 'PEROXOL', FOR –OOH,:24316). puts the hydroperoxides (class 18) below the hydroxy compounds
and their chalcogen analogues (class 17) and above the amines (class 19).

The fragment namer spelled a bare –OOH 'dioxidanyl' (a hydride prefix, no PIN vocabulary), so a ring
hydroperoxide got no PIN ('4-dioxidanylbenzoic acid' declined); the benzene namer had no '-peroxol' suffix
('hydroperoxybenzene'); the partially saturated fused carbocycle namer had none either, and dropped the '-ol' suffix
beside a ring –OOH ('2-hydroperoxy-6-hydroxy-1,2,3,4-tetrahydronaphthalene' at the best-effort tier).

Every name is read back by a fresh OPSIN call to the full standard InChIKey of the input.
"""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

PINS = [
    ("OOc1ccccc1", "benzeneperoxol"),
    ("OOc1ccc(C)cc1", "4-methylbenzene-1-peroxol"),
    ("OOc1ccc(OO)cc1", "benzene-1,4-diperoxol"),
    ("OOc1ccc(O)cc1", "4-hydroperoxyphenol"),                         # -ol (class 17) is senior
    ("OOc1ccc(C(=O)O)cc1", "4-hydroperoxybenzoic acid"),
    ("OOC1CCCc2ccccc21", "1,2,3,4-tetrahydronaphthalene-1-peroxol"),  # the book's own example
    ("OOC1CCc2ccccc2C1", "1,2,3,4-tetrahydronaphthalene-2-peroxol"),
    ("OOC1CCc2cc(C)ccc2C1", "6-methyl-1,2,3,4-tetrahydronaphthalene-2-peroxol"),
    ("OOC1CCc2ccccc2C1OO", "1,2,3,4-tetrahydronaphthalene-1,2-diperoxol"),
    ("OOc1ccc2c(c1)CCCC2", "5,6,7,8-tetrahydronaphthalene-2-peroxol"),
    ("OOC1CCc2cc(O)ccc2C1", "6-hydroperoxy-5,6,7,8-tetrahydronaphthalen-2-ol"),
    ("OOCC(=O)O", "hydroperoxyacetic acid"),
    ("OOC1(O)CCc2ccccc2C1", "2-hydroperoxy-1,2,3,4-tetrahydronaphthalen-2-ol"),   # -OH and -OOH on one ring carbon
    ("OOC1(O)CCCc2ccccc21", "1-hydroperoxy-1,2,3,4-tetrahydronaphthalen-1-ol"),
    # (e): the suffix locant is cited inside a multiplicative name ('3,4'-disulfanediyldi(benzene-1-thiol)',
    #:27593; '4,4'-oxydi(benzene-1-sulfonic acid) (PIN)',:5827)
    ("OOc1ccc(cc1)Oc1ccc(OO)cc1", "4,4'-oxydi(benzene-1-peroxol)"),
]


@pytest.mark.parametrize("smiles,name", PINS)
def test_ring_hydroperoxides_get_their_pin_at_every_tier(smiles, name):
    for tier in ("pin", "best-effort"):
        row = Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)
        assert (row["name"], row["tier"]) == (name, "pin_verified"), (tier, row)
    assert_full_rt(name, smiles)


@pytest.mark.parametrize("smiles,name", [
    ("CC(C)(C)OOC(C)(C)C", "2-(tert-butylperoxy)-2-methylpropane"),   # a substituted peroxy group is no 'hydroperoxy'
    ("COOc1ccc(C(=O)O)cc1", "4-(methylperoxy)benzoic acid"),
])
def test_substituted_peroxy_groups_keep_their_prefix(smiles, name):
    row = Orthonym(style="pin", **_emit_tier_flags("pin")).name_tiered(smiles)
    assert row["name"] == name, row


def test_the_fragment_prefix_of_a_bare_hydroperoxy_group():
    from rdkit import Chem

    from orthonym.assembly.hetero_chain_prefix import _is_bare_hydroperoxy
    mol = Chem.MolFromSmiles("OOc1ccccc1")
    assert _is_bare_hydroperoxy(mol, [1, 0], 1)
    assert not _is_bare_hydroperoxy(Chem.MolFromSmiles("COOc1ccccc1"), [0, 1, 2], 2)
    assert not _is_bare_hydroperoxy(Chem.MolFromSmiles("[18OH]Oc1ccccc1"), [1, 0], 1)
