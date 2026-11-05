"""Fix round of the merged lane/l17 (item 12): the N-substituted sulfonamido and urea prefixes of an acyclic parent.

 (the Blue Book) 'Method (1) generates preferred IUPAC names': '(N-cyclopropylmethanesulfonamido)acetic acid'
(:33029); (:33338) the urea prefixes are formed systematically: '2-[(methylcarbamoyl)amino]naphthalene-1-carboxylic
acid (PIN)' (:33354). The PIN tier had no name for the N-substituted forms on a chain parent: the chain walk
(``composer._check_for_acylamino``) spelled an -N(R')-SO2-R branch '(methylamino)' (the alkylamino fallback counts the carbon
branches of the nitrogen and has no place for the sulfonyl), and the urea prefix of the functional-group loop dropped the group of
the proximal nitrogen ('3-(carbamoylamino)propanoic acid' for the N-methyl urea); the gate suppressed both, as names of another
molecule.
"""
import pytest

from orthonym.cli import _tiered_namer
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


def _row(smiles, tier):
    return _tiered_namer("pin", tier).name_tiered(smiles)


_ACYCLIC_PINS = [
    ("CS(=O)(=O)N(C)CCC(=O)O", "3-(N-methylmethanesulfonamido)propanoic acid"),
    ("CS(=O)(=O)N(C)C(C)C(=O)O", "2-(N-methylmethanesulfonamido)propanoic acid"),
    ("CS(=O)(=O)N(CC)CCC(=O)O", "3-(N-ethylmethanesulfonamido)propanoic acid"),
    ("NC(=O)N(C)CCC(=O)O", "3-[carbamoyl(methyl)amino]propanoic acid"),
    ("CN(C)C(=O)N(C)CCC(=O)O", "3-[(dimethylcarbamoyl)(methyl)amino]propanoic acid"),
    ("NC(=O)N(CC)CC(CC(=O)O)C", "4-[carbamoyl(ethyl)amino]-3-methylbutanoic acid"),
    ("NC(=O)NCCC(=O)O", "3-(carbamoylamino)propanoic acid"),
    ("CNC(=O)NCCC(=O)O", "3-[(methylcarbamoyl)amino]propanoic acid"),
]


@pytest.mark.parametrize("smiles,name", _ACYCLIC_PINS)
def test_the_n_substituted_sulfonamido_and_urea_prefixes_of_an_acyclic_parent(smiles, name):
    """ (:32991) 'Method (1) generates preferred IUPAC names': '(N-cyclopropylmethanesulfonamido)acetic acid'
    (:33029); (:33338) the urea prefixes are formed systematically: '2-[(methylcarbamoyl)amino]naphthalene-1-
    carboxylic acid (PIN)' (:33354), the groups of both nitrogens cited on the amino prefix, 'carbamoyl(methyl)amino'. The
    name that held no sulfonyl ('3-(methylamino)propanoic acid') and the one that dropped the methyl of the proximal
    nitrogen ('3-(carbamoylamino)propanoic acid') described another molecule and never shipped."""
    for tier in ("pin", "best-effort"):
        row = _row(smiles, tier)
        assert row["name"] == name, row
        assert row["tier"] == "pin_verified" and row["is_pin"], row
    assert_full_rt(name, smiles)


def test_a_thiourea_with_a_substituted_proximal_nitrogen_is_not_named_with_a_lossy_prefix():
    """'3-(carbamothioylamino)propanoic acid' would drop the methyl of ``NC(=S)N(C)CCC(=O)O``: the prefix is refused
    (the molecule is left to the producers that name it whole), never emitted for another molecule."""
    smiles = "NC(=S)N(C)CCC(=O)O"
    for tier in ("pin", "best-effort"):
        row = _row(smiles, tier)
        assert row["name"] != "3-(carbamothioylamino)propanoic acid", row
        if row["name"] and row["tier"] != "abstain":
            assert_full_rt(row["name"], smiles)


