"""A substituted fused-heterocycle '-yl' prefix joins its inner prefixes to the ring stem without
a hyphen when the stem starts with a letter or a bracket (lane W, item 20).

 "Hyphens" (the Blue Book,:6938): hyphens are used in substitutive names "(a) to
separate locants from words or word fragments". A stem that starts with a letter follows the
prefixes directly: '7-methylnaphthalen-2-yl (preferred prefix)' "Compound substituted
substituent groups",:16210), '(7-chloroquinolin-4-yl)' (:4675); a stem that starts with a
locant keeps the hyphen: '5-methyl-1H-indol-3-yl', '2-methyl-1-benzothiophen-5-yl'.
The writer, ``get_substituted_fused_het_prefix``, used to put a hyphen in front of every stem and
the name was labelled pin_verified."""
import pytest

from tests.support.default_tier import default_tier_row
from tests.support.rt_assert import assert_full_rt, name_best_effort

pytestmark = pytest.mark.opsin_gate

# (smiles, name); the name main gave is in the comment
NO_HYPHEN = [
    ("Cc1cc2cc(CC(=O)O)c(O)cc2cn1", "(7-hydroxy-3-methylisoquinolin-6-yl)acetic acid"),       # 3-methyl-isoquinolin
    ("Cc1cc2cc(CCC(=O)O)ccc2cn1", "3-(3-methylisoquinolin-6-yl)propanoic acid"),
    ("Cc1cnc2ccc(CCC(=O)O)cc2c1", "3-(3-methylquinolin-6-yl)propanoic acid"),
    ("Cc1nc2ccccn2c1CC(=O)O", "(2-methylimidazo[1,2-a]pyridin-3-yl)acetic acid"),             # 2-methyl-imidazo[...]
    ("Cc1cc2c(s1)nccc2CC(=O)O", "(2-methylthieno[2,3-b]pyridin-4-yl)acetic acid"),
    ("Cc1cc2c(cc1CC(=O)O)oc1ccccc12", "(2-methyldibenzo[b,d]furan-3-yl)acetic acid"),
    # an inner prefix in its own enclosing marks; the book joins a bracket without a hyphen
    # (4'-cyano[1,1'-biphenyl], the Blue Book)
    ("OC(=O)Cc1ccc2ncc(C(F)(F)F)cc2c1", "[3-(trifluoromethyl)quinolin-6-yl]acetic acid"),
]

# a stem that starts with a locant keeps the hyphen; main's names, unchanged
KEEPS_HYPHEN = [
    ("Cc1ccc2c(c1)ccn2CC(=O)O", "(5-methyl-1H-indol-1-yl)acetic acid"),
    ("Cc1cc2c(s1)ccc(CC(=O)O)c2", "(2-methyl-1-benzothiophen-5-yl)acetic acid"),
    ("Cc1ccc2c(c1)[nH]c1ccc(CC(=O)O)cc12", "(2-methyl-9H-carbazol-6-yl)acetic acid"),
]


@pytest.mark.parametrize("smiles,expected", NO_HYPHEN + KEEPS_HYPHEN)
def test_prefix_hyphenation_both_tiers(smiles, expected):
    row = default_tier_row(smiles)
    assert row["name"] == expected and row["tier"] == "pin_verified", row
    assert name_best_effort(smiles)["name"] == expected
    assert_full_rt(expected, smiles)


def test_no_inner_prefix_is_unchanged():
    from orthonym.data.fused_heterocycles import (
        FUSED_HETEROCYCLE_PREFIX_STEMS, get_substituted_fused_het_prefix)
    core = next(k for k, v in FUSED_HETEROCYCLE_PREFIX_STEMS.items() if v == "isoquinolin")
    assert get_substituted_fused_het_prefix(core, 0, {0: 6}, "") == "(isoquinolin-6-yl)"


@pytest.mark.parametrize("stem,inner,locant,expected", [
    ("isoquinolin", "3-methyl-", 6, "(3-methylisoquinolin-6-yl)"),
    ("isoquinolin", "7-hydroxy-3-methyl", 6, "(7-hydroxy-3-methylisoquinolin-6-yl)"),
    ("1H-indol", "5-methyl-", 3, "(5-methyl-1H-indol-3-yl)"),
    ("[1,2,4]triazolo[1,5-a]pyrimidin", "2-methyl", 6, "(2-methyl[1,2,4]triazolo[1,5-a]pyrimidin-6-yl)"),
    ("quinolin", "4-(dimethylamino)", 2, "(4-(dimethylamino)quinolin-2-yl)"),
])
def test_writer_joins_by_the_next_character(monkeypatch, stem, inner, locant, expected):
    from orthonym.data import fused_heterocycles as fh
    monkeypatch.setitem(fh.FUSED_HETEROCYCLE_PREFIX_STEMS, "X", stem)
    assert fh.get_substituted_fused_het_prefix("X", 0, {0: locant}, inner) == expected
