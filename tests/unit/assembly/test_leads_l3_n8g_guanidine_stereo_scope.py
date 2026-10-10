"""Leads program L3, item N8g: a guanidine's stereodescriptors belong inside its
N-substituent prefixes, never in a numeric front-of-name block.

Guanidine has no numbered skeleton: "Guanidine and its derivatives",
 (the Blue Book) "the locants N, N' and N'' are used in preferred
IUPAC names. The locants 1, 2, and 3 have been used but are no longer recommended".
 "NAMING OF STEREOISOMERS" (:44643): descriptors that "relate to substituent groups
... are cited at the front of the corresponding prefix". The handler used to leave the
parent scope undeclared, both candidate scopes agreed, and a numeric block was prepended:
``(1E)-N,N'-di[(1E)-prop-1-en-1-yl]guanidine`` (OPSIN rejects it, 'Could not find bond
that:... locant=1'). The urea, thiourea, cyanamide and pnictogen handlers declare
``parent_scope='retained_no_locants'``; so does guanidine now.

Every expected PIN below is read back by a fresh OPSIN call to the input's full InChIKey.
"""
import pytest

from tests.support.pin_tiers import (
    assert_pin_at_both_tiers,
    name_breadth,
    name_default,
)
from tests.support.rt_assert import name_is_rt_exact

pytestmark = pytest.mark.opsin_gate

PIN_ROWS = [
    # the substituent descriptors are cited inside each prefix, no numeric front block
    ("C/C=C/NC(=N)N/C=C/C", "N,N'-di[(1E)-prop-1-en-1-yl]guanidine"),
    ("C/C=C/NC(=N/C)N/C=C/C", "N-methyl-N',N''-di[(1E)-prop-1-en-1-yl]guanidine"),
    # a stereocentre in an N-substituent is cited in its own prefix (control, unchanged)
    ("N=C(N)N[C@@H](C)c1ccc(cc1)C", "N-[(1S)-1-(4-methylphenyl)ethyl]guanidine"),
    ("N=C(N)NC[C@H](C)CC", "N-[(2R)-2-methylbutyl]guanidine"),
]


@pytest.mark.parametrize("smiles, pin", PIN_ROWS)
def test_guanidine_substituent_stereo_is_cited_inside_the_prefix(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles", [s for s, _ in PIN_ROWS])
def test_no_numeric_front_block_on_a_guanidine(smiles):
    # a '(1E)-N...guanidine' name cites a numeric locant against a parent that has none
    for res in (name_default(smiles), name_breadth(smiles)):
        name = res["name"]
        assert name.endswith("guanidine"), res
        assert not name.startswith("("), res


def test_ring_substituent_stereo_row_never_ships_a_non_pin_pin_verified():
    """'N=C(N)NC[C@H]1CCCO1': the Blue Book spelling cites the ring descriptor inside the
    bracket, 'N-{[(2R)-oxolan-2-yl]methyl}guanidine'. The old name '(2R)-N-[(oxolan-2-yl)
    methyl]guanidine' was labelled pin_verified with a front block that is no PIN spelling;
    the label is withdrawn (the default tier declines, as it does for the urea sibling) and
    the best-effort tier still gives a name that reads back to the full InChIKey."""
    smiles = "N=C(N)NC[C@H]1CCCO1"
    old = "(2R)-N-[(oxolan-2-yl)methyl]guanidine"
    d = name_default(smiles)
    assert not (d["name"] == old and d["tier"] == "pin_verified"), d
    assert d["tier"] != "pin_verified" or d["name"] == "N-{[(2R)-oxolan-2-yl]methyl}guanidine", d
    b = name_breadth(smiles)
    assert b["name"] != old, b
    assert b["name"] and name_is_rt_exact(b["name"], smiles), b


def test_urea_control_is_unchanged():
    # the scope was already declared for urea: its stereo row keeps its inside-the-prefix PIN
    smiles = "NC(=O)NC[C@H](C)CC"
    d = name_default(smiles)
    assert d["name"] == "[(2R)-2-methylbutyl]urea" and d["tier"] == "pin_verified", d
