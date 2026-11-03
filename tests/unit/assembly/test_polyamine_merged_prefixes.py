"""A polyamine cites an identical simple prefix on its nitrogens and on its chain as ONE multiplied prefix.

: identical prefixes are one multiplied prefix with all their locants; the N-locants come first
('N,4-dimethyl-N-(3-methylphenyl)benzamide', the Blue Book; 'N,2-dimethylpropanamide',:21624).
``composition_primitives.merged_n_and_c_entries``; the old form 'N1,N4-dimethyl-2-methyl...' failed the check.
"""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles, name", [
    ("CNCC(C)CCNC", "N1,N4,2-trimethylbutane-1,4-diamine"),
    ("CNCC(C)(C)CNC", "N1,N3,2,2-tetramethylpropane-1,3-diamine"),
    ("CN(C)CC(C)(C)CN(C)C", "N1,N1,N3,N3,2,2-hexamethylpropane-1,3-diamine"),
    ("CNCC(Cl)CCNC", "2-chloro-N1,N4-dimethylbutane-1,4-diamine"),    # different names stay apart
], ids=lambda v: v)
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
def test_one_multiplied_prefix(smiles, name, tier):
    row = Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)
    assert row["name"] == name, row
    assert row["tier"] == "pin_verified", row


@pytest.mark.parametrize("tier", ["pin"])
def test_an_n_locant_that_moves_with_the_numbering_is_not_merged(tier):
    """'N4,2-dimethylpentane-2,4-diamine (PIN)' (the Blue Book): reversing the chain moves the N-locant
    (N2 <-> N4), and the book's choice between the numberings is not built, so no merged name is labelled the PIN."""
    row = Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered("CNC(C)CC(C)(C)N")
    assert row["name"] != "N2,4-dimethylpentane-2,4-diamine", row
    assert row["tier"] != "pin_verified" or row["name"] == "N4,2-dimethylpentane-2,4-diamine", row
